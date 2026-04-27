"""
Miracle 1.0.2 - Adaptive Evaluator
====================================
Factor and pattern evaluation, overfitting detection

Features:
1. Factor IC evaluation
2. Pattern performance evaluation
3. Overfitting detection
4. Performance reporting
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from learner import (
    WalkForwardValidator,
    calc_information_coefficient,
)

logger = logging.getLogger("miracle.adaptive_learner.evaluator")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================
# Factor Evaluator
# ============================================================

class FactorEvaluator:
    """
    因子评估器
    
    评估各因子的预测能力和表现
    """
    
    def __init__(self, min_sample_size: int = 20):
        self.min_sample_size = min_sample_size
        self.factor_performance = defaultdict(lambda: {
            "signals": [],
            "returns": [],
            "ic_history": []
        })
    
    def update(self, factor_name: str, signal: float, actual_return: float):
        """
        更新因子表现
        
        Args:
            factor_name: 因子名称
            signal: 因子信号值
            actual_return: 实际收益
        """
        perf = self.factor_performance[factor_name]
        perf["signals"].append(signal)
        perf["returns"].append(actual_return)
        
        # 保持最近100个样本
        if len(perf["signals"]) > 100:
            perf["signals"] = perf["signals"][-100:]
            perf["returns"] = perf["returns"][-100:]
    
    def get_ic(self, factor_name: str) -> Tuple[float, float]:
        """
        获取因子IC
        
        Args:
            factor_name: 因子名称
            
        Returns:
            (ic, p_value)
        """
        perf = self.factor_performance.get(factor_name)
        if not perf or len(perf["signals"]) < self.min_sample_size:
            return 0.0, 1.0
        
        return calc_information_coefficient(perf["signals"], perf["returns"])
    
    def get_report(self) -> Dict[str, Any]:
        """
        获取所有因子IC报告
        
        Returns:
            各因子的IC统计报告
        """
        report = {}
        for factor_name, perf in self.factor_performance.items():
            if len(perf["signals"]) >= self.min_sample_size:
                ic, p_value = calc_information_coefficient(perf["signals"], perf["returns"])
                report[factor_name] = {
                    "ic": ic,
                    "p_value": p_value,
                    "sample_size": len(perf["signals"]),
                    "ic_history_avg": float(np.mean(perf["ic_history"])) if perf["ic_history"] else 0.0
                }
        return report


# ============================================================
# Pattern Evaluator
# ============================================================

class PatternEvaluator:
    """
    模式评估器
    
    评估各交易模式的表现
    """
    
    def __init__(self, min_sample_size: int = 5):
        self.min_sample_size = min_sample_size
        self.pattern_performance = defaultdict(lambda: {
            "total": 0,
            "wins": 0,
            "total_rr": 0.0,
            "win_rate": 0.5
        })
    
    def update(self, pattern_key: str, won: bool, actual_rr: float):
        """
        更新模式表现
        
        Args:
            pattern_key: 模式键
            won: 是否盈利
            actual_rr: 实际盈亏比
        """
        perf = self.pattern_performance[pattern_key]
        perf["total"] += 1
        if won:
            perf["wins"] += 1
        perf["total_rr"] += actual_rr
        if perf["total"] > 0:
            perf["win_rate"] = perf["wins"] / perf["total"]
    
    def is_allowed(self, pattern_key: str) -> bool:
        """
        检查模式是否允许交易
        
        Args:
            pattern_key: 模式键
            
        Returns:
            是否允许交易
        """
        perf = self.pattern_performance.get(pattern_key)
        
        # 样本不足，允许交易
        if not perf or perf["total"] < self.min_sample_size:
            return True
        
        # 胜率低于40%，禁止交易
        if perf["win_rate"] < 0.4:
            logger.warning(f"Pattern {pattern_key} blocked due to low win rate: {perf['win_rate']:.2%}")
            return False
        
        return True
    
    def get_stats(self, pattern_key: str) -> Dict[str, Any]:
        """
        获取模式统计
        
        Args:
            pattern_key: 模式键
            
        Returns:
            模式统计信息
        """
        perf = self.pattern_performance.get(pattern_key, {
            "total": 0, "wins": 0, "total_rr": 0.0, "win_rate": 0.5
        })
        avg_rr = perf["total_rr"] / perf["total"] if perf["total"] > 0 else 0.0
        return {
            "pattern": pattern_key,
            "total_trades": perf["total"],
            "wins": perf["wins"],
            "losses": perf["total"] - perf["wins"],
            "win_rate": perf["win_rate"],
            "avg_rr": avg_rr
        }
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """获取所有模式统计"""
        return [
            self.get_stats(pk)
            for pk in self.pattern_performance.keys()
        ]


# ============================================================
# Overfitting Detector
# ============================================================

class OverfittingDetector:
    """
    过拟合检测器
    
    使用Walk-Forward分析检测策略是否过拟合
    """
    
    def __init__(self, walk_forward_validator: WalkForwardValidator = None):
        self.walk_forward_validator = walk_forward_validator or WalkForwardValidator(
            train_window=50, test_window=20
        )
    
    def detect(self, factor_evaluator: FactorEvaluator) -> Dict[str, Any]:
        """
        检测过拟合
        
        Args:
            factor_evaluator: 因子评估器
            
        Returns:
            {
                "is_overfitting": bool,
                "train_ic_avg": float,
                "test_ic_avg": float,
                "ic_decay": float,
                "reason": str
            }
        """
        # 准备数据
        all_signals = []
        all_returns = []
        for perf in factor_evaluator.factor_performance.values():
            all_signals.extend(perf["signals"])
            all_returns.extend(perf["returns"])
        
        if len(all_signals) < 50:
            return {
                "is_overfitting": False,
                "reason": "样本不足，无法判断",
                "train_ic_avg": 0.0,
                "test_ic_avg": 0.0,
                "ic_decay": 0.0
            }
        
        # Walk-Forward验证
        data = [{"signal": s, "return": r} for s, r in zip(all_signals, all_returns)]
        
        def strategy_func(window_data):
            """Walk-forward验证：训练参数在train窗口，测试在test窗口"""
            n = len(window_data)
            if n < 10:
                return {"train_ic": 0.0, "test_ic": 0.0}
            
            split = n // 2
            train_window = window_data[:split]
            test_window = window_data[split:]
            
            # Train: compute IC on train window (in-sample)
            train_signals = [d["signal"] for d in train_window]
            train_returns = [d["return"] for d in train_window]
            train_ic, _ = calc_information_coefficient(train_signals, train_returns)
            
            # Test: use trained "params" (mean signal from train) on test window (out-of-sample)
            train_mean_signal = sum(train_signals) / len(train_signals)
            test_signals = [d["signal"] for d in test_window]
            test_returns = [d["return"] for d in test_window]
            test_ic, _ = calc_information_coefficient(test_signals, test_returns)
            
            return {
                "train_ic": train_ic,
                "test_ic": test_ic
            }
        
        wf_results = self.walk_forward_validator.validate(strategy_func, data, n_windows=5)
        
        ic_decay = wf_results.get("ic_decay", 0.0)
        is_overfitting = ic_decay > 0.3  # IC衰减超过30%认为过拟合
        
        return {
            "is_overfitting": is_overfitting,
            "train_ic_avg": wf_results.get("train_ic_avg", 0.0),
            "test_ic_avg": wf_results.get("test_ic_avg", 0.0),
            "ic_decay": ic_decay,
            "reason": "IC衰减超过30%" if is_overfitting else "未检测到过拟合"
        }


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "FactorEvaluator",
    "PatternEvaluator",
    "OverfittingDetector",
]
