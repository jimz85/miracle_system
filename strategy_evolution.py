"""
Miracle 1.0.2 - Strategy Evolution
===================================
Strategy evolution, factor weight adjustment, pattern recognition

Features:
1. Dynamic factor weight adjustment (with bounds)
2. Pattern recognition and tracking
3. Strategy evolution based on performance
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from datetime import datetime
from collections import defaultdict

from learner import (
    DecisionJournal,
    DecisionJournalEntry,
    WalkForwardValidator,
    calc_information_coefficient,
)

logger = logging.getLogger("miracle.adaptive_learner.strategy_evolution")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================
# Strategy Evolution
# ============================================================

class StrategyEvolution:
    """
    策略演化器
    
    负责:
    1. 因子权重动态调整（有上下限）
    2. 模式识别
    3. 策略学习记录
    """
    
    def __init__(self, 
                 config: Dict,
                 min_weight: float = 0.05,
                 max_weight: float = 1.0,
                 min_sample_size: int = 20):
        """
        Args:
            config: 交易配置字典
            min_weight: 因子权重下限
            max_weight: 因子权重上限
            min_sample_size: 最少样本数
        """
        self.config = config
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_sample_size = min_sample_size
        
        # 因子权重历史（用于学习追踪）
        self.factor_weight_history: List[Dict[str, Any]] = []
    
    def adjust_factor_weights(self, 
                             current_factors: Dict[str, Dict],
                             factor_evaluator) -> Dict[str, float]:
        """
        基于IC表现调整因子权重
        
        Args:
            current_factors: 当前因子配置 {因子名: {weight, ...}}
            factor_evaluator: 因子评估器
            
        Returns:
            新的因子权重字典
        """
        weights = {}
        
        for factor_name, perf in factor_evaluator.factor_performance.items():
            if len(perf["signals"]) < self.min_sample_size:
                # 样本不足，保持默认权重
                weights[factor_name] = current_factors.get(factor_name, {}).get("weight", 0.1)
                continue
            
            # 计算IC
            ic, p_value = calc_information_coefficient(perf["signals"], perf["returns"])
            perf["ic_history"].append(ic)
            
            # 根据IC调整权重
            default_weight = current_factors.get(factor_name, {}).get("weight", 0.1)
            
            if ic < 0.02:  # IC太低，因子无效
                new_weight = default_weight * 0.5  # 降权50%
                logger.info(f"Factor {factor_name} IC too low ({ic:.4f}), reducing weight")
            elif ic > 0.05:  # IC不错
                new_weight = default_weight * 1.1  # 加权10%
                logger.info(f"Factor {factor_name} IC good ({ic:.4f}), increasing weight")
            else:
                new_weight = default_weight  # 保持
            
            # 限制上下限
            weights[factor_name] = max(self.min_weight, min(self.max_weight, new_weight))
        
        # 确保所有因子都有权重
        for factor_name in current_factors:
            if factor_name not in weights:
                weights[factor_name] = current_factors[factor_name].get("weight", 0.1)
        
        # 记录权重历史
        self.factor_weight_history.append({
            "ts": datetime.now().isoformat(),
            "weights": weights.copy()
        })
        
        # 保持最近500条权重历史
        if len(self.factor_weight_history) > 500:
            self.factor_weight_history = self.factor_weight_history[-500:]
        
        return weights
    
    def get_pattern_key(self, 
                       direction: str, 
                       factors: Dict[str, float],
                       market_regime: str = "RANGE") -> str:
        """
        生成模式键
        
        Args:
            direction: 交易方向 (long/short)
            factors: 因子值字典
            market_regime: 市场状态
            
        Returns:
            模式键字符串
        """
        rsi = factors.get("rsi", 50)
        adx = factors.get("adx", 25)
        
        rsi_bucket = "low" if rsi < 40 else ("mid" if rsi < 60 else "high")
        adx_bucket = "low" if adx < 25 else ("mid" if adx < 40 else "high")
        
        return f"{direction}_{rsi_bucket}_rsi_{adx_bucket}_adx_{market_regime}"
    
    def get_factor_weight_evolution(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取因子权重演变历史
        
        Returns:
            {因子名: [{ts, weight}, ...]}
        """
        history = defaultdict(list)
        
        for entry in self.factor_weight_history:
            ts = entry["ts"]
            for factor, weight in entry["weights"].items():
                history[factor].append({
                    "ts": ts,
                    "weight": weight
                })
        
        return dict(history)


# ============================================================
# Trade Hooks
# ============================================================

class TradeHooks:
    """
    交易钩子 - 在交易入场/出场时调用
    
    用于记录学习和更新评估器
    """
    
    def __init__(self, 
                 factor_evaluator,
                 pattern_evaluator,
                 strategy_evolution: StrategyEvolution):
        self.factor_evaluator = factor_evaluator
        self.pattern_evaluator = pattern_evaluator
        self.strategy_evolution = strategy_evolution
    
    def on_trade_entry(self, trade_data: Dict) -> Tuple[str, bool, Optional[str]]:
        """
        交易入场时调用 - 记录入场信息
        
        Args:
            trade_data: {
                "symbol": str,
                "direction": str,
                "entry_time": str,
                "entry_price": float,
                "stop_loss": float,
                "take_profit": float,
                "factors": Dict,
                "market_regime": str
            }
            
        Returns:
            (pattern_key, is_allowed, trade_id)
        """
        pattern_key = self.strategy_evolution.get_pattern_key(
            trade_data['direction'],
            trade_data.get("factors", {}),
            trade_data.get("market_regime", "RANGE")
        )
        trade_id = f"{trade_data['symbol']}_{trade_data['entry_time']}"
        
        # 检查模式是否允许交易
        is_allowed = self.pattern_evaluator.is_allowed(pattern_key)
        
        logger.info(f"Trade entry recorded: pattern={pattern_key}, allowed={is_allowed}")
        return pattern_key, is_allowed, trade_id
    
    def on_trade_exit(self, 
                     trade_id: str, 
                     exit_data: Dict,
                     direction: str,
                     factors: Dict):
        """
        交易出场时调用 - 更新学习和统计
        
        Args:
            trade_id: 交易ID
            exit_data: {
                "exit_time": str,
                "exit_price": float,
                "pnl": float,
                "pnl_pct": float,
                "stop_triggered": str,
                "pattern_key": str,
                "factors": Dict
            }
            direction: 交易方向
            factors: 因子值字典
        """
        won = exit_data.get("pnl", 0) > 0
        actual_rr = self._calculate_actual_rr(exit_data)
        
        # 获取模式键
        pattern_key = exit_data.get("pattern_key") 
        if not pattern_key:
            pattern_key = self.strategy_evolution.get_pattern_key(
                direction, factors, exit_data.get("market_regime", "RANGE")
            )
        
        # 更新模式表现
        self.pattern_evaluator.update(pattern_key, won, actual_rr)
        
        # 更新因子表现
        for factor_name, value in factors.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # 方向性因子: RSI需要反转 (RSI低=超卖=强做多信号)
                if factor_name == "RSI" and value <= 100:
                    signal = (100 - value) / 100.0  # RSI30 → 0.70 (强做多信号)
                else:
                    signal = value / 100.0 if value > 1 else value
                self.factor_evaluator.update(factor_name, signal, exit_data.get("pnl_pct", 0))
    
    def _calculate_actual_rr(self, exit_data: Dict) -> float:
        """计算实际盈亏比"""
        pnl = exit_data.get("pnl", 0)
        if pnl == 0:
            return 0.0
        
        # 简化：用PnL的符号和大小估算RR
        # 盈利时RR为正，亏损时RR为负
        risk = exit_data.get("risk_amount", abs(pnl * 2))  # 估算风险金额
        if risk == 0:
            return 0.0
        
        return pnl / risk


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "StrategyEvolution",
    "TradeHooks",
]
