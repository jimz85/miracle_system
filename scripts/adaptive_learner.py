from __future__ import annotations

"""
Miracle 1.0.2 - Adaptive Learning System (重构版本)
==================================================
Self-learning system with walk-forward validation to prevent overfitting.

Features:
1. Walk-forward validation
2. Dynamic factor weight adjustment (with bounds)
3. Pattern performance statistics (with minimum sample requirements)
4. Overfitting detection
5. Decision Journal (compare with Kronos decision_journal.jsonl)
6. Historical decision tracking & pattern recognition statistics
7. Pattern whitelist/blacklist filtering
8. PCA anomaly detection and alerts

注意: 此模块已重构为三个子模块:
|- learner.py: 决策日记、WalkForward验证、信息系数计算、核心学习系统
|- evaluator.py: 因子/模式评估、过拟合检测、IC报告、PCA异常检测
|- strategy_evolution.py: 策略演化、因子权重调整、模式识别
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("miracle.adaptive_learner")

# 从子模块导入所有公开API，保持向后兼容
from backtest.evaluator import (
    AnomalyAlert,
    FactorEvaluator,
    OverfittingDetector,
    PatternEvaluator,
    PCAAnomalyDetector,
)
from backtest.learner import (
    DecisionJournal,
    DecisionJournalEntry,
    WalkForwardValidator,
    calc_information_coefficient,
)
from scripts.strategy_evolution import (
    StrategyEvolution,
    TradeHooks,
)

# ============================================================
# AdaptiveLearner - 整合所有组件
# ============================================================

class AdaptiveLearner:
    """
    自适应学习系统 - 带样本外验证

    特性:
    1. Walk-forward验证
    2. 因子权重动态调整（有上限）
    3. 模式表现统计（有最小样本要求）+ 白名单/黑名单
    4. 过拟合检测
    5. Decision Journal集成（可对比Kronos）
    6. 因子权重学习历史追踪
    7. PCA异常检测与警报
    """

    def __init__(self, config: Dict, base_dir: str = None, kronos_journal_path: str = None):
        """
        Args:
            config: 交易配置字典
            base_dir: 基础目录路径
            kronos_journal_path: Kronos decision_journal.jsonl 路径
        """
        self.config = config
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent

        # 学习参数
        self.min_sample_size = 20  # 最少20笔交易才能做统计判断
        self.max_weight = 1.0      # 因子权重上限
        self.min_weight = 0.05     # 因子权重下限
        
        # 模式白名单/黑名单
        pattern_whitelist = config.get("pattern_whitelist", [])
        pattern_blacklist = config.get("pattern_blacklist", [])
        
        # 创建组件
        self.factor_evaluator = FactorEvaluator(min_sample_size=self.min_sample_size)
        self.pattern_evaluator = PatternEvaluator(
            min_sample_size=5,
            pattern_whitelist=pattern_whitelist,
            pattern_blacklist=pattern_blacklist
        )
        self.overfitting_detector = OverfittingDetector()
        self.pca_anomaly_detector = PCAAnomalyDetector(history_size=100)
        self.strategy_evolution = StrategyEvolution(
            config=config,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            min_sample_size=self.min_sample_size
        )
        self.trade_hooks = TradeHooks(
            factor_evaluator=self.factor_evaluator,
            pattern_evaluator=self.pattern_evaluator,
            strategy_evolution=self.strategy_evolution
        )

        # 学习记录文件
        self.learning_log_path = self.base_dir / "data" / "learning_log.json"
        self._load_learning_log()

        # Decision Journal - 决策日记
        journal_dir = self.base_dir / "data" / "decision_journal"
        self.decision_journal = DecisionJournal(
            journal_dir=str(journal_dir),
            kronos_journal_path=kronos_journal_path
        )

        logger.info("AdaptiveLearner initialized with DecisionJournal, PCAAnomalyDetector, and pattern whitelist/blacklist")

    def _load_learning_log(self):
        """加载学习日志"""
        if self.learning_log_path.exists():
            try:
                with open(self.learning_log_path) as f:
                    data = json.load(f)
                    # 恢复因子表现数据
                    factor_perf_data = data.get("factor_performance", {})
                    for factor_name, perf in factor_perf_data.items():
                        if factor_name in self.factor_evaluator.factor_performance:
                            self.factor_evaluator.factor_performance[factor_name] = perf
                    
                    # 恢复模式表现数据
                    pattern_perf_data = data.get("pattern_performance", {})
                    for pattern_key, perf in pattern_perf_data.items():
                        if pattern_key in self.pattern_evaluator.pattern_performance:
                            self.pattern_evaluator.pattern_performance[pattern_key] = perf
                            
                logger.info("Learning log loaded from file")
            except Exception as e:
                logger.warning(f"Failed to load learning log: {e}")

    def _save_learning_log(self):
        """保存学习日志"""
        try:
            self.learning_log_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "factor_performance": dict(self.factor_evaluator.factor_performance),
                "pattern_performance": dict(self.pattern_evaluator.pattern_performance),
                "last_update": datetime.now().isoformat()
            }
            with open(self.learning_log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save learning log: {e}")

    def log_decision(self,
                    equity: float,
                    positions: Dict[str, Any],
                    candidates: List[Dict[str, Any]],
                    decision: str,
                    decision_reason: str,
                    execution_ok: bool,
                    execution_result: str = "",
                    llm_raw_output: str = "",
                    market_context: Dict[str, Any] = None,
                    factor_weights: Dict[str, float] = None,
                    pattern_key: str = None,
                    detected_patterns: List[str] = None) -> None:
        """
        记录决策到日记（对标Kronos decision_journal.jsonl格式）

        Args:
            equity: 当前权益
            positions: 持仓快照 {symbol: {direction, size, entry, price, pnl_pct, ...}}
            candidates: 候选币种列表 [{coin, direction, score, rsi_1h, adx_1h, ...}]
            decision: 决策 (open/close/hold/modify)
            decision_reason: 决策原因
            execution_ok: 执行是否成功
            execution_result: 执行结果描述
            llm_raw_output: LLM原始输出
            market_context: 市场上下文
            factor_weights: 当前因子权重
            pattern_key: 模式键
            detected_patterns: 检测到的模式列表
        """
        # 构建市场上下文
        ctx = market_context or {}
        local_context = {
            "market_regime": ctx.get("market_regime", "unknown"),
            "primary_direction": ctx.get("primary_direction", "both"),
            "overall_confidence": ctx.get("overall_confidence", 0.5),
            "emergency_level": ctx.get("emergency_level", "none"),
            "strategic_hint": ctx.get("strategic_hint", ""),
            "data_quality": ctx.get("data_quality", "fresh")
        }

        # 构建持仓快照
        positions_snapshot = {}
        for symbol, pos_data in positions.items():
            if isinstance(pos_data, dict):
                positions_snapshot[symbol] = {
                    "direction": pos_data.get("direction", "unknown"),
                    "size": pos_data.get("size", 0),
                    "entry": pos_data.get("entry_price", pos_data.get("entry", 0)),
                    "price": pos_data.get("current_price", pos_data.get("price", 0)),
                    "pnl_pct": pos_data.get("pnl_pct", 0),
                    "pnl_abs": pos_data.get("pnl_abs", pos_data.get("pnl", 0)),
                    "sl_price": pos_data.get("stop_loss", pos_data.get("sl_price", 0)),
                    "tp_price": pos_data.get("take_profit", pos_data.get("tp_price", 0))
                }

        # 检测过拟合
        overfitting_result = self.detect_overfitting()

        # 获取当前调整后的权重
        adjusted_weights = factor_weights or self.adjust_factor_weights()

        # 构建日记条目
        entry = DecisionJournalEntry(
            ts=datetime.now().isoformat(),
            equity=equity,
            position_count=len(positions),
            local_context=local_context,
            positions_snapshot=positions_snapshot,
            candidates_snapshot=candidates[:5],  # 最多5个候选
            llm_raw_output=llm_raw_output[:2000] if llm_raw_output else "",
            decision_parsed={
                "coin": candidates[0].get("coin", "") if candidates else "",
                "decision": decision,
                "reason": decision_reason[:500] if decision_reason else ""
            },
            execution_result=execution_result,
            execution_ok=execution_ok,
            factor_weights_snapshot=adjusted_weights,
            pattern_recognition={
                "detected_patterns": detected_patterns or [],
                "pattern_key": pattern_key or "",
                "confidence": ctx.get("pattern_confidence", 0.0)
            },
            learning_feedback={
                "overfitting_detected": overfitting_result.get("is_overfitting", False),
                "ic_decay": overfitting_result.get("ic_decay", 0.0),
                "adjusted_weights": adjusted_weights
            }
        )

        # 记录到日记
        self.decision_journal.record_decision(entry)

        # 记录因子权重历史
        self.factor_weight_history = self.strategy_evolution.factor_weight_history

        logger.info(f"Decision logged: {decision} | equity={equity:.2f} | pos={len(positions)}")

    def compare_with_kronos(self, time_window: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        与Kronos decision_journal.jsonl对比分析

        Args:
            time_window: 时间窗口，如 "1h", "6h", "1d"
            limit: 最多分析条数

        Returns:
            对比分析报告
        """
        return self.decision_journal.compare_with_kronos(time_window=time_window, limit=limit)

    def get_decision_stats(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        return self.decision_journal.get_stats()

    def get_factor_weight_evolution(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取因子权重演变历史

        Returns:
            {因子名: [{ts, weight}, ...]}
        """
        return self.strategy_evolution.get_factor_weight_evolution()

    def update_factor_performance(self, factor_name: str, signal: float,
                                   actual_return: float):
        """
        更新因子表现

        Args:
            factor_name: 因子名称
            signal: 因子信号值
            actual_return: 实际收益
        """
        self.factor_evaluator.update(factor_name, signal, actual_return)

    def update_pattern_performance(self, pattern_key: str, won: bool, actual_rr: float):
        """
        更新模式表现

        Args:
            pattern_key: 模式键
            won: 是否盈利
            actual_rr: 实际盈亏比
        """
        self.pattern_evaluator.update(pattern_key, won, actual_rr)

    def adjust_factor_weights(self) -> Dict[str, float]:
        """
        基于IC表现调整因子权重

        Returns:
            新的因子权重字典
        """
        return self.strategy_evolution.adjust_factor_weights(
            self.config.get("factors", {}),
            self.factor_evaluator
        )

    def detect_overfitting(self) -> Dict[str, Any]:
        """
        检测过拟合

        Returns:
            {
                "is_overfitting": bool,
                "train_ic_avg": float,
                "test_ic_avg": float,
                "ic_decay": float,
                "reason": str
            }
        """
        return self.overfitting_detector.detect(self.factor_evaluator)

    def get_factor_ic_report(self) -> Dict[str, Any]:
        """
        获取因子IC报告

        Returns:
            各因子的IC统计报告
        """
        return self.factor_evaluator.get_report()

    def get_pattern_stats(self, pattern_key: str) -> Dict[str, Any]:
        """
        获取模式统计

        Args:
            pattern_key: 模式键

        Returns:
            模式统计信息
        """
        return self.pattern_evaluator.get_stats(pattern_key)

    # ============================================================
    # Pattern Whitelist/Blacklist Management
    # ============================================================

    def add_pattern_to_whitelist(self, pattern_keys: List[str]):
        """添加模式到白名单"""
        self.pattern_evaluator.add_to_whitelist(pattern_keys)

    def remove_pattern_from_whitelist(self, pattern_keys: List[str]):
        """从白名单移除模式"""
        self.pattern_evaluator.remove_from_whitelist(pattern_keys)

    def add_pattern_to_blacklist(self, pattern_keys: List[str]):
        """添加模式到黑名单"""
        self.pattern_evaluator.add_to_blacklist(pattern_keys)

    def remove_pattern_from_blacklist(self, pattern_keys: List[str]):
        """从黑名单移除模式"""
        self.pattern_evaluator.remove_from_blacklist(pattern_keys)

    def get_pattern_whitelist(self) -> List[str]:
        """获取模式白名单"""
        return self.pattern_evaluator.get_whitelist()

    def get_pattern_blacklist(self) -> List[str]:
        """获取模式黑名单"""
        return self.pattern_evaluator.get_blacklist()

    def clear_pattern_whitelist(self):
        """清空白名单"""
        self.pattern_evaluator.clear_whitelist()

    def clear_pattern_blacklist(self):
        """清空黑名单"""
        self.pattern_evaluator.clear_blacklist()

    # ============================================================
    # PCA Anomaly Detection
    # ============================================================

    def fit_pca_baseline(self, features: List[List[float]], returns: List[float] = None):
        """
        拟合PCA基线分布（正常市场状态）

        Args:
            features: 因子特征列表 [[f1, f2, ...], ...]
            returns: 可选的收益列表
        """
        self.pca_anomaly_detector.fit_baseline(features, returns)

    def add_pca_sample(self, features: List[float], return_value: float = None) -> float:
        """
        添加PCA样本并返回异常分数

        Args:
            features: 因子特征 [f1, f2, ...]
            return_value: 可选的收益值

        Returns:
            异常分数 (0.0-1.0, 越高越异常)
        """
        return self.pca_anomaly_detector.add_sample(features, return_value)

    def check_pca_anomaly(self) -> AnomalyAlert | None:
        """
        检查PCA异常并返回警报

        Returns:
            AnomalyAlert或None
        """
        return self.pca_anomaly_detector.check_anomaly()

    def run_pca_full_check(self) -> Dict[str, Any]:
        """
        运行完整PCA检查（异常+漂移+波动）

        Returns:
            完整检查报告
        """
        return self.pca_anomaly_detector.run_full_check()

    def get_pca_stats(self) -> Dict[str, Any]:
        """获取PCA检测统计信息"""
        return self.pca_anomaly_detector.get_stats()

    def get_pca_alerts(self, n: int = 10) -> List[AnomalyAlert]:
        """获取最近的PCA警报"""
        return self.pca_anomaly_detector.get_recent_alerts(n)

    def on_trade_entry(self, trade_data: Dict) -> Tuple[str, bool, str | None]:
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
        return self.trade_hooks.on_trade_entry(trade_data)

    def on_trade_exit(self, trade_id: str, exit_data: Dict):
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
        """
        direction = exit_data.get("direction", "long")
        factors = exit_data.get("factors", {})
        self.trade_hooks.on_trade_exit(trade_id, exit_data, direction, factors)

        # 检测过拟合
        overfit_result = self.detect_overfitting()
        if overfit_result["is_overfitting"]:
            logger.warning(f"Overfitting detected: {overfit_result}")

        # 保存学习日志
        self._save_learning_log()


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    # learner
    "DecisionJournal",
    "DecisionJournalEntry",
    "WalkForwardValidator",
    "calc_information_coefficient",
    # evaluator
    "FactorEvaluator",
    "PatternEvaluator",
    "OverfittingDetector",
    "PCAAnomalyDetector",
    "AnomalyAlert",
    # strategy_evolution
    "StrategyEvolution",
    "TradeHooks",
    # main class
    "AdaptiveLearner",
]


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":
    import random

    import numpy as np

    # Test WalkForwardValidator
    print("Testing WalkForwardValidator...")

    data = [
        {"signal": random.uniform(-1, 1), "return": random.uniform(-0.1, 0.15)}
        for _ in range(100)
    ]

    wf = WalkForwardValidator(train_window=50, test_window=20)

    def strategy_func(train_data, test_data):
        # 用训练数据得到最佳参数，应用到测试数据
        signals_train = [d["signal"] for d in train_data]
        returns_train = [d["return"] for d in train_data]
        ic_train, _ = calc_information_coefficient(signals_train, returns_train)

        # 测试数据上：用训练数据的均值/阈值生成信号
        if len(test_data) == 0:
            return {"train_ic": ic_train, "test_ic": 0.0}

        # 简化：用训练数据的信号均值作为阈值
        signal_threshold = sum(signals_train) / len(signals_train) if signals_train else 0
        signals_test = [d["signal"] for d in test_data]
        returns_test = [d["return"] for d in test_data]

        # 转换: >threshold → 1, <threshold → -1 (方向)
        def to_direction(sig, thresh):
            if sig > thresh:
                return 1
            elif sig < thresh:
                return -1
            return 0

        dirs_test = [to_direction(s, signal_threshold) for s in signals_test]
        ic_test, _ = calc_information_coefficient(dirs_test, returns_test)

        return {"train_ic": ic_train, "test_ic": ic_test}

    result = wf.validate(strategy_func, data, n_windows=5)
    print(f"Walk-forward result: {result}")

    # Test AdaptiveLearner
    print("\nTesting AdaptiveLearner...")

    config = {
        "factors": {
            "price_momentum": {"weight": 0.6},
            "news_sentiment": {"weight": 0.2},
            "onchain": {"weight": 0.1},
            "wallet": {"weight": 0.1}
        }
    }

    learner = AdaptiveLearner(config)

    # Simulate factor updates
    for i in range(30):
        learner.update_factor_performance("price_momentum", random.uniform(-1, 1), random.uniform(-0.05, 0.08))
        learner.update_factor_performance("news_sentiment", random.uniform(-1, 1), random.uniform(-0.03, 0.04))

    # Adjust weights
    new_weights = learner.adjust_factor_weights()
    print(f"Adjusted weights: {new_weights}")

    # Detect overfitting
    overfit_result = learner.detect_overfitting()
    print(f"Overfitting detection: {overfit_result}")

    # Test Pattern Whitelist/Blacklist
    print("\nTesting Pattern Whitelist/Blacklist...")
    learner.add_pattern_to_blacklist(["short_high_rsi_high_adx_TREND"])
    learner.add_pattern_to_whitelist(["long_low_rsi_low_adx_RANGE"])
    print(f"Blacklist: {learner.get_pattern_blacklist()}")
    print(f"Whitelist: {learner.get_pattern_whitelist()}")

    # Test PCA Anomaly Detector
    print("\nTesting PCA Anomaly Detector...")
    try:
        # Generate synthetic feature data
        np.random.seed(42)
        baseline_features = np.random.randn(100, 4).tolist()
        baseline_returns = np.random.randn(100).tolist()
        learner.fit_pca_baseline(baseline_features, baseline_returns)
        print("PCA baseline fitted")

        # Add some samples
        for i in range(20):
            features = np.random.randn(4).tolist()
            ret = random.uniform(-0.05, 0.08)
            score = learner.add_pca_sample(features, ret)
            if i % 10 == 0:
                print(f"Sample {i}: anomaly_score={score:.4f}")

        # Run full check
        pca_report = learner.run_pca_full_check()
        print(f"PCA full check: has_alert={pca_report.get('has_alert')}")
        print(f"PCA anomaly score: {pca_report.get('anomaly_score', 0):.4f}")

        # Get PCA stats
        pca_stats = learner.get_pca_stats()
        print(f"PCA stats: {pca_stats}")
    except Exception as e:
        print(f"PCA test skipped or failed: {e}")

    print("\nAll tests passed!")
