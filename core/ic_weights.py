#!/usr/bin/env python3
from __future__ import annotations

"""
IC-based Factor Weight System (P1.4)
====================================

基于Memory Log决策-结果反馈闭环的IC权重动态调整

功能:
    - calculate_ic() - 计算信息系数 (预测方向 vs 实际方向)
    - update_weights() - 基于IC更新因子权重 (指数平滑)
    - decay_factor=0.7 exponential smoothing
    - min_samples=10 最小样本保护
    - 输出因子权重: rsi, macd, adx, bollinger, momentum

权重更新公式:
    new_weight = decay_factor * old_weight + (1 - decay_factor) * ic_value

Usage:
    from core.ic_weights import ICWeightManager

    manager = ICWeightManager()
    weights = manager.get_weights()  # 获取当前权重
    manager.update_weights()         # 从Memory Log更新IC并刷新权重
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Tuple


# 确保项目根目录在path中 (用于直接运行此脚本时)
# Python会自动将脚本所在目录加入sys.path[0],这会遮挡真正的memory/目录
# 因此需要将sys.path[0]替换为项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _project_root:
    # 移除脚本目录,替换为项目根目录
    sys.path[0] = _project_root

# Import memory.fusion_memory directly to avoid namespace package issues
import memory.fusion_memory as _fusion_memory

get_all_entries = _fusion_memory.get_all_entries
get_ic_feedback = _fusion_memory.get_ic_feedback

logger = logging.getLogger(__name__)

# ==================== 常量 ====================

CACHE_FILE = os.path.expanduser('~/.miracle_memory/ic_factor_weights.json')
DECAY_FACTOR = 0.7
MIN_SAMPLES = 10
MIN_WEIGHT = 0.05  # 最小权重阈值，IC≤0时权重不低于此值
FACTORS = ['rsi', 'macd', 'adx', 'bollinger', 'momentum']

# 默认权重 (因子IC未知时使用)
DEFAULT_WEIGHTS = {
    'rsi': 0.20,
    'macd': 0.20,
    'adx': 0.20,
    'bollinger': 0.20,
    'momentum': 0.20,
}


# ==================== 数据结构 ====================

@dataclass
class FactorStats:
    """因子统计"""
    correct: int = 0
    total: int = 0
    ic_value: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class ICWeights:
    """IC权重状态"""
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ic_values: Dict[str, float] = field(default_factory=lambda: {f: 0.0 for f in FACTORS})
    sample_counts: Dict[str, int] = field(default_factory=lambda: {f: 0 for f in FACTORS})
    last_updated: str | None = None


# ==================== IC权重管理器 ====================

class ICWeightManager:
    """
    基于Memory Log的IC动态权重管理器

    工作流程:
    1. 从Memory Log获取历史决策 (有outcome的)
    2. 对每个因子,计算预测方向与实际结果的一致性
    3. 用IC值通过指数平滑更新权重
    4. 输出权重用于FusionDecision
    """

    _instance: ICWeightManager | None = None
    _lock = Lock()

    def __init__(self):
        self._state = ICWeights()
        self._load()

    @classmethod
    def get_instance(cls) -> ICWeightManager:
        """单例获取"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        """从磁盘加载"""
        if not os.path.exists(CACHE_FILE):
            return
        try:
            with open(CACHE_FILE, encoding='utf-8') as f:
                data = json.load(f)
            self._state.weights = data.get('weights', dict(DEFAULT_WEIGHTS))
            self._state.ic_values = data.get('ic_values', {f: 0.0 for f in FACTORS})
            self._state.sample_counts = data.get('sample_counts', {f: 0 for f in FACTORS})
            self._state.last_updated = data.get('last_updated')
            logger.info(f"[IC] 加载权重: {self._state.weights}")
        except Exception as e:
            logger.warning(f"[IC] 加载失败: {e}")

    def _save(self) -> None:
        """持久化到磁盘"""
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'weights': self._state.weights,
                    'ic_values': self._state.ic_values,
                    'sample_counts': self._state.sample_counts,
                    'last_updated': self._state.last_updated,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[IC] 保存失败: {e}")

    def calculate_ic(self, factor_name: str, entries: list = None) -> float:
        """
        计算因子IC (信息系数)

        IC = 预测方向与实际结果的一致率

        预测方向判断:
        - rsi < 30 → 预测LONG (+1), rsi > 70 → 预测SHORT (-1)
        - macd > 0 → 预测LONG (+1), macd < 0 → 预测SHORT (-1)
        - adx > 25 → 趋势确认, 高ADX强化信号
        - bollinger 价格触及下轨→LONG, 上轨→SHORT
        - momentum 正→LONG, 负→SHORT

        Args:
            factor_name: 因子名称 (rsi/macd/adx/bollinger/momentum)
            entries: 可选,指定决策条目列表,默认从Memory Log获取

        Returns:
            float: IC值 0-1, 样本不足时返回0.0
        """
        if entries is None:
            entries = get_all_entries(limit=1000)

        # 过滤有outcome的决策
        valid_entries = [e for e in entries if e.get('outcome') in ('WIN', 'LOSS')]
        if len(valid_entries) < MIN_SAMPLES:
            logger.debug(f"[IC] {factor_name} 样本不足: {len(valid_entries)} < {MIN_SAMPLES}")
            return 0.0

        correct = 0
        total = 0

        for entry in valid_entries:
            factors = entry.get('factors', {})
            factor_val = factors.get(factor_name)
            if factor_val is None:
                continue

            verdict = entry.get('verdict', '')
            outcome = entry.get('outcome', '')

            # Skip HOLD/WAIT verdicts
            if verdict in ('HOLD', 'WAIT', ''):
                continue

            predicted = self._predict_direction(factor_name, factor_val)
            if predicted == 0:
                continue

            actual_correct = self._check_outcome(predicted, verdict, outcome)
            if actual_correct is not None:
                total += 1
                if actual_correct:
                    correct += 1

        ic = correct / total if total >= MIN_SAMPLES else 0.0
        logger.debug(f"[IC] {factor_name}: IC={ic:.3f} (n={total})")
        return ic

    def direction_accuracy(self, factor_name: str, entries: list = None) -> float:
        """
        计算因子方向正确率 (向后兼容)

        与calculate_ic()相同逻辑，仅作为显式命名保留

        Args:
            factor_name: 因子名称
            entries: 可选，指定决策条目列表

        Returns:
            float: 方向正确率 0-1
        """
        return self.calculate_ic(factor_name, entries)

    def _get_ic_pairs(self, factor_name: str, entries: list = None) -> Tuple[List[float], List[int]]:
        """
        提取因子预测信号强度与实际结果的配对数据

        用于计算真IC（Pearson/Spearman相关系数）

        预测信号:
        - rsi: (rsi - 50) / 50 → 归一化到 [-1, 1]，0=中性
        - macd: sign(macd) → -1/0/1
        - adx: adx / 100 → [0, 1]
        - bollinger: 2*position - 1 → 归一化到 [-1, 1]
        - momentum: sign(momentum) → -1/0/1

        实际结果: WIN=+1, LOSS=-1

        Returns:
            (signals, outcomes): 两个等长列表
        """
        if entries is None:
            entries = get_all_entries(limit=1000)

        valid_entries = [e for e in entries if e.get('outcome') in ('WIN', 'LOSS')]
        if len(valid_entries) < MIN_SAMPLES:
            return [], []

        signals = []
        outcomes = []

        for entry in valid_entries:
            factors = entry.get('factors', {})
            factor_val = factors.get(factor_name)
            if factor_val is None:
                continue

            verdict = entry.get('verdict', '')
            outcome = entry.get('outcome', '')

            if verdict in ('HOLD', 'WAIT', ''):
                continue

            signal = self._factor_to_signal(factor_name, factor_val)
            if signal == 0:
                continue

            # 实际结果需要按verdict方向对齐
            # BUY(做多): WIN=+1, LOSS=-1
            # SELL(做空): WIN=-1 (价格下跌=盈利), LOSS=+1 (价格上涨=亏损)
            if verdict == 'BUY':
                outcome_val = 1 if outcome == 'WIN' else -1
            elif verdict == 'SELL':
                outcome_val = -1 if outcome == 'WIN' else 1
            else:
                continue

            signals.append(signal)
            outcomes.append(outcome_val)

        return signals, outcomes

    def _factor_to_signal(self, factor_name: str, factor_val) -> float:
        """
        将因子值转换为信号强度 [-1, 1]

        用于真IC计算
        """
        try:
            val = float(factor_val)
        except (TypeError, ValueError):
            val_str = str(factor_val).lower()
            if 'bull' in val_str or 'long' in val_str:
                return 1.0
            elif 'bear' in val_str or 'short' in val_str:
                return -1.0
            return 0.0

        if factor_name == 'rsi':
            # (rsi - 50) / 50 → [-1, 1], 50=neutral
            return (val - 50) / 50.0

        elif factor_name == 'macd':
            # sign(macd) → -1/1, 0→0
            if val > 0:
                return 1.0
            elif val < 0:
                return -1.0
            return 0.0

        elif factor_name == 'adx':
            # ADX already [0, 100] → normalize to [0, 1]
            return val / 100.0

        elif factor_name == 'bollinger':
            # position [0, 1] → [-1, 1]
            return 2.0 * val - 1.0

        elif factor_name == 'momentum':
            if val > 0:
                return 1.0
            elif val < 0:
                return -1.0
            return 0.0

        return 0.0

    def rank_ic(self, factor_name: str, entries: list = None) -> float:
        """
        计算Spearman秩相关系数 (Rank IC)

        非线性单调关系，不受异常值影响

        Args:
            factor_name: 因子名称
            entries: 可选，指定决策条目列表

        Returns:
            float: Spearman IC [-1, 1]，样本不足返回0.0
        """
        signals, outcomes = self._get_ic_pairs(factor_name, entries)

        if len(signals) < MIN_SAMPLES:
            logger.debug(f"[IC] {factor_name} rank_ic 样本不足: {len(signals)} < {MIN_SAMPLES}")
            return 0.0

        # scipy spearmanr returns (correlation, p-value)
        # 延迟导入(spearmanr)避免冷启动耗时 ~7-18s
        from scipy.stats import spearmanr
        corr, _ = spearmanr(signals, outcomes)
        ic = corr if corr is not None else 0.0
        logger.debug(f"[IC] {factor_name}: rank_ic={ic:.3f} (n={len(signals)})")
        return ic

    def pearson_ic(self, factor_name: str, entries: list = None) -> float:
        """
        计算Pearson线性相关系数 (Pearson IC)

        衡量线性相关性

        Args:
            factor_name: 因子名称
            entries: 可选，指定决策条目列表

        Returns:
            float: Pearson IC [-1, 1]，样本不足返回0.0
        """
        signals, outcomes = self._get_ic_pairs(factor_name, entries)

        if len(signals) < MIN_SAMPLES:
            logger.debug(f"[IC] {factor_name} pearson_ic 样本不足: {len(signals)} < {MIN_SAMPLES}")
            return 0.0

        # 延迟导入(pearsonr)避免冷启动耗时 ~7-18s
        from scipy.stats import pearsonr
        corr, _ = pearsonr(signals, outcomes)
        ic = corr if corr is not None else 0.0
        logger.debug(f"[IC] {factor_name}: pearson_ic={ic:.3f} (n={len(signals)})")
        return ic

    def information_ratio(self, factor_name: str, entries: list = None,
                         ic_type: str = 'rank_ic') -> float:
        """
        计算信息比率 IR = IC_mean / IC_std

        衡量IC的稳定性，IR越高说明因子越稳定有效

        Args:
            factor_name: 因子名称
            entries: 可选，指定决策条目列表
            ic_type: 'rank_ic' (默认) 或 'pearson_ic'

        Returns:
            float: 信息比率，样本不足或std=0时返回0.0
        """
        import statistics

        if entries is None:
            entries = get_all_entries(limit=1000)

        # 按时间窗口计算rolling IC，取多段样本
        # 将entries分成多个窗口，计算每个窗口的IC
        window_size = MIN_SAMPLES
        valid_entries = [e for e in entries if e.get('outcome') in ('WIN', 'LOSS')]

        if len(valid_entries) < window_size * 2:
            # 样本不足，无法计算IR
            logger.debug(f"[IC] {factor_name} IR 样本不足: {len(valid_entries)} < {window_size * 2}")
            return 0.0

        ic_values = []
        for i in range(0, len(valid_entries) - window_size + 1, window_size // 2):
            window = valid_entries[i:i + window_size]
            if ic_type == 'pearson_ic':
                ic = self.pearson_ic(factor_name, entries=window)
            else:
                ic = self.rank_ic(factor_name, entries=window)

            if ic != 0.0:
                ic_values.append(ic)

        if len(ic_values) < 2:
            return 0.0

        ic_mean = statistics.mean(ic_values)
        ic_std = statistics.stdev(ic_values)

        if ic_std == 0.0:
            return 0.0

        ir = ic_mean / ic_std
        logger.debug(f"[IC] {factor_name}: IR={ir:.3f} (mean={ic_mean:.3f}, std={ic_std:.3f}, n_windows={len(ic_values)})")
        return ir

    def _predict_direction(self, factor_name: str, factor_val) -> int:
        """
        根据因子值预测方向

        Returns:
            1 = LONG, -1 = SHORT, 0 = NEUTRAL
        """
        try:
            val = float(factor_val)
        except (TypeError, ValueError):
            # 非数值型因子 (如 "bullish", "bearish")
            val_str = str(factor_val).lower()
            if 'bull' in val_str or 'long' in val_str:
                return 1
            elif 'bear' in val_str or 'short' in val_str:
                return -1
            return 0

        if factor_name == 'rsi':
            if val < 30:
                return 1   # 超卖 → LONG
            elif val > 70:
                return -1  # 超买 → SHORT
            return 0

        elif factor_name == 'macd':
            if val > 0:
                return 1
            elif val < 0:
                return -1
            return 0

        elif factor_name == 'adx':
            # ADX > 25 表示趋势确认, 但不指明方向
            # 结合其他因子判断, 这里返回0表示中立
            if val > 25:
                return 1   # 有趋势,正向处理
            return 0

        elif factor_name == 'bollinger':
            # val是价格在布林带的位置 (0-1)
            if val < 0.2:
                return 1   # 触及下轨 → LONG
            elif val > 0.8:
                return -1  # 触及上轨 → SHORT
            return 0

        elif factor_name == 'momentum':
            if val > 0:
                return 1
            elif val < 0:
                return -1
            return 0

        return 0

    def _check_outcome(self, predicted: int, verdict: str, outcome: str) -> bool | None:
        """
        检查预测是否正确

        Args:
            predicted: 1=LONG, -1=SHORT
            verdict: 决策裁决 BUY/SELL/HOLD
            outcome: WIN/LOSS

        Returns:
            True=正确, False=错误, None=无法判断
        """
        # 预测方向
        if predicted == 1:
            predicted_is_long = True
        else:
            predicted_is_long = False

        # 决策方向
        if verdict == 'BUY':
            decision_is_long = True
        elif verdict == 'SELL':
            decision_is_long = False
        else:
            return None

        # 预测与决策一致
        if predicted_is_long != decision_is_long:
            return None  # 预测方向与决策不符,跳过

        # 结果是否盈利
        if outcome == 'WIN':
            result_is_profit = True
        elif outcome == 'LOSS':
            result_is_profit = False
        else:
            return None

        # 正确预测 = 预测LONG+WIN 或 预测SHORT+LOSS
        return predicted_is_long == result_is_profit

    def update_weights(self) -> Dict[str, float]:
        """
        基于IC更新因子权重 (指数平滑)

        公式: new_weight = decay_factor * old_weight + (1 - decay_factor) * ic_value

        逻辑:
        1. 计算各因子Rank IC (Spearman相关系数)
        2. 用指数平滑更新权重
        3. IC为负的因子权重置MIN_WEIGHT
        4. 归一化使总和=1
        5. 最小样本保护: 样本不足时保持默认权重

        注意: 使用rank_ic()而非direction_accuracy()，因为真IC更准确反映因子预测能力

        Returns:
            Dict[str, float]: 更新后的权重
        """
        from datetime import datetime

        old_weights = dict(self._state.weights)
        new_weights = {}
        new_ic_values = {}
        new_sample_counts = {}

        total_samples = 0

        # 第一步：计算所有因子的Rank IC和权重
        raw_weights = {}
        negative_ic_factors = []  # 记录IC≤0的因子

        for factor in FACTORS:
            ic = self.rank_ic(factor)  # 使用真IC (Spearman相关系数)
            sample_count = self._count_samples(factor)

            new_ic_values[factor] = ic
            new_sample_counts[factor] = sample_count
            total_samples += sample_count

            old_weight = old_weights.get(factor, 1.0 / len(FACTORS))

            if sample_count >= MIN_SAMPLES and ic > 0:
                # IC为正,正常更新
                new_weight = DECAY_FACTOR * old_weight + (1 - DECAY_FACTOR) * ic
                raw_weights[factor] = new_weight
            elif sample_count >= MIN_SAMPLES and ic <= 0:
                # IC为负或零,权重设为最低阈值，确保差因子被淘汰
                raw_weights[factor] = MIN_WEIGHT
                negative_ic_factors.append(factor)
            else:
                # 样本不足,保持旧权重
                raw_weights[factor] = old_weight

        # 第二步：归一化
        # IC≤0的因子保持MIN_WEIGHT，IC>0的因子分配剩余权重
        num_negative = len(negative_ic_factors)
        num_positive = len(FACTORS) - num_negative

        if num_negative == len(FACTORS):
            # 所有因子IC都≤0，全部设为均等权重
            new_weights = {f: 1.0 / len(FACTORS) for f in FACTORS}
        elif num_negative > 0:
            # 部分因子IC≤0，这些保持MIN_WEIGHT，其余分配剩余
            reserved_weight = num_negative * MIN_WEIGHT
            remaining_weight = 1.0 - reserved_weight

            if num_positive > 0 and remaining_weight > 0:
                # 计算IC>0因子的原始权重总和
                positive_sum = sum(raw_weights[f] for f in FACTORS if f not in negative_ic_factors)
                if positive_sum > 0:
                    for f in FACTORS:
                        if f in negative_ic_factors:
                            new_weights[f] = MIN_WEIGHT
                        else:
                            # 按比例分配剩余权重
                            new_weights[f] = (raw_weights[f] / positive_sum) * remaining_weight
                else:
                    new_weights = {f: 1.0 / len(FACTORS) for f in FACTORS}
            else:
                new_weights = {f: 1.0 / len(FACTORS) for f in FACTORS}
        else:
            # 没有IC≤0的因子，正常归一化
            total = sum(raw_weights.values())
            if total > 0:
                new_weights = {k: v / total for k, v in raw_weights.items()}
            else:
                new_weights = dict(DEFAULT_WEIGHTS)

        # 更新状态
        self._state.weights = new_weights
        self._state.ic_values = new_ic_values
        self._state.sample_counts = new_sample_counts
        self._state.last_updated = datetime.now().isoformat()

        self._save()

        logger.info(f"[IC] 权重更新: {new_weights}")
        logger.info(f"[IC] IC值: {new_ic_values}")
        logger.info(f"[IC] 样本数: {new_sample_counts}")

        return new_weights

    def _count_samples(self, factor_name: str) -> int:
        """统计某因子的有效样本数"""
        entries = get_all_entries(limit=1000)
        valid_entries = [e for e in entries if e.get('outcome') in ('WIN', 'LOSS')]
        count = 0
        for entry in valid_entries:
            if entry.get('factors', {}).get(factor_name) is not None:
                verdict = entry.get('verdict', '')
                if verdict not in ('HOLD', 'WAIT', ''):
                    count += 1
        return count

    def get_weights(self) -> Dict[str, float]:
        """获取当前因子权重"""
        return dict(self._state.weights)

    def get_ic_values(self) -> Dict[str, float]:
        """获取当前IC值"""
        return dict(self._state.ic_values)

    def get_sample_counts(self) -> Dict[str, int]:
        """获取各因子样本计数"""
        return dict(self._state.sample_counts)

    def get_info(self) -> Dict:
        """获取完整IC信息"""
        return {
            'weights': self.get_weights(),
            'ic_values': self.get_ic_values(),
            'sample_counts': self.get_sample_counts(),
            'last_updated': self._state.last_updated,
            'decay_factor': DECAY_FACTOR,
            'min_samples': MIN_SAMPLES,
        }

    def reset_to_default(self) -> Dict[str, float]:
        """重置为默认权重"""
        self._state.weights = dict(DEFAULT_WEIGHTS)
        self._state.ic_values = {f: 0.0 for f in FACTORS}
        self._state.sample_counts = {f: 0 for f in FACTORS}
        self._state.last_updated = None
        self._save()
        logger.info("[IC] 权重已重置为默认")
        return dict(DEFAULT_WEIGHTS)


# ==================== 便捷函数 ====================

_manager: ICWeightManager | None = None


def get_ic_manager() -> ICWeightManager:
    """获取IC权重管理器单例"""
    global _manager
    if _manager is None:
        _manager = ICWeightManager.get_instance()
    return _manager


def get_weights() -> Dict[str, float]:
    """获取当前IC权重"""
    return get_ic_manager().get_weights()


def update_weights() -> Dict[str, float]:
    """从Memory Log更新IC并刷新权重"""
    return get_ic_manager().update_weights()


def get_ic_values() -> Dict[str, float]:
    """获取当前IC值"""
    return get_ic_manager().get_ic_values()


def calculate_ic(factor_name: str) -> float:
    """计算指定因子的IC (方向正确率，向后兼容)"""
    return get_ic_manager().calculate_ic(factor_name)


def direction_accuracy(factor_name: str) -> float:
    """计算指定因子的方向正确率 (向后兼容)"""
    return get_ic_manager().direction_accuracy(factor_name)


def rank_ic(factor_name: str) -> float:
    """计算指定因子的Spearman Rank IC (真IC)"""
    return get_ic_manager().rank_ic(factor_name)


def pearson_ic(factor_name: str) -> float:
    """计算指定因子的Pearson IC (真IC)"""
    return get_ic_manager().pearson_ic(factor_name)


def information_ratio(factor_name: str, ic_type: str = 'rank_ic') -> float:
    """计算指定因子的信息比率 IR = IC_mean / IC_std"""
    return get_ic_manager().information_ratio(factor_name, ic_type=ic_type)


def reset_weights() -> Dict[str, float]:
    """重置为默认权重"""
    return get_ic_manager().reset_to_default()


# ==================== 自检 ====================

if __name__ == '__main__':
    import pprint

    print("=== IC动态因子权重系统 (P1.4) ===")

    manager = ICWeightManager.get_instance()
    print("\n当前状态:")
    pprint.pprint(manager.get_info())

    print("\n从Memory Log更新权重...")
    weights = manager.update_weights()

    print("\n更新后权重:")
    pprint.pprint(weights)

    print("\n=== 自检完成 ===")