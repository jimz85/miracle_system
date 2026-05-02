#!/usr/bin/env python3
from __future__ import annotations

"""
Centralized Confidence & Fusion Module (P1.4)
==============================================
统一置信度计算和加权融合逻辑，消除多处重复实现。

功能:
    - weighted_fusion()      — 通用加权平均融合（替代3处重复实现）
    - kronos_confidence()    — Kronos 7因子投票风格的置信度计算
    - signal_base_confidence() — SignalGenerator风格的基础置信度
    - pattern_adjust_confidence() — Pattern历史胜率置信度调整
    - multi_tf_adjust()     — 多时间框架置信度调整

用法:
    from core.confidence import weighted_fusion, kronos_confidence

    # 加权融合
    combined = weighted_fusion(
        {"price": 0.5, "news": 0.3},
        {"price": 0.6, "news": 0.4}
    )

    # Kronos置信度
    conf = kronos_confidence(
        score=0.8, extreme=False, direction="long",
        factors={"_4h_direction": "bull", "_4h_strength": 0.6}
    )
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def weighted_fusion(scores: Dict[str, float],
                    weights: Dict[str, float]) -> float:
    """
    通用加权平均融合。

    Args:
        scores: 因子得分字典 {name: score}
        weights: 因子权重字典 {name: weight}

    Returns:
        加权平均后的综合得分，闭区间 [0, 1]（若全零返回 0.0）
    """
    if not scores or not weights:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for name, score in scores.items():
        w = weights.get(name, 0.0)
        weighted_sum += score * w
        total_weight += w

    if total_weight == 0.0:
        return 0.0

    return weighted_sum / total_weight


def kronos_confidence(score: float, extreme: bool,
                      direction: str,
                      factors: Dict) -> float:
    """
    Kronos 7因子投票风格的置信度计算。

    规则:
        - extreme RSI方向非wait → 0.80（固定高信心）
        - direction == 'wait'   → 0.0（被过滤的信号）
        - 普通信号                → min(abs(score) / 2.0, 1.0)
        注意：4H时间框架惩罚由调用方（voting_vote）处理，因为同时需要修改score和日志

    Args:
        score: 加权投票得分（原始值）
        extreme: 是否为极端RSI信号
        direction: 信号方向 ('long'/'short'/'wait')
        factors: 因子字典（未使用，保留接口兼容）

    Returns:
        float: [0, 1] 范围的置信度
    """
    if extreme and direction != 'wait':
        return 0.80
    elif direction == 'wait':
        return 0.0
    else:
        return min(abs(score) / 2.0, 1.0)


def signal_base_confidence(trend_strength: float,
                           signal_score: float,
                           confidence_modifier: float = 1.0,
                           real_data_score: float = 1.0,
                           volume_penalty: float = 0.0) -> float:
    """
    SignalGenerator 风格的基础置信度计算。

    公式:
        base = trend_strength/100 * 0.4 + signal_score * 0.6
        conf = base * confidence_modifier * real_data_score * (1 - volume_penalty)
        conf = clamp(conf, 0, 1)

    Args:
        trend_strength: 趋势强度 [0, 100]
        signal_score: 信号得分（已含pattern调整）
        confidence_modifier: 白名单过滤器调整因子
        real_data_score: 真实数据接入程度 [0, 1]
        volume_penalty: 成交量惩罚 [0, 1]

    Returns:
        float: [0, 1] 范围的置信度
    """
    # 基础置信度 = 趋势强度 + 信号得分
    base_confidence = (trend_strength / 100 * 0.4 +
                       signal_score * 0.6)
    # 调整
    confidence = (base_confidence * confidence_modifier *
                  real_data_score * (1.0 - volume_penalty))
    # 截断到 [0, 1]
    return max(0.0, min(confidence, 1.0))


def pattern_adjust_confidence(signal_score: float,
                              pattern_win_rate: float,
                              has_pattern_history: bool) -> float:
    """
    Pattern历史胜率置信度调整。

    规则:
        - 有历史: signal_score *= (0.5 + win_rate)，范围 [0.5, 1.5]
        - 无历史: signal_score *= 0.5（降低信心）

    Args:
        signal_score: 原始信号得分（绝对值）
        pattern_win_rate: pattern历史胜率 [0, 1]
        has_pattern_history: 是否有该pattern的历史数据

    Returns:
        float: 调整后的信号得分
    """
    if has_pattern_history:
        history_factor = 0.5 + pattern_win_rate
    else:
        history_factor = 0.5
    return signal_score * history_factor


def multi_tf_adjust(confidence: float, direction: str,
                    mt_result: Dict,
                    inplace_boost: float = 0.2) -> float:
    """
    多时间框架确认后的置信度调整。

    规则:
        - 确认通过: conf *= (1 + boost * 0.2)
        - 确认失败: conf *= confidence_boost（按比例降低）

    Args:
        confidence: 当前置信度
        direction: 信号方向
        mt_result: 多周期过滤结果字典（含 confirmed, confidence_boost）
        inplace_boost: 确认通过时的额外加成系数

    Returns:
        float: 调整后的置信度
    """
    if not mt_result.get("applied", False):
        return confidence

    boost = mt_result.get("confidence_boost", 0.0)
    confirmed = mt_result.get("confirmed", True)

    if confirmed:
        confidence = confidence * (1.0 + boost * inplace_boost)
    else:
        confidence = confidence * boost

    return min(confidence, 1.0)


# ==================== G4: 市场情报推荐逻辑 (统一) ====================

_INTEL_RECOMMENDATIONS = {
    (True, True): "看多",
    (True, False): "观望",
    (False, True): "看空",
    (False, False): "观望",
}

_INTEL_CAUTIOUS = {
    "positive": "谨慎看多",
    "negative": "谨慎看空",
    "neutral": "观望",
}


def market_intel_recommendation(
    combined_score: float,
    signal_values: list,
    alignment_boost: float = 0.0,
    threshold: float = 0.3,
    min_confidence: float = 0.3,
    max_confidence: float = 0.95,
) -> tuple:
    """
    统一的市场情报推荐方向+置信度生成。

    从 agent_market_intel.py 和 market_intel_llm_agent.py 的重复实现中提取。

    规则:
        - combined_score > 0.3 + majority positive → 看多, conf由score计算
        - combined_score < -0.3 + majority negative → 看空, conf由score计算
        - 混合信号 → 谨慎方向, conf = 0.35 + count*0.05
        - 完全中性 → 观望, conf = 0.5

    Args:
        combined_score: 加权综合评分 [-1, 1]
        signal_values: 各信号评分列表，用于判断方向一致性
        alignment_boost: 一致性对齐加成 (默认0，LLM版传 alignment_strength*0.1)
        threshold: 信号显著阈值 (默认0.3)
        min_confidence: 最小置信度 (默认0.3)
        max_confidence: 最大置信度 (默认0.95)

    Returns:
        tuple: (recommendation: str, confidence: float)
    """
    positive_count = sum(1 for s in signal_values if s > 0.2)
    negative_count = sum(1 for s in signal_values if s < -0.2)

    if combined_score > threshold:
        if positive_count >= 2:
            recommendation = "看多"
            confidence = 0.6 + (combined_score - threshold) * 0.5 + alignment_boost
        else:
            recommendation = "观望"
            confidence = 0.4
    elif combined_score < -threshold:
        if negative_count >= 2:
            recommendation = "看空"
            confidence = 0.6 + abs(combined_score) - threshold * 0.5 + alignment_boost
        else:
            recommendation = "观望"
            confidence = 0.4
    else:
        # 混合信号或中性区域
        if positive_count > negative_count:
            recommendation = "谨慎看多"
            confidence = 0.35 + positive_count * 0.05 + alignment_boost
        elif negative_count > positive_count:
            recommendation = "谨慎看空"
            confidence = 0.35 + negative_count * 0.05 + alignment_boost
        else:
            recommendation = "观望"
            confidence = 0.5

    confidence = max(min_confidence, min(max_confidence, confidence))
    return recommendation, confidence
