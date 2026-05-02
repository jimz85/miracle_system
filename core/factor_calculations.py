from __future__ import annotations

"""
Factor Calculations: 价格技术因子统一计算模块
从 miracle_core.py 提取并整合

职责：所有价格因子的标准计算
- calc_rsi / calc_adx / calc_macd / calc_atr / normalize_macd_histogram
- calc_combined_score

返回格式统一：所有函数返回值不含 tuple（避免与 core/price_factors.py 混淆）

代码拆分说明（2026-05-01）：
- 实际计算已迁移至 core/price_factors.py::PriceFactors 类
- 本文件保留公共API，委托给PriceFactors实现（保持向后兼容）
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.price_factors import PriceFactors


def calc_rsi(prices: List[float], period: int = 14) -> float:
    """RSI (Wilder平滑) — 委托给PriceFactors实现"""
    return PriceFactors.calc_rsi(prices, period)


def calc_adx(highs: List[float], lows: List[float], closes: List[float],
              period: int = 14) -> Dict[str, float]:
    """ADX (Wilder平滑) — 委托给PriceFactors实现"""
    return PriceFactors.calc_adx(highs, lows, closes, period)


def calc_macd(prices: List[float], fast: int = 12, slow: int = 26,
              signal: int = 9) -> Dict[str, float]:
    """MACD — 委托给PriceFactors实现"""
    return PriceFactors.calc_macd(prices, fast, slow, signal)


def calc_atr(highs: List[float], lows: List[float], closes: List[float],
              period: int = 14) -> float:
    """ATR (Wilder平滑) — 委托给PriceFactors实现"""
    return PriceFactors.calc_atr(highs, lows, closes, period)


def normalize_macd_histogram(hist: float, price: float) -> float:
    """
    归一化MACD直方图到-1~1范围
    使用价格百分比的ATR等价物作为标准化基准
    """
    if price <= 0 or not np.isfinite(hist):
        return 0.0
    normalized = hist / price
    return float(np.clip(normalized, -1.0, 1.0))


def calc_combined_score(price_score: float, news_score: float,
                        onchain_score: float, wallet_score: float,
                        weights: Dict[str, float]) -> float:
    """
    多因子加权综合得分（委托 core.confidence.weighted_fusion）
    """
    from core.confidence import weighted_fusion

    return weighted_fusion(
        {"price_momentum": price_score,
         "news_sentiment": news_score,
         "onchain": onchain_score,
         "wallet": wallet_score},
        {"price_momentum": weights.get("price_momentum", 0.6),
         "news_sentiment": weights.get("news_sentiment", 0.2),
         "onchain": weights.get("onchain", 0.1),
         "wallet": weights.get("wallet", 0.1)}
    )
