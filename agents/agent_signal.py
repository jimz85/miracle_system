"""
Agent-S: Signal Generation Agent for Miracle 1.0.1
高频趋势跟踪+事件驱动混合系统

职责：
1. 接收Agent-M的市场情报报告
2. 计算价格因子（RSI/ADX/MACD/布林带）
3. 多因子融合，生成综合信号
4. 白名单/黑名单过滤
5. 输出高置信度交易信号给Agent-R
"""

import uuid
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Core modules
from core.regime_classifier import RegimeClassifier, MarketRegime
from core.ic_weights import get_weights as get_ic_weights


# ============================================================================
# 1. 价格因子计算
# ============================================================================

class PriceFactors:
    """价格技术指标计算器"""

    @staticmethod
    def calc_rsi(prices: List[float], period: int = 14) -> float:
        """
        RSI (Relative Strength Index) 计算
        RSI < 30: 超卖 → 潜在买入机会
        RSI > 70: 超买 → 潜在卖出机会
        """
        if len(prices) < period + 1:
            return 50.0  # 数据不足返回中性值

        prices = np.array(prices)
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Wilder平滑 — 使用递归形式，避免pandas依赖
        # 正确Wilder：smooth[i] = smooth[i-1] + alpha*(value[i] - smooth[i-1])，alpha=1/period
        # 初始化：第一period用SMA（Wilder原始方法）
        alpha = 1.0 / period
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for i in range(period, len(gains)):
            avg_gain = avg_gain + alpha * (gains[i] - avg_gain)
            avg_loss = avg_loss + alpha * (losses[i] - avg_loss)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def calc_adx(highs: List[float], lows: List[float], closes: List[float],
                  period: int = 14) -> Dict[str, float]:
        """
        ADX (Average Directional Index) 计算 — 正确实现Wilder平滑

        ADX公式（Wilder原始）：
        1. TR = max(H-L, |H-PCP|, |L-PCP|)
        2. +DM = H - PH (if up move > down move and up move > 0)
        3. -DM = PL - L (if down move > up move and down move > 0)
        4. ATR, +DI, -DI 全部用Wilder平滑：SMA_prev * (period-1) / period + TR / period
        5. ADX = Wilder平滑(DX)，取14周期平均

        +DI > -DI: 多头趋势
        -DI > +DI: 空头趋势
        """
        if len(closes) < 2 * period + 1:
            return {"adx": 25.0, "plus_di": 25.0, "minus_di": 25.0}

        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)

        # True Range
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # Directional Movement
        plus_dm = np.zeros(len(closes) - 1)
        minus_dm = np.zeros(len(closes) - 1)

        for i in range(1, len(closes)):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i - 1] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i - 1] = down_move

        n = len(tr)
        period_int = int(period)

        if n < period_int + 1:
            return {"adx": 25.0, "plus_di": 25.0, "minus_di": 25.0}

        alpha = 1.0 / period_int

        # ── 收集每个时间点的DX值 ──
        dx_values = []
        plus_di_series = []
        minus_di_series = []

        # 第一个周期的初始值
        atr = float(np.mean(tr[:period_int]))
        plus_di_smooth = float(np.mean(plus_dm[:period_int]))
        minus_di_smooth = float(np.mean(minus_dm[:period_int]))

        # 后续每个时间点进行Wilder递推并计算DX
        for i in range(period_int, n):
            atr = atr * (1 - alpha) + tr[i] * alpha
            plus_di_smooth = plus_di_smooth * (1 - alpha) + plus_dm[i] * alpha
            minus_di_smooth = minus_di_smooth * (1 - alpha) + minus_dm[i] * alpha

            di_sum = plus_di_smooth + minus_di_smooth
            if di_sum == 0 or atr == 0:
                dx_values.append(0.0)
                plus_di_series.append(0.0)
                minus_di_series.append(0.0)
            else:
                # +DI = 100 × (+DM_smoothed / ATR)
                # -DI = 100 × (-DM_smoothed / ATR)
                # DX = 100 × |+DI - -DI| / (+DI + -DI)
                # 代入后: DX = 100 × |+DM - -DM| / (+DM + -DM)
                plus_di_pct = 100.0 * plus_di_smooth / atr
                minus_di_pct = 100.0 * minus_di_smooth / atr
                dx_i = 100.0 * abs(plus_di_smooth - minus_di_smooth) / (plus_di_smooth + minus_di_smooth)
                dx_values.append(dx_i)
                plus_di_series.append(plus_di_pct)
                minus_di_series.append(minus_di_pct)

        if len(dx_values) == 0:
            return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

        # ── ADX = Wilder平滑(DX) ──
        # 第一个ADX是前period个DX的均值
        if len(dx_values) >= period_int:
            adx = float(np.mean(dx_values[:period_int]))
        else:
            adx = float(np.mean(dx_values))

        # 后续ADX用Wilder递推（只进行period次）
        for i in range(period_int):
            if i < len(dx_values):
                adx = adx * (1 - alpha) + dx_values[i] * alpha

        return {
            "adx": float(adx),
            "plus_di": float(plus_di_series[-1]) if plus_di_series else 0.0,
            "minus_di": float(minus_di_series[-1]) if minus_di_series else 0.0
        }

    @staticmethod
    def calc_macd(prices: List[float], fast: int = 12, slow: int = 26,
                   signal: int = 9) -> Dict[str, float]:
        """
        MACD (Moving Average Convergence Divergence) 计算
        返回: {macd, signal, histogram}
        MACD > Signal: 金叉 → 多头信号
        MACD < Signal: 死叉 → 空头信号
        """
        if len(prices) < slow + signal:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

        prices = np.array(prices)

        # EMA计算
        def ema(data, n):
            ema_values = np.zeros_like(data, dtype=float)
            ema_values[0] = data[0]
            multiplier = 2 / (n + 1)
            for i in range(1, len(data)):
                ema_values[i] = (data[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
            return ema_values

        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {
            "macd": float(macd_line[-1]),
            "signal": float(signal_line[-1]),
            "histogram": float(histogram[-1])
        }

    @staticmethod
    def calc_atr(highs: List[float], lows: List[float], closes: List[float],
                  period: int = 14) -> float:
        """
        计算ATR（平均真实波幅）— 正确实现Wilder平滑

        Wilder ATR = ATR_prev * (period-1)/period + TR_current / period
        不是简单均值（简单均值会使ATR偏小，导致止损设得太紧）
        """
        n = len(closes)
        if n < period + 1:
            return (max(highs) - min(lows)) / min(lows) * closes[-1] if min(lows) > 0 else 0.01 * closes[-1]

        trs = []
        for i in range(1, n):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
            trs.append(tr)

        alpha = 1.0 / period
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0.0

        atr = sum(trs[:period]) / period
        for i in range(period, len(trs)):
            atr = atr * (1 - alpha) + trs[i] * alpha

        return float(atr)

    @staticmethod
    def calc_bollinger(prices: List[float], period: int = 20,
                        std_mult: float = 2.0) -> Dict[str, float]:
        """
        布林带计算
        价格触及下轨: 潜在买入
        价格触及上轨: 潜在卖出
        """
        if len(prices) < period:
            last_price = prices[-1] if prices else 0
            return {
                "upper": last_price * 1.05,
                "middle": last_price,
                "lower": last_price * 0.95,
                "bandwidth": 0.10
            }

        prices_arr = np.array(prices[-period:])
        middle = np.mean(prices_arr)
        std = np.std(prices_arr)

        return {
            "upper": float(middle + std_mult * std),
            "middle": float(middle),
            "lower": float(middle - std_mult * std),
            "bandwidth": float(2 * std_mult * std / middle) if middle != 0 else 0.0
        }

    @staticmethod
    def calc_momentum(prices: List[float], period: int = 10) -> float:
        """
        价格动量: 涨跌幅百分比
        正值: 上涨动量
        负值: 下跌动量
        """
        if len(prices) < period + 1:
            return 0.0

        current = prices[-1]
        past = prices[-period - 1]
        momentum = ((current - past) / past) * 100
        return float(momentum)

    @staticmethod
    def calc_volume_filter(volumes: List[float], period: int = 20) -> Dict[str, Any]:
        """
        成交量过滤器 — 识别假突破
        
        真实突破需要成交量放大确认：
        - 放量突破 = 真实信号
        - 缩量突破 = 假信号，降低置信度
        
        Returns:
            {
                "volume_ratio": float,   # 当前量/20日均量
                "is_confirmed": bool,     # 是否放量确认
                "confidence_penalty": float  # 置信度惩罚（0-0.3）
            }
        """
        if len(volumes) < period + 1:
            return {"volume_ratio": 1.0, "is_confirmed": True, "confidence_penalty": 0.0}
        
        recent_volumes = np.array(volumes[-period:])
        avg_volume = np.mean(recent_volumes)
        current_volume = volumes[-1]
        
        if avg_volume <= 0:
            return {"volume_ratio": 1.0, "is_confirmed": True, "confidence_penalty": 0.0}
        
        volume_ratio = current_volume / avg_volume
        
        # 放量标准：成交量 > 1.5倍均线
        is_confirmed = volume_ratio >= 1.5
        
        # 缩量惩罚：成交量 < 0.7倍均线时降置信度
        if volume_ratio < 0.7:
            confidence_penalty = 0.3  # 降低30%置信度
        elif volume_ratio < 1.0:
            confidence_penalty = 0.15
        elif volume_ratio >= 1.5:
            confidence_penalty = 0.0  # 放量，不惩罚
        else:
            confidence_penalty = 0.0
        
        return {
            "volume_ratio": float(volume_ratio),
            "is_confirmed": is_confirmed,
            "confidence_penalty": confidence_penalty,
            "avg_volume": float(avg_volume),
            "current_volume": float(current_volume)
        }

    @staticmethod
    def calc_all(prices: List[float], highs: List[float],
                  lows: List[float], timeframe: str = "1H") -> Dict[str, Any]:
        """一次性计算所有价格因子
        
        Args:
            prices: 价格列表
            highs: 最高价列表
            lows: 最低价列表
            timeframe: 时间周期 ("1H" 或 "4H")
        """
        pf = PriceFactors()
        rsi = pf.calc_rsi(prices)
        adx_data = pf.calc_adx(highs, lows, prices)
        macd_data = pf.calc_macd(prices)
        bollinger = pf.calc_bollinger(prices)
        momentum = pf.calc_momentum(prices)

        # EMA计算（用于趋势判断）
        ema20 = pf._calc_ema(prices, 20)
        ema50 = pf._calc_ema(prices, 50)
        ema200 = pf._calc_ema(prices, 200)

        # MACD方向（用于多周期确认）
        macd_direction = "bull" if macd_data["histogram"] > 0 else "bear"

        return {
            "rsi": rsi,
            "adx": adx_data["adx"],
            "plus_di": adx_data["plus_di"],
            "minus_di": adx_data["minus_di"],
            "macd": macd_data["macd"],
            "macd_signal": macd_data["signal"],
            "macd_histogram": macd_data["histogram"],
            "macd_direction": macd_direction,
            "bollinger_upper": bollinger["upper"],
            "bollinger_middle": bollinger["middle"],
            "bollinger_lower": bollinger["lower"],
            "bollinger_bandwidth": bollinger["bandwidth"],
            "momentum": momentum,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "current_price": prices[-1] if prices else 0,
            "timeframe": timeframe
        }

    @staticmethod
    def calc_all_4h(prices: List[float], highs: List[float],
                    lows: List[float], volumes: List[float] = None) -> Dict[str, Any]:
        """
        计算4H时间周期的价格因子（用于趋势确认）
        
        4H因子用于:
        - 趋势确认（ADX > 20表示趋势有效）
        - RSI极端区域确认（4H RSI超买/超卖强化信号）
        - 成交量确认（4H放量验证突破）
        """
        pf = PriceFactors()
        rsi = pf.calc_rsi(prices)
        adx_data = pf.calc_adx(highs, lows, prices)
        macd_data = pf.calc_macd(prices)
        bollinger = pf.calc_bollinger(prices)
        momentum = pf.calc_momentum(prices)

        # EMA计算（用于趋势判断）
        ema20 = pf._calc_ema(prices, 20)
        ema50 = pf._calc_ema(prices, 50)
        ema200 = pf._calc_ema(prices, 200)

        # MACD方向
        macd_direction = "bull" if macd_data["histogram"] > 0 else "bear"

        # 成交量比率（4H）
        volume_ratio = 1.0
        if volumes and len(volumes) >= 20:
            vol_filter = pf.calc_volume_filter(volumes)
            volume_ratio = vol_filter.get("volume_ratio", 1.0)

        return {
            "rsi": rsi,
            "adx": adx_data["adx"],
            "plus_di": adx_data["plus_di"],
            "minus_di": adx_data["minus_di"],
            "macd": macd_data["macd"],
            "macd_signal": macd_data["signal"],
            "macd_histogram": macd_data["histogram"],
            "macd_direction": macd_direction,
            "bollinger_upper": bollinger["upper"],
            "bollinger_middle": bollinger["middle"],
            "bollinger_lower": bollinger["lower"],
            "bollinger_bandwidth": bollinger["bandwidth"],
            "momentum": momentum,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "current_price": prices[-1] if prices else 0,
            "volume_ratio": volume_ratio,
            "timeframe": "4H"
        }

    @staticmethod
    def _calc_ema(prices: List[float], period: int) -> float:
        """计算EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)


# ============================================================================
# 2. 趋势判断
# ============================================================================

class TrendDetector:
    """趋势方向和强度检测"""

    @staticmethod
    def detect_trend(prices: List[float], highs: List[float],
                     lows: List[float]) -> Dict[str, Any]:
        """
        检测趋势方向和强度
        返回: {
            "trend": "bull" / "bear" / "range",
            "strength": 0-100,
            "ema20": float,
            "ema50": float,
            "ema200": float
        }
        """
        pf = PriceFactors()
        factors = pf.calc_all(prices, highs, lows)

        ema20 = factors["ema20"]
        ema50 = factors["ema50"]
        ema200 = factors["ema200"]
        adx = factors["adx"]
        current = factors["current_price"]

        # 趋势判断逻辑
        # 多头: EMA20 > EMA50 > EMA200, 价格在各均线上方
        # 空头: EMA20 < EMA50 < EMA200, 价格在各均线下方

        bull_signals = 0
        bear_signals = 0

        if ema20 > ema50:
            bull_signals += 1
        else:
            bear_signals += 1

        if ema50 > ema200:
            bull_signals += 1
        else:
            bear_signals += 1

        if current > ema20:
            bull_signals += 1
        else:
            bear_signals += 1

        # MACD方向
        if factors["macd_histogram"] > 0:
            bull_signals += 1
        else:
            bear_signals += 1

        # 趋势强度计算（高频模式：降低ADX依赖，更注重价格动量）
        # 基础分 = ADX * 0.8（降低ADX权重）
        trend_strength_base = adx * 0.8

        # 加上动量加分（如果价格有明显趋势）
        momentum = factors.get("momentum", 0)
        momentum_bonus = min(abs(momentum) * 3, 15)  # 最多加15分

        # 多头信号多 → 加分
        trend_strength = min(trend_strength_base + momentum_bonus + (bull_signals - 2) * 8, 100)

        if bull_signals >= 3 and adx > 15:
            trend = "bull"
        elif bear_signals >= 3 and adx > 15:
            trend = "bear"
        else:
            trend = "range"

        return {
            "trend": trend,
            "strength": float(trend_strength),
            "ema20": float(ema20),
            "ema50": float(ema50),
            "ema200": float(ema200),
            "adx": float(adx),
            "bull_signals": bull_signals,
            "bear_signals": bear_signals
        }


# ============================================================================
# 3. 白名单/黑名单过滤
# ============================================================================

class WhitelistFilter:
    """
    基于历史表现的白名单模式过滤

    规则:
    - RSI 35-45 + ADX > 30 → 通过 (低估反弹模式)
    - RSI 55-65 + ADX 30-40 → 通过 (上升中继模式)
    - RSI < 25 或 RSI > 75 → 降低置信度 (极端信号)
    - 其他模式 → 拒绝或降低权重
    """

    def __init__(self):
        # 历史模式表现缓存
        self.pattern_stats: Dict[str, Dict[str, Any]] = {}
        self.blacklist: set = set()

    def check(self, signal: Dict, factors: Dict,
              factors_4h: Optional[Dict] = None) -> Dict[str, Any]:
        """
        检查信号是否通过白名单过滤
        返回: {"passed": bool, "confidence_modifier": float, "reason": str}

        RSI多空阈值设计（与calc_price_score保持一致）:
        - RSI < 30: 强烈买入（最佳赔率点）
        - RSI 30-40: 偏低买入
        - RSI 40-50: 中性偏多
        - RSI 50-60: 中性
        - RSI 60-70: 偏高卖出
        - RSI > 70: 强烈卖出
        
        Args:
            signal: 信号字典
            factors: 1H价格因子
            factors_4h: 4H价格因子（可选，用于更精细的4H regime过滤）
        """
        rsi = factors.get("rsi", 50)
        adx = factors.get("adx", 0)
        trend = factors.get("trend", "range")

        passed = False
        confidence_modifier = 1.0
        reason = ""

        # ── 白名单模式检查 ──
        # ADX < 20 表示趋势不够强，直接拒绝
        if adx < 20:
            passed = False
            confidence_modifier = 0.0
            reason = "no_trend_adx_low"

        # 强烈做多信号: RSI < 40
        elif rsi < 40:
            passed = True
            confidence_modifier = 1.4
            reason = "oversold_strong_long"

        # 强烈做空信号: RSI > 60
        elif rsi > 60:
            passed = True
            confidence_modifier = 1.3
            reason = "overbought_strong_short"

        # 偏低买入: RSI 40-50 + ADX >= 20
        elif 40 <= rsi < 50 and adx >= 20:
            passed = True
            confidence_modifier = 1.2
            reason = "oversold_rebound_pattern"

        # 偏高卖出: RSI 50-60 + ADX >= 20
        elif 50 < rsi <= 60 and adx >= 20:
            passed = True
            confidence_modifier = 1.1
            reason = "overbought_reversal_pattern"

        # 中性区域: RSI 40-60，ADX >= 20 时可以通过
        elif 40 <= rsi <= 60 and adx >= 20:
            passed = True
            confidence_modifier = 0.85
            reason = "neutral_with_trend"

        # 其他模式拒绝
        else:
            passed = False
            confidence_modifier = 0.0
            reason = "non_ideal_pattern"

        # === 黑名单检查（支持4H regime） ===
        pattern_key = self._get_pattern_key(rsi, adx, trend, factors_4h)
        if pattern_key in self.blacklist:
            passed = False
            confidence_modifier = 0.0
            reason = f"blacklisted_pattern:{reason}"

        return {
            "passed": passed,
            "confidence_modifier": confidence_modifier,
            "reason": reason,
            "pattern_key": pattern_key
        }

    def _get_pattern_key(self, rsi: float, adx: float, trend: str,
                         factors_4h: Optional[Dict] = None) -> str:
        """
        生成模式键
        
        Args:
            rsi: RSI值
            adx: ADX值
            trend: 趋势方向
            factors_4h: 4H价格因子（可选）
        """
        rsi_bucket = "low" if rsi < 40 else ("mid" if rsi < 60 else "high")
        adx_bucket = "low" if adx < 25 else ("mid" if adx < 40 else "high")
        
        # 如果有4H因子，加入4H regime信息
        if factors_4h is not None:
            regime_4h = MultiTimeframeFilter.get_4h_regime(factors_4h)
            return f"{trend}_{rsi_bucket}_rsi_{adx_bucket}_adx_4h_{regime_4h}"
        
        return f"{trend}_{rsi_bucket}_rsi_{adx_bucket}_adx"

    def update_pattern_db(self, pattern_key: str, won: bool, actual_rr: float):
        """
        交易结束后更新模式数据库
        用于自学习，调整模式权重
        """
        if pattern_key not in self.pattern_stats:
            self.pattern_stats[pattern_key] = {
                "total": 0,
                "wins": 0,
                "total_rr": 0.0,
                "win_rate": 0.5
            }

        stats = self.pattern_stats[pattern_key]
        stats["total"] += 1
        if won:
            stats["wins"] += 1
        stats["total_rr"] += actual_rr
        stats["win_rate"] = stats["wins"] / stats["total"]
        stats["avg_rr"] = stats["total_rr"] / stats["total"]

        # 如果胜率 < 40%，加入黑名单（最少20笔才能做统计判断）
        if stats["total"] >= 20 and stats["win_rate"] < 0.4:
            self.blacklist.add(pattern_key)

    def get_pattern_stats(self, pattern_key: str) -> Dict[str, Any]:
        """查询模式历史表现"""
        return self.pattern_stats.get(pattern_key, {
            "total": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0
        })

    def add_to_blacklist(self, pattern_key: str):
        """手动加入黑名单"""
        self.blacklist.add(pattern_key)

    def remove_from_blacklist(self, pattern_key: str):
        """从黑名单移除"""
        self.blacklist.discard(pattern_key)


# ============================================================================
# 3b. 多周期过滤 (1H + 4H 双重确认)
# ============================================================================

class MultiTimeframeFilter:
    """
    多周期过滤器 - 基于Kronos 1H+4H双层确认架构
    
    原理:
    - 1H: 主要趋势/入场信号
    - 4H: 趋势确认，避免假信号
    - 入场条件: 1H和4H方向一致
    
    确认逻辑:
    1. 方向一致性: 1H信号方向 == 4H MACD方向
    2. 趋势强度: 4H ADX > 20 表示趋势有效
    3. RSI极端区域: 如果1H RSI极端，4H RSI应在同一区域
    4. 成交量确认: 4H成交量放大验证突破
    """

    @staticmethod
    def confirm(signal_1h: Dict, factors_1h: Dict, factors_4h: Dict) -> Dict:
        """
        多周期确认检查
        
        Args:
            signal_1h: 1H信号 dict，包含 direction
            factors_1h: 1H价格因子 dict
            factors_4h: 4H价格因子 dict
            
        Returns:
            {
                "confirmed": bool,          # 是否通过确认
                "confidence_boost": float,  # 置信度调整 (0.0-1.0)
                "confirmations": int,       # 通过的确认项数量
                "total_checks": int,        # 总检查项数量
                "check_details": dict       # 各检查项详情
            }
        """
        confirmations = 0
        total_checks = 4
        
        check_details = {}
        
        # === Check 1: 方向一致性 ===
        # 1H信号方向与4H MACD方向一致
        signal_direction = signal_1h.get('direction', 'wait')
        macd_direction_4h = factors_4h.get('macd_direction', 'bear')
        
        if signal_direction == 'wait':
            check_details['direction_check'] = {
                'passed': False,
                'reason': 'no_1h_signal'
            }
        elif signal_direction == macd_direction_4h:
            confirmations += 1
            check_details['direction_check'] = {
                'passed': True,
                'reason': f'1H {signal_direction} aligns with 4H {macd_direction_4h}'
            }
        else:
            check_details['direction_check'] = {
                'passed': False,
                'reason': f'1H {signal_direction} conflicts with 4H {macd_direction_4h}'
            }
        
        # === Check 2: 4H趋势强度 ===
        # ADX > 20 表示趋势有效
        adx_4h = factors_4h.get('adx', 0)
        if adx_4h > 20:
            confirmations += 1
            check_details['trend_strength_check'] = {
                'passed': True,
                'reason': f'4H ADX {adx_4h:.1f} > 20, trend is valid'
            }
        else:
            check_details['trend_strength_check'] = {
                'passed': False,
                'reason': f'4H ADX {adx_4h:.1f} <= 20, weak trend'
            }
        
        # === Check 3: RSI极端区域一致性 ===
        # 如果1H RSI极端，4H RSI应该在同一区域（4H更慢所以更可靠）
        rsi_1h = factors_1h.get('rsi', 50)
        rsi_4h = factors_4h.get('rsi', 50)
        
        rsi_1h_extreme = rsi_1h < 35 or rsi_1h > 65
        rsi_4h_extreme = rsi_4h < 40 or rsi_4h > 60  # 4H阈值更宽松
        
        if rsi_1h_extreme:
            # 1H极端时，4H也应该极端
            if rsi_4h_extreme:
                # 进一步检查是否在同一方向
                both_oversold = rsi_1h < 35 and rsi_4h < 40
                both_overbought = rsi_1h > 65 and rsi_4h > 60
                if both_oversold or both_overbought:
                    confirmations += 1
                    check_details['rsi_extreme_check'] = {
                        'passed': True,
                        'reason': f'Both in extreme zone: 1H RSI {rsi_1h:.1f}, 4H RSI {rsi_4h:.1f}'
                    }
                else:
                    check_details['rsi_extreme_check'] = {
                        'passed': False,
                        'reason': f'RSI mismatch: 1H {rsi_1h:.1f}, 4H {rsi_4h:.1f}'
                    }
            else:
                check_details['rsi_extreme_check'] = {
                    'passed': False,
                    'reason': f'1H extreme but 4H not: 1H {rsi_1h:.1f}, 4H {rsi_4h:.1f}'
                }
        else:
            # 1H非极端，4H极端可以作为额外确认
            if rsi_4h_extreme:
                confirmations += 0.5  # 部分确认
                check_details['rsi_extreme_check'] = {
                    'passed': True,
                    'partial': True,
                    'reason': f'4H extreme zone confirms: 4H RSI {rsi_4h:.1f}'
                }
            else:
                confirmations += 1  # 两者都不是极端，中性通过
                check_details['rsi_extreme_check'] = {
                    'passed': True,
                    'reason': f'Neither in extreme zone: neutral confirmation'
                }
        
        # === Check 4: 4H成交量确认 ===
        # 放量突破需要成交量放大确认
        volume_ratio_4h = factors_4h.get('volume_ratio', 1.0)
        if volume_ratio_4h > 1.3:
            confirmations += 1
            check_details['volume_check'] = {
                'passed': True,
                'reason': f'4H volume ratio {volume_ratio_4h:.2f} > 1.3'
            }
        elif volume_ratio_4h > 1.0:
            check_details['volume_check'] = {
                'passed': True,
                'partial': True,
                'reason': f'4H volume ratio {volume_ratio_4h:.2f} > 1.0, marginal'
            }
        else:
            check_details['volume_check'] = {
                'passed': False,
                'reason': f'4H volume ratio {volume_ratio_4h:.2f} <= 1.0, weak'
            }
        
        # 计算置信度调整
        confidence_boost = confirmations / total_checks  # 0.0 to 1.0
        
        # 判断是否确认通过（至少3/4项通过）
        confirmed = confirmations >= 3
        
        return {
            "confirmed": confirmed,
            "confidence_boost": confidence_boost,
            "confirmations": confirmations,
            "total_checks": total_checks,
            "check_details": check_details
        }

    @staticmethod
    def get_4h_regime(factors_4h: Dict) -> str:
        """
        获取4H市场状态（用于WhitelistFilter模式键）
        
        Returns:
            "bull_trending" | "bear_trending" | "bull_ranging" | "bear_ranging" | "range_bound"
        """
        adx = factors_4h.get('adx', 0)
        macd_dir = factors_4h.get('macd_direction', 'bear')
        rsi = factors_4h.get('rsi', 50)
        
        if adx < 20:
            return "range_bound"
        
        if macd_dir == 'bull':
            return "bull_trending" if adx > 30 else "bull_ranging"
        else:
            return "bear_trending" if adx > 30 else "bear_ranging"


# ============================================================================
# 4. 多因子融合信号生成器
# ============================================================================

class SignalGenerator:
    """
    多因子融合信号生成器

    权重配置:
    - price_momentum: 0.6 (价格动量权重)
    - news_sentiment: 0.2 (新闻情绪权重)
    - onchain: 0.1 (链上数据权重)
    - wallet: 0.1 (钱包数据权重)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 加载IC权重，替换硬编码的0.6/0.2/0.1/0.1
        # IC权重反映各因子历史预测精度
        self._load_ic_weights()
        
        self.whitelist = WhitelistFilter()

        # 自学习模式数据库
        self.pattern_db: Dict[str, List[Dict]] = defaultdict(list)
        
        # RegimeClassifier实例
        self._regime_classifier = RegimeClassifier()

    def _load_ic_weights(self) -> None:
        """
        从IC权重系统加载动态权重，替换硬编码权重
        
        IC权重因子: rsi, macd, adx, bollinger, momentum
        信号因子: price_momentum, news_sentiment, onchain, wallet
        
        映射策略:
        - price_momentum: 基于IC权重最高的三个技术因子(rsi, macd, momentum)的平均
        - news_sentiment: 基于IC权重中的macd(技术信号)
        - onchain: 使用固定较低权重(链上数据置信度低)
        - wallet: 使用固定较低权重(钱包数据置信度低)
        """
        try:
            ic_weights = get_ic_weights()
            
            # price_momentum: 综合RSI、MACD、动量的IC权重
            price_ic = (ic_weights.get('rsi', 0.2) + 
                        ic_weights.get('macd', 0.2) + 
                        ic_weights.get('momentum', 0.2)) / 3.0
            
            # news_sentiment: 使用ADX的IC权重(趋势确认类似新闻信号)
            news_ic = ic_weights.get('adx', 0.2)
            
            # onchain和wallet使用较低的固定权重(数据质量和覆盖率问题)
            onchain_ic = 0.1
            wallet_ic = 0.1
            
            # 归一化使总和为1.0
            total = price_ic + news_ic + onchain_ic + wallet_ic
            if total > 0:
                self.weights = {
                    "price_momentum": price_ic / total,
                    "news_sentiment": news_ic / total,
                    "onchain": onchain_ic / total,
                    "wallet": wallet_ic / total
                }
            else:
                # 回退到默认值
                self.weights = {
                    "price_momentum": 0.6,
                    "news_sentiment": 0.2,
                    "onchain": 0.1,
                    "wallet": 0.1
                }
                
            logger = logging.getLogger(__name__)
            logger.info(f"[SignalGenerator] IC权重已加载: {self.weights}")
            
        except Exception as e:
            # 如果IC加载失败，使用硬编码默认值
            self.weights = {
                "price_momentum": 0.6,
                "news_sentiment": 0.2,
                "onchain": 0.1,
                "wallet": 0.1
            }
            logger = logging.getLogger(__name__)
            logger.warning(f"[SignalGenerator] IC权重加载失败，使用默认权重: {e}")

    def calc_price_score(self, factors: Dict) -> float:
        """
        计算价格因子得分 (-1 到 1)
        正分 = 多头信号, 负分 = 空头信号
        """
        score = 0.0
        weights_sum = 0.0

        # RSI 评分 (权重 0.25)
        # 与WhitelistFilter保持一致的阈值设计
        rsi = factors.get("rsi", 50)
        if rsi < 30:
            rsi_score = 1.0  # 超卖 → 强烈买入（最佳赔率点）
        elif rsi < 40:
            rsi_score = 0.6  # 偏低 → 买入
        elif rsi <= 50:
            rsi_score = 0.3  # 中性偏多
        elif rsi <= 60:
            rsi_score = 0.0  # 中性
        elif rsi <= 70:
            rsi_score = -0.6  # 偏高 → 卖出
        else:
            rsi_score = -1.0  # 超买 → 强烈卖出
        score += rsi_score * 0.25
        weights_sum += 0.25

        # MACD 评分 (权重 0.25)
        macd_hist = factors.get("macd_histogram", 0)
        current_price = factors.get("current_price", 100)
        # 正确归一化：MACD直方图 / 价格 = 价格变动百分比
        # 这样可以消除价格量级的影响
        if abs(current_price) > 0:
            macd_normalized = macd_hist / current_price
        else:
            macd_normalized = 0.0
        # 映射到 -1 ~ 1 范围（0.01 = 1%价格变动）
        macd_score = max(-1.0, min(1.0, macd_normalized / 0.01))
        score += macd_score * 0.25
        weights_sum += 0.25

        # 动量评分 (权重 0.25)
        momentum = factors.get("momentum", 0)
        momentum_score = max(min(momentum / 10, 1.0), -1.0)  # ±10% 归一化
        score += momentum_score * 0.25
        weights_sum += 0.25

        # 趋势评分 (权重 0.25)
        trend = factors.get("trend", "range")
        if trend == "bull":
            trend_score = 1.0
        elif trend == "bear":
            trend_score = -1.0
        else:
            trend_score = 0.0
        score += trend_score * 0.25
        weights_sum += 0.25

        return score / weights_sum if weights_sum > 0 else 0.0

    def calc_news_score(self, intel_report: Dict) -> float:
        """
        从情报报告提取新闻情绪得分
        返回: -1 (极度利空) 到 1 (极度利好)

        兼容两种格式：
        - Agent-M 格式: intel_report['news_sentiment']['score']
        - 旧格式: intel_report['sentiment'] + intel_report['sentiment_score']
        """
        # Agent-M 格式优先（兼容新接口）
        news_sentiment = intel_report.get("news_sentiment", {})
        if isinstance(news_sentiment, dict):
            score = news_sentiment.get("score", 0.0)
            # 同时提取 sentiment 字符串用于方向判断
            labels = news_sentiment.get("labels", [])
            sentiment_str = "neutral"
            for label in labels:
                if "利好" in label or "bullish" in label.lower():
                    sentiment_str = "bullish"
                    break
                elif "利空" in label or "bearish" in label.lower():
                    sentiment_str = "bearish"
                    break
        else:
            # 旧格式兼容
            sentiment_str = intel_report.get("sentiment", "neutral")
            score = intel_report.get("sentiment_score", 0.0)

        if sentiment_str == "bullish":
            return min(score, 1.0)
        elif sentiment_str == "bearish":
            return max(score, -1.0)
        else:
            return score  # 中性

    def calc_onchain_score(self, intel_report: Dict) -> float:
        """
        从情报报告提取链上数据得分
        返回: -1 到 1

        兼容 Agent-M 格式: exchange_flow_signal (float)
        兼容旧格式: cvd_change, exchange_flow_ratio, active_address_change
        """
        onchain_data = intel_report.get("onchain", {})

        score = 0.0
        count = 0

        # Agent-M 格式: exchange_flow_signal（-1~1范围）
        if "exchange_flow_signal" in onchain_data:
            flow_signal = onchain_data["exchange_flow_signal"]
            score += flow_signal
            count += 1

        # 旧格式兼容
        if "cvd_change" in onchain_data:
            cvd = onchain_data["cvd_change"]
            score += (1.0 if cvd > 0 else -1.0) * min(abs(cvd) / 1000, 1.0)
            count += 1

        if "exchange_flow_ratio" in onchain_data:
            flow = onchain_data["exchange_flow_ratio"]
            if flow < 0.3:
                score += 0.5
            elif flow > 0.7:
                score -= 0.5
            count += 1

        if "active_address_change" in onchain_data:
            change = onchain_data["active_address_change"]
            score += max(min(change / 20, 1.0), -1.0)
            count += 1

        return score / count if count > 0 else 0.0

    def calc_wallet_score(self, intel_report: Dict) -> float:
        """
        从情报报告提取钱包/机构数据得分
        返回: -1 到 1

        兼容 Agent-M 格式: concentration_signal (float)
        兼容旧格式: institution_holding_change, whale_activity, etf_net_flow
        """
        wallet_data = intel_report.get("wallet", {})

        score = 0.0
        count = 0

        # Agent-M 格式: concentration_signal（-1~1范围）
        if "concentration_signal" in wallet_data:
            conc_signal = wallet_data["concentration_signal"]
            score += conc_signal
            count += 1

        # 旧格式兼容
        if "institution_holding_change" in wallet_data:
            change = wallet_data["institution_holding_change"]
            score += max(min(change / 5, 1.0), -1.0)
            count += 1

        if "whale_activity" in wallet_data:
            activity = wallet_data["whale_activity"]
            if activity == "accumulating":
                score += 0.7
            elif activity == "distributing":
                score -= 0.7
            count += 1

        if "etf_net_flow" in wallet_data:
            flow = wallet_data["etf_net_flow"]
            score += max(min(flow / 500, 1.0), -1.0)
            count += 1

        return score / count if count > 0 else 0.0

    def calc_combined_score(self, price_factors: Dict,
                             intel_report: Dict) -> Dict[str, float]:
        """
        多因子加权融合
        返回各因子得分和综合得分
        """
        price_score = self.calc_price_score(price_factors)
        news_score = self.calc_news_score(intel_report)
        onchain_score = self.calc_onchain_score(intel_report)
        wallet_score = self.calc_wallet_score(intel_report)

        # 归一化：price_score是-1~1，其他也是-1~1，权重和=1.0
        # 但实际calc_price_score返回-1~1，其他也是-1~1，直接加权即可
        # 唯一问题：price_score * 0.6 范围是-0.6~0.6，其他类似
        # 结果范围是-1~1，这是对的
        combined = (
            price_score * self.weights["price_momentum"] +
            news_score * self.weights["news_sentiment"] +
            onchain_score * self.weights["onchain"] +
            wallet_score * self.weights["wallet"]
        )

        # 额外检查：如果因子数据全为0（未接入真实API），降低置信度
        real_data_score = 1.0
        if abs(price_score) < 0.05:
            real_data_score *= 0.5  # 价格因子无效
        if abs(news_score) < 0.05:
            real_data_score *= 0.7  # 新闻因子无效（未接入）
        if abs(onchain_score) < 0.05:
            real_data_score *= 0.8  # 链上因子无效（未接入）
        if abs(wallet_score) < 0.05:
            real_data_score *= 0.9  # 钱包因子无效（未接入）

        return {
            "price_score": price_score,
            "news_score": news_score,
            "onchain_score": onchain_score,
            "wallet_score": wallet_score,
            "combined": combined,
            "weights_used": self.weights,
            "real_data_score": real_data_score  # 真实数据接入程度
        }

    def generate_signal(self, symbol: str, price_data: Dict,
                        intel_report: Dict,
                        price_data_4h: Optional[Dict] = None,
                        override_mt_filter: bool = False) -> Dict:
        """
        生成综合交易信号

        Args:
            symbol: 交易标的 (如 "BTC")
            price_data: 价格数据 {"prices": [], "highs": [], "lows": []}
            intel_report: Agent-M 情报报告
            price_data_4h: 4H价格数据 (可选，用于多周期确认)
            override_mt_filter: 跳过4H确认过滤器（用于特殊场景）

        Returns:
            交易信号字典
        """
        prices = price_data.get("prices", [])
        highs = price_data.get("highs", prices)
        lows = price_data.get("lows", prices)

        if len(prices) < 50:
            return self._wait_signal(symbol, "Insufficient price data")

        # === 1. 计算1H价格因子 ===
        factors = PriceFactors.calc_all(prices, highs, lows, timeframe="1H")

        # === 2. 趋势判断 ===
        trend_info = TrendDetector.detect_trend(prices, highs, lows)
        factors["trend"] = trend_info["trend"]
        factors["trend_strength"] = trend_info["strength"]

        # === 2b. 市场状态分类 (Regime Classification) ===
        # 使用RegimeClassifier进行市场状态分类
        regime_result = {"regime": "sideways", "confidence": 0.5, "metrics": None}
        try:
            if len(prices) >= 50:
                # 构建DataFrame供RegimeClassifier使用
                regime_df = pd.DataFrame({
                    'high': highs[-100:] if len(highs) >= 100 else highs,
                    'low': lows[-100:] if len(lows) >= 100 else lows,
                    'close': prices[-100:] if len(prices) >= 100 else prices
                })
                regime, regime_confidence, regime_metrics = self._regime_classifier.classify(regime_df)
                regime_result = {
                    "regime": regime.value if hasattr(regime, 'value') else str(regime),
                    "confidence": regime_confidence,
                    "metrics": {
                        "adx": regime_metrics.adx if regime_metrics else 0,
                        "plus_di": regime_metrics.plus_di if regime_metrics else 0,
                        "minus_di": regime_metrics.minus_di if regime_metrics else 0,
                        "momentum": regime_metrics.momentum if regime_metrics else 0
                    }
                }
                factors["regime"] = regime_result["regime"]
                factors["regime_confidence"] = regime_confidence
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"[SignalGenerator] RegimeClassification失败: {e}")

        # === 3. 成交量过滤 ===
        volumes = price_data.get("volumes", [])
        volume_filter_result = {"volume_ratio": 1.5, "is_confirmed": True, "confidence_penalty": 0.0}
        if len(volumes) >= 20:
            volume_filter_result = PriceFactors.calc_volume_filter(volumes)
        factors["volume_ratio"] = volume_filter_result["volume_ratio"]
        factors["volume_confirmed"] = volume_filter_result["is_confirmed"]

        # === 4. 多因子融合 ===
        scores = self.calc_combined_score(factors, intel_report)
        combined_score = scores["combined"]
        real_data_score = scores.get("real_data_score", 1.0)

        # === 5. 白名单过滤 ===
        whitelist_result = self.whitelist.check(scores, factors)
        confidence_modifier = whitelist_result["confidence_modifier"]

        # === 6. 计算基础置信度 ===
        # 基础置信度 = 趋势强度 + 综合得分
        base_confidence = (trend_info["strength"] / 100 * 0.5 +
                          (combined_score + 1) / 2 * 0.5)
        # 成交量惩罚（缩量突破降低置信度）
        volume_penalty = volume_filter_result["confidence_penalty"]
        # 真实数据接入程度因子（未接入API时降低置信度）
        confidence = base_confidence * confidence_modifier * real_data_score
        confidence = confidence * (1.0 - volume_penalty)  # 成交量惩罚
        confidence = max(0.0, min(confidence, 1.0))

        # === 6b. 多周期过滤 (1H + 4H) ===
        mt_filter_result = {
            "applied": False,
            "confirmed": True,  # 如果没有4H数据，默认通过
            "confidence_boost": 0.0,
            "confirmations": 0,
            "total_checks": 0,
            "check_details": {}
        }
        factors_4h = None
        
        if price_data_4h is not None and not override_mt_filter:
            prices_4h = price_data_4h.get("prices", [])
            highs_4h = price_data_4h.get("highs", prices_4h)
            lows_4h = price_data_4h.get("lows", prices_4h)
            volumes_4h = price_data_4h.get("volumes", [])
            
            if len(prices_4h) >= 50:
                # 计算4H因子
                factors_4h = PriceFactors.calc_all_4h(prices_4h, highs_4h, lows_4h, volumes_4h)
                
                # 构建临时信号用于确认
                temp_signal = {
                    "direction": "long" if combined_score > 0 else ("short" if combined_score < 0 else "wait")
                }
                
                # 运行多周期确认
                mt_filter_result = MultiTimeframeFilter.confirm(temp_signal, factors, factors_4h)
                
                # 根据确认结果调整置信度
                if mt_filter_result["confirmed"]:
                    # 确认通过：提升置信度
                    confidence_boost = mt_filter_result["confidence_boost"]
                    confidence = confidence * (1.0 + confidence_boost * 0.2)
                    confidence = min(confidence, 1.0)  # 不超过1.0
                else:
                    # 确认失败：降低置信度
                    confidence_boost = mt_filter_result["confidence_boost"]
                    confidence = confidence * confidence_boost  # 按比例降低
                
                mt_filter_result["applied"] = True

        # === 7. 方向判断 ===
        if abs(combined_score) < 0.05:  # 阈值降低到0.05以产生更多信号（高频模式）
            direction = "wait"
            entry_price = None
            stop_loss = None
            take_profit = None
            rr_ratio = 0.0
        else:
            entry_price = prices[-1]

            # 使用ATR计算止损（更科学，考虑波动率）
            # ATR计算：使用布林带中轨作为波动率代理
            pf = PriceFactors()
            atr_value = factors.get("atr", pf.calc_atr(
                factors.get("highs", prices),
                factors.get("lows", prices),
                prices
            ))
            atr_multiplier = 3  # 3倍ATR作为止损

            # 计算风险金额（止损距离）
            stop_distance = atr_value * atr_multiplier

            # 目标RR = 2.5（赔率优先）
            target_rr = 2.5

            if combined_score > 0:
                direction = "long"
                # 止损：入场价 - 3倍ATR
                stop_loss = entry_price - stop_distance
                # 止盈：入场价 + 止损距离 * RR
                take_profit = entry_price + stop_distance * target_rr
            else:
                direction = "short"
                # 止损：入场价 + 3倍ATR
                stop_loss = entry_price + stop_distance
                # 止盈：入场价 - 止损距离 * RR
                take_profit = entry_price - stop_distance * target_rr

            rr_ratio = target_rr

            # 更新factors中的atr值（供风控模块使用）
            factors["atr"] = atr_value

        # === 8. 多周期确认后的最终方向判断 ===
        # 只有在置信度 >= 0.3 且 (确认通过 或 override) 时才执行
        if direction != "wait":
            if not override_mt_filter and mt_filter_result["applied"]:
                if confidence < 0.3 or not mt_filter_result["confirmed"]:
                    # 多周期确认失败，降低为wait
                    direction = "wait"
                    entry_price = None
                    stop_loss = None
                    take_profit = None
                    rr_ratio = 0.0

        # === 9. 建议杠杆和仓位 ===
        leverage = 1
        if confidence > 0.75 and trend_info["strength"] > 60:
            leverage = 2
        if confidence > 0.85 and trend_info["strength"] > 75 and abs(combined_score) > 0.5:
            leverage = 3

        position_size = confidence * 0.2  # 最大20%仓位

        # === 10. 生成信号理由 ===
        reason = self._generate_reason(direction, scores, factors, whitelist_result)

        # === 11. 组装信号 ===
        signal = {
            "symbol": symbol,
            "direction": direction,
            "signal_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry_price": round(entry_price, 2) if entry_price else None,
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "take_profit": round(take_profit, 2) if take_profit else None,
            "rr_ratio": round(rr_ratio, 2),
            "confidence": round(confidence, 4),
            "trend_strength": round(trend_info["strength"], 2),
            "leverage_recommended": leverage,
            "position_size_pct": round(position_size, 4),
            "factors": {
                "price_score": round(scores["price_score"], 4),
                "news_score": round(scores["news_score"], 4),
                "onchain_score": round(scores["onchain_score"], 4),
                "wallet_score": round(scores["wallet_score"], 4),
                "combined": round(scores["combined"], 4),
                "atr": atr_value if 'atr_value' in dir() else factors.get("atr", 0),
                "rsi": factors.get("rsi", 50),
                "adx": factors.get("adx", 25),
                "trend": factors.get("trend", "range")
            },
            "reason": reason,
            "whitelist_passed": whitelist_result["passed"],
            "pattern_key": whitelist_result.get("pattern_key", ""),
            "trend_info": {
                "trend": trend_info["trend"],
                "ema20": round(trend_info["ema20"], 2),
                "ema50": round(trend_info["ema50"], 2),
                "ema200": round(trend_info["ema200"], 2)
            },
            "regime_info": {
                "regime": regime_result["regime"],
                "confidence": round(regime_result["confidence"], 4),
                "metrics": regime_result["metrics"]
            },
            "volume_info": {
                "volume_ratio": round(volume_filter_result["volume_ratio"], 2),
                "is_confirmed": volume_filter_result["is_confirmed"],
                "confidence_penalty": round(volume_filter_result["confidence_penalty"], 3),
                "has_data": len(volumes) >= 20
            },
            "multi_timeframe": {
                "applied": mt_filter_result["applied"],
                "confirmed": mt_filter_result["confirmed"],
                "confidence_boost": round(mt_filter_result["confidence_boost"], 4),
                "confirmations": mt_filter_result["confirmations"],
                "total_checks": mt_filter_result["total_checks"],
                "check_details": mt_filter_result["check_details"],
                "4h_regime": MultiTimeframeFilter.get_4h_regime(factors_4h) if factors_4h else None,
                "factors_4h": {
                    "rsi": round(factors_4h["rsi"], 1) if factors_4h else None,
                    "adx": round(factors_4h["adx"], 1) if factors_4h else None,
                    "macd_direction": factors_4h.get("macd_direction") if factors_4h else None,
                    "volume_ratio": round(factors_4h.get("volume_ratio", 1.0), 2) if factors_4h else None
                } if factors_4h else None
            }
        }

        return signal

    def _wait_signal(self, symbol: str, reason: str) -> Dict:
        """生成等待信号"""
        return {
            "symbol": symbol,
            "direction": "wait",
            "signal_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "rr_ratio": 0.0,
            "confidence": 0.0,
            "trend_strength": 0.0,
            "leverage_recommended": 1,
            "position_size_pct": 0.0,
            "factors": {
                "price_score": 0.0,
                "news_score": 0.0,
                "onchain_score": 0.0,
                "wallet_score": 0.0,
                "combined": 0.0
            },
            "reason": reason,
            "whitelist_passed": False
        }

    def _generate_reason(self, direction: str, scores: Dict,
                          factors: Dict, whitelist: Dict) -> str:
        """生成信号理由文本"""
        reasons = []

        if direction == "wait":
            return "No clear signal - combined score below threshold"

        # 价格因子理由
        if scores["price_score"] > 0.3:
            reasons.append(f"Price momentum positive ({scores['price_score']:.2f})")
        elif scores["price_score"] < -0.3:
            reasons.append(f"Price momentum negative ({scores['price_score']:.2f})")

        # RSI理由
        rsi = factors.get("rsi", 50)
        if rsi < 30:
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # MACD理由
        if factors.get("macd_histogram", 0) > 0:
            reasons.append("MACD bullish")
        else:
            reasons.append("MACD bearish")

        # 趋势理由
        trend = factors.get("trend", "range")
        if trend != "range":
            reasons.append(f"{trend.capitalize()} trend confirmed")

        # 新闻理由
        if abs(scores["news_score"]) > 0.3:
            direction_word = "bullish" if scores["news_score"] > 0 else "bearish"
            reasons.append(f"News sentiment {direction_word}")

        # 白名单理由
        if whitelist["passed"]:
            reasons.append(f"Pattern: {whitelist['reason']}")
        else:
            reasons.append(f"Pattern rejected: {whitelist['reason']}")

        return "; ".join(reasons)

    # =========================================================================
    # 自学习接口
    # =========================================================================

    def update_pattern_db(self, trade_result: Dict):
        """
        交易结束后更新模式数据库
        trade_result: {
            "pattern_key": "long_RSI35-45_ADX30+",
            "actual_rr": 2.3,
            "won": bool
        }
        """
        pattern_key = trade_result.get("pattern_key", "")
        won = trade_result.get("won", False)
        actual_rr = trade_result.get("actual_rr", 0.0)

        self.pattern_db[pattern_key].append({
            "won": won,
            "rr": actual_rr,
            "timestamp": datetime.utcnow().isoformat()
        })

        # 更新白名单过滤器
        self.whitelist.update_pattern_db(pattern_key, won, actual_rr)

    def get_pattern_stats(self, pattern_key: str) -> Dict[str, Any]:
        """
        查询模式历史表现
        返回: {total, wins, win_rate, avg_rr}
        """
        records = self.pattern_db.get(pattern_key, [])
        if not records:
            return self.whitelist.get_pattern_stats(pattern_key)

        total = len(records)
        wins = sum(1 for r in records if r["won"])
        total_rr = sum(r["rr"] for r in records)

        return {
            "total": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "avg_rr": total_rr / total if total > 0 else 0.0,
            "records": records
        }

    def get_all_patterns_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模式的统计"""
        return {
            key: {
                "total": len(records),
                "wins": sum(1 for r in records if r["won"]),
                "win_rate": sum(1 for r in records if r["won"]) / len(records) if records else 0,
                "avg_rr": sum(r["rr"] for r in records) / len(records) if records else 0
            }
            for key, records in self.pattern_db.items()
        }


# ============================================================================
# 5. Agent-S 主类（对接Agent-M和Agent-R）
# ============================================================================

class AgentSignal:
    """
    Agent-S: 信号生成Agent

    对接:
    - 输入: Agent-M 的市场情报报告
    - 输出: 高置信度交易信号给 Agent-R
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.generator = SignalGenerator(self.config)

    def process_intel(self, symbol: str, price_data: Dict,
                       intel_report: Dict,
                       price_data_4h: Optional[Dict] = None) -> Dict:
        """
        处理Agent-M情报，生成交易信号

        Args:
            symbol: 交易标的
            price_data: 价格数据 {"prices": [], "highs": [], "lows": []}
            intel_report: Agent-M情报报告
            price_data_4h: 4H价格数据 (可选，用于多周期确认)

        Returns:
            高置信度交易信号
        """
        signal = self.generator.generate_signal(symbol, price_data, intel_report,
                                               price_data_4h=price_data_4h)
        return signal

    def feedback(self, signal_id: str, trade_result: Dict):
        """
        接收Agent-R的交易结果反馈，用于自学习

        Args:
            signal_id: 信号ID
            trade_result: {pattern_key, actual_rr, won}
        """
        self.generator.update_pattern_db(trade_result)

    def get_stats(self) -> Dict:
        """获取信号统计"""
        return self.generator.get_all_patterns_stats()


# ============================================================================
# 6. 演示/测试代码
# ============================================================================

if __name__ == "__main__":
    import random

    # 模拟价格数据
    base_price = 72000
    prices = [base_price * (1 + random.uniform(-0.02, 0.025)) for _ in range(100)]
    highs = [p * 1.005 for p in prices]
    lows = [p * 0.995 for p in prices]

    price_data = {
        "prices": prices,
        "highs": highs,
        "lows": lows
    }

    # 模拟Agent-M情报报告
    intel_report = {
        "sentiment": "bullish",
        "sentiment_score": 0.6,
        "onchain": {
            "cvd_change": 500,
            "exchange_flow_ratio": 0.25,
            "active_address_change": 15
        },
        "wallet": {
            "institution_holding_change": 3.2,
            "whale_activity": "accumulating",
            "etf_net_flow": 300
        }
    }

    # 生成信号
    agent = AgentSignal()
    signal = agent.process_intel("BTC", price_data, intel_report)

    print("=" * 60)
    print("Agent-S Signal Generation Test")
    print("=" * 60)
    print(f"Symbol: {signal['symbol']}")
    print(f"Direction: {signal['direction']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Trend Strength: {signal['trend_strength']:.1f}")
    print(f"Entry Price: {signal['entry_price']}")
    print(f"Stop Loss: {signal['stop_loss']}")
    print(f"Take Profit: {signal['take_profit']}")
    print(f"RR Ratio: {signal['rr_ratio']}")
    print(f"Leverage: {signal['leverage_recommended']}x")
    print(f"Position Size: {signal['position_size_pct']:.2%}")
    print(f"Whitelist Passed: {signal['whitelist_passed']}")
    print("-" * 60)
    print(f"Reason: {signal['reason']}")
    print("-" * 60)
    print("Factor Scores:")
    for k, v in signal['factors'].items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)
