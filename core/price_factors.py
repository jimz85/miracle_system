from __future__ import annotations

"""
PriceFactors: 价格技术指标计算器
从 agents/agent_signal.py 提取

职责：RSI / ADX / MACD / ATR / 布林带 / 动量 / 成交量 核心计算
依赖：numpy, pandas (optional, for vectorized calculations via pandas.ewm)
"""

from typing import Any, Dict, List

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class PriceFactors:
    """价格技术指标计算器"""

    @staticmethod
    def calc_rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0

        # Try talib first (fastest, C-level)
        if HAS_TALIB:
            try:
                result = talib.RSI(np.array(prices, dtype=np.float64), period)
                return float(result)
            except Exception:
                pass

        # Fallback: pandas ewm vectorized (no Python for-loop)
        if HAS_PANDAS:
            prices_arr = np.array(prices)
            deltas = np.diff(prices_arr)
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)

            avg_gains = pd.Series(gains).ewm(span=period, adjust=False).mean().values
            avg_losses = pd.Series(losses).ewm(span=period, adjust=False).mean().values

            rs = avg_gains[-1] / (avg_losses[-1] + 1e-10)
            if avg_losses[-1] < 1e-10:
                return 100.0
            return float(100 - (100 / (1 + rs)))

        # Pure numpy fallback (no pandas, no for-loop)
        prices_arr = np.array(prices)
        deltas = np.diff(prices_arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        alpha = 1.0 / period
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for i in range(period, len(gains)):
            avg_gain = avg_gain + alpha * (gains[i] - avg_gain)
            avg_loss = avg_loss + alpha * (losses[i] - avg_loss)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def calc_adx(highs: List[float], lows: List[float], closes: List[float],
                  period: int = 14) -> Dict[str, float]:
        if len(closes) < 2 * period + 1:
            return {"adx": 25.0, "plus_di": 25.0, "minus_di": 25.0}

        # Try talib first (fastest, C-level)
        if HAS_TALIB:
            try:
                high_arr = np.array(highs, dtype=np.float64)
                low_arr = np.array(lows, dtype=np.float64)
                close_arr = np.array(closes, dtype=np.float64)
                adx_val = talib.ADX(high_arr, low_arr, close_arr, period)
                plus_di = talib.PLUS_DI(high_arr, low_arr, close_arr, period)
                minus_di = talib.MINUS_DI(high_arr, low_arr, close_arr, period)
                return {
                    "adx": float(adx_val),
                    "plus_di": float(plus_di),
                    "minus_di": float(minus_di),
                }
            except Exception:
                pass

        # Vectorized numpy (no Python for-loops for main smoothing)
        highs_arr = np.array(highs)
        lows_arr = np.array(lows)
        closes_arr = np.array(closes)

        tr1 = highs_arr[1:] - lows_arr[1:]
        tr2 = np.abs(highs_arr[1:] - closes_arr[:-1])
        tr3 = np.abs(lows_arr[1:] - closes_arr[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        up_move = highs_arr[1:] - highs_arr[:-1]
        down_move = lows_arr[:-1] - lows_arr[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        n = len(tr)
        period_int = int(period)
        alpha = 1.0 / period_int

        if n < period_int + 1:
            return {"adx": 25.0, "plus_di": 25.0, "minus_di": 25.0}

        # Use pandas ewm for vectorized smoothing (Wilder's smoothing via span=period, adjust=False)
        if HAS_PANDAS:
            tr_series = pd.Series(tr)
            plus_dm_series = pd.Series(plus_dm)
            minus_dm_series = pd.Series(minus_dm)

            atr_smooth = tr_series.ewm(span=period_int, adjust=False).mean()
            plus_dm_smooth = plus_dm_series.ewm(span=period_int, adjust=False).mean()
            minus_dm_smooth = minus_dm_series.ewm(span=period_int, adjust=False).mean()

            plus_di_vals = 100.0 * plus_dm_smooth / (atr_smooth + 1e-10)
            minus_di_vals = 100.0 * minus_dm_smooth / (atr_smooth + 1e-10)
            dx_vals = 100.0 * np.abs(plus_di_vals - minus_di_vals) / (plus_di_vals + minus_di_vals + 1e-10)

            # ADX is smoothed DX over `period` bars (Wilder's smoothing again)
            adx_series = dx_vals.ewm(span=period_int, adjust=False).mean()
            adx_val = float(adx_series.iloc[-1])
            plus_di_val = float(plus_di_vals.iloc[-1])
            minus_di_val = float(minus_di_vals.iloc[-1])
            return {
                "adx": adx_val,
                "plus_di": plus_di_val,
                "minus_di": minus_di_val,
            }

        # Pure numpy fallback (loop-based Wilder smoothing)
        atr = float(np.mean(tr[:period_int]))
        plus_di_smooth = float(np.mean(plus_dm[:period_int]))
        minus_di_smooth = float(np.mean(minus_dm[:period_int]))
        dx_vals = np.zeros(n)

        for i in range(period_int, n):
            atr = atr * (1 - alpha) + tr[i] * alpha
            plus_di_smooth = plus_di_smooth * (1 - alpha) + plus_dm[i] * alpha
            minus_di_smooth = minus_di_smooth * (1 - alpha) + minus_dm[i] * alpha
            di_sum = plus_di_smooth + minus_di_smooth
            if di_sum > 0 and atr > 0:
                dx_vals[i] = 100.0 * abs(plus_di_smooth - minus_di_smooth) / di_sum
            else:
                dx_vals[i] = 0.0

        # Smooth DX into ADX
        adx = float(np.mean(dx_vals[period_int:period_int * 2])) if n >= period_int * 2 else float(np.mean(dx_vals[period_int:]))
        for i in range(period_int):
            idx = period_int + i
            if idx < n:
                adx = adx * (1 - alpha) + dx_vals[idx] * alpha

        return {
            "adx": float(adx),
            "plus_di": float(100.0 * plus_di_smooth / (atr + 1e-10)),
            "minus_di": float(100.0 * minus_di_smooth / (atr + 1e-10)),
        }

    @staticmethod
    def calc_macd(prices: List[float], fast: int = 12, slow: int = 26,
                   signal: int = 9) -> Dict[str, float]:
        if len(prices) < slow + signal:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

        prices = np.array(prices)

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
        if len(prices) < period + 1:
            return 0.0

        current = prices[-1]
        past = prices[-period - 1]
        momentum = ((current - past) / past) * 100
        return float(momentum)

    @staticmethod
    def calc_volume_filter(volumes: List[float], period: int = 20) -> Dict[str, Any]:
        if len(volumes) < period + 1:
            return {"volume_ratio": 1.0, "is_confirmed": True, "is_rejected": False, "confidence_penalty": 0.0, "filter_reason": ""}

        recent_volumes = np.array(volumes[-period:])
        avg_volume = np.mean(recent_volumes)
        current_volume = volumes[-1]

        if avg_volume <= 0:
            return {"volume_ratio": 1.0, "is_confirmed": True, "is_rejected": False, "confidence_penalty": 0.0, "filter_reason": ""}

        volume_ratio = current_volume / avg_volume

        if volume_ratio < 0.3:
            return {
                "volume_ratio": float(volume_ratio),
                "is_confirmed": False,
                "is_rejected": True,
                "confidence_penalty": 0.5,
                "filter_reason": "volume_below_30pct_avg",
                "avg_volume": float(avg_volume),
                "current_volume": float(current_volume)
            }

        is_confirmed = volume_ratio >= 1.5

        if volume_ratio < 0.7:
            confidence_penalty = 0.3
        elif volume_ratio < 1.0:
            confidence_penalty = 0.15
        else:
            confidence_penalty = 0.0

        return {
            "volume_ratio": float(volume_ratio),
            "is_confirmed": is_confirmed,
            "is_rejected": False,
            "confidence_penalty": confidence_penalty,
            "filter_reason": "",
            "avg_volume": float(avg_volume),
            "current_volume": float(current_volume)
        }

    @staticmethod
    def calc_all(prices: List[float], highs: List[float],
                  lows: List[float], timeframe: str = "1H") -> Dict[str, Any]:
        pf = PriceFactors()
        rsi = pf.calc_rsi(prices)
        adx_data = pf.calc_adx(highs, lows, prices)
        macd_data = pf.calc_macd(prices)
        bollinger = pf.calc_bollinger(prices)
        momentum = pf.calc_momentum(prices)

        ema20 = pf._calc_ema(prices, 20)
        ema50 = pf._calc_ema(prices, 50)
        ema200 = pf._calc_ema(prices, 200)

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
        pf = PriceFactors()
        rsi = pf.calc_rsi(prices)
        adx_data = pf.calc_adx(highs, lows, prices)
        macd_data = pf.calc_macd(prices)
        bollinger = pf.calc_bollinger(prices)
        momentum = pf.calc_momentum(prices)

        ema20 = pf._calc_ema(prices, 20)
        ema50 = pf._calc_ema(prices, 50)
        ema200 = pf._calc_ema(prices, 200)

        macd_direction = "bull" if macd_data["histogram"] > 0 else "bear"

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
        if len(prices) < period:
            return prices[-1] if prices else 0
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)

    # ========== Funding Rate & OI Factors ==========

    @staticmethod
    def calc_funding_rate_factor(
        funding_rates: List[float],
        side: str = "long",
        high_funding_threshold: float = 0.0003
    ) -> Dict[str, Any]:
        """
        计算资金费率因子

        Args:
            funding_rates: 历史资金费率列表 (最新在最后), e.g. [0.0001, 0.0002, 0.0001]
            side: 当前交易方向 "long" or "short"
            high_funding_threshold: 高资金费率阈值，默认0.0003 (0.03%)

        Returns:
            dict: {
                "funding_rate": float,           # 当前资金费率
                "funding_rate_trend": float,     # 3期均值
                "funding_rate_direction": str,   # "increasing" / "decreasing" / "stable"
                "short_high_funding_boost": float,  # 做空时高funding信心加成
                "confidence_boost": float,       # 综合信心加成
                "is_high_funding": bool,         # 是否高资金费率
            }
        """
        if not funding_rates or len(funding_rates) == 0:
            return {
                "funding_rate": 0.0,
                "funding_rate_trend": 0.0,
                "funding_rate_direction": "stable",
                "short_high_funding_boost": 0.0,
                "confidence_boost": 0.0,
                "is_high_funding": False,
            }

        current_fr = funding_rates[-1]
        is_high_funding = abs(current_fr) > high_funding_threshold

        # 计算3期均值 (trend)
        lookback = min(3, len(funding_rates))
        funding_rate_trend = sum(funding_rates[-lookback:]) / lookback

        # 判断方向
        if len(funding_rates) >= 3:
            recent = sum(funding_rates[-3:]) / 3
            prev = sum(funding_rates[-4:-1]) / 3 if len(funding_rates) >= 4 else recent
            if recent > prev * 1.05:
                direction = "increasing"
            elif recent < prev * 0.95:
                direction = "decreasing"
            else:
                direction = "stable"
        else:
            direction = "stable"

        # 做空时高funding加成逻辑：
        # 做空者收到资金费率，如果资金费率>阈值，做空更安全
        short_high_funding_boost = 0.0
        if side.lower() == "short" and is_high_funding:
            # funding为正意味着多头支付空头，funding越高做空越有利
            short_high_funding_boost = min(abs(current_fr) / high_funding_threshold * 0.1, 0.2)
        elif side.lower() == "long" and is_high_funding:
            # 做多时高funding意味着持仓成本高，轻微惩罚
            short_high_funding_boost = -0.05

        # 方向加成：funding正在上升对做空有利
        if side.lower() == "short" and direction == "increasing":
            short_high_funding_boost += 0.05

        confidence_boost = short_high_funding_boost

        return {
            "funding_rate": float(current_fr),
            "funding_rate_trend": float(funding_rate_trend),
            "funding_rate_direction": direction,
            "short_high_funding_boost": float(short_high_funding_boost),
            "confidence_boost": float(confidence_boost),
            "is_high_funding": is_high_funding,
        }

    @staticmethod
    def calc_oi_change_rate(
        oi_history: List[float],
        period: int = 3
    ) -> Dict[str, Any]:
        """
        计算OI变化率作为过滤因子

        Args:
            oi_history: 历史OI列表 (最新在最后)
            period: 计算变化率的回看期

        Returns:
            dict: {
                "oi_current": float,        # 当前OI
                "oi_change_rate": float,    # 变化率 (百分比)
                "oi_direction": str,        # "increasing" / "decreasing" / "stable"
                "is_filtered": bool,        # 是否被过滤 (OI急剧下降可能意味趋势反转)
                "filter_reason": str,       # 过滤原因
                "confidence_penalty": float, # 信心惩罚
            }
        """
        if not oi_history or len(oi_history) < 2:
            return {
                "oi_current": oi_history[-1] if oi_history else 0.0,
                "oi_change_rate": 0.0,
                "oi_direction": "stable",
                "is_filtered": False,
                "filter_reason": "",
                "confidence_penalty": 0.0,
            }

        current_oi = oi_history[-1]

        if len(oi_history) < period + 1:
            prev_oi = oi_history[0]
        else:
            prev_oi = oi_history[-period - 1]

        if prev_oi > 0:
            oi_change_rate = (current_oi - prev_oi) / prev_oi
        else:
            oi_change_rate = 0.0

        # 判断方向
        if oi_change_rate > 0.05:  # 5%以上
            oi_direction = "increasing"
        elif oi_change_rate < -0.05:
            oi_direction = "decreasing"
        else:
            oi_direction = "stable"

        # 过滤逻辑：OI急剧下降(超过20%)可能是趋势反转信号，轻微惩罚
        is_filtered = False
        filter_reason = ""
        confidence_penalty = 0.0

        if oi_change_rate < -0.20:
            is_filtered = True
            filter_reason = "oi_sharp_decline"
            confidence_penalty = 0.15
        elif oi_change_rate < -0.10:
            filter_reason = "oi_moderate_decline"
            confidence_penalty = 0.05

        return {
            "oi_current": float(current_oi),
            "oi_change_rate": float(oi_change_rate),
            "oi_direction": oi_direction,
            "is_filtered": is_filtered,
            "filter_reason": filter_reason,
            "confidence_penalty": float(confidence_penalty),
        }
