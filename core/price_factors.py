"""
PriceFactors: 价格技术指标计算器
从 agents/agent_signal.py 提取

职责：RSI / ADX / MACD / ATR / 布林带 / 动量 / 成交量 核心计算
依赖：numpy
"""

from typing import Any, Dict, List

import numpy as np


class PriceFactors:
    """价格技术指标计算器"""

    @staticmethod
    def calc_rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0

        prices = np.array(prices)
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

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
        if len(closes) < 2 * period + 1:
            return {"adx": 25.0, "plus_di": 25.0, "minus_di": 25.0}

        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)

        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

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

        dx_values = []
        plus_di_series = []
        minus_di_series = []

        atr = float(np.mean(tr[:period_int]))
        plus_di_smooth = float(np.mean(plus_dm[:period_int]))
        minus_di_smooth = float(np.mean(minus_dm[:period_int]))

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
                plus_di_pct = 100.0 * plus_di_smooth / atr
                minus_di_pct = 100.0 * minus_di_smooth / atr
                dx_i = 100.0 * abs(plus_di_smooth - minus_di_smooth) / (plus_di_smooth + minus_di_smooth)
                dx_values.append(dx_i)
                plus_di_series.append(plus_di_pct)
                minus_di_series.append(minus_di_pct)

        if len(dx_values) == 0:
            return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

        if len(dx_values) >= period_int:
            adx = float(np.mean(dx_values[:period_int]))
        else:
            adx = float(np.mean(dx_values))

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
