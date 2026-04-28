"""
Signal Filters: 趋势检测 + 白名单过滤 + 多周期确认
从 agents/agent_signal.py 提取

职责：
- TrendDetector: 趋势方向和强度检测
- WhitelistFilter: 历史表现白名单模式过滤
- MultiTimeframeFilter: 1H+4H 双重确认

依赖：core.price_factors.PriceFactors
"""

from typing import Any, Dict, Optional

from core.price_factors import PriceFactors


class TrendDetector:
    """趋势方向和强度检测"""

    @staticmethod
    def detect_trend(prices: list, highs: list,
                     lows: list) -> Dict[str, Any]:
        pf = PriceFactors()
        factors = pf.calc_all(prices, highs, lows)

        ema20 = factors["ema20"]
        ema50 = factors["ema50"]
        ema200 = factors["ema200"]
        adx = factors["adx"]
        current = factors["current_price"]

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

        if factors["macd_histogram"] > 0:
            bull_signals += 1
        else:
            bear_signals += 1

        trend_strength_base = adx * 0.8
        momentum = factors.get("momentum", 0)
        momentum_bonus = min(abs(momentum) * 3, 15)
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
        self.pattern_stats: Dict[str, Dict[str, Any]] = {}
        self.blacklist: set = set()

    def check(self, signal: Dict, factors: Dict,
              factors_4h: Optional[Dict] = None) -> Dict[str, Any]:
        rsi = factors.get("rsi", 50)
        adx = factors.get("adx", 0)
        trend = factors.get("trend", "range")

        passed = False
        confidence_modifier = 1.0
        reason = ""

        if adx < 20:
            passed = False
            confidence_modifier = 0.0
            reason = "no_trend_adx_low"
        elif rsi < 40:
            passed = True
            confidence_modifier = 1.4
            reason = "oversold_strong_long"
        elif rsi > 60:
            passed = True
            confidence_modifier = 1.3
            reason = "overbought_strong_short"
        elif 40 <= rsi < 50 and adx >= 20:
            passed = True
            confidence_modifier = 1.2
            reason = "oversold_rebound_pattern"
        elif 50 < rsi <= 60 and adx >= 20:
            passed = True
            confidence_modifier = 1.1
            reason = "overbought_reversal_pattern"
        elif 40 <= rsi <= 60 and adx >= 20:
            passed = True
            confidence_modifier = 0.85
            reason = "neutral_with_trend"
        else:
            passed = False
            confidence_modifier = 0.0
            reason = "non_ideal_pattern"

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
        rsi_bucket = "low" if rsi < 40 else ("mid" if rsi < 60 else "high")
        adx_bucket = "low" if adx < 25 else ("mid" if adx < 40 else "high")

        if factors_4h is not None:
            regime_4h = MultiTimeframeFilter.get_4h_regime(factors_4h)
            return f"{trend}_{rsi_bucket}_rsi_{adx_bucket}_adx_4h_{regime_4h}"

        return f"{trend}_{rsi_bucket}_rsi_{adx_bucket}_adx"

    def update_pattern_db(self, pattern_key: str, won: bool, actual_rr: float):
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

        if stats["total"] >= 20 and stats["win_rate"] < 0.4:
            self.blacklist.add(pattern_key)

    def get_pattern_stats(self, pattern_key: str) -> Dict[str, Any]:
        return self.pattern_stats.get(pattern_key, {
            "total": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0
        })

    def add_to_blacklist(self, pattern_key: str):
        self.blacklist.add(pattern_key)

    def remove_from_blacklist(self, pattern_key: str):
        self.blacklist.discard(pattern_key)


class MultiTimeframeFilter:
    """
    多周期过滤器 - 基于 Kronos 1H+4H 双层确认架构

    原理:
    - 1H: 主要趋势/入场信号
    - 4H: 趋势确认，避免假信号
    - 入场条件: 1H 和 4H 方向一致
    """

    @staticmethod
    def confirm(signal_1h: Dict, factors_1h: Dict, factors_4h: Dict) -> Dict:
        confirmations = 0
        total_checks = 4
        check_details = {}

        # Check 1: 方向一致性
        signal_direction = signal_1h.get('direction', 'wait')
        macd_direction_4h = factors_4h.get('macd_direction', 'bear')

        if signal_direction == 'wait':
            check_details['direction_check'] = {'passed': False, 'reason': 'no_1h_signal'}
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

        # Check 2: 4H趋势强度
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

        # Check 3: RSI极端区域一致性
        rsi_1h = factors_1h.get('rsi', 50)
        rsi_4h = factors_4h.get('rsi', 50)

        rsi_1h_extreme = rsi_1h < 35 or rsi_1h > 65
        rsi_4h_extreme = rsi_4h < 40 or rsi_4h > 60

        if rsi_1h_extreme:
            if rsi_4h_extreme:
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
            if rsi_4h_extreme:
                confirmations += 0.5
                check_details['rsi_extreme_check'] = {
                    'passed': True,
                    'partial': True,
                    'reason': f'4H extreme zone confirms: 4H RSI {rsi_4h:.1f}'
                }
            else:
                confirmations += 1
                check_details['rsi_extreme_check'] = {
                    'passed': True,
                    'reason': f'Neither in extreme zone: neutral confirmation'
                }

        # Check 4: 4H成交量确认
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

        confidence_boost = confirmations / total_checks
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
        adx = factors_4h.get('adx', 0)
        macd_dir = factors_4h.get('macd_direction', 'bear')

        if adx < 20:
            return "range_bound"

        if macd_dir == 'bull':
            return "bull_trending" if adx > 30 else "bull_ranging"
        else:
            return "bear_trending" if adx > 30 else "bear_ranging"
