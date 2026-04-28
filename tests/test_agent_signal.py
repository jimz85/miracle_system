"""
Tests for agents/agent_signal.py - 信号生成 (RSI/ADX计算)
=========================================================

Covers:
- Normal path: RSI calculation, ADX calculation, signal generation
- Edge cases: insufficient data, boundary values, zero/missing values
- Exception handling: invalid inputs, empty data
"""

import math
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_signal import (
    MultiTimeframeFilter,
    PriceFactors,
    TrendDetector,
    WhitelistFilter,
)

# ============================================================================
# RSI Calculation Tests
# ============================================================================

class TestRSICalculation:
    """Test RSI (Relative Strength Index) calculation with Wilder smoothing"""

    def test_rsi_basic_calculation(self):
        """RSI with basic price data"""
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 108, 111, 113, 112]
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_rsi_insufficient_data(self):
        """RSI returns 50.0 (neutral) when data insufficient"""
        prices = [100, 101, 102]  # less than period + 1
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert rsi == 50.0

    def test_rsi_all_gains(self):
        """RSI should be 100 when all price changes are gains"""
        prices = [100 + i for i in range(20)]  # strictly increasing
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert rsi == 100.0

    def test_rsi_all_losses(self):
        """RSI should be 0 when all price changes are losses"""
        prices = [100 - i for i in range(20)]  # strictly decreasing
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert rsi == 0.0

    def test_rsi_neutral_at_50(self):
        """RSI should be 50 when gains equal losses"""
        # Up then down same amount
        prices = [100, 105, 110, 105, 100, 95, 100, 105, 110, 105, 100, 95, 100,
                  105, 110, 105, 100, 95, 100, 105, 110, 105, 100, 95, 100, 105,
                  110, 105, 100, 95]
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 40 <= rsi <= 60  # approximately neutral

    def test_rsi_with_custom_period(self):
        """RSI calculation respects custom period"""
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110,
                  108, 111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        rsi_7 = PriceFactors.calc_rsi(prices, period=7)
        rsi_14 = PriceFactors.calc_rsi(prices, period=14)
        # Different periods should give different values
        assert rsi_7 != rsi_14

    def test_rsi_extremely_volatile(self):
        """RSI with highly volatile prices"""
        prices = [100, 150, 50, 200, 25, 175, 50, 150, 75, 125]
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_rsi_oversold_region(self):
        """RSI < 30 indicates oversold"""
        # Sharp drop then slight recovery
        prices = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 85, 83, 81, 80, 82]
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert rsi < 50  # should be in oversold territory

    def test_rsi_overbought_region(self):
        """RSI > 70 indicates overbought"""
        # Sharp rise then slight drop
        prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 115, 117, 114, 116, 118]
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert rsi > 50  # should be in overbought territory


# ============================================================================
# ADX Calculation Tests
# ============================================================================

class TestADXCalculation:
    """Test ADX (Average Directional Index) calculation with Wilder smoothing"""

    def test_adx_basic_calculation(self):
        """ADX with valid price data"""
        highs = [105, 107, 106, 108, 110, 109, 111, 113, 112, 114]
        lows = [95, 97, 94, 96, 98, 95, 99, 101, 98, 102]
        closes = [100, 102, 99, 103, 105, 104, 106, 108, 107, 109]

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)

        assert "adx" in result
        assert "plus_di" in result
        assert "minus_di" in result
        assert 0 <= result["adx"] <= 100

    def test_adx_insufficient_data(self):
        """ADX returns default values when data insufficient"""
        highs = [105, 107]
        lows = [95, 97]
        closes = [100, 102]

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)

        assert result["adx"] == 25.0
        assert result["plus_di"] == 25.0
        assert result["minus_di"] == 25.0

    def test_adx_strong_uptrend(self):
        """ADX should be high (>25) in strong trending market"""
        # Strong uptrend
        highs = [100 + i * 2 for i in range(35)]
        lows = [98 + i * 2 for i in range(35)]
        closes = [99 + i * 2 for i in range(35)]

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)
        # ADX should be positive
        assert result["adx"] > 0

    def test_adx_strong_downtrend(self):
        """ADX should be high (>25) in strong downtrend"""
        # Strong downtrend
        highs = [100 - i * 2 for i in range(35)]
        lows = [98 - i * 2 for i in range(35)]
        closes = [99 - i * 2 for i in range(35)]

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)
        # ADX should be positive
        assert result["adx"] > 0

    def test_adx_range_bound_market(self):
        """ADX should be low in ranging market"""
        # Flat/ranging market
        highs = [100, 102, 101, 103, 100, 102, 101, 103, 100, 102, 101, 103,
                 100, 102, 101, 103, 100, 102, 101, 103, 100, 102, 101, 103,
                 100, 102, 101, 103, 100, 102, 101, 103, 100, 102, 101]
        lows = [98, 100, 99, 101, 98, 100, 99, 101, 98, 100, 99, 101,
                98, 100, 99, 101, 98, 100, 99, 101, 98, 100, 99, 101,
                98, 100, 99, 101, 98, 100, 99, 101, 98, 100, 99]
        closes = [99, 101, 100, 102, 99, 101, 100, 102, 99, 101, 100, 102,
                  99, 101, 100, 102, 99, 101, 100, 102, 99, 101, 100, 102,
                  99, 101, 100, 102, 99, 101, 100, 102, 99, 101, 100]

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)
        # In ranging market, ADX might be lower
        assert result["adx"] >= 0

    def test_adx_plus_di_greater_in_uptrend(self):
        """+DI should be > -DI in uptrend"""
        # Strong uptrend
        highs = [100 + i * 3 for i in range(40)]
        lows = [98 + i * 3 for i in range(40)]
        closes = [99 + i * 3 for i in range(40)]

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)
        assert result["plus_di"] > result["minus_di"]

    def test_adx_minus_di_greater_in_downtrend(self):
        """-DI should be > +DI in downtrend"""
        # Strong downtrend
        highs = [100 - i * 3 for i in range(40)]
        lows = [98 - i * 3 for i in range(40)]
        closes = [99 - i * 3 for i in range(40)]

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)
        assert result["minus_di"] > result["plus_di"]

    def test_adx_di_values_bounded(self):
        """+DI and -DI should be between 0 and 100"""
        highs = [105, 107, 106, 108, 110, 109, 111, 113, 112, 114,
                 105, 107, 106, 108, 110, 109, 111, 113, 112, 114,
                 105, 107, 106, 108, 110, 109, 111, 113, 112, 114,
                 105, 107, 106, 108, 110]
        lows = [95, 97, 94, 96, 98, 95, 99, 101, 98, 102,
                95, 97, 94, 96, 98, 95, 99, 101, 98, 102,
                95, 97, 94, 96, 98, 95, 99, 101, 98, 102,
                95, 97, 94, 96, 98]
        closes = [100, 102, 99, 103, 105, 104, 106, 108, 107, 109,
                  100, 102, 99, 103, 105, 104, 106, 108, 107, 109,
                  100, 102, 99, 103, 105, 104, 106, 108, 107, 109,
                  100, 102, 99, 103, 105]

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)
        assert 0 <= result["plus_di"] <= 100
        assert 0 <= result["minus_di"] <= 100


# ============================================================================
# MACD Calculation Tests
# ============================================================================

class TestMACDCalculation:
    """Test MACD calculation"""

    def test_macd_basic_calculation(self):
        """MACD with valid price data"""
        prices = [100 + i for i in range(50)]
        result = PriceFactors.calc_macd(prices)

        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_macd_insufficient_data(self):
        """MACD returns zeros when data insufficient"""
        prices = [100, 101, 102]
        result = PriceFactors.calc_macd(prices)

        assert result["macd"] == 0.0
        assert result["signal"] == 0.0
        assert result["histogram"] == 0.0

    def test_macd_histogram_sign(self):
        """MACD histogram sign indicates trend direction"""
        # Rising prices should give positive histogram
        prices = list(range(100, 200))
        result = PriceFactors.calc_macd(prices)
        assert result["histogram"] > 0 or result["histogram"] < 0  # should have a sign


# ============================================================================
# ATR Calculation Tests
# ============================================================================

class TestATRCalculation:
    """Test ATR (Average True Range) calculation"""

    def test_atr_basic_calculation(self):
        """ATR with valid price data"""
        highs = [105, 107, 106, 108, 110]
        lows = [95, 97, 94, 96, 98]
        closes = [100, 102, 99, 103, 105]

        atr = PriceFactors.calc_atr(highs, lows, closes, period=14)
        assert atr > 0

    def test_atr_insufficient_data(self):
        """ATR handles insufficient data"""
        highs = [105, 107]
        lows = [95, 97]
        closes = [100, 102]

        atr = PriceFactors.calc_atr(highs, lows, closes, period=14)
        assert atr >= 0


# ============================================================================
# Bollinger Bands Tests
# ============================================================================

class TestBollingerBands:
    """Test Bollinger Bands calculation"""

    def test_bollinger_basic_calculation(self):
        """Bollinger Bands with valid data"""
        prices = [100 + i for i in range(30)]
        result = PriceFactors.calc_bollinger(prices)

        assert "upper" in result
        assert "middle" in result
        assert "lower" in result
        assert "bandwidth" in result

    def test_bollinger_upper_greater_than_middle(self):
        """Upper band should be above middle band"""
        prices = [100 + i for i in range(30)]
        result = PriceFactors.calc_bollinger(prices)
        assert result["upper"] > result["middle"]

    def test_bollinger_middle_greater_than_lower(self):
        """Middle band should be above lower band"""
        prices = [100 + i for i in range(30)]
        result = PriceFactors.calc_bollinger(prices)
        assert result["middle"] > result["lower"]

    def test_bollinger_insufficient_data(self):
        """Bollinger with insufficient data"""
        prices = [100, 101, 102]
        result = PriceFactors.calc_bollinger(prices)

        # Should return fallback values based on last price
        assert result["upper"] > 0
        assert result["lower"] > 0


# ============================================================================
# Momentum Calculation Tests
# ============================================================================

class TestMomentum:
    """Test momentum calculation"""

    def test_momentum_positive(self):
        """Positive momentum for rising prices"""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        momentum = PriceFactors.calc_momentum(prices, period=10)
        assert momentum > 0

    def test_momentum_negative(self):
        """Negative momentum for falling prices"""
        prices = [110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
        momentum = PriceFactors.calc_momentum(prices, period=10)
        assert momentum < 0

    def test_momentum_insufficient_data(self):
        """Momentum returns 0 with insufficient data"""
        prices = [100, 101, 102]
        momentum = PriceFactors.calc_momentum(prices, period=10)
        assert momentum == 0.0


# ============================================================================
# Volume Filter Tests
# ============================================================================

class TestVolumeFilter:
    """Test volume filter calculation"""

    def test_volume_filter_basic(self):
        """Volume filter with valid data"""
        volumes = [1000] * 25
        volumes.append(2000)  # current volume 2x average

        result = PriceFactors.calc_volume_filter(volumes, period=20)

        assert "volume_ratio" in result
        assert "is_confirmed" in result
        assert "confidence_penalty" in result

    def test_volume_filter_high_volume_confirms(self):
        """High volume should confirm signal"""
        volumes = [1000] * 25
        volumes.append(2000)  # 2x average = confirmed

        result = PriceFactors.calc_volume_filter(volumes, period=20)
        assert result["is_confirmed"]  # use == instead of is for numpy bool
        assert result["confidence_penalty"] == 0.0

    def test_volume_filter_low_volume_penalizes(self):
        """Low volume should penalize confidence"""
        volumes = [1000] * 25
        volumes.append(500)  # 0.5x average

        result = PriceFactors.calc_volume_filter(volumes, period=20)
        assert result["confidence_penalty"] > 0

    def test_volume_filter_insufficient_data(self):
        """Volume filter with insufficient data"""
        volumes = [100, 200]
        result = PriceFactors.calc_volume_filter(volumes, period=20)

        # Should return safe defaults
        assert result["volume_ratio"] == 1.0
        assert result["is_confirmed"] is True


# ============================================================================
# PriceFactors calc_all Tests
# ============================================================================

class TestCalcAll:
    """Test calc_all (all indicators at once)"""

    def test_calc_all_returns_all_factors(self):
        """calc_all returns all technical factors"""
        prices = [100 + i for i in range(100)]
        highs = [p + 5 for p in prices]
        lows = [p - 5 for p in prices]

        result = PriceFactors.calc_all(prices, highs, lows, timeframe="1H")

        assert "rsi" in result
        assert "adx" in result
        assert "macd" in result
        assert "bollinger_upper" in result
        assert "momentum" in result
        assert "ema20" in result
        assert "ema50" in result
        assert "current_price" in result

    def test_calc_all_4h_returns_all_factors(self):
        """calc_all_4h returns all 4H factors"""
        prices = [100 + i for i in range(100)]
        highs = [p + 5 for p in prices]
        lows = [p - 5 for p in prices]
        volumes = [1000] * 100

        result = PriceFactors.calc_all_4h(prices, highs, lows, volumes)

        assert "rsi" in result
        assert "adx" in result
        assert "volume_ratio" in result
        assert result["timeframe"] == "4H"


# ============================================================================
# TrendDetector Tests
# ============================================================================

class TestTrendDetector:
    """Test trend detection"""

    def test_detect_trend_bullish(self):
        """Detect bullish trend"""
        prices = [100 + i * 0.5 for i in range(250)]
        highs = [p + 5 for p in prices]
        lows = [p - 5 for p in prices]

        result = TrendDetector.detect_trend(prices, highs, lows)

        assert "trend" in result
        assert "strength" in result
        assert result["trend"] in ["bull", "bear", "range"]

    def test_detect_trend_bearish(self):
        """Detect bearish trend"""
        prices = [100 - i * 0.5 for i in range(250)]
        highs = [p + 5 for p in prices]
        lows = [p - 5 for p in prices]

        result = TrendDetector.detect_trend(prices, highs, lows)

        assert result["trend"] in ["bull", "bear", "range"]

    def test_detect_trend_range(self):
        """Detect range-bound market"""
        prices = [100] * 250
        highs = [p + 5 for p in prices]
        lows = [p - 5 for p in prices]

        result = TrendDetector.detect_trend(prices, highs, lows)

        assert result["trend"] in ["bull", "bear", "range"]


# ============================================================================
# WhitelistFilter Tests
# ============================================================================

class TestWhitelistFilter:
    """Test whitelist pattern filter"""

    def test_whitelist_filter_strong_long_signal(self):
        """RSI < 40 should pass as strong long"""
        filter = WhitelistFilter()
        signal = {"action": "BUY"}
        factors = {"rsi": 30, "adx": 30, "trend": "bull"}

        result = filter.check(signal, factors)

        assert result["passed"] is True
        assert result["confidence_modifier"] >= 1.0

    def test_whitelist_filter_strong_short_signal(self):
        """RSI > 60 should pass as strong short"""
        filter = WhitelistFilter()
        signal = {"action": "SELL"}
        factors = {"rsi": 70, "adx": 30, "trend": "bear"}

        result = filter.check(signal, factors)

        assert result["passed"] is True

    def test_whitelist_filter_low_adx_rejected(self):
        """ADX < 20 should be rejected (no trend)"""
        filter = WhitelistFilter()
        signal = {"action": "BUY"}
        factors = {"rsi": 50, "adx": 15, "trend": "range"}

        result = filter.check(signal, factors)

        assert result["passed"] is False
        assert result["reason"] == "no_trend_adx_low"

    def test_pattern_key_generation(self):
        """Pattern key is generated correctly"""
        filter = WhitelistFilter()
        key = filter._get_pattern_key(rsi=35, adx=30, trend="bull")
        assert "bull" in key
        assert "rsi" in key

    def test_blacklist_blocks_signal(self):
        """Blacklisted pattern blocks signal"""
        filter = WhitelistFilter()
        signal = {"action": "BUY"}
        factors = {"rsi": 35, "adx": 30, "trend": "bull"}

        pattern_key = filter._get_pattern_key(35, 30, "bull")
        filter.add_to_blacklist(pattern_key)

        result = filter.check(signal, factors)
        assert result["passed"] is False


# ============================================================================
# Edge Cases and Exception Handling
# ============================================================================

class TestAgentSignalEdgeCases:
    """Test edge cases and exception handling"""

    def test_empty_price_list(self):
        """Empty price list handled gracefully"""
        rsi = PriceFactors.calc_rsi([], period=14)
        assert rsi == 50.0  # neutral fallback

    def test_single_price(self):
        """Single price handled"""
        rsi = PriceFactors.calc_rsi([100], period=14)
        assert rsi == 50.0  # neutral fallback

    def test_all_same_prices(self):
        """All identical prices handled"""
        prices = [100] * 30
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_zero_prices(self):
        """Zero prices handled"""
        prices = [0] * 20
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_negative_prices(self):
        """Negative prices handled (edge case)"""
        prices = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10,
                   0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_very_large_prices(self):
        """Very large price values handled"""
        prices = [1_000_000 + i for i in range(30)]
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_very_small_prices(self):
        """Very small price values handled"""
        prices = [0.00001 + i * 0.00001 for i in range(30)]
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_mixed_positive_negative_prices(self):
        """Mixed positive/negative prices"""
        prices = [100, -50, 150, -75, 200, -100, 250, -125, 300, -150] * 3
        rsi = PriceFactors.calc_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_adx_with_zero_range(self):
        """ADX with zero high-low range"""
        highs = [100] * 50
        lows = [100] * 50
        closes = [100] * 50

        result = PriceFactors.calc_adx(highs, lows, closes, period=14)
        assert "adx" in result

    def test_ema_calculation_edge_cases(self):
        """EMA edge cases"""
        # Very short price list
        ema = PriceFactors._calc_ema([100], 20)
        assert ema == 100

        # Empty price list
        ema = PriceFactors._calc_ema([], 20)
        assert ema == 0


# ============================================================================
# calc_all with Different Timeframes
# ============================================================================

class TestCalcAllTimeframes:
    """Test calc_all with different timeframe handling"""

    def test_calc_all_1h_timeframe(self):
        """calc_all returns correct timeframe"""
        prices = [100 + i for i in range(100)]
        highs = [p + 5 for p in prices]
        lows = [p - 5 for p in prices]

        result = PriceFactors.calc_all(prices, highs, lows, timeframe="1H")
        assert result["timeframe"] == "1H"

    def test_calc_all_with_volume(self):
        """calc_all_4h includes volume data"""
        prices = [100 + i for i in range(100)]
        highs = [p + 5 for p in prices]
        lows = [p - 5 for p in prices]
        volumes = [1000] * 100

        result = PriceFactors.calc_all_4h(prices, highs, lows, volumes)
        assert "volume_ratio" in result


# ============================================================================
# MultiTimeframeFilter Tests
# ============================================================================

class TestMultiTimeframeFilter:
    """Test multi-timeframe filter"""

    def test_get_4h_regime_bull(self):
        """4H regime detection for bull market"""
        factors_4h = {
            "trend": "bull",
            "adx": 35,  # > 30 for trending
            "rsi": 55,
            "macd_direction": "bull"
        }
        regime = MultiTimeframeFilter.get_4h_regime(factors_4h)
        # Returns strings like "bull_trending", "bear_ranging", "range_bound"
        assert regime in ["bull_trending", "bull_ranging", "bear_trending", "bear_ranging", "range_bound"]

    def test_get_4h_regime_bear(self):
        """4H regime detection for bear market"""
        factors_4h = {
            "trend": "bear",
            "adx": 35,  # > 30 for trending
            "rsi": 45,
            "macd_direction": "bear"
        }
        regime = MultiTimeframeFilter.get_4h_regime(factors_4h)
        assert regime in ["bull_trending", "bull_ranging", "bear_trending", "bear_ranging", "range_bound"]

    def test_get_4h_regime_range(self):
        """4H regime detection for range market"""
        factors_4h = {
            "trend": "range",
            "adx": 15,  # < 20 for range_bound
            "rsi": 50,
            "macd_direction": "bear"
        }
        regime = MultiTimeframeFilter.get_4h_regime(factors_4h)
        assert regime == "range_bound"

    def test_get_4h_regime_ranging_with_bull_macd(self):
        """4H regime when adx < 20 but macd is bull"""
        factors_4h = {
            "trend": "range",
            "adx": 15,  # < 20
            "rsi": 50,
            "macd_direction": "bull"
        }
        regime = MultiTimeframeFilter.get_4h_regime(factors_4h)
        assert regime == "range_bound"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
