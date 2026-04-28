"""
Miracle 1.0.1 - Core Functions Unit Tests
==========================================
Tests for RSI, ADX, MACD, ATR, factor calculations, and risk metrics.

Run with: pytest tests/test_miracle_core.py -v
"""

import pytest
import sys
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miracle_core import (
    calc_rsi, calc_adx, calc_macd, calc_atr,
    calc_factors, calc_trend_strength, calc_leverage,
    calc_position_size, check_stops, can_trade,
    check_risk_limits, format_trade_signal, update_factor_weights,
    Direction, TradeSignal, Trade
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    base = 100
    return {
        "highs": [base + i * 2 + 5 for i in range(50)],
        "lows": [base + i * 2 - 5 for i in range(50)],
        "closes": [base + i * 2 for i in range(50)]
    }


@pytest.fixture
def trending_up_data():
    """Generate upward trending data"""
    return {
        "highs": [100 + i * 3 + 5 for i in range(50)],
        "lows": [100 + i * 3 - 5 for i in range(50)],
        "closes": [100 + i * 3 for i in range(50)]
    }


@pytest.fixture
def trending_down_data():
    """Generate downward trending data"""
    return {
        "highs": [200 - i * 3 + 5 for i in range(50)],
        "lows": [200 - i * 3 - 5 for i in range(50)],
        "closes": [200 - i * 3 for i in range(50)]
    }


@pytest.fixture
def flat_data():
    """Generate flat/ranging data"""
    return {
        "highs": [100 + math.sin(i / 5) * 2 + 3 for i in range(50)],
        "lows": [100 + math.sin(i / 5) * 2 - 3 for i in range(50)],
        "closes": [100 + math.sin(i / 5) * 2 for i in range(50)]
    }


# =============================================================================
# RSI Tests
# =============================================================================

class TestRSI:
    """RSI calculation tests"""

    def test_rsi_basic(self):
        """Basic RSI calculation should return value in 0-100 range"""
        prices = [100 + i for i in range(20)]
        rsi = calc_rsi(prices)
        assert 0 <= rsi <= 100

    def test_rsi_oversold(self):
        """RSI should be high (>50) in continuous uptrend"""
        prices = [100 + i * 2 for i in range(20)]
        rsi = calc_rsi(prices)
        assert rsi > 50

    def test_rsi_overbought(self):
        """RSI should be low (<50) in continuous downtrend"""
        prices = [100 - i * 2 for i in range(20)]
        rsi = calc_rsi(prices)
        assert rsi < 50

    def test_rsi_insufficient_data(self):
        """RSI should return 50.0 when data is insufficient"""
        prices = [100, 102, 101]
        rsi = calc_rsi(prices, period=14)
        assert rsi == 50.0

    def test_rsi_empty_data(self):
        """RSI should return 50.0 for empty data"""
        rsi = calc_rsi([])
        assert rsi == 50.0

    def test_rsi_extreme_oversold(self):
        """RSI should approach 0 for severe downtrend"""
        # Continuous decline
        prices = [100 - i * 5 for i in range(30)]
        rsi = calc_rsi(prices)
        assert rsi < 30

    def test_rsi_extreme_overbought(self):
        """RSI should approach 100 for strong uptrend"""
        # Continuous rise
        prices = [100 + i * 5 for i in range(30)]
        rsi = calc_rsi(prices)
        assert rsi > 70

    def test_rsi_no_change(self):
        """RSI should be 100 when price doesn't change (no losses)"""
        prices = [100] * 20
        rsi = calc_rsi(prices)
        # When price is flat with no changes, RSI = 100 (no losses)
        assert rsi == 100.0


# =============================================================================
# ADX Tests
# =============================================================================

class TestADX:
    """ADX calculation tests"""

    def test_adx_basic(self, trending_up_data):
        """ADX should return non-negative values"""
        data = trending_up_data
        result = calc_adx(data["highs"], data["lows"], data["closes"])
        assert isinstance(result, dict), "calc_adx should return dict (unified with core/price_factors.py)"
        assert result["adx"] >= 0
        assert result["plus_di"] >= 0
        assert result["minus_di"] >= 0

    def test_adx_trending_up(self, trending_up_data):
        """In uptrend, +DI should be greater than -DI"""
        data = trending_up_data
        result = calc_adx(data["highs"], data["lows"], data["closes"])
        assert result["plus_di"] > result["minus_di"]

    def test_adx_trending_down(self, trending_down_data):
        """In downtrend, -DI should be greater than +DI"""
        data = trending_down_data
        result = calc_adx(data["highs"], data["lows"], data["closes"])
        assert result["minus_di"] > result["plus_di"]

    def test_adx_flat_data(self, flat_data):
        """ADX should be non-negative in ranging market"""
        data = flat_data
        result = calc_adx(data["highs"], data["lows"], data["closes"])
        assert isinstance(result, dict)
        # ADX should be non-negative regardless of market condition
        assert result["adx"] >= 0

    def test_adx_insufficient_data(self):
        """ADX should return default values for insufficient data"""
        highs = [100, 101]
        lows = [99, 100]
        closes = [100, 101]
        result = calc_adx(highs, lows, closes)
        assert isinstance(result, dict)
        assert result["adx"] == 25.0
        assert result["plus_di"] == 25.0
        assert result["minus_di"] == 25.0

    def test_adx_calculation_consistency(self):
        """ADX calculation should be consistent for same input"""
        highs = [105 + i for i in range(40)]
        lows = [95 + i for i in range(40)]
        closes = [100 + i for i in range(40)]

        result1 = calc_adx(highs, lows, closes)
        result2 = calc_adx(highs, lows, closes)
        assert result1 == result2


# =============================================================================
# MACD Tests
# =============================================================================

class TestMACD:
    """MACD calculation tests"""

    def test_macd_basic(self):
        """Basic MACD should return dict with three values"""
        prices = [100 + (i % 10 - 5) for i in range(50)]
        result = calc_macd(prices)
        assert isinstance(result, dict), "calc_macd should return dict"
        assert isinstance(result["macd"], float)
        assert isinstance(result["signal"], float)
        assert isinstance(result["histogram"], float)

    def test_macd_histogram_sign(self):
        """MACD histogram should be positive when MACD > signal"""
        prices = [100 + i * 2 for i in range(50)]
        result = calc_macd(prices)
        assert result["histogram"] > 0  # In uptrend, MACD should be above signal

    def test_macd_insufficient_data(self):
        """MACD should return zeros for insufficient data"""
        prices = [100, 101, 102]
        result = calc_macd(prices)
        assert result["macd"] == 0.0
        assert result["signal"] == 0.0
        assert result["histogram"] == 0.0

    def test_macd_below_signal(self):
        """MACD histogram should be negative when MACD < signal"""
        prices = [100 - i * 2 for i in range(50)]
        result = calc_macd(prices)
        assert result["histogram"] < 0


# =============================================================================
# ATR Tests
# =============================================================================

class TestATR:
    """ATR calculation tests"""

    def test_atr_basic(self):
        """ATR should be positive for valid data"""
        highs = [105, 110, 108, 112, 115]
        lows = [95, 100, 98, 102, 105]
        closes = [100, 105, 103, 108, 110]
        atr = calc_atr(highs, lows, closes)
        assert atr > 0

    def test_atr_zero_handling(self):
        """ATR should handle flat market without errors"""
        highs = [100, 100, 100, 100, 100]
        lows = [100, 100, 100, 100, 100]
        closes = [100, 100, 100, 100, 100]
        atr = calc_atr(highs, lows, closes)
        assert atr >= 0

    def test_atr_insufficient_data(self):
        """ATR should return calculated value for short data"""
        highs = [105, 110]
        lows = [95, 100]
        closes = [100, 105]
        atr = calc_atr(highs, lows, closes)
        assert atr > 0

    def test_atr_volatility_reflects_range(self):
        """ATR should be higher for more volatile data"""
        # Low volatility
        low_vol_highs = [100 + i * 0.5 for i in range(20)]
        low_vol_lows = [100 + i * 0.5 - 1 for i in range(20)]
        low_vol_closes = [100 + i * 0.5 for i in range(20)]
        atr_low = calc_atr(low_vol_highs, low_vol_lows, low_vol_closes)

        # High volatility
        high_vol_highs = [100 + i * 5 for i in range(20)]
        high_vol_lows = [100 + i * 5 - 10 for i in range(20)]
        high_vol_closes = [100 + i * 5 for i in range(20)]
        atr_high = calc_atr(high_vol_highs, high_vol_lows, high_vol_closes)

        assert atr_high > atr_low


# =============================================================================
# Factor Calculation Tests
# =============================================================================

class TestFactorCalculation:
    """Factor calculation tests"""

    def test_calc_factors_basic(self, sample_price_data):
        """calc_factors should return all required keys"""
        factors = calc_factors(sample_price_data)

        required_keys = ["rsi", "adx", "macd_hist", "composite_score"]
        for key in required_keys:
            assert key in factors

    def test_calc_factors_score_range(self, sample_price_data):
        """Composite score should be in 0-100 range"""
        factors = calc_factors(sample_price_data)
        assert 0 <= factors["composite_score"] <= 100

    def test_calc_factors_with_news(self, sample_price_data):
        """calc_factors should accept news data"""
        news_data = {"sentiment": 0.5}
        factors = calc_factors(sample_price_data, news_data=news_data)
        assert "news_sentiment" in factors

    def test_calc_factors_with_onchain(self, sample_price_data):
        """calc_factors should accept onchain data"""
        onchain_data = {"exchange_flow": 0.3, "large_transfer": 0.1}
        factors = calc_factors(sample_price_data, onchain_data=onchain_data)
        assert "exchange_flow" in factors


# =============================================================================
# Trend Strength Tests
# =============================================================================

class TestTrendStrength:
    """Trend strength calculation tests"""

    def test_strong_trend(self):
        """Strong trend (ADX > 70) should be identified"""
        factors = {"adx": 80, "plus_di": 70, "minus_di": 20}
        strength, label = calc_trend_strength(factors)
        assert strength >= 70
        assert label == "strong"

    def test_medium_trend(self):
        """Medium trend (40 <= ADX < 70) should be identified"""
        factors = {"adx": 50, "plus_di": 40, "minus_di": 30}
        strength, label = calc_trend_strength(factors)
        assert 40 <= strength < 70
        assert label == "medium"

    def test_weak_trend(self):
        """Weak trend (ADX < 40) should be identified"""
        factors = {"adx": 20, "plus_di": 30, "minus_di": 30}
        strength, label = calc_trend_strength(factors)
        assert strength < 40
        assert label == "weak"

    def test_trend_strength_bounds(self):
        """Trend strength should always be 0-100"""
        factors = {"adx": 100, "plus_di": 100, "minus_di": 0}
        strength, label = calc_trend_strength(factors)
        assert 0 <= strength <= 100


# =============================================================================
# Leverage Calculation Tests
# =============================================================================

class TestLeverage:
    """Leverage calculation tests"""

    def test_leverage_strong_trend(self):
        """High leverage for strong trend with high confidence"""
        leverage, multiplier = calc_leverage(80, 90)
        assert leverage == 3  # strong_trend leverage
        assert multiplier > 1.0

    def test_leverage_medium_trend(self):
        """Medium leverage for medium trend"""
        leverage, multiplier = calc_leverage(50, 50)
        assert leverage == 2  # medium_trend leverage
        # multiplier = base_multiplier * (0.5 + confidence_factor * 0.5)
        # multiplier = 1.0 * (0.5 + 0.5 * 0.5) = 0.75
        assert multiplier == 0.75

    def test_leverage_weak_trend(self):
        """Low leverage for weak trend"""
        leverage, multiplier = calc_leverage(20, 30)
        assert leverage == 1  # weak_trend leverage
        assert multiplier < 1.0

    def test_leverage_confidence_effect(self):
        """Higher confidence should increase multiplier"""
        _, mult_low = calc_leverage(50, 30)
        _, mult_high = calc_leverage(50, 90)
        assert mult_high > mult_low


# =============================================================================
# Position Size Tests
# =============================================================================

class TestPositionSize:
    """Position size calculation tests"""

    def test_position_size_basic(self):
        """Position size should be positive"""
        pos, contracts = calc_position_size(10000, 100, 0.02, 2)
        assert pos > 0
        assert contracts > 0

    def test_position_size_zero_stop_loss(self):
        """Position size should handle zero stop loss"""
        pos, contracts = calc_position_size(10000, 100, 0, 2)
        assert pos >= 0

    def test_position_size_max_limit(self):
        """Position size should be capped at max percentage"""
        pos, contracts = calc_position_size(10000, 100, 0.5, 10)
        # Should be limited by max_position_pct (15%)
        assert pos <= 10000 * 0.15


# =============================================================================
# Stop Check Tests
# =============================================================================

class TestStopCheck:
    """Stop loss check tests"""

    def test_stop_loss_long_hit(self):
        """Long position should exit when price drops to stop loss"""
        position = {
            "direction": "long",
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "entry_time": "2024-01-01T00:00:00"
        }
        should_exit, reason = check_stops(position, 94)
        assert should_exit
        assert reason == "sl"

    def test_stop_loss_short_hit(self):
        """Short position should exit when price rises to stop loss"""
        position = {
            "direction": "short",
            "entry_price": 100,
            "stop_loss": 105,
            "take_profit": 90,
            "entry_time": "2024-01-01T00:00:00"
        }
        should_exit, reason = check_stops(position, 106)
        assert should_exit
        assert reason == "sl"

    def test_take_profit_long_hit(self):
        """Long position should exit when price rises to take profit"""
        position = {
            "direction": "long",
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "entry_time": "2024-01-01T00:00:00"
        }
        should_exit, reason = check_stops(position, 111)
        assert should_exit
        assert reason == "tp"

    def test_stop_not_hit(self):
        """Position should not exit when neither stop nor take profit is hit"""
        from datetime import datetime, timedelta
        recent_time = (datetime.now() - timedelta(hours=1)).isoformat()
        position = {
            "direction": "long",
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "entry_time": recent_time
        }
        should_exit, reason = check_stops(position, 102)
        assert not should_exit
        assert reason == "none"


# =============================================================================
# Trading Control Tests
# =============================================================================

class TestTradingControl:
    """Trading control tests"""

    def test_can_trade_empty_history(self):
        """Should be able to trade with empty history"""
        result, reason = can_trade([])
        assert result
        assert reason == "ok"

    def test_can_trade_daily_limit(self):
        """Should not trade when daily limit is reached"""
        from datetime import datetime
        today = datetime.now().date()
        trade_history = [
            {"exit_time": f"{today}T{10+i}:00:00", "direction": "long", "pnl": 100}
            for i in range(5)
        ]
        result, reason = can_trade(trade_history)
        assert not result
        assert "daily_limit" in reason

    def test_can_trade_interval(self):
        """Should respect minimum trade interval"""
        from datetime import datetime, timedelta
        now = datetime.now()
        recent_time = (now - timedelta(hours=1)).isoformat()
        trade_history = [{"exit_time": recent_time, "direction": "long", "pnl": 100}]
        result, reason = can_trade(trade_history)
        assert not result
        assert "interval" in reason


# =============================================================================
# Risk Limits Tests
# =============================================================================

class TestRiskLimits:
    """Risk limit check tests"""

    def test_risk_limits_ok(self):
        """Should pass when within limits"""
        can_trade, reason = check_risk_limits(10000, 100, 500, 10000)
        assert can_trade
        assert reason == "ok"

    def test_risk_limits_daily_loss(self):
        """Should trigger daily loss limit"""
        can_trade, reason = check_risk_limits(10000, -600, 0, 10000)
        assert not can_trade
        assert "daily_loss" in reason

    def test_risk_limits_drawdown(self):
        """Should trigger total drawdown limit"""
        can_trade, reason = check_risk_limits(10000, -100, -2500, 10000)
        assert not can_trade
        assert "drawdown" in reason


# =============================================================================
# Trade Signal Tests
# =============================================================================

class TestTradeSignal:
    """Trade signal formatting tests"""

    def test_format_signal_long(self, sample_price_data):
        """Should format long signal correctly"""
        factors = calc_factors(sample_price_data)
        signal = format_trade_signal("long", 100, factors, 10000)

        assert signal.direction == Direction.LONG
        assert signal.entry_price == 100
        assert signal.stop_loss < 100
        assert signal.take_profit > 100
        assert signal.leverage >= 1

    def test_format_signal_short(self, sample_price_data):
        """Should format short signal correctly"""
        factors = calc_factors(sample_price_data)
        signal = format_trade_signal("short", 100, factors, 10000)

        assert signal.direction == Direction.SHORT
        assert signal.entry_price == 100
        assert signal.stop_loss > 100
        assert signal.take_profit < 100

    def test_signal_has_valid_rr(self, sample_price_data):
        """Signal RR ratio should be positive"""
        factors = calc_factors(sample_price_data)
        signal = format_trade_signal("long", 100, factors, 10000)
        assert signal.rr_ratio >= 2.0  # Min RR from config


# =============================================================================
# Factor Weight Update Tests
# =============================================================================

class TestFactorWeightUpdate:
    """Factor weight update tests"""

    def test_update_low_win_rate(self):
        """Low win rate should reduce weights"""
        from datetime import datetime
        recent_trades = [
            Trade(
                id=f"t{i}", direction=Direction.LONG,
                entry_price=100, exit_price=95,
                position_size=1000, leverage=1,
                entry_time=datetime.now().isoformat(),
                exit_time=datetime.now().isoformat(),
                pnl=-100, pnl_pct=-10,
                factors={}, stop_triggered="sl"
            )
            for i in range(5)
        ]
        # use_ic=False returns nested dict format: {'price_momentum': {'weight': 0.6}}
        weights = update_factor_weights(recent_trades, use_ic=False)
        # Weights should be reduced (price_momentum['weight'] < 0.6)
        assert weights["price_momentum"]["weight"] < 0.6

    def test_update_high_win_rate(self):
        """High win rate should increase weights"""
        from datetime import datetime
        recent_trades = [
            Trade(
                id=f"t{i}", direction=Direction.LONG,
                entry_price=100, exit_price=110,
                position_size=1000, leverage=1,
                entry_time=datetime.now().isoformat(),
                exit_time=datetime.now().isoformat(),
                pnl=100, pnl_pct=10,
                factors={}, stop_triggered="tp"
            )
            for i in range(5)
        ]
        # use_ic=False returns nested dict format: {'price_momentum': {'weight': 0.6}}
        weights = update_factor_weights(recent_trades, use_ic=False)
        # Weights should be increased (price_momentum['weight'] > 0.6)
        assert weights["price_momentum"]["weight"] > 0.6

    def test_update_insufficient_data(self):
        """Should return current weights with insufficient data"""
        weights = update_factor_weights([])
        assert "price_momentum" in weights


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
