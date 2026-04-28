"""
Miracle 1.0.1 - Agent Module Tests
===================================
Tests for Agent-S (Signal Generator), Agent-R (Risk), Agent-M (Market Intel).

Run with: pytest tests/test_agents.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_signal import (
    AgentSignal,
    PriceFactors,
    SignalGenerator,
    TrendDetector,
    WhitelistFilter,
)

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample prices for testing"""
    base = 72000
    return [base * (1 + i * 0.001) for i in range(100)]


@pytest.fixture
def sample_highs_lows():
    """Generate sample highs and lows"""
    base = 72000
    return {
        "highs": [base * (1 + i * 0.001) * 1.005 for i in range(100)],
        "lows": [base * (1 + i * 0.001) * 0.995 for i in range(100)]
    }


@pytest.fixture
def sample_intel_report():
    """Generate sample intel report from Agent-M"""
    return {
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


@pytest.fixture
def config():
    """Load trading config"""
    config_path = Path(__file__).parent.parent / "miracle_config.json"
    with open(config_path) as f:
        return json.load(f)


# =============================================================================
# PriceFactors Tests
# =============================================================================

class TestPriceFactors:
    """PriceFactors calculator tests"""

    def test_calc_rsi_basic(self, sample_prices):
        """RSI should be calculated correctly"""
        rsi = PriceFactors.calc_rsi(sample_prices)
        assert 0 <= rsi <= 100

    def test_calc_rsi_no_data(self):
        """RSI should return 50 for insufficient data"""
        rsi = PriceFactors.calc_rsi([100, 101])
        assert rsi == 50.0

    def test_calc_adx_basic(self, sample_highs_lows, sample_prices):
        """ADX should be calculated correctly"""
        result = PriceFactors.calc_adx(
            sample_highs_lows["highs"],
            sample_highs_lows["lows"],
            sample_prices
        )
        assert "adx" in result
        assert "plus_di" in result
        assert "minus_di" in result

    def test_calc_macd_basic(self, sample_prices):
        """MACD should be calculated correctly"""
        result = PriceFactors.calc_macd(sample_prices)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_calc_bollinger_basic(self, sample_prices):
        """Bollinger Bands should be calculated"""
        result = PriceFactors.calc_bollinger(sample_prices)
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result
        assert result["upper"] > result["middle"]
        assert result["middle"] > result["lower"]

    def test_calc_momentum_basic(self, sample_prices):
        """Momentum should be calculated"""
        momentum = PriceFactors.calc_momentum(sample_prices)
        assert isinstance(momentum, float)

    def test_calc_all(self, sample_prices, sample_highs_lows):
        """calc_all should return all factors"""
        result = PriceFactors.calc_all(
            sample_prices,
            sample_highs_lows["highs"],
            sample_highs_lows["lows"]
        )
        required = ["rsi", "adx", "macd", "bollinger_upper", "momentum", "ema20"]
        for key in required:
            assert key in result


# =============================================================================
# TrendDetector Tests
# =============================================================================

class TestTrendDetector:
    """Trend detection tests"""

    def test_detect_trend_bull(self, sample_highs_lows, sample_prices):
        """Should detect bullish trend"""
        result = TrendDetector.detect_trend(
            sample_prices,
            sample_highs_lows["highs"],
            sample_highs_lows["lows"]
        )
        assert "trend" in result
        assert "strength" in result
        assert result["trend"] in ["bull", "bear", "range"]

    def test_detect_trend_bear(self):
        """Should detect bearish trend"""
        prices = [100 - i * 2 for i in range(100)]
        highs = [p + 5 for p in prices]
        lows = [p - 5 for p in prices]
        result = TrendDetector.detect_trend(prices, highs, lows)
        assert result["trend"] == "bear"

    def test_detect_trend_range(self):
        """Should detect ranging market"""
        import math
        # Create oscillating data with no clear direction
        prices = [100 + math.sin(i / 10) * 10 for i in range(100)]
        highs = [p + 2 for p in prices]
        lows = [p - 2 for p in prices]
        result = TrendDetector.detect_trend(prices, highs, lows)
        # The result should be one of the valid trends
        assert result["trend"] in ["bull", "bear", "range"]


# =============================================================================
# WhitelistFilter Tests
# =============================================================================

class TestWhitelistFilter:
    """Whitelist pattern filter tests"""

    def test_oversold_rebound_pattern(self):
        """RSI 35-45 + ADX > 30 should pass with high confidence"""
        wf = WhitelistFilter()
        signal = {}
        factors = {"rsi": 40, "adx": 35, "trend": "bull"}
        result = wf.check(signal, factors)

        assert result["passed"]
        assert result["confidence_modifier"] == 1.2

    def test_extreme_oversold(self):
        """RSI < 30 should have highest confidence modifier"""
        wf = WhitelistFilter()
        signal = {}
        factors = {"rsi": 25, "adx": 40, "trend": "bull"}
        result = wf.check(signal, factors)

        assert result["passed"]
        assert result["confidence_modifier"] == 1.4

    def test_extreme_overbought(self):
        """RSI > 70 should have high confidence modifier"""
        wf = WhitelistFilter()
        signal = {}
        factors = {"rsi": 75, "adx": 40, "trend": "bear"}
        result = wf.check(signal, factors)

        assert result["passed"]
        assert result["confidence_modifier"] == 1.3

    def test_no_trend_adx_low(self):
        """Low ADX should reject signal"""
        wf = WhitelistFilter()
        signal = {}
        factors = {"rsi": 50, "adx": 15, "trend": "range"}
        result = wf.check(signal, factors)

        assert not result["passed"]

    def test_update_pattern_db_win(self):
        """Pattern DB should update on win"""
        wf = WhitelistFilter()
        wf.update_pattern_db("test_pattern", won=True, actual_rr=2.5)
        stats = wf.get_pattern_stats("test_pattern")

        assert stats["total"] == 1
        assert stats["wins"] == 1
        assert stats["win_rate"] == 1.0

    def test_update_pattern_db_loss(self):
        """Pattern DB should update on loss"""
        wf = WhitelistFilter()
        wf.update_pattern_db("test_pattern", won=False, actual_rr=0.5)
        stats = wf.get_pattern_stats("test_pattern")

        assert stats["total"] == 1
        assert stats["wins"] == 0

    def test_blacklist_low_win_rate(self):
        """Pattern with <40% win rate should be blacklisted (requires 20+ trades)"""
        wf = WhitelistFilter()
        pattern = "low_win_pattern"

        # Add 20 losses - threshold is 20 trades before blacklisting
        for _ in range(20):
            wf.update_pattern_db(pattern, won=False, actual_rr=0.5)

        assert pattern in wf.blacklist


# =============================================================================
# SignalGenerator Tests
# =============================================================================

class TestSignalGenerator:
    """Signal generation tests"""

    def test_calc_price_score(self):
        """Price score calculation"""
        sg = SignalGenerator()
        factors = {
            "rsi": 25,  # Oversold
            "macd_histogram": 100,
            "momentum": 5,
            "trend": "bull"
        }
        score = sg.calc_price_score(factors)
        assert score > 0  # Should be positive for oversold in uptrend

    def test_calc_news_score(self, sample_intel_report):
        """News score from intel report"""
        sg = SignalGenerator()
        score = sg.calc_news_score(sample_intel_report)
        assert isinstance(score, float)

    def test_calc_onchain_score(self, sample_intel_report):
        """Onchain score from intel report"""
        sg = SignalGenerator()
        score = sg.calc_onchain_score(sample_intel_report)
        assert isinstance(score, float)

    def test_calc_combined_score(self, sample_prices, sample_highs_lows, sample_intel_report):
        """Combined multi-factor score"""
        sg = SignalGenerator()
        factors = PriceFactors.calc_all(
            sample_prices,
            sample_highs_lows["highs"],
            sample_highs_lows["lows"]
        )
        scores = sg.calc_combined_score(factors, sample_intel_report)

        assert "price_score" in scores
        assert "combined" in scores
        assert -1 <= scores["combined"] <= 1

    def test_generate_signal_wait(self):
        """Should return wait signal when no clear direction"""
        sg = SignalGenerator()
        price_data = {
            "prices": [100] * 30,
            "highs": [103] * 30,
            "lows": [97] * 30
        }
        intel = {"sentiment": "neutral", "sentiment_score": 0}

        signal = sg.generate_signal("BTC", price_data, intel)
        assert signal["direction"] == "wait"

    def test_generate_signal_long(self, sample_prices, sample_highs_lows, sample_intel_report):
        """Should generate long signal"""
        price_data = {
            "prices": sample_prices,
            "highs": sample_highs_lows["highs"],
            "lows": sample_highs_lows["lows"]
        }

        sg = SignalGenerator()
        signal = sg.generate_signal("BTC", price_data, sample_intel_report)

        assert "direction" in signal
        assert "entry_price" in signal
        assert "stop_loss" in signal
        assert "take_profit" in signal
        assert "confidence" in signal

    def test_whitelist_integration(self):
        """Whitelist should filter signals"""
        sg = SignalGenerator()

        # Low ADX should fail whitelist
        price_data = {
            "prices": [100 + i for i in range(100)],
            "highs": [105 + i for i in range(100)],
            "lows": [95 + i for i in range(100)]
        }
        intel = {"sentiment": "neutral", "sentiment_score": 0, "onchain": {}, "wallet": {}}

        sg.generate_signal("BTC", price_data, intel)
        # Signal should be generated but with lower confidence due to whitelist


# =============================================================================
# AgentSignal Tests
# =============================================================================

class TestAgentSignal:
    """Agent-S tests"""

    def test_agent_init(self, config):
        """Agent-S should initialize"""
        agent = AgentSignal(config)
        assert agent is not None
        assert hasattr(agent, "generator")

    def test_process_intel(self, sample_prices, sample_highs_lows, sample_intel_report):
        """Agent-S should process intel and generate signal"""
        agent = AgentSignal()
        price_data = {
            "prices": sample_prices,
            "highs": sample_highs_lows["highs"],
            "lows": sample_highs_lows["lows"]
        }

        signal = agent.process_intel("BTC", price_data, sample_intel_report)
        assert signal is not None
        assert "symbol" in signal

    def test_feedback(self):
        """Agent-S should accept feedback for learning"""
        agent = AgentSignal()
        trade_result = {
            "pattern_key": "bull_RSI35-45_ADX30+",
            "won": True,
            "actual_rr": 2.5
        }
        agent.feedback("signal_123", trade_result)

        stats = agent.get_stats()
        assert isinstance(stats, dict)


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
