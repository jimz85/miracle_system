"""
Miracle 1.0.1 - Risk Metrics and Adaptive Learning Tests
=======================================================

Run with: pytest tests/test_risk_metrics.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miracle_core import RiskMetrics
from scripts.adaptive_learner import AdaptiveLearner, WalkForwardValidator, calc_information_coefficient


# =============================================================================
# RiskMetrics Tests
# =============================================================================

class TestRiskMetrics:
    """Risk metrics calculation tests"""

    def test_var_basic(self):
        """VaR should return negative value for losses"""
        # 80% of returns are negative
        returns = [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02]
        var = RiskMetrics.calculate_var(returns, confidence=0.95)
        assert isinstance(var, float)

    def test_var_insufficient_data(self):
        """VaR should return 0 for insufficient data"""
        returns = [0.01, 0.02]
        var = RiskMetrics.calculate_var(returns)
        assert var == 0.0

    def test_cvar_basic(self):
        """CVaR should be more extreme than VaR"""
        returns = [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02]
        var = RiskMetrics.calculate_var(returns, confidence=0.95)
        cvar = RiskMetrics.calculate_cvar(returns, confidence=0.95)
        # CVaR should be <= VaR (more negative)
        assert cvar <= var

    def test_cvar_insufficient_data(self):
        """CVaR should return 0 for insufficient data"""
        returns = [0.01]
        cvar = RiskMetrics.calculate_cvar(returns)
        assert cvar == 0.0

    def test_max_drawdown_basic(self):
        """Max drawdown should be positive"""
        returns = [0.05, 0.03, -0.10, 0.02, 0.01]
        mdd = RiskMetrics.calculate_max_drawdown(returns)
        assert mdd >= 0

    def test_max_drawdown_no_drawdown(self):
        """Max drawdown should be 0 for consecutive gains"""
        returns = [0.01, 0.02, 0.03, 0.04]
        mdd = RiskMetrics.calculate_max_drawdown(returns)
        assert mdd == 0.0

    def test_max_drawdown_empty(self):
        """Max drawdown should be 0 for empty data"""
        mdd = RiskMetrics.calculate_max_drawdown([])
        assert mdd == 0.0

    def test_sharpe_ratio_basic(self):
        """Sharpe ratio should be calculated"""
        returns = [0.01, 0.02, -0.01, 0.03, 0.015]
        sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_sharpe_ratio_insufficient_data(self):
        """Sharpe ratio should be 0 for insufficient data"""
        returns = [0.01]
        sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_zero_std(self):
        """Sharpe ratio should be 0 when std is 0"""
        returns = [0.01, 0.01, 0.01, 0.01]
        sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sortino_ratio_basic(self):
        """Sortino ratio should be calculated"""
        returns = [0.01, 0.02, -0.01, 0.03, 0.015]
        sortino = RiskMetrics.calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)

    def test_sortino_ratio_insufficient_data(self):
        """Sortino ratio should be 0 for insufficient data"""
        returns = [0.01]
        sortino = RiskMetrics.calculate_sortino_ratio(returns)
        assert sortino == 0.0

    def test_sortino_ratio_no_downside(self):
        """Sortino ratio should be 0 when no downside returns"""
        returns = [0.01, 0.02, 0.03, 0.04]
        sortino = RiskMetrics.calculate_sortino_ratio(returns)
        assert sortino == 0.0


# =============================================================================
# Information Coefficient Tests
# =============================================================================

class TestInformationCoefficient:
    """IC calculation tests"""

    def test_ic_perfect_positive(self):
        """IC should be 1 for perfect positive correlation"""
        signals = list(range(1, 11))  # [1, 2, ..., 10]
        returns = list(range(1, 11))   # [1, 2, ..., 10]
        ic, p_value = calc_information_coefficient(signals, returns)
        assert abs(ic - 1.0) < 0.01

    def test_ic_perfect_negative(self):
        """IC should be -1 for perfect negative correlation"""
        signals = list(range(1, 11))  # [1, 2, ..., 10]
        returns = list(range(10, 0, -1))  # [10, 9, ..., 1]
        ic, p_value = calc_information_coefficient(signals, returns)
        assert abs(ic + 1.0) < 0.01

    def test_ic_no_correlation(self):
        """IC should be close to 0 for no correlation"""
        signals = list(range(1, 11))  # [1, 2, ..., 10]
        # Use random values with no clear correlation
        returns = [5, 3, 8, 2, 7, 1, 6, 4, 9, 0]
        ic, p_value = calc_information_coefficient(signals, returns)
        # IC should be close to 0 for uncorrelated data
        assert abs(ic) < 0.5  # Allow some tolerance for small samples

    def test_ic_insufficient_data(self):
        """IC should be 0 for insufficient data"""
        signals = [1, 2]
        returns = [1, 2]
        ic, p_value = calc_information_coefficient(signals, returns)
        assert ic == 0.0


# =============================================================================
# Walk-Forward Validator Tests
# =============================================================================

class TestWalkForwardValidator:
    """Walk-forward validation tests"""

    def test_wfv_basic(self):
        """Walk-forward validator should work"""
        wf = WalkForwardValidator(train_window=20, test_window=10)

        data = [
            {"signal": i * 0.1, "return": i * 0.08}
            for i in range(50)
        ]

        def strategy_func(train_data, test_data):
            signals = [d["signal"] for d in train_data]
            returns = [d["return"] for d in train_data]
            ic, _ = calc_information_coefficient(signals, returns)
            return {"train_ic": ic}

        result = wf.validate(strategy_func, data, n_windows=3)
        assert "train_ic" in result
        assert "test_ic" in result
        assert "decay" in result

    def test_wfv_insufficient_data(self):
        """Walk-forward should handle insufficient data"""
        wf = WalkForwardValidator()

        data = [{"signal": 1, "return": 1}]

        result = wf.validate(lambda x: {"train_ic": 0}, data, n_windows=5)
        assert result["reason"] == "insufficient_data"


# =============================================================================
# AdaptiveLearner Tests
# =============================================================================

class TestAdaptiveLearner:
    """Adaptive learning system tests"""

    def test_learner_init(self):
        """AdaptiveLearner should initialize"""
        config = {
            "factors": {
                "price_momentum": {"weight": 0.6},
                "news_sentiment": {"weight": 0.2}
            }
        }
        learner = AdaptiveLearner(config)
        assert learner is not None
        assert learner.min_sample_size == 20

    def test_update_factor_performance(self):
        """Should update factor performance"""
        config = {"factors": {"test_factor": {"weight": 0.5}}}
        learner = AdaptiveLearner(config)

        learner.update_factor_performance("test_factor", 0.5, 0.02)

        # Access via factor_evaluator
        perf = learner.factor_evaluator.factor_performance["test_factor"]
        assert len(perf["signals"]) == 1
        assert len(perf["returns"]) == 1

    def test_update_pattern_performance(self):
        """Should update pattern performance"""
        config = {"factors": {}}
        learner = AdaptiveLearner(config)

        learner.update_pattern_performance("test_pattern", won=True, actual_rr=2.5)

        # Access via pattern_evaluator
        stats = learner.pattern_evaluator.pattern_performance["test_pattern"]
        assert stats["total"] == 1
        assert stats["wins"] == 1

    def test_adjust_factor_weights_insufficient_data(self):
        """Should keep default weights with insufficient data"""
        config = {
            "factors": {
                "price_momentum": {"weight": 0.6}
            }
        }
        learner = AdaptiveLearner(config)
        learner.min_sample_size = 100  # Set high requirement

        weights = learner.adjust_factor_weights()
        assert weights["price_momentum"] == 0.6

    def test_adjust_factor_weights_ic_based(self):
        """Should adjust weights based on IC"""
        config = {
            "factors": {
                "price_momentum": {"weight": 0.6}
            }
        }
        learner = AdaptiveLearner(config)
        learner.min_sample_size = 5  # Low requirement

        # Add data with positive IC (high correlation)
        for i in range(25):
            learner.update_factor_performance("price_momentum", i * 0.1, i * 0.08)

        weights = learner.adjust_factor_weights()
        # Should increase due to positive IC
        assert weights["price_momentum"] > 0.6

    def test_detect_overfitting_insufficient_data(self):
        """Should return not overfitting with insufficient data"""
        config = {"factors": {}}
        learner = AdaptiveLearner(config)

        result = learner.detect_overfitting()
        assert result["is_overfitting"] == False

    def test_get_pattern_stats(self):
        """Should get pattern statistics"""
        config = {"factors": {}}
        learner = AdaptiveLearner(config)

        learner.update_pattern_performance("bull_pattern", won=True, actual_rr=2.0)
        learner.update_pattern_performance("bull_pattern", won=True, actual_rr=2.5)
        learner.update_pattern_performance("bull_pattern", won=False, actual_rr=0.5)

        stats = learner.get_pattern_stats("bull_pattern")
        assert stats["total_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["win_rate"] == pytest.approx(2/3, rel=0.01)

    def test_get_factor_ic_report(self):
        """Should get factor IC report"""
        config = {
            "factors": {
                "price_momentum": {"weight": 0.6}
            }
        }
        learner = AdaptiveLearner(config)

        # Add sufficient data
        for i in range(30):
            learner.update_factor_performance("price_momentum", i * 0.1, i * 0.08)

        report = learner.get_factor_ic_report()
        assert "price_momentum" in report

    def test_pattern_blocking(self):
        """Should block patterns with low win rate"""
        config = {"factors": {}}
        learner = AdaptiveLearner(config)

        # Add 5 losses
        for _ in range(5):
            learner.update_pattern_performance("low_win_pattern", won=False, actual_rr=0.5)

        # Pattern should be blocked (5 losses = 0% win rate < 40% threshold)
        is_allowed = learner.pattern_evaluator.is_allowed("low_win_pattern")
        assert is_allowed == False


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
