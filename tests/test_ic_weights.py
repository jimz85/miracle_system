"""
Tests for core/ic_weights.py - IC权重系统
=========================================

Covers:
- Normal path: IC calculation, weight updates, direction prediction
- Edge cases: insufficient samples, boundary IC values, default weights
- Exception handling: invalid entries, missing data, file errors
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ic_weights import (
    DECAY_FACTOR,
    DEFAULT_WEIGHTS,
    FACTORS,
    MIN_SAMPLES,
    MIN_WEIGHT,
    FactorStats,
    ICWeightManager,
    ICWeights,
    calculate_ic,
    get_ic_manager,
    get_ic_values,
    get_weights,
    reset_weights,
    update_weights,
)

# ============================================================================
# ICWeightManager Basic Tests
# ============================================================================

class TestICWeightManagerBasics:
    """Test ICWeightManager basic functionality"""

    def test_get_instance_returns_singleton(self):
        """get_instance returns same instance"""
        manager1 = ICWeightManager.get_instance()
        manager2 = ICWeightManager.get_instance()
        assert manager1 is manager2

    def test_get_weights_returns_dict(self):
        """get_weights returns weights dictionary"""
        manager = ICWeightManager()
        weights = manager.get_weights()
        assert isinstance(weights, dict)
        assert all(f in weights for f in FACTORS)

    def test_default_weights_are_equal(self):
        """Default weights should be equal (0.20 each)"""
        manager = ICWeightManager()
        weights = manager.get_weights()
        assert len(set(weights.values())) == 1  # all equal
        assert weights['rsi'] == 0.20

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0"""
        manager = ICWeightManager()
        weights = manager.get_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.0001

    def test_get_ic_values_returns_dict(self):
        """get_ic_values returns IC values dictionary"""
        manager = ICWeightManager()
        ic_values = manager.get_ic_values()
        assert isinstance(ic_values, dict)
        assert all(f in ic_values for f in FACTORS)

    def test_get_sample_counts_returns_dict(self):
        """get_sample_counts returns counts dictionary"""
        manager = ICWeightManager()
        counts = manager.get_sample_counts()
        assert isinstance(counts, dict)
        assert all(f in counts for f in FACTORS)

    def test_get_info_returns_complete_info(self):
        """get_info returns complete IC system info"""
        manager = ICWeightManager()
        info = manager.get_info()
        assert "weights" in info
        assert "ic_values" in info
        assert "sample_counts" in info
        assert "decay_factor" in info
        assert "min_samples" in info
        assert info["decay_factor"] == DECAY_FACTOR
        assert info["min_samples"] == MIN_SAMPLES


# ============================================================================
# IC Calculation Tests
# ============================================================================

class TestICCalculation:
    """Test IC (Information Coefficient) calculation"""

    def test_calculate_ic_with_insufficient_samples(self):
        """IC returns 0 when samples < MIN_SAMPLES"""
        manager = ICWeightManager()
        # Create mock entries with less than MIN_SAMPLES
        mock_entries = [
            {"factors": {"rsi": 25}, "verdict": "BUY", "outcome": "WIN"},
            {"factors": {"rsi": 75}, "verdict": "SELL", "outcome": "WIN"},
        ]
        ic = manager.calculate_ic("rsi", entries=mock_entries)
        assert ic == 0.0

    def test_calculate_ic_rsi_oversold_long(self):
        """RSI < 30 should predict LONG"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", 25)
        assert predicted == 1  # LONG

    def test_calculate_ic_rsi_overbought_short(self):
        """RSI > 70 should predict SHORT"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", 75)
        assert predicted == -1  # SHORT

    def test_calculate_ic_rsi_neutral(self):
        """RSI 30-70 should predict NEUTRAL"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", 50)
        assert predicted == 0  # NEUTRAL

    def test_calculate_ic_macd_positive(self):
        """MACD > 0 should predict LONG"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("macd", 10)
        assert predicted == 1

    def test_calculate_ic_macd_negative(self):
        """MACD < 0 should predict SHORT"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("macd", -10)
        assert predicted == -1

    def test_calculate_ic_adx_strong_trend(self):
        """ADX > 25 indicates strong trend"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("adx", 30)
        assert predicted == 1  # has trend

    def test_calculate_ic_bollinger_lower_band(self):
        """Bollinger < 0.2 indicates lower band touch = LONG"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("bollinger", 0.1)
        assert predicted == 1

    def test_calculate_ic_bollinger_upper_band(self):
        """Bollinger > 0.8 indicates upper band touch = SHORT"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("bollinger", 0.9)
        assert predicted == -1

    def test_calculate_ic_momentum_positive(self):
        """Positive momentum predicts LONG"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("momentum", 5)
        assert predicted == 1

    def test_calculate_ic_momentum_negative(self):
        """Negative momentum predicts SHORT"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("momentum", -5)
        assert predicted == -1

    def test_calculate_ic_preserves_entry_data(self):
        """calculate_ic doesn't modify input entries"""
        manager = ICWeightManager()
        mock_entries = [
            {"factors": {"rsi": 25}, "verdict": "BUY", "outcome": "WIN", "id": "test1"},
        ]
        original = json.dumps(mock_entries, sort_keys=True)
        manager.calculate_ic("rsi", entries=mock_entries)
        after = json.dumps(mock_entries, sort_keys=True)
        assert original == after


# ============================================================================
# Direction Prediction Edge Cases
# ============================================================================

class TestDirectionPredictionEdgeCases:
    """Test direction prediction with edge case inputs"""

    def test_string_factor_value_bullish(self):
        """String 'bullish' predicts LONG"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", "bullish")
        assert predicted == 1

    def test_string_factor_value_bearish(self):
        """String 'bearish' predicts SHORT"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", "bearish")
        assert predicted == -1

    def test_string_factor_value_long(self):
        """String 'long' predicts LONG"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", "long signal")
        assert predicted == 1

    def test_string_factor_value_short(self):
        """String 'short' predicts SHORT"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", "short signal")
        assert predicted == -1

    def test_unknown_factor_returns_neutral(self):
        """Unknown factor returns 0 (NEUTRAL)"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("unknown_factor", 50)
        assert predicted == 0

    def test_rsi_exactly_30_boundary(self):
        """RSI exactly at 30 should be NEUTRAL (boundary case)"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", 30)
        assert predicted == 0

    def test_rsi_exactly_70_boundary(self):
        """RSI exactly at 70 should be NEUTRAL (boundary case)"""
        manager = ICWeightManager()
        predicted = manager._predict_direction("rsi", 70)
        assert predicted == 0


# ============================================================================
# Outcome Checking Tests
# ============================================================================

class TestOutcomeChecking:
    """Test outcome correctness checking"""

    def test_predicted_long_and_win_is_correct(self):
        """Predict LONG + WIN = correct"""
        result = ICWeightManager()._check_outcome(1, "BUY", "WIN")
        assert result is True

    def test_predicted_short_and_loss_is_correct(self):
        """Predict SHORT + LOSS = correct"""
        result = ICWeightManager()._check_outcome(-1, "SELL", "LOSS")
        assert result is True

    def test_predicted_long_but_loss_is_wrong(self):
        """Predict LONG + LOSS = wrong"""
        result = ICWeightManager()._check_outcome(1, "BUY", "LOSS")
        assert result is False

    def test_predicted_short_but_win_is_wrong(self):
        """Predict SHORT + WIN = wrong"""
        result = ICWeightManager()._check_outcome(-1, "SELL", "WIN")
        assert result is False

    def test_hold_verdict_returns_none(self):
        """HOLD verdict cannot be checked"""
        result = ICWeightManager()._check_outcome(1, "HOLD", "WIN")
        assert result is None

    def test_wait_verdict_returns_none(self):
        """WAIT verdict cannot be checked"""
        result = ICWeightManager()._check_outcome(1, "WAIT", "LOSS")
        assert result is None

    def test_empty_verdict_returns_none(self):
        """Empty verdict cannot be checked"""
        result = ICWeightManager()._check_outcome(1, "", "WIN")
        assert result is None


# ============================================================================
# Weight Update Tests
# ============================================================================

class TestWeightUpdate:
    """Test weight update mechanism"""

    def test_reset_to_default_weights(self):
        """reset_to_default restores DEFAULT_WEIGHTS"""
        manager = ICWeightManager()
        # Modify weights
        manager._state.weights = {f: 0.5 for f in FACTORS}
        manager.reset_to_default()

        weights = manager.get_weights()
        for f in FACTORS:
            assert weights[f] == DEFAULT_WEIGHTS[f]

    def test_update_weights_with_insufficient_samples(self):
        """update_weights preserves old weights when samples insufficient"""
        manager = ICWeightManager()
        old_weights = manager.get_weights()

        # Mock get_all_entries to return fewer than MIN_SAMPLES entries
        with patch('core.ic_weights.get_all_entries', return_value=[]):
            manager.update_weights()

        # With no samples, weights should remain unchanged
        assert manager.get_weights() == old_weights

    def test_update_weights_affects_state(self):
        """update_weights changes internal state"""
        manager = ICWeightManager()

        # First update should save
        with patch('core.ic_weights.get_all_entries', return_value=[]):
            manager.update_weights()

        assert manager._state.last_updated is not None


# ============================================================================
# IC Weight Module Functions Tests
# ============================================================================

class TestModuleFunctions:
    """Test module-level convenience functions"""

    def test_get_weights_function(self):
        """get_weights() module function works"""
        weights = get_weights()
        assert isinstance(weights, dict)

    def test_update_weights_function(self):
        """update_weights() module function works"""
        with patch('core.ic_weights.get_all_entries', return_value=[]):
            weights = update_weights()
        assert isinstance(weights, dict)

    def test_get_ic_values_function(self):
        """get_ic_values() module function works"""
        ic_values = get_ic_values()
        assert isinstance(ic_values, dict)

    def test_calculate_ic_function(self):
        """calculate_ic() module function works"""
        with patch('core.ic_weights.get_all_entries', return_value=[]):
            ic = calculate_ic("rsi")
        assert isinstance(ic, float)
        assert 0.0 <= ic <= 1.0

    def test_reset_weights_function(self):
        """reset_weights() module function works"""
        result = reset_weights()
        assert isinstance(result, dict)


# ============================================================================
# FactorStats Tests
# ============================================================================

class TestFactorStats:
    """Test FactorStats dataclass"""

    def test_accuracy_with_no_samples(self):
        """Accuracy is 0 when total is 0"""
        stats = FactorStats()
        assert stats.accuracy == 0.0

    def test_accuracy_calculation(self):
        """Accuracy = correct / total"""
        stats = FactorStats(correct=7, total=10)
        assert stats.accuracy == 0.7


# ============================================================================
# File Operations Tests
# ============================================================================

class TestFileOperations:
    """Test file save/load operations"""

    def test_load_nonexistent_file_uses_defaults(self):
        """Loading nonexistent file uses default weights"""
        # Mock the CACHE_FILE path to nonexistent
        with patch('core.ic_weights.CACHE_FILE', '/nonexistent/path/weights.json'):
            manager = ICWeightManager()
            weights = manager.get_weights()
            assert weights == DEFAULT_WEIGHTS

    def test_save_creates_directory(self):
        """_save creates directory if needed"""
        manager = ICWeightManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'weights.json')
            with patch('core.ic_weights.CACHE_FILE', cache_path):
                manager._save()
                assert os.path.exists(cache_path)


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestICWeightsEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_all_factors_have_weight(self):
        """All FACTORS have a weight entry"""
        manager = ICWeightManager()
        weights = manager.get_weights()
        for factor in FACTORS:
            assert factor in weights

    def test_weights_are_non_negative(self):
        """All weights should be >= 0"""
        manager = ICWeightManager()
        weights = manager.get_weights()
        for w in weights.values():
            assert w >= 0

    def test_ic_values_initialized_to_zero(self):
        """IC values start at 0.0"""
        manager = ICWeightManager()
        ic_values = manager.get_ic_values()
        for ic in ic_values.values():
            assert ic == 0.0

    def test_sample_counts_initialized_to_zero(self):
        """Sample counts start at 0"""
        manager = ICWeightManager()
        counts = manager.get_sample_counts()
        for count in counts.values():
            assert count == 0

    def test_min_weight_constant(self):
        """MIN_WEIGHT constant is defined"""
        assert MIN_WEIGHT == 0.05
        assert MIN_WEIGHT > 0

    def test_decay_factor_constant(self):
        """DECAY_FACTOR constant is defined"""
        assert 0 < DECAY_FACTOR < 1

    def test_min_samples_constant(self):
        """MIN_SAMPLES constant is defined"""
        assert MIN_SAMPLES >= 1

    def test_factor_list_complete(self):
        """FACTORS list contains expected factors"""
        expected = ['rsi', 'macd', 'adx', 'bollinger', 'momentum']
        assert FACTORS == expected


# ============================================================================
# Concurrent Access Tests
# ============================================================================

class TestConcurrentAccess:
    """Test thread safety and concurrent access"""

    def test_get_instance_thread_safe(self):
        """get_instance should be thread-safe (basic test)"""
        import threading

        results = []

        def get_manager():
            m = ICWeightManager.get_instance()
            results.append(id(m))

        threads = [threading.Thread(target=get_manager) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should get same instance
        assert len(set(results)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
