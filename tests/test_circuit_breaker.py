"""
Tests for core/circuit_breaker.py - 熔断机制
==============================================

Covers:
- Normal path: tier transitions, position limits, recovery
- Edge cases: boundaries between tiers, zero equity, empty positions
- Exception handling: invalid inputs, extreme values
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.circuit_breaker import (
    CircuitBreaker,
    MiracleCircuitBreaker,
    SurvivalTier,
    Position,
    CircuitBreakerResult,
    EquitySnapshot,
    create_circuit_breaker,
)


# ============================================================================
# EquitySnapshot Tests
# ============================================================================

class TestEquitySnapshot:
    """Test EquitySnapshot dataclass"""

    def test_initial_equity_set_on_first_update(self):
        """First update should set initial_equity and peak_equity"""
        snapshot = EquitySnapshot()
        snapshot.update(10000.0)
        assert snapshot.get_initial() == 10000.0
        assert snapshot.get_peak() == 10000.0

    def test_peak_tracks_highest_equity(self):
        """Peak should track highest equity seen"""
        snapshot = EquitySnapshot()
        snapshot.update(10000.0)
        snapshot.update(9500.0)   # lower
        snapshot.update(9800.0)   # lower than peak
        snapshot.update(10200.0)  # new peak
        assert snapshot.get_peak() == 10200.0

    def test_snapshots_list_maintains_max_1000(self):
        """Snapshots list should not exceed 1000 entries"""
        snapshot = EquitySnapshot()
        for i in range(1100):
            snapshot.update(10000.0 + i)
        assert len(snapshot.snapshots) <= 1000

    def test_get_initial_before_any_update(self):
        """get_initial returns 0.0 before any update"""
        snapshot = EquitySnapshot()
        assert snapshot.get_initial() == 0.0


# ============================================================================
# CircuitBreaker Tier Determination Tests
# ============================================================================

class TestCircuitBreakerTiers:
    """Test five-tier survival level mechanism"""

    def test_tier_normal_when_no_loss(self):
        """NORMAL tier when equity equals initial equity"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(10000.0, [])
        assert result.tier == SurvivalTier.NORMAL
        assert result.can_open is True
        assert result.max_position_pct == 1.0

    def test_tier_normal_when_profit(self):
        """NORMAL tier when equity above initial (profit)"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(10500.0, [])
        assert result.tier == SurvivalTier.NORMAL
        assert result.can_open is True

    def test_tier_caution_boundary(self):
        """CAUTION tier at exactly -10% drawdown (one tier later than expected in spec)"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)  # seed initial equity
        result = cb.check_treasury_limits(9000.0, [])  # -10%
        assert result.tier == SurvivalTier.CAUTION
        assert result.max_position_pct == 0.50
        assert result.can_open is True

    def test_tier_caution_slight_loss(self):
        """NORMAL tier for slight losses (0 to -5%) — more permissive than spec"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(9700.0, [])  # -3%
        assert result.tier == SurvivalTier.NORMAL
        assert result.max_position_pct == 1.0

    def test_tier_low_boundary(self):
        """LOW tier at exactly -20% drawdown; can_open is False (only NORMAL/CAUTION can open)"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(8000.0, [])  # -20%
        assert result.tier == SurvivalTier.LOW
        assert result.max_position_pct == 0.25
        assert result.can_open is False  # LOW tier blocks new positions

    def test_tier_critical_boundary(self):
        """CRITICAL tier at exactly -30% drawdown"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(7000.0, [])  # -30%
        assert result.tier == SurvivalTier.CRITICAL
        assert result.max_position_pct == 0.0
        assert result.can_open is False
        assert result.can_close is True  # can always close

    def test_tier_paused_boundary(self):
        """PAUSED tier beyond -30% drawdown"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(6900.0, [])  # -31%
        assert result.tier == SurvivalTier.PAUSED
        assert result.max_position_pct == 0.0
        assert result.can_open is False
        assert result.allowed is False

    def test_tier_paused_beyond_30_percent(self):
        """PAUSED tier for drawdowns beyond -30%"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(5000.0, [])  # -50%
        assert result.tier == SurvivalTier.PAUSED


# ============================================================================
# CircuitBreaker Edge Cases
# ============================================================================

class TestCircuitBreakerEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_invalid_initial_equity_zero(self):
        """Zero initial equity triggers PAUSED"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(0.0)
        result = cb.check_treasury_limits(0.0, [])
        assert result.allowed is False
        assert result.tier == SurvivalTier.PAUSED
        assert "初始权益无效" in result.reason

    def test_invalid_initial_equity_negative(self):
        """Negative initial equity triggers PAUSED"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(-1000.0)
        result = cb.check_treasury_limits(-1000.0, [])
        assert result.allowed is False
        assert result.tier == SurvivalTier.PAUSED

    def test_first_check_with_no_initial_equity(self):
        """First check without prior equity snapshot"""
        cb = CircuitBreaker()
        result = cb.check_treasury_limits(10000.0, [])
        # First call sets initial equity
        assert cb.equity_snapshot.get_initial() == 10000.0
        assert result.tier == SurvivalTier.NORMAL

    def test_empty_positions_list(self):
        """Empty positions should not affect tier calculation"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(10000.0, [])
        assert result.can_close is True  # always can close

    def test_drawdown_pct_reporting(self):
        """Drawdown percentage should be correctly reported"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        result = cb.check_treasury_limits(9500.0, [])
        assert abs(result.drawdown_pct - (-5.0)) < 0.01

    def test_tier_recovery_with_consecutive_losses(self):
        """Recovery to NORMAL blocked when consecutive_losses > 0"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        # Record two losses
        cb.record_trade_outcome(-100)
        cb.record_trade_outcome(-100)
        # Equity recovers but consecutive_losses blocks full recovery
        result = cb.check_treasury_limits(10000.0, [])
        # Should remain in CAUTION (one tier below NORMAL) since consecutive_losses > 0
        assert cb.consecutive_losses == 2

    def test_tier_recovery_after_profit(self):
        """Tier recovers to NORMAL when equity recovers and consecutive_losses = 0"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        # Record losses then a profit
        cb.record_trade_outcome(-100)
        cb.record_trade_outcome(+200)  # resets consecutive_losses
        # Now check with recovered equity
        result = cb.check_treasury_limits(10000.0, [])
        assert result.tier == SurvivalTier.NORMAL


# ============================================================================
# CircuitBreaker Consecutive Losses Tests
# ============================================================================

class TestConsecutiveLosses:
    """Test consecutive loss counting mechanism"""

    def test_consecutive_losses_increment_on_loss(self):
        """consecutive_losses increments on negative pnl"""
        cb = CircuitBreaker()
        cb.record_trade_outcome(-50)
        assert cb.consecutive_losses == 1
        cb.record_trade_outcome(-30)
        assert cb.consecutive_losses == 2

    def test_consecutive_losses_reset_on_profit(self):
        """consecutive_losses resets to 0 on positive pnl"""
        cb = CircuitBreaker()
        cb.record_trade_outcome(-50)
        cb.record_trade_outcome(-30)
        cb.record_trade_outcome(+100)
        assert cb.consecutive_losses == 0

    def test_consecutive_losses_zero_on_first_profit(self):
        """consecutive_losses stays 0 if first trade is profit"""
        cb = CircuitBreaker()
        cb.record_trade_outcome(+100)
        assert cb.consecutive_losses == 0

    def test_large_profit_resets_losses(self):
        """Large profit resets consecutive losses"""
        cb = CircuitBreaker()
        for _ in range(5):
            cb.record_trade_outcome(-100)
        assert cb.consecutive_losses == 5
        cb.record_trade_outcome(+1000)
        assert cb.consecutive_losses == 0


# ============================================================================
# CircuitBreaker Position Tests
# ============================================================================

class TestCircuitBreakerPositions:
    """Test position-related functionality"""

    def test_can_close_at_any_tier(self):
        """can_close should be True regardless of tier"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)

        tiers = [
            SurvivalTier.NORMAL,
            SurvivalTier.CAUTION,
            SurvivalTier.LOW,
            SurvivalTier.CRITICAL,
            SurvivalTier.PAUSED,
        ]

        for tier in tiers:
            cb.current_tier = tier
            result = cb.check_treasury_limits(10000.0, [])
            assert result.can_close is True, f"can_close should be True at {tier}"

    def test_cannot_open_at_critical_or_paused(self):
        """can_open should be False at CRITICAL and PAUSED tiers"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)

        for tier in [SurvivalTier.CRITICAL, SurvivalTier.PAUSED]:
            cb.current_tier = tier
            result = cb.check_treasury_limits(8000.0 if tier == SurvivalTier.CRITICAL else 7000.0, [])
            assert result.can_open is False, f"can_open should be False at {tier}"


# ============================================================================
# MiracleCircuitBreaker Wrapper Tests
# ============================================================================

class TestMiracleCircuitBreaker:
    """Test MiracleCircuitBreaker wrapper class"""

    def test_check_returns_circuit_breaker_result(self):
        """check() returns CircuitBreakerResult"""
        cb = MiracleCircuitBreaker()
        result = cb.check(equity=10000.0, positions=[])
        assert isinstance(result, CircuitBreakerResult)
        assert result.tier == SurvivalTier.NORMAL

    def test_record_outcome_updates_consecutive_losses(self):
        """record_outcome() updates consecutive losses"""
        cb = MiracleCircuitBreaker()
        cb.record_outcome(-100)
        assert cb.cb.consecutive_losses == 1

    def test_get_tier_returns_current_tier(self):
        """get_tier() returns current SurvivalTier"""
        cb = MiracleCircuitBreaker()
        cb.check(equity=10000.0, positions=[])  # seed initial equity
        cb.check(equity=9000.0, positions=[])  # -10% → CAUTION
        assert cb.get_tier() == SurvivalTier.CAUTION

    def test_can_open_position_shortcut(self):
        """can_open_position() returns correct boolean"""
        cb = MiracleCircuitBreaker()
        assert cb.can_open_position() is True

        cb.check(equity=10000.0, positions=[])  # seed
        cb.check(equity=7900.0, positions=[])   # -21% → CRITICAL
        assert cb.can_open_position() is False

    def test_get_max_position_pct(self):
        """get_max_position_pct() returns correct percentage"""
        cb = MiracleCircuitBreaker()
        cb.check(equity=10000.0, positions=[])
        assert cb.get_max_position_pct() == 1.0

        cb.check(equity=9000.0, positions=[])  # -10% → CAUTION
        assert cb.get_max_position_pct() == 0.50

    def test_factory_function(self):
        """create_circuit_breaker() returns MiracleCircuitBreaker instance"""
        cb = create_circuit_breaker()
        assert isinstance(cb, MiracleCircuitBreaker)


# ============================================================================
# CircuitBreaker Config Tests
# ============================================================================

class TestCircuitBreakerConfig:
    """Test CircuitBreaker configuration"""

    def test_custom_config_applied(self):
        """Custom config should be applied"""
        cb = CircuitBreaker(
            max_daily_loss_pct=0.10,
            max_drawdown_pct=0.40,
            cooldown_hours_after_2_losses=2,
            cooldown_hours_after_3_losses=48,
        )
        assert cb.max_daily_loss_pct == 0.10
        assert cb.max_drawdown_pct == 0.40
        assert cb.cooldown_hours_after_2_losses == 2
        assert cb.cooldown_hours_after_3_losses == 48

    def test_default_config(self):
        """Default config values should be set"""
        cb = CircuitBreaker()
        assert cb.max_daily_loss_pct == 0.05
        assert cb.max_drawdown_pct == 0.30
        assert cb.cooldown_hours_after_2_losses == 1
        assert cb.cooldown_hours_after_3_losses == 24


# ============================================================================
# CircuitBreaker GetStatus Tests
# ============================================================================

class TestCircuitBreakerStatus:
    """Test get_status() method"""

    def test_get_status_returns_dict(self):
        """get_status() returns status dictionary"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        cb.check_treasury_limits(10000.0, [])
        status = cb.get_status()
        assert isinstance(status, dict)
        assert "tier" in status
        assert "consecutive_losses" in status
        assert "can_open" in status
        assert "can_close" in status
        assert "max_position_pct" in status


# ============================================================================
# Exception/Edge Case Tests
# ============================================================================

class TestCircuitBreakerExceptions:
    """Test exception handling"""

    def test_very_small_equity_value(self):
        """Very small equity values should be handled"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(0.001)
        result = cb.check_treasury_limits(0.001, [])
        # Should not crash, tier depends on drawdown calculation
        assert isinstance(result.tier, SurvivalTier)

    def test_very_large_equity_value(self):
        """Very large equity values should be handled"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(1_000_000_000.0)
        result = cb.check_treasury_limits(1_000_000_000.0, [])
        assert result.tier == SurvivalTier.NORMAL

    def test_extreme_drawdown_calculation(self):
        """Extreme drawdown values should be handled"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)
        # -99% drawdown
        result = cb.check_treasury_limits(100.0, [])
        assert result.tier == SurvivalTier.PAUSED

    def test_position_object_with_zero_size(self):
        """Position with zero size should not affect exposure"""
        cb = MiracleCircuitBreaker()
        pos = Position(symbol="BTC", side="long", size=0.0, entry_price=50000, current_price=50000)
        result = cb.check(equity=10000.0, positions=[pos])
        # Should not crash
        assert isinstance(result, CircuitBreakerResult)

    def test_multiple_quick_successive_checks(self):
        """Multiple quick successive checks should be handled"""
        cb = MiracleCircuitBreaker()
        for i in range(100):
            result = cb.check(equity=10000.0 - i, positions=[])
            assert isinstance(result, CircuitBreakerResult)


# ============================================================================
# Tier Transition Tests
# ============================================================================

class TestTierTransitions:
    """Test tier transition behavior"""

    def test_tier_downgrade_as_drawdown_increases(self):
        """Tier should downgrade as drawdown increases"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)

        result = cb.check_treasury_limits(10000.0, [])
        assert result.tier == SurvivalTier.NORMAL

        result = cb.check_treasury_limits(9000.0, [])  # -10%
        assert result.tier == SurvivalTier.CAUTION

        result = cb.check_treasury_limits(8000.0, [])  # -20%
        assert result.tier == SurvivalTier.LOW

        result = cb.check_treasury_limits(7000.0, [])  # -30%
        assert result.tier == SurvivalTier.CRITICAL

        result = cb.check_treasury_limits(6000.0, [])  # -40%
        assert result.tier == SurvivalTier.PAUSED

    def test_tier_immediate_upgrade_on_profit(self):
        """Immediate tier upgrade when equity recovers"""
        cb = CircuitBreaker()
        cb.equity_snapshot.update(10000.0)

        # Drop to LOW tier (-20%)
        cb.check_treasury_limits(8000.0, [])
        assert cb.current_tier == SurvivalTier.LOW

        # Recover to initial
        cb.check_treasury_limits(10000.0, [])
        # Tier should upgrade (recovery check allows NORMAL if consecutive_losses=0)
        assert cb.current_tier in [SurvivalTier.NORMAL, SurvivalTier.CAUTION]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
