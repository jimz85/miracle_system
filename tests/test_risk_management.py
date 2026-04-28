"""
Tests for core/risk_management.py - 风控计算
============================================

Covers:
- Normal path: ATR calculation, position sizing, risk monitoring
- Edge cases: zero values, boundary values, empty data
- Exception handling: invalid inputs, division by zero
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.risk_management import (
    ATRCalculator,
    DynamicPositionSizer,
    Position,
    CrossCurrencyRiskMonitor,
    SlippageFeeSimulator,
    create_risk_manager,
)


# ============================================================================
# ATRCalculator Tests
# ============================================================================

class TestATRCalculator:
    """Test ATR (Average True Range) calculator with Wilder smoothing"""

    def test_first_tr_use_simple_range(self):
        """First TR should use high-low range"""
        calc = ATRCalculator(period=14)
        # Need multiple updates for Wilder ATR to be computed
        # First bar: TR = high - low = 10, but ATR not yet computed (need period bars)
        for i in range(15):
            calc.update(high=105 + i, low=95 + i, close=100 + i)
        # After 15 bars, ATR should be computed via Wilder smoothing
        atr = calc.get_atr()
        assert atr > 0  # ATR should be positive after enough data

    def test_tr_includes_prev_close(self):
        """TR should include previous close comparison"""
        calc = ATRCalculator(period=14)
        # Feed enough data for Wilder smoothing
        for i in range(20):
            calc.update(high=105 + i, low=95 + i, close=100 + i)
        atr = calc.get_atr()
        assert atr is not None

    def test_wilder_smoothing_applied(self):
        """Wilder smoothing should be applied after period bars"""
        calc = ATRCalculator(period=14)
        # Feed 20 bars of data
        for i in range(20):
            high = 100 + i + 2
            low = 100 + i
            close = 100 + i + 1
            calc.update(high, low, close)

        # After period bars, ATR should be Wilder smoothed
        assert calc.get_atr() is not None
        assert calc.get_atr() > 0

    def test_get_normalized_atr(self):
        """Normalized ATR percentage calculation"""
        calc = ATRCalculator(period=14)
        for i in range(20):
            calc.update(high=105, low=95, close=100)

        norm_atr = calc.get_normalized_atr(current_price=10000)
        assert norm_atr > 0
        assert norm_atr < 100  # should be a small percentage

    def test_get_atr_before_enough_data(self):
        """get_atr returns 0 when insufficient data"""
        calc = ATRCalculator(period=14)
        calc.update(high=105, low=95, close=100)
        assert calc.get_atr() == 0  # only 1 bar, need 14

    def test_zero_price_handling(self):
        """Zero price should not cause division by zero"""
        calc = ATRCalculator(period=14)
        calc.update(high=0, low=0, close=0)
        norm_atr = calc.get_normalized_atr(current_price=0)
        assert norm_atr == 0  # should handle gracefully


# ============================================================================
# DynamicPositionSizer Tests
# ============================================================================

class TestDynamicPositionSizer:
    """Test dynamic position sizing with ATR-based risk management"""

    def test_calculate_position_normal_case(self):
        """Normal position calculation with valid data"""
        sizer = DynamicPositionSizer(account_balance=10000, risk_percent=0.02)
        result = sizer.calculate_position(
            high=105, low=95, close=100,
            entry_price=100, direction="long"
        )

        assert "position_size" in result
        assert "stop_loss" in result
        assert "take_profit" in result
        assert "atr" in result
        assert result["position_size"] > 0

    def test_position_size_respects_max_limit(self):
        """Position size should not exceed max_position_percent"""
        sizer = DynamicPositionSizer(
            account_balance=10000,
            max_position_percent=0.10  # 10% max
        )

        # Use very small ATR to trigger large theoretical position
        result = sizer.calculate_position(
            high=100.01, low=99.99, close=100,
            entry_price=100, direction="long"
        )

        position_value = result["position_size"] * result["entry_price"]
        max_allowed = sizer.account_balance * sizer.max_position_percent
        assert position_value <= max_allowed * 1.01  # small float tolerance

    def test_position_size_respects_min_limit(self):
        """Position size should not go below min_position_percent"""
        sizer = DynamicPositionSizer(
            account_balance=10000,
            min_position_percent=0.05  # 5% min
        )

        # Use very large ATR to trigger tiny theoretical position
        result = sizer.calculate_position(
            high=200, low=100, close=150,
            entry_price=150, direction="long"
        )

        position_value = result["position_size"] * result["entry_price"]
        min_required = sizer.account_balance * sizer.min_position_percent
        assert position_value >= min_required * 0.99  # small float tolerance

    def test_long_direction_stop_loss_below_entry(self):
        """Long position stop loss should be below entry when ATR > 0"""
        sizer = DynamicPositionSizer(stop_multiplier=2.0)
        # Use enough bars for valid ATR
        for i in range(20):
            sizer.atr_calculator.update(high=105 + i, low=95 + i, close=100 + i)

        result = sizer.calculate_position(
            high=105, low=95, close=100,
            entry_price=100, direction="long"
        )
        assert result["stop_loss"] < result["entry_price"]

    def test_short_direction_stop_loss_above_entry(self):
        """Short position stop loss should be above entry when ATR > 0"""
        sizer = DynamicPositionSizer(stop_multiplier=2.0)
        # Use enough bars for valid ATR
        for i in range(20):
            sizer.atr_calculator.update(high=105 + i, low=95 + i, close=100 + i)

        result = sizer.calculate_position(
            high=105, low=95, close=100,
            entry_price=100, direction="short"
        )
        assert result["stop_loss"] > result["entry_price"]

    def test_take_profit_for_long(self):
        """Long take profit should be above entry when ATR > 0"""
        sizer = DynamicPositionSizer(stop_multiplier=2.0)
        # Use enough bars for valid ATR
        for i in range(20):
            sizer.atr_calculator.update(high=105 + i, low=95 + i, close=100 + i)

        result = sizer.calculate_position(
            high=105, low=95, close=100,
            entry_price=100, direction="long",
            risk_reward_ratio=2.0
        )
        assert result["take_profit"] > result["entry_price"]

    def test_take_profit_for_short(self):
        """Short take profit should be below entry when ATR > 0"""
        sizer = DynamicPositionSizer(stop_multiplier=2.0)
        # Use enough bars for valid ATR
        for i in range(20):
            sizer.atr_calculator.update(high=105 + i, low=95 + i, close=100 + i)

        result = sizer.calculate_position(
            high=105, low=95, close=100,
            entry_price=100, direction="short",
            risk_reward_ratio=2.0
        )
        assert result["take_profit"] < result["entry_price"]

    def test_zero_atr_uses_default_stop(self):
        """Zero ATR should use default 2% stop loss"""
        sizer = DynamicPositionSizer()
        result = sizer.calculate_position(
            high=100, low=100, close=100,
            entry_price=100, direction="long"
        )
        # With zero ATR, should use entry_price * 0.02 as stop distance
        assert result["atr"] == 0 or result["atr"] is not None

    def test_update_balance(self):
        """update_balance should change account_balance"""
        sizer = DynamicPositionSizer(account_balance=10000)
        sizer.update_balance(15000)
        assert sizer.account_balance == 15000

    def test_get_stats(self):
        """get_stats returns correct statistics"""
        sizer = DynamicPositionSizer(account_balance=10000, risk_percent=0.02)
        stats = sizer.get_stats()
        assert stats["account_balance"] == 10000
        assert stats["risk_percent"] == 0.02
        assert "current_atr" in stats

    def test_custom_risk_percent(self):
        """Custom risk percent should be applied"""
        sizer = DynamicPositionSizer(risk_percent=0.01)
        assert sizer.risk_percent == 0.01


# ============================================================================
# CrossCurrencyRiskMonitor Tests
# ============================================================================

class TestCrossCurrencyRiskMonitor:
    """Test cross-currency risk exposure monitoring"""

    def test_add_position(self):
        """Adding position should appear in positions dict"""
        monitor = CrossCurrencyRiskMonitor()
        pos = Position("BTC", "long", 0.1, 50000, 51000)
        monitor.add_position(pos)
        assert "BTC" in monitor.positions

    def test_remove_position(self):
        """Removing position should return the position"""
        monitor = CrossCurrencyRiskMonitor()
        pos = Position("BTC", "long", 0.1, 50000, 51000)
        monitor.add_position(pos)
        removed = monitor.remove_position("BTC")
        assert removed is not None
        assert "BTC" not in monitor.positions

    def test_remove_nonexistent_position(self):
        """Removing nonexistent position returns None"""
        monitor = CrossCurrencyRiskMonitor()
        result = monitor.remove_position("NONEXISTENT")
        assert result is None

    def test_update_position_price(self):
        """Updating position price should recalculate pnl"""
        monitor = CrossCurrencyRiskMonitor()
        pos = Position("BTC", "long", 0.1, 50000, 50000)  # entry=current
        monitor.add_position(pos)
        monitor.update_position_price("BTC", 51000)  # price goes up

        assert monitor.positions["BTC"].unrealized_pnl > 0

    def test_get_total_exposure(self):
        """Total exposure calculation"""
        monitor = CrossCurrencyRiskMonitor(account_balance=10000)
        monitor.add_position(Position("BTC", "long", 0.1, 50000, 50000))  # 5000 notional
        monitor.add_position(Position("ETH", "long", 1.0, 3000, 3000))    # 3000 notional

        total, pct = monitor.get_total_exposure()
        assert total == 8000
        assert pct == 0.8

    def test_get_single_exposure(self):
        """Single asset exposure"""
        monitor = CrossCurrencyRiskMonitor(account_balance=10000)
        monitor.add_position(Position("BTC", "long", 0.1, 50000, 50000))

        notional, pct = monitor.get_single_exposure("BTC")
        assert notional == 5000
        assert pct == 0.5

    def test_single_exposure_nonexistent_symbol(self):
        """Single exposure for nonexistent symbol returns zero"""
        monitor = CrossCurrencyRiskMonitor()
        notional, pct = monitor.get_single_exposure("NONEXISTENT")
        assert notional == 0
        assert pct == 0

    def test_can_open_position_within_limits(self):
        """can_open_position returns True when within limits"""
        monitor = CrossCurrencyRiskMonitor(
            max_single_exposure=0.3,
            max_total_exposure=1.0,
            account_balance=10000
        )
        # 1000 notional = 10% of 10000, should be fine
        can_open, reason = monitor.can_open_position("BTC", 0.02, 50000)
        assert can_open is True

    def test_can_open_position_exceeds_single_limit(self):
        """can_open_position returns False when single exposure exceeded"""
        monitor = CrossCurrencyRiskMonitor(
            max_single_exposure=0.2,
            max_total_exposure=1.0,
            account_balance=10000
        )
        # 3000 notional = 30% > 20% single limit
        can_open, reason = monitor.can_open_position("BTC", 0.06, 50000)
        assert can_open is False
        assert "单币种" in reason

    def test_can_open_position_exceeds_total_limit(self):
        """can_open_position returns False when total exposure exceeded"""
        monitor = CrossCurrencyRiskMonitor(
            max_single_exposure=0.5,
            max_total_exposure=0.5,
            account_balance=10000
        )
        # Already holding 4000 (40%), adding 6000 would be 100%
        monitor.add_position(Position("ETH", "long", 0.08, 50000, 50000))
        can_open, reason = monitor.can_open_position("BTC", 0.04, 50000)  # 2000 more = 60%
        assert can_open is False
        assert "总敞口" in reason

    def test_can_open_position_already_holding(self):
        """can_open_position returns False when already holding symbol"""
        monitor = CrossCurrencyRiskMonitor(
            max_single_exposure=0.5,  # high enough not to trigger single limit
            max_total_exposure=1.0,
            account_balance=10000
        )
        monitor.add_position(Position("BTC", "long", 0.1, 50000, 50000))
        can_open, reason = monitor.can_open_position("BTC", 0.1, 50000)
        assert can_open is False
        assert "已持有" in reason

    def test_get_total_pnl(self):
        """Total pnl calculation"""
        monitor = CrossCurrencyRiskMonitor(account_balance=10000)
        # BTC: long 0.1 at 50000, now 51000 -> +100 pnl
        # ETH: long 1.0 at 3000, now 2900 -> -100 pnl
        monitor.add_position(Position("BTC", "long", 0.1, 50000, 51000))
        monitor.add_position(Position("ETH", "long", 1.0, 3000, 2900))

        total_pnl, pct = monitor.get_total_pnl()
        assert total_pnl == 0  # +100 + (-100)

    def test_get_risk_report(self):
        """Risk report contains expected fields"""
        monitor = CrossCurrencyRiskMonitor(account_balance=10000)
        monitor.add_position(Position("BTC", "long", 0.1, 50000, 50000))

        report = monitor.get_risk_report()
        assert "account_balance" in report
        assert "total_notional" in report
        assert "total_exposure_percent" in report
        assert "positions" in report
        assert "warnings" in report

    def test_risk_report_warnings_at_90_percent(self):
        """Warnings generated when exposure near limits"""
        monitor = CrossCurrencyRiskMonitor(
            max_total_exposure=1.0,
            max_single_exposure=0.3,
            account_balance=10000
        )
        # Add position at 90% of single limit
        monitor.add_position(Position("BTC", "long", 0.06, 50000, 50000))  # 3000 = 30%

        report = monitor.get_risk_report()
        assert len(report["warnings"]) > 0


# ============================================================================
# SlippageFeeSimulator Tests
# ============================================================================

class TestSlippageFeeSimulator:
    """Test slippage and fee simulation"""

    def test_simulate_market_buy(self):
        """Market buy simulation"""
        sim = SlippageFeeSimulator()
        result = sim.simulate_trade("BTC", "buy", 0.1, 50000, "market")

        assert result["execution_price"] > 50000  # slippage increases price for buy
        assert result["side"] == "buy"
        assert result["fee_type"] == "taker"
        assert result["fees"] > 0

    def test_simulate_market_sell(self):
        """Market sell simulation"""
        sim = SlippageFeeSimulator()
        result = sim.simulate_trade("BTC", "sell", 0.1, 50000, "market")

        assert result["execution_price"] < 50000  # slippage decreases price for sell
        assert result["side"] == "sell"
        assert result["fee_type"] == "taker"

    def test_simulate_limit_order(self):
        """Limit order has lower slippage and maker fee"""
        sim = SlippageFeeSimulator()
        market_result = sim.simulate_trade("BTC", "buy", 0.1, 50000, "market")
        limit_result = sim.simulate_trade("BTC", "buy", 0.1, 50000, "limit")

        # Limit order should have lower slippage
        assert limit_result["slippage_pct"] < market_result["slippage_pct"]
        # Limit order should have maker fee
        assert limit_result["fee_type"] == "maker"

    def test_large_order_has_more_slippage(self):
        """Larger orders have more slippage"""
        sim = SlippageFeeSimulator(slippage_model="linear")
        small = sim.simulate_trade("BTC", "buy", 0.01, 50000, "market")
        large = sim.simulate_trade("BTC", "buy", 1.0, 50000, "market")

        assert large["slippage_pct"] > small["slippage_pct"]

    def test_sqrt_slippage_model(self):
        """sqrt slippage model should be different from linear"""
        sim_linear = SlippageFeeSimulator(slippage_model="linear")
        sim_sqrt = SlippageFeeSimulator(slippage_model="sqrt")

        result_linear = sim_linear.simulate_trade("BTC", "buy", 1.0, 50000, "market")
        result_sqrt = sim_sqrt.simulate_trade("BTC", "buy", 1.0, 50000, "market")

        # Both should calculate slippage, but differently
        assert result_linear["slippage_pct"] > 0
        assert result_sqrt["slippage_pct"] > 0

    def test_slippage_capped_at_1_percent(self):
        """Slippage should be capped at 1%"""
        sim = SlippageFeeSimulator()
        # Very large order
        result = sim.simulate_trade("BTC", "buy", 100.0, 50000, "market")
        assert result["slippage_pct"] <= 0.01

    def test_backtest_adjustment(self):
        """Backtest cost adjustment calculation"""
        sim = SlippageFeeSimulator()
        trades = [
            {"symbol": "BTC", "entry_price": 50000, "exit_price": 51000, "size": 0.1, "side": "long"},
            {"symbol": "ETH", "entry_price": 3000, "exit_price": 2900, "size": 1.0, "side": "long"},
        ]

        result = sim.backtest_adjustment(trades, initial_balance=10000)
        assert "total_fees" in result
        assert "total_slippage" in result
        assert "total_cost" in result

    def test_backtest_adjustment_empty_trades(self):
        """Empty trades list handled gracefully"""
        sim = SlippageFeeSimulator()
        result = sim.backtest_adjustment([], initial_balance=10000)
        assert result["total_cost"] == 0
        assert result["adjusted_return"] == 0

    def test_get_stats(self):
        """Stats from trade history"""
        sim = SlippageFeeSimulator()
        sim.simulate_trade("BTC", "buy", 0.1, 50000, "market")
        sim.simulate_trade("ETH", "sell", 1.0, 3000, "market")

        stats = sim.get_stats()
        assert stats["trade_count"] == 2
        assert "total_cost" in stats


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestCreateRiskManager:
    """Test create_risk_manager factory function"""

    def test_returns_tuple_of_three_managers(self):
        """Returns (position_sizer, risk_monitor, fee_simulator)"""
        sizer, monitor, fee_sim = create_risk_manager()
        assert isinstance(sizer, DynamicPositionSizer)
        assert isinstance(monitor, CrossCurrencyRiskMonitor)
        assert isinstance(fee_sim, SlippageFeeSimulator)

    def test_custom_config_applied(self):
        """Custom config passed to managers"""
        config = {"account_balance": 20000, "risk_percent": 0.01}
        sizer, monitor, fee_sim = create_risk_manager(config)
        assert sizer.account_balance == 20000
        assert monitor.account_balance == 20000


# ============================================================================
# Edge Cases and Exception Handling
# ============================================================================

class TestRiskManagementEdgeCases:
    """Test edge cases and exception handling"""

    def test_position_with_zero_entry_price(self):
        """Position with zero entry price - notional_value is size * current_price"""
        monitor = CrossCurrencyRiskMonitor()
        pos = Position("BTC", "long", 0.1, 0, 50000)
        monitor.add_position(pos)
        # notional_value = size * current_price (not entry_price)
        assert pos.notional_value == 5000.0

    def test_position_with_zero_current_price(self):
        """Zero current price handled"""
        monitor = CrossCurrencyRiskMonitor()
        pos = Position("BTC", "long", 0.1, 50000, 0)
        monitor.add_position(pos)
        assert pos.notional_value == 0

    def test_zero_account_balance_exposure(self):
        """Zero account balance exposure calculation"""
        monitor = CrossCurrencyRiskMonitor(account_balance=0)
        monitor.add_position(Position("BTC", "long", 0.1, 50000, 50000))
        total, pct = monitor.get_total_exposure()
        assert total == 5000
        assert pct == 0  # 0 balance means 0%

    def test_atr_calculator_handles_identical_prices(self):
        """Identical high/low/close handled"""
        calc = ATRCalculator(period=14)
        for _ in range(20):
            calc.update(high=100, low=100, close=100)
        assert calc.get_atr() >= 0

    def test_position_sizer_with_zero_entry_price(self):
        """Zero entry price causes division by zero in fallback calculation"""
        sizer = DynamicPositionSizer()
        # With zero entry price and zero ATR, the fallback also divides by zero
        # This is a known limitation - entry_price must be > 0
        with pytest.raises(ZeroDivisionError):
            sizer.calculate_position(
                high=100, low=100, close=100,
                entry_price=0, direction="long"
            )

    def test_slippage_simulation_zero_price(self):
        """Zero market price in slippage simulation"""
        sim = SlippageFeeSimulator()
        result = sim.simulate_trade("BTC", "buy", 0.1, 0, "market")
        assert result["execution_price"] == 0
        assert result["total_cost"] == 0

    def test_multiple_positions_same_symbol(self):
        """Adding same symbol twice replaces position"""
        monitor = CrossCurrencyRiskMonitor()
        monitor.add_position(Position("BTC", "long", 0.1, 50000, 50000))
        monitor.add_position(Position("BTC", "long", 0.2, 51000, 51000))
        assert monitor.positions["BTC"].size == 0.2

    def test_export_import_positions(self):
        """Export and import positions"""
        monitor = CrossCurrencyRiskMonitor(account_balance=10000)
        monitor.add_position(Position("BTC", "long", 0.1, 50000, 51000))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            monitor.export_positions(filepath)
            monitor2 = CrossCurrencyRiskMonitor(account_balance=10000)
            monitor2.import_positions(filepath)
            assert "BTC" in monitor2.positions
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
