"""
Miracle 1.0.1 - Integration Tests
=================================
End-to-end integration tests for the trading system.

Run with: pytest tests/test_integration.py -v
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_coordinator import CoordinatorState, MiracleCoordinator

# =============================================================================
# Coordinator Integration Tests
# =============================================================================

class TestCoordinatorIntegration:
    """Coordinator integration tests"""

    def test_coordinator_init(self):
        """Coordinator should initialize with symbols"""
        coordinator = MiracleCoordinator(symbols=["BTC", "ETH"])
        assert coordinator is not None
        assert "BTC" in coordinator.symbols
        assert "ETH" in coordinator.symbols

    def test_coordinator_default_symbols(self):
        """Coordinator should have default symbols"""
        coordinator = MiracleCoordinator()
        assert len(coordinator.symbols) > 0
        assert "BTC" in coordinator.symbols

    def test_state_load(self, tmp_path):
        """CoordinatorState should load from file"""
        state_path = tmp_path / "test_state.json"

        # Create initial state
        coordinator = CoordinatorState(state_path)
        assert coordinator.state is not None

    def test_can_trade_initially(self, tmp_path):
        """Should be able to trade initially"""
        state_path = tmp_path / "test_state.json"
        state = CoordinatorState(state_path)
        can_trade, reason = state.can_trade()
        assert can_trade
        assert reason == "可以交易"

    def test_can_trade_daily_limit(self, tmp_path):
        """Should respect daily trade limit"""
        state_path = tmp_path / "test_state.json"
        state = CoordinatorState(state_path)
        state.state["today_trades"] = 5

        can_trade, reason = state.can_trade()
        assert not can_trade
        assert "上限" in reason

    def test_can_trade_pause(self, tmp_path):
        """Should respect pause state"""
        state_path = tmp_path / "test_state.json"
        state = CoordinatorState(state_path)
        state.state["is_paused"] = True
        state.state["pause_until"] = (datetime.now() + timedelta(hours=1)).isoformat()

        can_trade, reason = state.can_trade()
        assert not can_trade
        assert "暂停" in reason

    def test_record_trade_win(self, tmp_path):
        """Recording win should reset consecutive losses"""
        state_path = tmp_path / "test_state.json"
        state = CoordinatorState(state_path)
        state.state["consecutive_losses"] = 2

        state.record_trade({"pnl": 100})

        assert state.state["consecutive_losses"] == 0
        assert state.state["today_trades"] == 1

    def test_record_trade_loss(self, tmp_path):
        """Recording loss should increment consecutive losses"""
        state_path = tmp_path / "test_state.json"
        state = CoordinatorState(state_path)

        state.record_trade({"pnl": -100})

        assert state.state["consecutive_losses"] == 1

    def test_record_trade_double_loss_pause(self, tmp_path):
        """Double loss should trigger pause"""
        state_path = tmp_path / "test_state.json"
        state = CoordinatorState(state_path)

        state.record_trade({"pnl": -100})
        state.record_trade({"pnl": -100})

        assert state.state["is_paused"]
        assert state.state["pause_until"] is not None

    def test_scan_symbol_returns_result(self):
        """Scanning symbol should return result dict"""
        coordinator = MiracleCoordinator(symbols=["BTC"])
        result = coordinator.scan("BTC")

        assert "symbol" in result
        assert result["symbol"] == "BTC"
        assert "trades_executed" in result
        assert "signals" in result

    def test_scan_all_symbols(self):
        """Scanning all symbols should return results for each"""
        coordinator = MiracleCoordinator(symbols=["BTC", "ETH"])
        results = coordinator.scan_all()

        assert len(results) == 2
        symbols = [r["symbol"] for r in results]
        assert "BTC" in symbols
        assert "ETH" in symbols

    def test_demo_signal(self):
        """Demo signal should have required fields"""
        coordinator = MiracleCoordinator()
        signal = coordinator._demo_signal("BTC")

        required = ["symbol", "direction", "entry_price", "stop_loss",
                    "take_profit", "confidence", "trend_strength"]
        for field in required:
            assert field in signal

    def test_demo_risk_approval(self):
        """Demo risk approval should work"""
        coordinator = MiracleCoordinator()
        signal = coordinator._demo_signal("BTC")
        approval = coordinator._demo_risk_approval(signal)

        assert "approved" in approval
        assert "modified_signal" in approval


# =============================================================================
# Full Trading Flow Tests
# =============================================================================

class TestTradingFlow:
    """Full trading flow integration tests"""

    def test_signal_to_execution_flow(self):
        """Test complete flow from signal to execution decision"""
        coordinator = MiracleCoordinator()

        # Get demo signal
        signal = coordinator._demo_signal("BTC")
        assert signal is not None

        # Get risk approval
        approval = coordinator._demo_risk_approval(signal)
        assert approval is not None

        # Check if state allows trading
        can_trade, reason = coordinator.state.can_trade()
        assert isinstance(can_trade, bool)
        assert isinstance(reason, str)

    def test_config_loading(self):
        """Config should be loaded correctly"""
        coordinator = MiracleCoordinator()
        assert coordinator.config is not None
        assert "factors" in coordinator.config
        assert "risk" in coordinator.config
        assert "position" in coordinator.config


# =============================================================================
# State Persistence Tests
# =============================================================================

class TestStatePersistence:
    """State persistence tests"""

    def test_state_save_load(self, tmp_path):
        """State should save and load correctly"""
        state_path = tmp_path / "state.json"

        # Create and modify state
        state1 = CoordinatorState(state_path)
        state1.state["today_trades"] = 3
        state1.state["total_trades"] = 10
        state1.save()

        # Load in new instance
        state2 = CoordinatorState(state_path)
        assert state2.state["today_trades"] == 3
        assert state2.state["total_trades"] == 10

    def test_state_creates_directory(self, tmp_path):
        """State should create parent directory if needed"""
        state_path = tmp_path / "subdir" / "state.json"
        state = CoordinatorState(state_path)
        state.save()

        assert state_path.exists()


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
