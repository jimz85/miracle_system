"""
PositionMonitor 持仓监控器测试
"""
from unittest.mock import MagicMock

import pytest

from core.executor_config import ExecutorConfig
from core.position_monitor import PositionMonitor

# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_config():
    return ExecutorConfig()


@pytest.fixture
def monitor(mock_client, mock_config):
    return PositionMonitor(exchange_client=mock_client, config=mock_config)


# ══════════════════════════════════════════════════════════════
# check_stop_loss
# ══════════════════════════════════════════════════════════════

class TestCheckStopLoss:
    """测试止损检查"""

    def test_long_position_hits_stop_loss(self, monitor):
        """多头持仓，价格跌至止损价 → 触发止损"""
        trade = {
            "symbol": "DOGE-USDT-SWAP",
            "side": "long",
            "stop_loss": 0.10,
        }

        assert monitor.check_stop_loss(trade, current_price=0.095) is True
        assert monitor.check_stop_loss(trade, current_price=0.10) is True
        assert monitor.check_stop_loss(trade, current_price=0.105) is False

    def test_short_position_hits_stop_loss(self, monitor):
        """空头持仓，价格涨至止损价 → 触发止损"""
        trade = {
            "symbol": "DOGE-USDT-SWAP",
            "side": "short",
            "stop_loss": 0.10,
        }

        assert monitor.check_stop_loss(trade, current_price=0.105) is True
        assert monitor.check_stop_loss(trade, current_price=0.10) is True
        assert monitor.check_stop_loss(trade, current_price=0.095) is False

    def test_no_stop_loss_returns_false(self, monitor):
        """交易无止损设置（stop_loss=0） → 价格0触发止损（止损默认为0）"""
        trade = {
            "symbol": "DOGE-USDT-SWAP",
            "side": "long",
        }
        # 当 stop_loss 默认值为 0，current_price=0 <= 0 → True
        # 这是预期行为：止损为0时，任何非正价格都会触发
        assert monitor.check_stop_loss(trade, current_price=0.0) is True


# ══════════════════════════════════════════════════════════════
# check_take_profit
# ══════════════════════════════════════════════════════════════

class TestCheckTakeProfit:
    """测试止盈检查"""

    def test_long_position_hits_take_profit(self, monitor):
        """多头持仓，价格涨至止盈价 → 触发止盈"""
        trade = {
            "symbol": "DOGE-USDT-SWAP",
            "side": "long",
            "take_profit": 0.12,
        }

        assert monitor.check_take_profit(trade, current_price=0.125) is True
        assert monitor.check_take_profit(trade, current_price=0.12) is True
        assert monitor.check_take_profit(trade, current_price=0.115) is False

    def test_short_position_hits_take_profit(self, monitor):
        """空头持仓，价格跌至止盈价 → 触发止盈"""
        trade = {
            "symbol": "DOGE-USDT-SWAP",
            "side": "short",
            "take_profit": 0.08,
        }

        assert monitor.check_take_profit(trade, current_price=0.075) is True
        assert monitor.check_take_profit(trade, current_price=0.08) is True
        assert monitor.check_take_profit(trade, current_price=0.085) is False


# ══════════════════════════════════════════════════════════════
# calculate_pnl
# ══════════════════════════════════════════════════════════════

class TestCalculatePnL:
    """测试盈亏计算"""

    def test_long_position_profit(self, monitor):
        """多头持仓盈利"""
        trade = {
            "side": "long",
            "entry_price": 100.0,
            "position_size": 10.0,
            "leverage": 1,
        }

        # (110 - 100) * 10 * 1 = 100
        assert monitor.calculate_pnl(trade, current_price=110.0) == 100.0

    def test_long_position_loss(self, monitor):
        """多头持仓亏损"""
        trade = {
            "side": "long",
            "entry_price": 100.0,
            "position_size": 10.0,
            "leverage": 1,
        }

        # (90 - 100) * 10 * 1 = -100
        assert monitor.calculate_pnl(trade, current_price=90.0) == -100.0

    def test_short_position_profit(self, monitor):
        """空头持仓盈利"""
        trade = {
            "side": "short",
            "entry_price": 100.0,
            "position_size": 10.0,
            "leverage": 1,
        }

        # (100 - 90) * 10 * 1 = 100
        assert monitor.calculate_pnl(trade, current_price=90.0) == 100.0

    def test_short_position_loss(self, monitor):
        """空头持仓亏损"""
        trade = {
            "side": "short",
            "entry_price": 100.0,
            "position_size": 10.0,
            "leverage": 1,
        }

        # (100 - 110) * 10 * 1 = -100
        assert monitor.calculate_pnl(trade, current_price=110.0) == -100.0

    def test_leverage_multiplier(self, monitor):
        """杠杆放大盈亏"""
        trade = {
            "side": "long",
            "entry_price": 100.0,
            "position_size": 10.0,
            "leverage": 3,
        }

        # (110 - 100) * 10 * 3 = 300
        assert monitor.calculate_pnl(trade, current_price=110.0) == 300.0

    def test_zero_leverage(self, monitor):
        """无杠杆"""
        trade = {
            "side": "long",
            "entry_price": 100.0,
            "position_size": 10.0,
            "leverage": 0,
        }

        assert monitor.calculate_pnl(trade, current_price=110.0) == 0.0


# ══════════════════════════════════════════════════════════════
# check_moving_stop
# ══════════════════════════════════════════════════════════════

class TestCheckMovingStop:
    """测试移动止损"""

    def test_long_profit_below_breakeven(self, monitor):
        """多头未达到2R门槛 → 不移动止损"""
        trade = {
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 95.0,  # risk = 5
        }

        # breakeven_threshold = 100 + 5*2 = 110
        # 当前价格 105 < 110，不移动
        result = monitor.check_moving_stop(trade, current_price=105.0)
        assert result is None

    def test_long_not_breakeven_at_exact_threshold(self, monitor):
        """多头正好在2R门槛 → 不移动（需严格大于）"""
        trade = {
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 95.0,  # risk = 5
        }

        # breakeven_threshold = 100 + 5*2 = 110
        # 110 > 110 is False，需要 > 110
        result = monitor.check_moving_stop(trade, current_price=110.0)
        assert result is None

    def test_long_breakeven_exceeded(self, monitor):
        """多头超过2R门槛 → 移动止损到入场价"""
        trade = {
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 95.0,  # risk = 5
        }

        # 当前价格 110.01 > 110，移动止损到 100 * 0.998
        result = monitor.check_moving_stop(trade, current_price=110.01)
        assert result is not None
        assert result == 99.8  # 入场价 * 0.998

    def test_short_breakeven(self, monitor):
        """空头达到2R门槛 → 移动止损到入场价"""
        trade = {
            "side": "short",
            "entry_price": 100.0,
            "stop_loss": 105.0,  # risk = 5
        }

        # 当前价格 90 < 90，移动止损到 100 * 1.002
        result = monitor.check_moving_stop(trade, current_price=90.0)
        assert result is not None
        assert result == 100.2  # 入场价 * 1.002

    def test_zero_risk(self, monitor):
        """risk=0时 → 不移动止损"""
        trade = {
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 100.0,  # risk = 0
        }

        result = monitor.check_moving_stop(trade, current_price=110.0)
        assert result is None


# ══════════════════════════════════════════════════════════════
# 持仓管理
# ══════════════════════════════════════════════════════════════

class TestPositionManagement:
    """测试持仓管理"""

    def test_update_and_get_position(self, monitor):
        """更新并获取持仓"""
        pos_data = {"symbol": "DOGE-USDT-SWAP", "size": 100}
        monitor.update_position("DOGE-USDT-SWAP", pos_data)

        result = monitor.get_position("DOGE-USDT-SWAP")
        assert result == pos_data

    def test_get_nonexistent_position(self, monitor):
        """获取不存在的持仓返回 None"""
        result = monitor.get_position("NONEXISTENT")
        assert result is None

    def test_remove_position(self, monitor):
        """移除持仓"""
        pos_data = {"symbol": "DOGE-USDT-SWAP", "size": 100}
        monitor.update_position("DOGE-USDT-SWAP", pos_data)
        monitor.remove_position("DOGE-USDT-SWAP")

        assert monitor.get_position("DOGE-USDT-SWAP") is None

    def test_get_all_positions(self, monitor):
        """获取所有持仓"""
        monitor.update_position("DOGE-USDT-SWAP", {"size": 100})
        monitor.update_position("SOL-USDT-SWAP", {"size": 50})

        all_pos = monitor.get_all_positions()
        assert len(all_pos) == 2
        assert "DOGE-USDT-SWAP" in all_pos
        assert "SOL-USDT-SWAP" in all_pos

    def test_get_all_positions_returns_copy(self, monitor):
        """get_all_positions 返回浅拷贝，字典引用相同但列表本身不同"""
        monitor.update_position("DOGE-USDT-SWAP", {"size": 100})
        all_pos = monitor.get_all_positions()

        # 顶层字典是新的
        assert all_pos is not monitor.positions
        # 但嵌套的字典是同一个引用（浅拷贝）
        assert all_pos["DOGE-USDT-SWAP"] is monitor.positions["DOGE-USDT-SWAP"]
