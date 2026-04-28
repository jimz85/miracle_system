"""
OrderManager 订单管理器测试
"""
from unittest.mock import MagicMock

import pytest

from core.executor_config import ExecutorConfig
from core.order_manager import OrderManager


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_config():
    return ExecutorConfig()


@pytest.fixture
def manager(mock_client, mock_config):
    return OrderManager(exchange_client=mock_client, config=mock_config)


class TestCreateMarketOrder:
    """测试市价单创建"""

    def test_create_market_order_calls_exchange(self, manager, mock_client):
        """市价单调用 exchange_client.place_order"""
        mock_client.place_order.return_value = {"order_id": "TEST-001"}
        result = manager.create_market_order(
            symbol="DOGE-USDT-SWAP",
            side="long",
            size=100.0,
            leverage=3,
        )
        mock_client.place_order.assert_called_once_with(
            symbol="DOGE-USDT-SWAP",
            side="long",
            order_type="market",
            price=None,
            size=100.0,
            leverage=3,
        )
        assert result == {"order_id": "TEST-001"}

    def test_create_market_order_default_leverage(self, manager, mock_client):
        """市价单默认杠杆 1"""
        mock_client.place_order.return_value = {"order_id": "TEST-002"}
        manager.create_market_order(symbol="BTC-USDT-SWAP", side="short", size=0.1)
        call = mock_client.place_order.call_args
        assert call.kwargs["leverage"] == 1

    def test_create_market_order_returns_none_on_error(self, manager, mock_client):
        """exchange_client 返回 None 时市价单返回 None"""
        mock_client.place_order.return_value = None
        result = manager.create_market_order(symbol="ETH-USDT-SWAP", side="long", size=1.0)
        assert result is None


class TestCreateLimitOrder:
    """测试限价单创建"""

    def test_create_limit_order_calls_exchange(self, manager, mock_client):
        """限价单正确传递 price 参数"""
        mock_client.place_order.return_value = {"order_id": "TEST-003"}
        result = manager.create_limit_order(
            symbol="SOL-USDT-SWAP",
            side="short",
            price=95.5,
            size=10.0,
            leverage=5,
        )
        mock_client.place_order.assert_called_once_with(
            symbol="SOL-USDT-SWAP",
            side="short",
            order_type="limit",
            price=95.5,
            size=10.0,
            leverage=5,
        )
        assert result == {"order_id": "TEST-003"}

    def test_limit_order_passes_leverage(self, manager, mock_client):
        """限价单传递杠杆参数"""
        mock_client.place_order.return_value = {"order_id": "TEST-004"}
        manager.create_limit_order(
            symbol="AVAX-USDT-SWAP",
            side="long",
            price=25.0,
            size=50.0,
            leverage=10,
        )
        assert mock_client.place_order.call_args.kwargs["leverage"] == 10


class TestPendingOrders:
    """测试 pending_orders 状态管理"""

    def test_pending_orders_initialized_empty(self, manager):
        """pending_orders 初始为空字典"""
        assert manager.pending_orders == {}

    def test_pending_orders_mutable(self, manager):
        """pending_orders 可被外部修改（不阻止，但跟踪状态）"""
        manager.pending_orders["TEST-001"] = {"symbol": "DOGE-USDT-SWAP", "side": "long"}
        assert "TEST-001" in manager.pending_orders
