from __future__ import annotations

"""
Order Manager - 订单管理器
===========================

从 agents/agent_executor.py 提取

包含:
- OrderManager: 订单生命周期管理

用法:
    from core.order_manager import OrderManager
    from agents.agent_executor import OrderManager  # 向后兼容
"""

# 向前引用 ExchangeClient（运行时才检查）
from typing import TYPE_CHECKING, Dict, Optional

from core.executor_config import ExecutorConfig

if TYPE_CHECKING:
    from core.exchange_client import ExchangeClient


class OrderManager:
    """
    订单管理器
    负责订单生命周期管理
    """

    def __init__(self, exchange_client: ExchangeClient, config: ExecutorConfig):
        self.client = exchange_client
        self.config = config
        self.pending_orders: Dict[str, Dict] = {}

    def create_market_order(self, symbol: str, side: str, size: float,
                           leverage: int = 1) -> Dict | None:
        """创建市价单"""
        return self.client.place_order(
            symbol=symbol,
            side=side,
            order_type="market",
            price=None,
            size=size,
            leverage=leverage
        )

    def create_limit_order(self, symbol: str, side: str, price: float,
                          size: float, leverage: int = 1) -> Dict | None:
        """创建限价单"""
        return self.client.place_order(
            symbol=symbol,
            side=side,
            order_type="limit",
            price=price,
            size=size,
            leverage=leverage
        )

    def create_oco_order(self, symbol: str, side: str, size: float,
                        entry_price: float, sl_price: float, tp_price: float,
                        leverage: int = 1) -> Dict | None:
        """创建OCO订单（止损+止盈）"""
        return self.client.place_oco_order(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            leverage=leverage
        )

    def cancel_order(self, order_id: str, inst_id: str = None) -> bool:
        """取消订单"""
        if hasattr(self.client, 'cancel_algo_order') and inst_id:
            return self.client.cancel_algo_order(inst_id, order_id)
        return False

    def get_order_status(self, order_id: str) -> Dict | None:
        """获取订单状态"""
        for order in self.pending_orders.values():
            if order.get("order_id") == order_id:
                return order
        return None

    def add_pending_order(self, order_id: str, order_data: Dict):
        """添加待处理订单"""
        self.pending_orders[order_id] = order_data

    def remove_pending_order(self, order_id: str):
        """移除待处理订单"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]

    def get_pending_orders(self) -> Dict[str, Dict]:
        """获取所有待处理订单"""
        return self.pending_orders.copy()
