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

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0,
        leverage: int = 1,
        tp1_percent: float = 0.50,
        tp2_percent: float = 0.25,
        trailing_percent: float = 0.25,
        trailing_callback_rate: float = 0.01
    ) -> Dict | None:
        """
        分批止盈Bracket订单

        策略:
        - 50% 仓位 @ 1R 止盈
        - 25% 仓位 @ 2R 止盈
        - 25% 仓位 追踪止损

        Args:
            symbol: 币种符号
            side: "buy"(做多) / "sell"(做空)
            size: 合约总数量
            entry_price: 入场价格
            stop_loss: 止损价格
            risk_reward_ratio: 风险回报比（默认2.0）
            leverage: 杠杆倍数
            tp1_percent: 第一止盈比例（默认50%）
            tp2_percent: 第二止盈比例（默认25%）
            trailing_percent: 追踪止损比例（默认25%）
            trailing_callback_rate: 追踪止损回调率

        Returns:
            dict with entry_order, tp1_order, tp2_order, trailing_order
        """
        # 验证比例
        total_exit = tp1_percent + tp2_percent + trailing_percent
        if abs(total_exit - 1.0) > 0.001:
            raise ValueError(
                f"TP1({tp1_percent}) + TP2({tp2_percent}) + trailing({trailing_percent}) "
                f"= {total_exit}, must equal 1.0"
            )

        # 计算止盈价格
        # LONG: TP > entry (价格上涨触发止盈)
        # SHORT: TP < entry (价格下跌触发止盈)
        stop_distance = abs(entry_price - stop_loss)
        tp1_price = entry_price + stop_distance * risk_reward_ratio if side.upper() == "BUY" else entry_price - stop_distance * risk_reward_ratio
        tp2_price = entry_price + stop_distance * 2 * risk_reward_ratio if side.upper() == "BUY" else entry_price - stop_distance * 2 * risk_reward_ratio

        # 计算各部分数量
        tp1_size = size * tp1_percent
        tp2_size = size * tp2_percent
        trailing_size = size * trailing_percent

        results = {
            "symbol": symbol,
            "side": side,
            "total_size": size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_reward_ratio": risk_reward_ratio,
            "leverage": leverage,
            "tp1": {"percent": tp1_percent, "price": tp1_price, "size": tp1_size},
            "tp2": {"percent": tp2_percent, "price": tp2_price, "size": tp2_size},
            "trailing": {"percent": trailing_percent, "callback_rate": trailing_callback_rate, "size": trailing_size},
        }

        # 1. 市价入场
        entry_result = self.create_market_order(symbol, side, size, leverage)
        if entry_result is None:
            results["entry_order"] = None
            results["status"] = "entry_failed"
            return results

        results["entry_order"] = entry_result
        results["status"] = "open"

        # 2. 第一止盈 @ 1R (50%仓位)
        tp1_result = self.client.place_conditional_tp(
            symbol=symbol,
            side=side,
            size=tp1_size,
            tp_price=tp1_price,
            leverage=leverage
        )
        results["tp1_order"] = tp1_result

        # 3. 第二止盈 @ 2R (25%仓位)
        tp2_result = self.client.place_conditional_tp(
            symbol=symbol,
            side=side,
            size=tp2_size,
            tp_price=tp2_price,
            leverage=leverage
        )
        results["tp2_order"] = tp2_result

        # 4. 追踪止损 (25%仓位)
        # OKX追踪止损需要设置slTriggerPx=0配合szAuto=true
        # 注意：追踪止损触发后是市价平仓
        trailing_result = self.client.place_trailing_stop(
            symbol=symbol,
            side=side,
            size=trailing_size,
            callback_rate=trailing_callback_rate,
            leverage=leverage
        )
        results["trailing_order"] = trailing_result

        # 追踪止损也设置一个条件SL保护（当追踪止损未触发时）
        # 只有当tp1和tp2都没触发时，sl才会在止损价格触发
        sl_result = self.client.place_conditional_sl(
            symbol=symbol,
            side=side,
            size=trailing_size,
            sl_price=stop_loss,
            leverage=leverage
        )
        results["sl_order"] = sl_result

        return results
