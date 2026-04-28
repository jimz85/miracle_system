"""
Position Monitor - 持仓监控器
==============================

从 agents/agent_executor.py 提取

包含:
- PositionMonitor: 持仓监控、止损检查和自动平仓

用法:
    from core.position_monitor import PositionMonitor
    from agents.agent_executor import PositionMonitor  # 向后兼容
"""

from datetime import datetime
from typing import Dict, Optional, Tuple

from core.executor_config import ExecutorConfig

# 向前引用 ExchangeClient（运行时才检查）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.exchange_client import ExchangeClient


class PositionMonitor:
    """
    持仓监控器
    负责持仓监控、止损检查和自动平仓
    """

    def __init__(self, exchange_client: 'ExchangeClient', config: ExecutorConfig):
        self.client = exchange_client
        self.config = config
        self.positions: Dict[str, Dict] = {}

    def monitor(self, trade: Dict, current_price: float, atr: float = None) -> Tuple[bool, str]:
        """
        监控持仓

        Returns:
            (should_exit, reason)
            reason: "none" | "sl" | "tp" | "structure" | "atr"
        """
        symbol = trade.get("symbol")
        side = trade.get("side")
        entry_price = trade.get("entry_price")
        stop_loss = trade.get("stop_loss")
        take_profit = trade.get("take_profit")
        entry_time_str = trade.get("timestamp")

        # 价格止损
        if side == "long":
            if current_price <= stop_loss:
                return True, "sl"
            if current_price >= take_profit:
                return True, "tp"
        else:  # short
            if current_price >= stop_loss:
                return True, "sl"
            if current_price <= take_profit:
                return True, "tp"

        # 动态结构止损: 当持仓超过4h后，使用ATR动态跟踪止损替代时间止损
        # 结构止损原则：不在突破点反向，而是在趋势破坏点退出
        if entry_time_str:
            entry_time = datetime.fromisoformat(entry_time_str)
            hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
            if hold_hours >= 4:  # 前4小时用固定SL
                # ATR动态止损：使用1.5倍ATR作为结构止损距离
                if atr is None:
                    # 如果没有ATR，使用价格波动率估算
                    price_range = abs(take_profit - entry_price) if take_profit and entry_price else entry_price * 0.02
                    atr_stop_distance = price_range * 0.5  # 50%价格范围作为动态止损
                else:
                    atr_stop_distance = atr * 1.5
                
                if side == "long":
                    # 多头：结构止损 = 最高价 - 1.5*ATR（追踪高点）
                    struct_stop = current_price - atr_stop_distance
                    if struct_stop > stop_loss:  # 只跟踪不回头
                        if current_price <= struct_stop:
                            return True, "structure"
                else:  # short
                    struct_stop = current_price + atr_stop_distance
                    if struct_stop < stop_loss:  # 只跟踪不回头
                        if current_price >= struct_stop:
                            return True, "structure"

        return False, "none"

    def check_stop_loss(self, trade: Dict, current_price: float) -> bool:
        """检查是否触发止损"""
        side = trade.get("side")
        stop_loss = trade.get("stop_loss", 0)

        if side == "long" and current_price <= stop_loss:
            return True
        if side == "short" and current_price >= stop_loss:
            return True
        return False

    def check_take_profit(self, trade: Dict, current_price: float) -> bool:
        """检查是否触发止盈"""
        side = trade.get("side")
        take_profit = trade.get("take_profit", 0)

        if side == "long" and current_price >= take_profit:
            return True
        if side == "short" and current_price <= take_profit:
            return True
        return False

    def calculate_pnl(self, trade: Dict, current_price: float) -> float:
        """计算盈亏"""
        side = trade.get("side")
        entry_price = trade.get("entry_price")
        position_size = trade.get("position_size", 0)
        leverage = trade.get("leverage", 1)

        if side == "long":
            pnl = (current_price - entry_price) * position_size * leverage
        else:
            pnl = (entry_price - current_price) * position_size * leverage

        return pnl

    def check_moving_stop(self, trade: Dict, current_price: float,
                         atr: float = None) -> Optional[float]:
        """
        检查移动止损

        Returns:
            新止损价格，如果不需要移动则返回None
        """
        side = trade.get("side")
        entry_price = trade.get("entry_price")
        stop_loss = trade.get("stop_loss", 0)

        risk = abs(entry_price - stop_loss)
        if risk == 0:
            return None

        # 如果盈利超过2*R，移动止损到入场价
        breakeven_threshold = entry_price + risk * 2

        if side == "long" and current_price > breakeven_threshold:
            return entry_price * 0.998  # 微利保护
        if side == "short" and current_price < breakeven_threshold:
            return entry_price * 1.002

        return None

    def update_position(self, symbol: str, position_data: Dict):
        """更新持仓数据"""
        self.positions[symbol] = position_data

    def remove_position(self, symbol: str):
        """移除持仓数据"""
        if symbol in self.positions:
            del self.positions[symbol]

    def get_position(self, symbol: str) -> Optional[Dict]:
        """获取持仓数据"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """获取所有持仓"""
        return self.positions.copy()
