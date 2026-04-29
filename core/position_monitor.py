from __future__ import annotations

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

# 向前引用 ExchangeClient（运行时才检查）
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from core.executor_config import ExecutorConfig

if TYPE_CHECKING:
    from core.exchange_client import ExchangeClient


class PositionMonitor:
    """
    持仓监控器
    负责持仓监控、止损检查和自动平仓
    """

    def __init__(self, exchange_client: ExchangeClient, config: ExecutorConfig):
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
        trade.get("symbol")
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
                    # 跟踪运行最高价，不回头
                    highest_price = trade.get('highest_price', entry_price)
                    if current_price > highest_price:
                        trade['highest_price'] = current_price
                        highest_price = current_price
                    struct_stop = highest_price - atr_stop_distance
                    if struct_stop > stop_loss:  # 只跟踪不回头
                        if current_price <= struct_stop:
                            return True, "structure"
                else:  # short
                    # 空头：结构止损 = 最低价 + 1.5*ATR（追踪低点）
                    # 跟踪运行最低价，不回头
                    lowest_price = trade.get('lowest_price', entry_price)
                    if current_price < lowest_price:
                        trade['lowest_price'] = current_price
                        lowest_price = current_price
                    struct_stop = lowest_price + atr_stop_distance
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

        if not entry_price or not side:
            return 0.0

        if side == "long":
            pnl = (current_price - entry_price) * position_size * leverage
        else:
            pnl = (entry_price - current_price) * position_size * leverage

        return pnl

    def check_moving_stop(self, trade: Dict, current_price: float,
                         atr: float = None) -> float | None:
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

    def get_position(self, symbol: str) -> Dict | None:
        """获取持仓数据"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """获取所有持仓"""
        return self.positions.copy()

    # ─────────────────────────────────────────────────────────────
    # P0-FIX: 强平价与保证金率监控
    # 高杠杆下价格未触及SL但已触及强平价，导致穿仓
    # ─────────────────────────────────────────────────────────────

    def check_liquidation_risk(
        self,
        symbol: str,
        trade: Dict,
        margin_ratio_threshold: float = 0.30,
    ) -> Tuple[bool, str]:
        """
        检查保证金率与预估强平价

        P0-FIX: 每次监控循环必须拉取liqPx和marginRatio，
        当marginRatio < 阈值时强制减仓或告警。

        Args:
            symbol: 币种，如 BTC-USDT-SWAP
            trade: 交易记录字典，需包含 entry_price, position_size, leverage, side
            margin_ratio_threshold: 保证金率阈值，低于此值触发警告（默认30%）

        Returns:
            (is_dangerous, reason): 是否危险，危险原因
        """
        try:
            # 从交易所获取实时保证金数据
            positions = self.client.get_positions()
            for pos in positions:
                if pos.inst_id != symbol:
                    continue

                liq_price = getattr(pos, 'liq_price', 0) or 0
                margin_ratio = getattr(pos, 'margin_ratio', 0) or 0
                pos_side = getattr(pos, 'direction', 'long') or 'long'
                entry_price = trade.get("entry_price", 0)
                current_price = trade.get("current_price", 0)

                if liq_price == 0 or margin_ratio == 0:
                    return False, "none"  # 数据不可用，跳过

                # 检查保证金率是否过低
                if margin_ratio < margin_ratio_threshold:
                    reason = (
                        f"MARGIN_CRITICAL: {symbol} "
                        f"margin_ratio={margin_ratio:.2%} < {margin_ratio_threshold:.0%} "
                        f"(liq_price={liq_price}, entry={entry_price})"
                    )
                    return True, reason

                # 检查当前价格是否接近强平价（10%以内）
                if liq_price > 0 and current_price > 0:
                    distance_to_liq = abs(current_price - liq_price) / current_price
                    if distance_to_liq < 0.10:  # 10%以内
                        reason = (
                            f"MARGIN_WARNING: {symbol} "
                            f"距强平价{distance_to_liq:.1%} "
                            f"(liq={liq_price}, current={current_price})"
                        )
                        return True, reason

                return False, "none"

            # 持仓不在交易所（可能是phantom）
            return False, "position_not_on_exchange"

        except ConnectionError:
            # P0-FIX: get_balance现在会抛出异常而不是返回模拟数据
            # 上游应该感知到交易所断连
            return False, "exchange_connection_failed"
        except Exception as e:
            logging.getLogger(__name__).warning(f"检查强平风险失败: {e}")
            return False, f"check_failed: {e}"
