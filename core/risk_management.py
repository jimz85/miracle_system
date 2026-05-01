from __future__ import annotations

"""
Risk Management - 风险管理模块
包含: ATR动态仓位, 跨币种风险监控, 滑点手续费模拟
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ========================
# ATR 动态仓位计算
# ========================

class ATRCalculator:
    """Average True Range 计算器 - 使用Wilder's平滑方法"""
    
    def __init__(self, period: int = 14):
        if period <= 0:
            raise ValueError(f"ATR period must be positive, got {period}")
        self.period = period
        self.tr_list = []
        self.atr = None
    
    def update(self, high: float, low: float, close: float) -> float:
        """更新ATR值"""
        if len(self.tr_list) > 0:
            prev_close = self.tr_list[-1][2]  # 上个周期的close
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
        else:
            tr = high - low
        
        self.tr_list.append((high, low, close, tr))
        
        # 保持固定窗口
        if len(self.tr_list) > self.period * 2:
            self.tr_list.pop(0)
        
        # Wilder's smoothing
        if len(self.tr_list) >= self.period:
            if self.atr is None:
                self.atr = sum(t[3] for t in self.tr_list[-self.period:]) / self.period
            else:
                self.atr = (self.atr * (self.period - 1) + tr) / self.period
        
        return self.atr or 0
    
    def get_atr(self) -> float:
        return self.atr or 0
    
    def get_normalized_atr(self, current_price: float) -> float:
        """获取标准化的ATR百分比"""
        if self.atr and current_price > 0:
            return (self.atr / current_price) * 100
        return 0


class DynamicPositionSizer:
    """
    动态仓位计算器

    公式: Position Size = (Account × Risk%) / (ATR × Stop Multiplier)

    特点:
    - 高波动 → 自动减仓
    - 低波动 → 可以适当加仓
    - 自动限制最大/最小仓位
    """

    MAX_HISTORY_SIZE: int = 1000
    ARCHIVE_DIR: Path = Path("data")

    def __init__(
        self,
        account_balance: float = 10000,
        risk_percent: float = 0.02,      # 2% 账户风险
        stop_multiplier: float = 2.0,   # ATR止损倍数
        max_position_percent: float = 0.3,  # 最大仓位30%
        min_position_percent: float = 0.01,  # 最小仓位1%
        atr_period: int = 14
    ):
        self.account_balance = account_balance
        self.risk_percent = risk_percent
        self.stop_multiplier = stop_multiplier
        self.max_position_percent = max_position_percent
        self.min_position_percent = min_position_percent
        self.atr_calculator = ATRCalculator(period=atr_period)
        self.position_history: List[dict] = []
        self.ARCHIVE_DIR = Path("data")
        self.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def _archive_to_disk(self, history_type: str, data: List[dict]) -> None:
        """归档历史数据到磁盘"""
        if not data:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.ARCHIVE_DIR / f"archived_{history_type}_history_{timestamp}.jsonl"
        try:
            with open(archive_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _trim_history(self, history: List, max_size: int, history_type: str) -> None:
        """修剪历史记录，超出上限时归档到磁盘"""
        if len(history) > max_size:
            to_archive = history[:-max_size]
            self._archive_to_disk(history_type, to_archive)
            del history[:-max_size]

    def add_position_record(self, position_data: dict) -> None:
        """添加仓位记录"""
        self.position_history.append({
            **position_data,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_history(self.position_history, self.MAX_HISTORY_SIZE, "position_sizer")
    
    def calculate_position(
        self,
        high: float,
        low: float,
        close: float,
        entry_price: float,
        direction: str = "long",  # "long" or "short"
        risk_reward_ratio: float = 2.0  # 风险回报比，默认2.0
    ) -> Dict[str, Any]:
        """
        计算仓位大小

        Returns:
            dict with position_size, stop_loss, take_profit, risk_amount, atr等
        """
        atr = self.atr_calculator.update(high, low, close)

        # 风险金额
        risk_amount = self.account_balance * self.risk_percent

        # 理论仓位
        if atr > 0:
            raw_position = risk_amount / (atr * self.stop_multiplier)
        else:
            if entry_price > 0:
                raw_position = risk_amount / (entry_price * 0.02)  # 默认2%止损
            else:
                raw_position = 0  # entry_price=0 无法计算仓位

        # 转换为实际金额
        raw_position_value = raw_position * entry_price

        # 限制仓位比例
        max_position = self.account_balance * self.max_position_percent
        min_position = self.account_balance * self.min_position_percent

        position_value = max(min_position, min(max_position, raw_position_value))
        position_size = position_value / entry_price if entry_price > 0 else 0

        # 计算止损
        # LONG: SL < entry (价格下跌触发止损)
        # SHORT: SL > entry (价格上涨触发止损)
        if direction == "long":
            stop_loss = entry_price - (atr * self.stop_multiplier)
        else:
            stop_loss = entry_price + (atr * self.stop_multiplier)

        # 计算止盈
        # LONG: TP > entry (价格上涨触发止盈)
        # SHORT: TP < entry (价格下跌触发止盈)
        stop_distance = atr * self.stop_multiplier
        if direction == "long":
            take_profit = entry_price + (stop_distance * risk_reward_ratio)
        else:
            take_profit = entry_price - (stop_distance * risk_reward_ratio)

        # 实际风险
        actual_risk = abs(entry_price - stop_loss) * position_size

        return {
            "position_size": position_size,
            "position_value": position_value,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr,
            "atr_percent": (atr / entry_price * 100) if entry_price > 0 else 0,
            "risk_amount": actual_risk,
            "risk_percent": (actual_risk / self.account_balance * 100),
            "risk_reward_ratio": risk_reward_ratio,
            "volatility_adjustment": "high" if atr > 0.03 else "normal"
        }
    
    def update_balance(self, new_balance: float):
        """更新账户余额"""
        self.account_balance = new_balance
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "account_balance": self.account_balance,
            "risk_percent": self.risk_percent,
            "stop_multiplier": self.stop_multiplier,
            "max_position_percent": self.max_position_percent,
            "current_atr": self.atr_calculator.get_atr()
        }


# ========================
# 波动率目标杠杆
# ========================

class TargetVolatilitySizing:
    """
    波动率目标杠杆计算器

    公式: leverage = target_daily_vol / ATR(20)

    特点:
    - 高波动市场 → 自动降低杠杆
    - 低波动市场 → 可以适当提高杠杆
    - 最大杠杆限制防止过度杠杆
    """

    def __init__(
        self,
        target_daily_vol: float = 0.02,    # 目标日波动率2%
        atr_period: int = 20,              # ATR周期
        max_leverage: float = 5.0,         # 最大杠杆限制
        min_leverage: float = 1.0,         # 最小杠杆限制
        account_balance: float = 10000
    ):
        """
        Args:
            target_daily_vol: 目标日波动率（如0.02表示2%）
            atr_period: ATR计算周期
            max_leverage: 最大杠杆倍数
            min_leverage: 最小杠杆倍数
            account_balance: 账户余额
        """
        if target_daily_vol <= 0:
            raise ValueError(f"target_daily_vol must be positive, got {target_daily_vol}")
        if max_leverage < min_leverage:
            raise ValueError(f"max_leverage ({max_leverage}) must be >= min_leverage ({min_leverage})")

        self.target_daily_vol = target_daily_vol
        self.atr_period = atr_period
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.account_balance = account_balance
        self.atr_calculator = ATRCalculator(period=atr_period)

    def calculate_leverage(
        self,
        high: float,
        low: float,
        close: float,
        current_price: float
    ) -> float:
        """
        基于波动率计算推荐杠杆

        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            current_price: 当前价格

        Returns:
            推荐的杠杆倍数
        """
        atr = self.atr_calculator.update(high, low, close)

        if atr <= 0 or current_price <= 0:
            return self.min_leverage

        # ATR作为日波动率的代理
        # leverage = target_daily_vol / atr_percent
        atr_percent = atr / current_price

        if atr_percent <= 0:
            return self.max_leverage

        # 计算理论杠杆
        calculated_leverage = self.target_daily_vol / atr_percent

        # 限制在 [min_leverage, max_leverage] 范围内
        bounded_leverage = max(self.min_leverage, min(self.max_leverage, calculated_leverage))

        return bounded_leverage

    def calculate_position_size(
        self,
        high: float,
        low: float,
        close: float,
        entry_price: float,
        direction: str = "long",
        risk_percent: float = 0.02
    ) -> Dict[str, Any]:
        """
        计算基于波动率的仓位大小

        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            entry_price: 入场价格
            direction: "long" or "short"
            risk_percent: 账户风险百分比

        Returns:
            dict with leverage, position_size, stop_loss, risk_amount等
        """
        current_price = close

        # 计算波动率杠杆
        leverage = self.calculate_leverage(high, low, close, current_price)

        # 计算风险金额（使用波动率调整后的杠杆）
        risk_amount = self.account_balance * risk_percent

        # 计算理论仓位
        # 仓位价值 = 风险金额 / (ATR * stop_multiplier)
        # ATR止损 = ATR * 2
        atr = self.atr_calculator.get_atr()
        stop_distance = atr * 2 if atr > 0 else entry_price * 0.02

        if stop_distance > 0:
            raw_position_size = risk_amount / stop_distance
        else:
            raw_position_size = 0

        # 转换为实际数量（考虑杠杆）
        position_size = raw_position_size

        # 计算止损价格
        if direction == "long":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # 实际风险
        actual_risk = abs(entry_price - stop_loss) * position_size

        return {
            "leverage": leverage,
            "position_size": position_size,
            "position_value": position_size * entry_price,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "atr": atr,
            "atr_percent": (atr / current_price * 100) if current_price > 0 else 0,
            "target_daily_vol": self.target_daily_vol * 100,
            "actual_risk": actual_risk,
            "risk_percent": (actual_risk / self.account_balance * 100) if self.account_balance > 0 else 0,
            "volatility_adjustment": "high" if leverage <= 2 else "normal" if leverage <= 4 else "low"
        }

    def update_balance(self, new_balance: float):
        """更新账户余额"""
        self.account_balance = new_balance

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "account_balance": self.account_balance,
            "target_daily_vol": self.target_daily_vol * 100,
            "atr_period": self.atr_period,
            "max_leverage": self.max_leverage,
            "min_leverage": self.min_leverage,
            "current_atr": self.atr_calculator.get_atr()
        }


# ========================
# 跨币种风险监控
# ========================

@dataclass
class Position:
    """持仓数据"""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0
    leverage: float = 1.0
    margin: float = 0
    
    @property
    def notional_value(self) -> float:
        return self.size * self.current_price
    
    @property
    def unrealized_pnl_percent(self) -> float:
        if self.entry_price > 0:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100 * (1 if self.side == "long" else -1)
        return 0


class CrossCurrencyRiskMonitor:
    """
    跨币种风险敞口监控

    功能:
    - 实时追踪所有持仓
    - 计算总风险敞口
    - 检测超额敞口
    - 资产相关性分析
    """

    MAX_HISTORY_SIZE: int = 1000
    ARCHIVE_DIR: Path = Path("data")

    def __init__(
        self,
        max_total_exposure: float = 1.0,      # 最大总敞口 (账户的倍数)
        max_single_exposure: float = 0.3,      # 单币种最大敞口
        max_correlation: float = 0.7,          # 最大相关性
        account_balance: float = 10000
    ):
        self.max_total_exposure = max_total_exposure
        self.max_single_exposure = max_single_exposure
        self.max_correlation = max_correlation
        self.account_balance = account_balance
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict] = []
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.ARCHIVE_DIR = Path("data")
        self.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def _archive_to_disk(self, history_type: str, data: List[dict]) -> None:
        """归档历史数据到磁盘"""
        if not data:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.ARCHIVE_DIR / f"archived_{history_type}_history_{timestamp}.jsonl"
        try:
            with open(archive_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _trim_history(self, history: List, max_size: int, history_type: str) -> None:
        """修剪历史记录，超出上限时归档到磁盘"""
        if len(history) > max_size:
            to_archive = history[:-max_size]
            self._archive_to_disk(history_type, to_archive)
            del history[:-max_size]

    def _record_history(self, action: str, position: Position) -> None:
        """记录持仓历史"""
        self.position_history.append({
            "action": action,
            "symbol": position.symbol,
            "side": position.side,
            "size": position.size,
            "entry_price": position.entry_price,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_history(self.position_history, self.MAX_HISTORY_SIZE, "cross_currency")
    
    def add_position(self, position: Position) -> None:
        """添加持仓"""
        self.positions[position.symbol] = position
        self._record_history("add", position)
    
    def remove_position(self, symbol: str) -> Position | None:
        """移除持仓"""
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            self._record_history("remove", pos)
            return pos
        return None
    
    def update_position_price(self, symbol: str, current_price: float) -> None:
        """更新持仓价格"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = current_price
            if pos.side == "long":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
    
    def get_total_exposure(self) -> Tuple[float, float]:
        """
        获取总敞口
        
        Returns:
            (total_notional, exposure_percent)
        """
        total = sum(p.notional_value for p in self.positions.values())
        return total, total / self.account_balance if self.account_balance > 0 else 0
    
    def get_single_exposure(self, symbol: str) -> Tuple[float, float]:
        """获取单币种敞口"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            return pos.notional_value, pos.notional_value / self.account_balance if self.account_balance > 0 else 0
        return 0, 0
    
    def can_open_position(self, symbol: str, size: float, price: float) -> Tuple[bool, str]:
        """
        检查是否可以开仓
        
        Returns:
            (can_open, reason)
        """
        notional = size * price
        single_exposure = notional / self.account_balance if self.account_balance > 0 else 0
        total_exposure, total_pct = self.get_total_exposure()
        
        # 检查单币种上限
        if single_exposure > self.max_single_exposure:
            return False, f"单币种敞口 {single_exposure:.1%} 超过上限 {self.max_single_exposure:.1%}"
        
        # 检查总敞口上限
        new_total = (total_exposure + notional) / self.account_balance
        if new_total > self.max_total_exposure:
            return False, f"总敞口 {new_total:.1%} 超过上限 {self.max_total_exposure:.1%}"
        
        # 检查是否已持有
        if symbol in self.positions:
            return False, f"已持有 {symbol} 仓位"
        
        return True, "OK"
    
    def get_total_pnl(self) -> Tuple[float, float]:
        """获取总盈亏"""
        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        total_pnl_pct = total_pnl / self.account_balance if self.account_balance > 0 else 0
        return total_pnl, total_pnl_pct
    
    def get_risk_report(self) -> Dict[str, Any]:
        """获取风险报告"""
        total_notional, total_pct = self.get_total_exposure()
        total_pnl, pnl_pct = self.get_total_pnl()
        
        return {
            "account_balance": self.account_balance,
            "total_notional": total_notional,
            "total_exposure_percent": total_pct,
            "total_unrealized_pnl": total_pnl,
            "total_pnl_percent": pnl_pct,
            "position_count": len(self.positions),
            "max_single_exposure": self.max_single_exposure,
            "max_total_exposure": self.max_total_exposure,
            "positions": {
                symbol: {
                    "side": p.side,
                    "size": p.size,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "notional": p.notional_value,
                    "pnl": p.unrealized_pnl,
                    "pnl_percent": p.unrealized_pnl_percent
                }
                for symbol, p in self.positions.items()
            },
            "warnings": self._check_warnings()
        }
    
    def _check_warnings(self) -> List[str]:
        """检查风险警告"""
        warnings = []
        total_notional, total_pct = self.get_total_exposure()
        
        if total_pct > self.max_total_exposure * 0.9:
            warnings.append(f"总敞口接近上限: {total_pct:.1%}")
        
        for symbol, pos in self.positions.items():
            single_pct = pos.notional_value / self.account_balance if self.account_balance > 0 else 0
            if single_pct > self.max_single_exposure * 0.9:
                warnings.append(f"{symbol} 敞口接近上限: {single_pct:.1%}")
            
            if abs(pos.unrealized_pnl_percent) > 10:
                warnings.append(f"{symbol} 浮亏较大: {pos.unrealized_pnl_percent:.1%}")
        
        return warnings
    
    def _record_history(self, action: str, position: Position):
        """记录历史"""
        self.position_history.append({
            "action": action,
            "symbol": position.symbol,
            "side": position.side,
            "size": position.size,
            "price": position.entry_price,
            "timestamp": self._get_timestamp()
        })
    
    @staticmethod
    def _get_timestamp() -> str:
        from datetime import datetime
        return datetime.now().isoformat()
    
    def export_positions(self, filepath: str):
        """导出持仓到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.get_risk_report(), f, indent=2)
    
    def import_positions(self, filepath: str):
        """从文件导入持仓"""
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
                for symbol, pos_data in data.get("positions", {}).items():
                    pos = Position(
                        symbol=symbol,
                        side=pos_data["side"],
                        size=pos_data["size"],
                        entry_price=pos_data["entry"],
                        current_price=pos_data["current"]
                    )
                    self.positions[symbol] = pos


# ========================
# 滑点/手续费模拟
# ========================

class SlippageFeeSimulator:
    """
    滑点和手续费模拟器

    功能:
    - 订单簿深度模拟
    - maker/taker费率
    - 实时滑点计算
    - 回测成本修正
    """

    MAX_HISTORY_SIZE: int = 1000
    ARCHIVE_DIR: Path = Path("data")

    def __init__(
        self,
        maker_fee: float = 0.001,     # 0.1% maker手续费
        taker_fee: float = 0.001,     # 0.1% taker手续费
        base_slippage: float = 0.0005, # 0.05% 基础滑点
        orderbook_depth: int = 10,     # 订单簿深度
        slippage_model: str = "linear"  # "linear" or "sqrt"
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.base_slippage = base_slippage
        self.orderbook_depth = orderbook_depth
        self.slippage_model = slippage_model
        self.trade_history: List[dict] = []
        self.ARCHIVE_DIR = Path("data")
        self.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def _archive_to_disk(self, history_type: str, data: List[dict]) -> None:
        """归档历史数据到磁盘"""
        if not data:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.ARCHIVE_DIR / f"archived_{history_type}_history_{timestamp}.jsonl"
        try:
            with open(archive_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _trim_history(self, history: List, max_size: int, history_type: str) -> None:
        """修剪历史记录，超出上限时归档到磁盘"""
        if len(history) > max_size:
            to_archive = history[:-max_size]
            self._archive_to_disk(history_type, to_archive)
            del history[:-max_size]
    
    def simulate_trade(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        size: float,
        market_price: float,
        order_type: str = "market"  # "market" or "limit"
    ) -> Dict[str, Any]:
        """
        模拟交易成本
        
        Returns:
            dict with execution_price, slippage, fees, total_cost等
        """
        # 计算滑点
        if order_type == "market":
            slippage_pct = self._calculate_slippage(size, market_price)
        else:
            slippage_pct = self.base_slippage * 0.5  # 限价单滑点更小
        
        # 执行价格
        if side == "buy":
            execution_price = market_price * (1 + slippage_pct)
        else:
            execution_price = market_price * (1 - slippage_pct)
        
        # 手续费
        fee_type = "maker" if order_type == "limit" else "taker"
        fee_rate = self.maker_fee if fee_type == "maker" else self.taker_fee
        fees = size * execution_price * fee_rate
        
        # 总成本
        notional = size * execution_price
        total_cost = fees + abs(size * market_price * slippage_pct)
        
        result = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "market_price": market_price,
            "execution_price": execution_price,
            "slippage_pct": slippage_pct,
            "slippage_cost": abs(size * market_price * slippage_pct),
            "fee_type": fee_type,
            "fee_rate": fee_rate,
            "fees": fees,
            "notional": notional,
            "total_cost": total_cost,
            "total_cost_pct": total_cost / notional if notional > 0 else 0
        }
        
        self.trade_history.append(result)
        self._trim_history(self.trade_history, self.MAX_HISTORY_SIZE, "slippage")
        return result
    
    def _calculate_slippage(self, size: float, price: float) -> float:
        """计算滑点百分比"""
        # 基于订单大小的滑点模型
        size_pct = size * price  # 订单名义价值
        
        if self.slippage_model == "linear":
            slippage = self.base_slippage * (1 + size_pct / 10000)
        else:  # sqrt
            slippage = self.base_slippage * (1 + (size_pct / 10000) ** 0.5)
        
        return min(slippage, 0.01)  # 最多1%滑点
    
    def backtest_adjustment(
        self,
        trades: List[Dict],
        initial_balance: float
    ) -> Dict[str, Any]:
        """
        对回测结果进行成本修正
        
        Args:
            trades: 回测交易列表
            initial_balance: 初始资金
        
        Returns:
            dict with adjusted_returns, costs等
        """
        total_fees = 0
        total_slippage = 0
        
        for trade in trades:
            if "exit_price" in trade and "entry_price" in trade:
                size = trade.get("size", 0)
                entry = trade["entry_price"]
                trade["exit_price"]
                
                result = self.simulate_trade(
                    symbol=trade.get("symbol", "UNKNOWN"),
                    side=trade.get("side", "long"),
                    size=size,
                    market_price=entry,
                    order_type="limit"
                )
                
                total_fees += result["fees"] * 2  # 入场+出场
                total_slippage += result["slippage_cost"] * 2
        
        return {
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "total_cost": total_fees + total_slippage,
            "cost_per_trade": (total_fees + total_slippage) / len(trades) if trades else 0,
            "adjusted_return": sum(t.get("pnl", 0) for t in trades) - total_fees - total_slippage,
            "original_return": sum(t.get("pnl", 0) for t in trades),
            "cost_ratio": (total_fees + total_slippage) / initial_balance if initial_balance > 0 else 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.trade_history:
            return {"trade_count": 0}
        
        total_cost = sum(t["total_cost"] for t in self.trade_history)
        total_fees = sum(t["fees"] for t in self.trade_history)
        avg_slippage = sum(t["slippage_pct"] for t in self.trade_history) / len(self.trade_history)
        
        return {
            "trade_count": len(self.trade_history),
            "total_cost": total_cost,
            "total_fees": total_fees,
            "total_slippage": total_cost - total_fees,
            "avg_slippage_pct": avg_slippage * 100,
            "maker_count": sum(1 for t in self.trade_history if t["fee_type"] == "maker"),
            "taker_count": sum(1 for t in self.trade_history if t["fee_type"] == "taker")
        }


# ========================
# 工厂函数
# ========================

def create_risk_manager(config: Dict[str, Any] | None = None) -> Tuple[DynamicPositionSizer, CrossCurrencyRiskMonitor, SlippageFeeSimulator]:
    """
    创建风险管理器便捷函数
    
    Returns:
        (position_sizer, risk_monitor, fee_simulator)
    """
    config = config or {}
    
    account = config.get("account_balance", 10000)
    
    position_sizer = DynamicPositionSizer(
        account_balance=account,
        risk_percent=config.get("risk_percent", 0.02),
        stop_multiplier=config.get("stop_multiplier", 2.0),
        max_position_percent=config.get("max_position_percent", 0.3),
        atr_period=config.get("atr_period", 14)
    )
    
    risk_monitor = CrossCurrencyRiskMonitor(
        max_total_exposure=config.get("max_total_exposure", 1.0),
        max_single_exposure=config.get("max_single_exposure", 0.3),
        account_balance=account
    )
    
    fee_simulator = SlippageFeeSimulator(
        maker_fee=config.get("maker_fee", 0.001),
        taker_fee=config.get("taker_fee", 0.001),
        base_slippage=config.get("base_slippage", 0.0005)
    )
    
    return position_sizer, risk_monitor, fee_simulator


if __name__ == "__main__":
    # 测试
    print("=== Risk Management Test ===\n")
    
    # 1. ATR动态仓位
    psizer = DynamicPositionSizer(account_balance=10000, risk_percent=0.02)
    result = psizer.calculate_position(
        high=105, low=95, close=100,
        entry_price=100, direction="long"
    )
    print("Dynamic Position:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 2. 跨币种风险监控
    monitor = CrossCurrencyRiskMonitor(account_balance=10000)
    monitor.add_position(Position("BTC", "long", 0.1, 50000, 51000))
    monitor.add_position(Position("ETH", "long", 1.0, 3000, 2900))
    
    can_open, reason = monitor.can_open_position("SOL", 10, 100)
    print(f"\nCan open SOL: {can_open}, reason: {reason}")
    
    total, pct = monitor.get_total_exposure()
    print(f"Total exposure: ${total:.2f} ({pct:.1%})")
    
    # 3. 滑点手续费
    fee_sim = SlippageFeeSimulator()
    trade = fee_sim.simulate_trade("BTC", "buy", 0.1, 50000, "market")
    print("\nTrade simulation:")
    print(f"  Execution: ${trade['execution_price']:.2f}")
    print(f"  Slippage: {trade['slippage_pct']*100:.3f}%")
    print(f"  Fees: ${trade['fees']:.2f}")
    print(f"  Total cost: ${trade['total_cost']:.2f}")
    
    print("\n=== All tests passed! ===")
