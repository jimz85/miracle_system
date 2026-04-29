from __future__ import annotations

"""
TradeLogger - 交易日志记录器
===========================

从 agents/agent_executor.py 提取到 core/ 模块

用法:
    from core.trade_logger import TradeLogger
    from agents.agent_executor import TradeLogger  # 向后兼容导入
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from core.executor_config import ExecutorConfig

logger = logging.getLogger(__name__)


class TradeLogger:
    """交易日志记录器"""

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.trades: List[Dict] = []
        self._load_trades()

    def _load_trades(self):
        """从文件加载交易记录"""
        try:
            filepath = f"{self.config.log_dir}/{self.config.trade_log_file}"
            with open(filepath) as f:
                self.trades = json.load(f)
        except FileNotFoundError:
            self.trades = []

    def _save_trades(self):
        """保存交易记录到文件 - 使用原子写入防止崩溃损坏"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        filepath = f"{self.config.log_dir}/{self.config.trade_log_file}"
        tmp_filepath = filepath + ".tmp"
        with open(tmp_filepath, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)
        os.replace(tmp_filepath, filepath)  # 原子替换

    def log_entry(self, trade_record: Dict) -> str:
        """
        记录入场
        返回: trade_id
        """
        trade_id = trade_record.get("trade_id") or str(uuid.uuid4())
        record = {
            "trade_id": trade_id,
            "signal_id": trade_record.get("signal_id", ""),
            "timestamp": datetime.now().isoformat(),
            "symbol": trade_record.get("symbol", ""),
            "exchange": trade_record.get("exchange", "okx"),
            "side": trade_record.get("side", ""),
            "entry_price": trade_record.get("entry_price", 0),
            "entry_slippage": trade_record.get("entry_slippage", 0),
            "exit_price": None,
            "exit_slippage": None,
            "leverage": trade_record.get("leverage", 1),
            "position_size": trade_record.get("position_size", 0),
            "stop_loss": trade_record.get("stop_loss", 0),
            "take_profit": trade_record.get("take_profit", 0),
            "actual_rr": None,
            "pnl": None,
            "exit_reason": None,
            "hold_hours": None,
            "market_regime": trade_record.get("market_regime", "unknown"),
            "factors": trade_record.get("factors", {}),
            "status": "open"
        }

        self.trades.append(record)
        self._save_trades()

        logger.info(f"入场记录: {trade_id} {record['symbol']} {record['side']} @ {record['entry_price']}")
        return trade_id

    def log_exit(self, trade_id: str, exit_reason: str, pnl: float, hold_hours: float):
        """记录出场"""
        for trade in reversed(self.trades):
            if trade["trade_id"] == trade_id:
                trade["exit_reason"] = exit_reason
                trade["pnl"] = pnl
                trade["hold_hours"] = hold_hours

                # 计算实际RR
                if trade["entry_price"] > 0 and trade["stop_loss"] > 0:
                    risk = abs(trade["entry_price"] - trade["stop_loss"])
                    if trade["side"] == "long":
                        reward = trade["entry_price"] - (trade.get("exit_price") or trade["entry_price"])
                    else:
                        reward = (trade.get("exit_price") or trade["entry_price"]) - trade["entry_price"]

                    trade["actual_rr"] = reward / risk if risk > 0 else 0

                # 获取出场价格
                if trade.get("_exit_price"):
                    trade["exit_price"] = trade["_exit_price"]
                if trade.get("_exit_slippage"):
                    trade["exit_slippage"] = trade["_exit_slippage"]

                trade["status"] = "closed"
                trade["exit_timestamp"] = datetime.now().isoformat()

                self._save_trades()

                actual_rr_str = f"{trade['actual_rr']:.2f}" if trade.get("actual_rr") is not None else "N/A"
                logger.info(
                    f"出场记录: {trade_id} {exit_reason} PNL={pnl:.2f} RR={actual_rr_str}"
                )
                break

    def log_signal_miss(self, signal: Dict, reason: str):
        """记录信号放弃"""
        miss_record = {
            "type": "signal_miss",
            "signal_id": signal.get("signal_id", ""),
            "timestamp": datetime.now().isoformat(),
            "symbol": signal.get("symbol", ""),
            "reason": reason,
            "signal_data": signal
        }

        self.trades.append(miss_record)
        self._save_trades()

        logger.warning(f"信号放弃 [{signal.get('symbol')}]: {reason}")

    def get_open_trades(self) -> List[Dict]:
        """获取所有未平仓交易"""
        return [t for t in self.trades if t.get("status") == "open"]

    def get_trade(self, trade_id: str) -> Dict | None:
        """获取指定交易"""
        for t in self.trades:
            if t["trade_id"] == trade_id:
                return t
        return None

    def get_recent_trades(self, count: int = 20) -> List[Dict]:
        """获取最近N笔交易"""
        closed = [t for t in self.trades if t.get("status") == "closed"]
        return sorted(closed, key=lambda x: x.get("timestamp", ""), reverse=True)[:count]
