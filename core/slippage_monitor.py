from __future__ import annotations

"""
SlippageMonitor - 滑点监控
==========================

从 agents/agent_executor.py 提取到 core/ 模块

用法:
    from core.slippage_monitor import SlippageMonitor
    from agents.agent_executor import SlippageMonitor  # 向后兼容导入
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from core.executor_config import ExecutorConfig

logger = logging.getLogger(__name__)


class SlippageMonitor:
    """滑点监控"""

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.slippage_records: List[Dict] = []
        self._load_history()

    def _load_history(self):
        """从文件加载历史滑点记录"""
        try:
            filepath = f"{self.config.log_dir}/{self.config.slippage_log_file}"
            with open(filepath) as f:
                self.slippage_records = json.load(f)
        except FileNotFoundError:
            self.slippage_records = []

    def _save_history(self):
        """保存滑点记录到文件"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        filepath = f"{self.config.log_dir}/{self.config.slippage_log_file}"
        with open(filepath, "w") as f:
            json.dump(self.slippage_records[-1000:], f, indent=2)  # 保留最近1000条

    def record_execution(self, signal: Dict, planned_price: float, actual_price: float) -> Dict:
        """
        记录滑点
        返回: {
            "slippage_pct": float,
            "slippage_abs": float,
            "warning": bool
        }
        """
        if planned_price == 0:
            return {"slippage_pct": 0, "slippage_abs": 0, "warning": False}

        slippage_abs = actual_price - planned_price

        # 考虑方向
        # 做多时：实际价格 > 计划价格 = 不利滑点（负）
        # 做空时：实际价格 < 计划价格 = 不利滑点（负）
        side = signal.get("side", "")
        if side == "buy":
            slippage_pct = (actual_price - planned_price) / planned_price
        elif side == "sell":
            slippage_pct = (planned_price - actual_price) / planned_price
        else:
            slippage_pct = (actual_price - planned_price) / planned_price

        record = {
            "trade_id": signal.get("trade_id", ""),
            "symbol": signal.get("symbol", ""),
            "planned_price": planned_price,
            "actual_price": actual_price,
            "slippage_pct": slippage_pct,
            "slippage_abs": slippage_abs,
            "timestamp": datetime.now().isoformat()
        }

        self.slippage_records.append(record)
        self._save_history()

        warning = abs(slippage_pct) > self.config.slippage_warning_threshold

        if warning:
            logger.warning(
                f"滑点警告 [{signal.get('symbol')}]: "
                f"计划={planned_price}, 实际={actual_price}, "
                f"滑点={slippage_pct*100:.2f}%"
            )

        return {
            "slippage_pct": slippage_pct,
            "slippage_abs": slippage_abs,
            "warning": warning
        }

    def get_avg_slippage(self, symbol: str, lookback_trades: int = 20) -> float:
        """获取平均滑点"""
        symbol_records = [
            r for r in self.slippage_records[-lookback_trades*10:]
            if r.get("symbol") == symbol
        ][-lookback_trades:]

        if not symbol_records:
            return 0.0

        return sum(r["slippage_pct"] for r in symbol_records) / len(symbol_records)
