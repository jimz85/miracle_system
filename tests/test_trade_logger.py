"""
TradeLogger 交易日志记录器测试
"""
import json
import tempfile
from pathlib import Path

import pytest

from core.executor_config import ExecutorConfig
from core.trade_logger import TradeLogger


@pytest.fixture
def mock_config():
    # 显式创建临时目录，不用 tmp_path fixture，避免 pytest-asyncio 生命周期冲突
    tmpdir = Path(tempfile.mkdtemp())
    config = ExecutorConfig()
    config.log_dir = str(tmpdir / "logs")
    config.trade_log_file = "trade_log.json"
    yield config
    # 清理
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def logger(mock_config):
    return TradeLogger(config=mock_config)


class TestLogEntry:
    """测试交易入场记录"""

    def test_log_entry_creates_trade_id(self, logger):
        """log_entry 返回 trade_id 字符串"""
        trade = {
            "symbol": "DOGE-USDT-SWAP",
            "side": "long",
            "entry_price": 0.10,
            "size": 100.0,
            "leverage": 3,
        }
        trade_id = logger.log_entry(trade)
        assert isinstance(trade_id, str)
        assert len(trade_id) > 0

    def test_log_entry_saves_to_file(self, logger, mock_config):
        """log_entry 正确写入 trade_log.json"""
        trade = {
            "symbol": "SOL-USDT-SWAP",
            "side": "short",
            "entry_price": 95.0,
            "size": 10.0,
            "leverage": 5,
        }
        trade_id = logger.log_entry(trade)
        log_file = Path(mock_config.log_dir) / mock_config.trade_log_file
        assert log_file.exists()
        with open(log_file) as f:
            data = json.load(f)
        # trade_logger 存储为 list，找到包含该 trade_id 的记录
        assert isinstance(data, list)
        assert any(r.get("trade_id") == trade_id for r in data)
        entry = next(r for r in data if r.get("trade_id") == trade_id)
        assert entry["symbol"] == "SOL-USDT-SWAP"
        assert entry["side"] == "short"

    def test_log_entry_adds_metadata(self, logger):
        """log_entry 自动填充 timestamp 和 status"""
        trade = {"symbol": "BTC-USDT-SWAP", "side": "long", "entry_price": 60000.0, "size": 0.01}
        trade_id = logger.log_entry(trade)
        stored = logger.get_trade(trade_id)
        assert "timestamp" in stored
        assert stored["status"] == "open"

    def test_log_entry_with_stop_loss_take_profit(self, logger):
        """log_entry 正确记录 SL/TP"""
        trade = {
            "symbol": "ETH-USDT-SWAP",
            "side": "long",
            "entry_price": 2500.0,
            "size": 1.0,
            "stop_loss": 2400.0,
            "take_profit": 2750.0,
        }
        trade_id = logger.log_entry(trade)
        stored = logger.get_trade(trade_id)
        assert stored["stop_loss"] == 2400.0
        assert stored["take_profit"] == 2750.0


class TestLogExit:
    """测试交易出场记录"""

    def test_log_exit_updates_status_and_pnl(self, logger):
        """log_exit 将持仓状态更新为 closed 并记录 pnl"""
        entry = {
            "symbol": "DOGE-USDT-SWAP",
            "side": "long",
            "entry_price": 0.10,
            "size": 100.0,
            "leverage": 3,
        }
        trade_id = logger.log_entry(entry)
        logger.log_exit(trade_id, exit_reason="take_profit", pnl=150.5, hold_hours=2.5)
        stored = logger.get_trade(trade_id)
        assert stored["status"] == "closed"
        assert stored["pnl"] == 150.5
        assert stored["exit_reason"] == "take_profit"
        assert stored["hold_hours"] == 2.5

    def test_log_exit_nonexistent_trade_no_crash(self, logger):
        """log_exit 处理不存在的 trade_id 不崩溃"""
        logger.log_exit("NONEXISTENT-ID", exit_reason="stop_loss", pnl=-50.0, hold_hours=1.0)


class TestGetTrades:
    """测试查询接口"""

    def test_get_open_trades_filters_closed(self, logger):
        """get_open_trades 只返回 status=open 的交易"""
        t1 = logger.log_entry({"symbol": "DOGE-USDT-SWAP", "side": "long", "entry_price": 0.10, "size": 100.0})
        t2 = logger.log_entry({"symbol": "SOL-USDT-SWAP", "side": "short", "entry_price": 95.0, "size": 10.0})
        logger.log_exit(t1, exit_reason="stop_loss", pnl=-10.0, hold_hours=0.5)
        open_trades = logger.get_open_trades()
        open_symbols = [t["symbol"] for t in open_trades]
        assert "DOGE-USDT-SWAP" not in open_symbols
        assert "SOL-USDT-SWAP" in open_symbols

    def test_get_trade_returns_none_for_missing(self, logger):
        """get_trade 对不存在的 trade_id 返回 None"""
        assert logger.get_trade("MISSING-ID") is None

    def test_get_recent_trades_respects_count(self, logger):
        """get_recent_trades 按时间倒序返回指定数量（只返回 closed 交易）"""
        # 创建 5 笔交易，关闭其中 3 笔
        t_ids = []
        for i in range(5):
            t_ids.append(
                logger.log_entry(
                    {"symbol": f"COIN-{i}-USDT-SWAP", "side": "long", "entry_price": 10.0 + i, "size": 1.0}
                )
            )
        # 关闭前 3 笔
        for i in range(3):
            logger.log_exit(t_ids[i], exit_reason="take_profit", pnl=10.0, hold_hours=1.0)
        recent = logger.get_recent_trades(count=3)
        assert len(recent) == 3
        # 最新的 3 笔（按 timestamp 倒序）
        assert all(t["status"] == "closed" for t in recent)


class TestLogSignalMiss:
    """测试信号错失记录"""

    def test_log_signal_miss_saves_reason(self, logger):
        """log_signal_miss 追加到 trade_log.json 的 trades 列表"""
        signal = {"symbol": "DOGE-USDT-SWAP", "reason": "rsi_overbought", "price": 0.12}
        logger.log_signal_miss(signal, reason="insufficient_margin")
        # signal_miss 追加到 self.trades 列表，通过 _save_trades 保存
        assert any(
            r.get("type") == "signal_miss" and r.get("reason") == "insufficient_margin"
            for r in logger.trades
        )
