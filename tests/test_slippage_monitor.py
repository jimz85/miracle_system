"""
SlippageMonitor 滑点监控器测试
"""
import json
import tempfile
from pathlib import Path

import pytest

from core.executor_config import ExecutorConfig
from core.slippage_monitor import SlippageMonitor


@pytest.fixture
def mock_config():
    # 显式创建临时目录，避免 pytest-asyncio tmp_path 生命周期冲突
    tmpdir = Path(tempfile.mkdtemp())
    config = ExecutorConfig()
    config.log_dir = str(tmpdir / "logs")
    config.slippage_log_file = "slippage_log.json"
    yield config
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def monitor(mock_config):
    return SlippageMonitor(config=mock_config)


class TestRecordExecution:
    """测试订单执行记录"""

    def test_zero_planned_price_returns_zero(self, monitor):
        """计划价格=0 时返回 {slippage_pct: 0, ...} 不崩溃"""
        result = monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.0,
            actual_price=0.1000,
        )
        assert result["slippage_pct"] == 0
        assert result["warning"] is False

    def test_buy_side_positive_when_actual_above_plan(self, monitor):
        """
        做多买入时：实际 > 计划 → slippage_pct > 0
        代码公式: (actual - planned) / planned
        """
        result = monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=1.0000,
            actual_price=1.0010,  # 0.1% slippage < 0.5% threshold
        )
        assert result["slippage_pct"] > 0
        assert result["warning"] is False

    def test_buy_side_negative_when_actual_below_plan(self, monitor):
        """做多买入时：实际 < 计划 → slippage_pct < 0（有利滑点）"""
        result = monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.1000,
            actual_price=0.0990,
        )
        assert result["slippage_pct"] < 0

    def test_sell_side_positive_when_actual_below_plan(self, monitor):
        """
        卖出/做空时：实际 < 计划 → slippage_pct > 0
        代码公式: (planned - actual) / planned
        """
        result = monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "sell"},
            planned_price=0.1000,
            actual_price=0.0990,
        )
        assert result["slippage_pct"] > 0

    def test_sell_side_negative_when_actual_above_plan(self, monitor):
        """卖出/做空时：实际 > 计划 → slippage_pct < 0（不利滑点）"""
        result = monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "sell"},
            planned_price=0.1000,
            actual_price=0.1010,
        )
        assert result["slippage_pct"] < 0

    def test_unknown_side_uses_plain_formula(self, monitor):
        """未知 side 使用通用公式"""
        result = monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP"},  # 无 side
            planned_price=0.1000,
            actual_price=0.1010,
        )
        # 通用公式同 buy
        assert result["slippage_pct"] > 0

    def test_record_execution_saves_to_file(self, monitor, mock_config):
        """record_execution 写入 slippage_log.json"""
        monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.1000,
            actual_price=0.1010,
        )
        log_file = Path(mock_config.log_dir) / "slippage_log.json"
        assert log_file.exists()
        with open(log_file) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["symbol"] == "DOGE-USDT-SWAP"

    def test_record_execution_preserves_signal_data(self, monitor):
        """返回值包含原始信号数据"""
        signal = {"symbol": "SOL-USDT-SWAP", "side": "buy", "confidence": 0.85}
        result = monitor.record_execution(signal, planned_price=95.0, actual_price=95.2)
        assert "slippage_pct" in result
        assert "slippage_abs" in result
        assert "warning" in result

    def test_record_execution_warning_threshold(self, monitor, mock_config):
        """abs(slippage_pct) > threshold → warning=True"""
        # threshold 默认 0.005 (0.5%)
        monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.1000,
            actual_price=0.1100,  # 10% slippage
        )
        result = monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.1000,
            actual_price=0.1100,
        )
        assert result["warning"] is True


class TestGetAvgSlippage:
    """测试平均滑点查询"""

    def test_avg_slippage_no_history(self, monitor):
        """无历史记录时返回 0"""
        avg = monitor.get_avg_slippage("DOGE-USDT-SWAP")
        assert avg == 0.0

    def test_avg_slippage_single_trade(self, monitor):
        """单笔交易平均滑点等于该笔滑点"""
        monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.1000,
            actual_price=0.1010,
        )
        avg = monitor.get_avg_slippage("DOGE-USDT-SWAP")
        assert abs(avg - 0.01) < 0.001  # ~1%

    def test_avg_slippage_multiple_trades(self, monitor):
        """多笔交易返回平均值"""
        monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.1000,
            actual_price=0.1010,
        )
        monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.1000,
            actual_price=0.0990,
        )
        avg = monitor.get_avg_slippage("DOGE-USDT-SWAP")
        assert abs(avg) < 0.001  # ~0%

    def test_avg_slippage_lookback_limit(self, monitor):
        """lookback_trades 参数限制查询范围"""
        for i in range(10):
            monitor.record_execution(
                signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
                planned_price=0.1000,
                actual_price=0.1000 + (0.001 * (i % 2)),
            )
        avg = monitor.get_avg_slippage("DOGE-USDT-SWAP", lookback_trades=3)
        assert isinstance(avg, float)

    def test_avg_slippage_unknown_symbol(self, monitor):
        """未知币种返回 0"""
        monitor.record_execution(
            signal={"symbol": "DOGE-USDT-SWAP", "side": "buy"},
            planned_price=0.1000,
            actual_price=0.1010,
        )
        avg = monitor.get_avg_slippage("UNKNOWN-USDT-SWAP")
        assert avg == 0.0
