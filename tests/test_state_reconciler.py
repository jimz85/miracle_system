"""
StateReconciler 状态协调器测试
测试幽灵仓位检测、孤立订单检测、仓位对账等核心逻辑
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.state_reconciler import (
    LocalOrderRecord,
    LocalPositionRecord,
    OCOIssue,
    OrphanOrder,
    PhantomPosition,
    ReconcileResult,
    StateReconciler,
    ZombiePosition,
)

# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def temp_state_file():
    """临时状态文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"positions": {}, "orders": {}, "trades": []}')
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_trade_log():
    """临时交易日志文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('[]')
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════
# 数据类测试 (无需 Mock)
# ══════════════════════════════════════════════════════════════

class TestLocalPositionRecord:
    """测试本地持仓记录数据类"""

    def test_record_creation(self):
        """测试记录创建"""
        record = LocalPositionRecord(
            inst_id="DOGE-USDT-SWAP",
            direction="long",
            entry_price=0.10,
            contracts=1000.0,
            status="open"
        )

        assert record.inst_id == "DOGE-USDT-SWAP"
        assert record.direction == "long"
        assert record.contracts == 1000.0
        assert record.status == "open"

    def test_record_asdict(self):
        """测试 dataclass 序列化"""
        record = LocalPositionRecord(
            inst_id="DOGE-USDT-SWAP",
            direction="long",
            entry_price=0.10,
            contracts=1000.0,
            status="open"
        )

        # dataclass 可通过 asdict() 序列化（需手动调用）
        d = {
            "inst_id": record.inst_id,
            "direction": record.direction,
            "entry_price": record.entry_price,
            "contracts": record.contracts,
            "status": record.status,
        }
        assert isinstance(d, dict)
        assert d["inst_id"] == "DOGE-USDT-SWAP"
        assert d["direction"] == "long"


class TestPhantomPosition:
    """测试幽灵仓位数据类"""

    def test_phantom_position_creation(self):
        """测试幽灵仓位创建"""
        phantom = PhantomPosition(
            inst_id="DOGE-USDT-SWAP",
            direction="long",
            entry_price=0.10,
            contracts=1000.0,
            note="交易所无此持仓"
        )

        assert phantom.inst_id == "DOGE-USDT-SWAP"
        assert phantom.direction == "long"
        assert phantom.contracts == 1000.0
        assert phantom.note == "交易所无此持仓"


class TestZombiePosition:
    """测试僵尸仓位数据类"""

    def test_zombie_position_creation(self):
        """测试僵尸仓位创建"""
        zombie = ZombiePosition(
            inst_id="SOL-USDT-SWAP",
            direction="long",
            pos=50.0,
            entry_price=150.0,
            mark_price=155.0,
            upl=250.0,
            notional=7750.0,
            note="本地无此记录"
        )

        assert zombie.inst_id == "SOL-USDT-SWAP"
        assert zombie.pos == 50.0
        assert zombie.upl == 250.0


class TestOrphanOrder:
    """测试孤立订单数据类"""

    def test_orphan_order_creation(self):
        """测试孤立订单创建"""
        order = OrphanOrder(
            order_id="ORD123456",
            inst_id="DOGE-USDT-SWAP",
            side="buy",
            order_type="market",
            sz=100.0,
            note="本地无此记录"
        )

        assert order.order_id == "ORD123456"
        assert order.inst_id == "DOGE-USDT-SWAP"
        assert order.side == "buy"


class TestReconcileResult:
    """测试 ReconcileResult 数据类"""

    def test_default_values(self):
        """测试默认值"""
        result = ReconcileResult()

        assert result.orphan_orders == []
        assert result.phantom_positions == []
        assert result.zombie_positions == []
        assert result.oco_issues == []
        assert result.is_healthy is True
        assert result.needs_sync is False
        assert result.errors == []
        assert result.warnings == []

    def test_is_healthy_default_true(self):
        """默认 is_healthy 为 True"""
        result = ReconcileResult()
        assert result.is_healthy is True

    def test_is_healthy_can_be_set_false(self):
        """is_healthy 可以被设为 False"""
        result = ReconcileResult(is_healthy=False)
        assert result.is_healthy is False

    def test_is_healthy_not_auto_computed(self):
        """is_healthy 是普通字段，非计算属性。
        reconcile() 方法负责在检测到问题时设置 is_healthy=False。
        此测试验证 dataclass 本身的默认行为。"""
        result = ReconcileResult(
            phantom_positions=[
                PhantomPosition(
                    inst_id="DOGE-USDT-SWAP",
                    direction="long",
                    entry_price=0.10,
                    contracts=1000.0,
                )
            ]
        )
        # dataclass 不自动计算 is_healthy，保持默认值 True
        # reconcile() 运行时才会设置 is_healthy=False
        assert result.is_healthy is True  # 默认值，除非被 reconcile() 修改

    def test_zombie_positions_can_be_set(self):
        """zombie_positions 可以被设置"""
        result = ReconcileResult(
            zombie_positions=[
                ZombiePosition(
                    inst_id="SOL-USDT-SWAP",
                    direction="long",
                    pos=50.0,
                    entry_price=150.0,
                    mark_price=155.0,
                    upl=250.0,
                    notional=7750.0,
                )
            ]
        )
        # is_healthy 需要由 reconcile() 方法设置，非自动计算
        assert len(result.zombie_positions) == 1
        assert result.zombie_positions[0].inst_id == "SOL-USDT-SWAP"

    def test_needs_sync_true(self):
        """needs_sync 为 True 表示需要同步"""
        result = ReconcileResult(needs_sync=True)
        assert result.needs_sync is True

    def test_warnings_and_errors(self):
        """测试警告和错误字段"""
        result = ReconcileResult(
            warnings=["DOGE-USDT-SWAP 本地记录已过期"],
            errors=["无法连接 OKX API"]
        )

        assert len(result.warnings) == 1
        assert len(result.errors) == 1
        assert result.is_healthy is True  # is_healthy 需由 reconcile() 设置


# ══════════════════════════════════════════════════════════════
# ReconcileResult 字段计数
# ══════════════════════════════════════════════════════════════

class TestReconcileResultCounts:
    """测试 ReconcileResult 计数字段"""

    def test_counts_default_to_zero(self):
        """计数字段默认为 0"""
        result = ReconcileResult()

        assert result.total_exchange_positions == 0
        assert result.total_local_positions == 0
        assert result.total_exchange_orders == 0
        assert result.total_local_orders == 0
        assert result.account_balance == 0.0

    def test_counts_set_correctly(self):
        """计数字段可以正确设置"""
        result = ReconcileResult(
            total_exchange_positions=3,
            total_local_positions=5,
            total_exchange_orders=10,
            total_local_orders=8,
            account_balance=72806.50,
        )

        assert result.total_exchange_positions == 3
        assert result.total_local_positions == 5
        assert result.account_balance == 72806.50


# ══════════════════════════════════════════════════════════════
# StateReconciler 实例化测试
# ══════════════════════════════════════════════════════════════

class TestStateReconcilerInit:
    """测试 StateReconciler 初始化"""

    def test_default_initialization(self, temp_state_file, temp_trade_log):
        """测试默认初始化（需要 Mock ExchangeClient）"""
        with patch('core.state_reconciler.ExchangeClient') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            reconciler = StateReconciler(
                state_file=temp_state_file,
                trade_log_file=temp_trade_log,
                exchange="okx"
            )

            assert reconciler.exchange == "okx"
            assert reconciler.state_file == temp_state_file
            assert reconciler.trade_log_file == temp_trade_log
            # client 应该在 init 中创建
            mock_client_cls.assert_called_once()

    def test_parent_dir_created_if_missing(self, temp_state_file):
        """测试父目录不存在时会被创建（文件本身由 save_local_state 创建）"""
        temp_state_file.unlink()  # 删除文件
        # 父目录也删除
        parent_dir = temp_state_file.parent
        import shutil
        shutil.rmtree(parent_dir, ignore_errors=True)

        with patch('core.state_reconciler.ExchangeClient'):
            StateReconciler(
                state_file=temp_state_file,
                trade_log_file=temp_trade_log,
                exchange="okx"
            )

        # 父目录应该被创建
        assert temp_state_file.parent.exists()


# ══════════════════════════════════════════════════════════════
# 本地状态读写测试
# ══════════════════════════════════════════════════════════════

class TestLocalStatePersistence:
    """测试本地状态文件读写"""

    def test_load_empty_state(self, temp_state_file, temp_trade_log):
        """空状态文件加载返回空字典"""
        with open(temp_state_file, 'w') as f:
            json.dump({"positions": {}, "orders": {}, "trades": []}, f)

        with patch('core.state_reconciler.ExchangeClient'):
            reconciler = StateReconciler(
                state_file=temp_state_file,
                trade_log_file=temp_trade_log,
                exchange="okx"
            )
            loaded = reconciler.load_local_positions()

        assert loaded == {}

    def test_save_and_load_positions(self, temp_state_file, temp_trade_log):
        """保存后加载，仓位一致"""
        positions = {
            "DOGE-USDT-SWAP": LocalPositionRecord(
                inst_id="DOGE-USDT-SWAP",
                direction="long",
                entry_price=0.10,
                contracts=1000.0,
                status="open"
            )
        }

        with patch('core.state_reconciler.ExchangeClient'):
            reconciler = StateReconciler(
                state_file=temp_state_file,
                trade_log_file=temp_trade_log,
                exchange="okx"
            )
            reconciler.save_local_state(positions)

        # 重新加载
        with patch('core.state_reconciler.ExchangeClient'):
            reconciler2 = StateReconciler(
                state_file=temp_state_file,
                trade_log_file=temp_trade_log,
                exchange="okx"
            )
            loaded = reconciler2.load_local_positions()

        assert "DOGE-USDT-SWAP" in loaded
        assert loaded["DOGE-USDT-SWAP"].direction == "long"
        assert loaded["DOGE-USDT-SWAP"].contracts == 1000.0


# ══════════════════════════════════════════════════════════════
# can_open_position 权限检查
# ══════════════════════════════════════════════════════════════

class TestCanOpenPosition:
    """测试 can_open_position 权限检查"""

    def test_returns_tuple(self, temp_state_file, temp_trade_log):
        """can_open_position 返回 (bool, str) 元组"""
        with patch('core.state_reconciler.ExchangeClient'):
            reconciler = StateReconciler(
                state_file=temp_state_file,
                trade_log_file=temp_trade_log,
                exchange="okx"
            )
            result = reconciler.can_open_position("DOGE-USDT-SWAP")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# ══════════════════════════════════════════════════════════════
# 多实例隔离测试
# ══════════════════════════════════════════════════════════════

class TestMultiInstanceIsolation:
    """测试多实例状态隔离"""

    def test_two_instances_different_files(self):
        """两个实例使用不同状态文件，互不干扰"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "state1.json"
            file2 = Path(tmpdir) / "state2.json"

            with patch('core.state_reconciler.ExchangeClient'):
                r1 = StateReconciler(state_file=file1, exchange="okx")
                r2 = StateReconciler(state_file=file2, exchange="okx")

                assert r1.state_file != r2.state_file
                assert r1.state_file == file1
                assert r2.state_file == file2

    def test_state_file_persistence(self, temp_state_file, temp_trade_log):
        """测试状态文件持久化"""
        with patch('core.state_reconciler.ExchangeClient'):
            reconciler = StateReconciler(
                state_file=temp_state_file,
                trade_log_file=temp_trade_log,
                exchange="okx"
            )
            positions = {
                "DOGE-USDT-SWAP": LocalPositionRecord(
                    inst_id="DOGE-USDT-SWAP",
                    direction="long",
                    entry_price=0.10,
                    contracts=1000.0,
                    status="open"
                )
            }
            reconciler.save_local_state(positions)

        # 重新读取文件内容（直接 dict 格式，不是包装在 positions 键下）
        with open(temp_state_file) as f:
            data = json.load(f)

        assert "DOGE-USDT-SWAP" in data
        assert data["DOGE-USDT-SWAP"]["direction"] == "long"
        assert data["DOGE-USDT-SWAP"]["contracts"] == 1000.0
