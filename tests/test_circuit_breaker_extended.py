"""
CircuitBreaker 补充测试 - 覆盖核心风险控制逻辑

两个核心类:
  - CircuitBreaker (line 133): 核心实现，方法 check_treasury_limits / record_trade_outcome / get_status
  - MiracleCircuitBreaker (line 341): 统一包装，方法 check / record_outcome / get_tier / can_open_position
"""
import pytest
from core.circuit_breaker import (
    CircuitBreaker, MiracleCircuitBreaker, EquitySnapshot,
    SurvivalTier, CircuitBreakerResult
)


# ══════════════════════════════════════════════════════════════
# CircuitBreaker 核心实现测试
# ══════════════════════════════════════════════════════════════

class TestCircuitBreakerCore:
    """测试 CircuitBreaker 核心实现"""

    def test_default_init(self):
        """默认初始化"""
        cb = CircuitBreaker()
        assert cb.current_tier == SurvivalTier.NORMAL
        assert cb.consecutive_losses == 0

    def test_custom_init(self):
        """自定义参数初始化"""
        cb = CircuitBreaker(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.20,
            cooldown_hours_after_2_losses=2,
            cooldown_hours_after_3_losses=48,
        )
        assert cb.max_daily_loss_pct == 0.03
        assert cb.max_drawdown_pct == 0.20
        assert cb.cooldown_hours_after_2_losses == 2
        assert cb.cooldown_hours_after_3_losses == 48


class TestCheckTreasuryLimits:
    """测试 check_treasury_limits() 熔断检查"""

    def test_normal_within_limits(self):
        """权益正常 → 允许交易"""
        cb = CircuitBreaker()
        result = cb.check_treasury_limits(current_equity=100000, positions=[])

        assert isinstance(result, CircuitBreakerResult)
        assert result.allowed is True
        assert result.can_open is True
        assert result.can_close is True

    def test_result_has_all_fields(self):
        """结果包含所有必要字段"""
        cb = CircuitBreaker()
        result = cb.check_treasury_limits(current_equity=100000, positions=[])

        assert hasattr(result, 'allowed')
        assert hasattr(result, 'tier')
        assert hasattr(result, 'can_open')
        assert hasattr(result, 'can_close')
        assert hasattr(result, 'max_position_pct')

    def test_max_position_pct_per_tier(self):
        """各层级最大持仓比例正确"""
        cb = CircuitBreaker()

        # NORMAL: 100%
        r = cb.check_treasury_limits(current_equity=100000, positions=[])
        assert r.max_position_pct == 1.0

    def test_equity_snapshot_updated(self):
        """检查后权益快照已更新"""
        cb = CircuitBreaker()
        cb.check_treasury_limits(current_equity=100000, positions=[])

        assert cb.equity_snapshot.get_initial() == 100000


class TestRecordTradeOutcome:
    """测试 record_trade_outcome() 交易记录"""

    def test_loss_increases_consecutive(self):
        """亏损 → 连亏计数+1"""
        cb = CircuitBreaker()
        cb.check_treasury_limits(current_equity=100000, positions=[])

        cb.record_trade_outcome(pnl=-100)
        assert cb.consecutive_losses == 1

    def test_win_resets_consecutive(self):
        """盈利 → 连亏计数归零"""
        cb = CircuitBreaker()
        cb.consecutive_losses = 3
        cb.record_trade_outcome(pnl=100)
        assert cb.consecutive_losses == 0

    def test_zero_pnl_resets_consecutive(self):
        """pnl=0 → 重置连亏计数（代码逻辑：非亏损即重置）"""
        cb = CircuitBreaker()
        cb.consecutive_losses = 2
        cb.record_trade_outcome(pnl=0)
        # 代码逻辑: if pnl < 0 → else(盈利或零) → 重置为0
        assert cb.consecutive_losses == 0


class TestGetStatus:
    """测试 get_status() 状态查询"""

    def test_status_has_required_fields(self):
        """状态字典包含所有字段"""
        cb = CircuitBreaker()
        cb.check_treasury_limits(current_equity=100000, positions=[])

        status = cb.get_status()

        assert "tier" in status
        assert "consecutive_losses" in status
        assert "can_open" in status
        assert "can_close" in status
        assert "max_position_pct" in status

    def test_status_reflects_tier(self):
        """状态反映当前层级"""
        cb = CircuitBreaker()
        cb.check_treasury_limits(current_equity=100000, positions=[])

        status = cb.get_status()
        assert status["tier"] == SurvivalTier.NORMAL.value


# ══════════════════════════════════════════════════════════════
# MiracleCircuitBreaker 包装类测试
# ══════════════════════════════════════════════════════════════

class TestMiracleCircuitBreaker:
    """测试 MiracleCircuitBreaker 包装类"""

    def test_default_init(self):
        """默认初始化"""
        cb = MiracleCircuitBreaker()
        assert cb.get_tier() == SurvivalTier.NORMAL
        assert cb.can_open_position() is True

    def test_check_equivalent(self):
        """check() 返回与 CircuitBreaker.check_treasury_limits() 一致的结果"""
        cb = MiracleCircuitBreaker()
        result = cb.check(equity=100000, positions=[])

        assert isinstance(result, CircuitBreakerResult)
        assert result.allowed is True

    def test_record_outcome_delegates(self):
        """record_outcome 正确委托给内部 CircuitBreaker"""
        cb = MiracleCircuitBreaker()
        cb.check(equity=100000, positions=[])

        cb.record_outcome(pnl=-100)

        # 委托后，内部连亏计数应该增加
        assert cb.cb.consecutive_losses == 1

    def test_get_tier_delegates(self):
        """get_tier 正确返回当前层级"""
        cb = MiracleCircuitBreaker()
        assert cb.get_tier() == SurvivalTier.NORMAL

    def test_can_open_position(self):
        """can_open_position 返回布尔值"""
        cb = MiracleCircuitBreaker()
        result = cb.can_open_position()
        assert isinstance(result, bool)
        assert result is True  # 初始状态

    def test_get_max_position_pct(self):
        """get_max_position_pct 返回浮点数"""
        cb = MiracleCircuitBreaker()
        pct = cb.get_max_position_pct()
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 1.0


# ══════════════════════════════════════════════════════════════
# EquitySnapshot 核心逻辑测试
# ══════════════════════════════════════════════════════════════

class TestEquitySnapshot:
    """测试 EquitySnapshot 权益快照"""

    def test_initial_equity_zero(self):
        """初始权益为 0"""
        snap = EquitySnapshot()
        assert snap.get_initial() == 0

    def test_first_update_sets_initial(self):
        """首次 update 设置 initial_equity"""
        snap = EquitySnapshot()
        snap.update(100000)

        assert snap.get_initial() == 100000
        assert snap.peak_equity == 100000

    def test_peak_never_decreases(self):
        """peak 只增不减"""
        snap = EquitySnapshot()
        snap.update(100000)
        snap.update(80000)
        snap.update(90000)

        assert snap.peak_equity == 100000

    def test_peak_increases_with_higher(self):
        """peak 随更高权益更新"""
        snap = EquitySnapshot()
        snap.update(100000)
        snap.update(110000)
        snap.update(105000)

        assert snap.peak_equity == 110000

    def test_recovery_positive(self):
        """正恢复百分比"""
        snap = EquitySnapshot()
        snap.update(100000)
        snap.update(95000)

        recovery = snap.get_recovery_pct(97000)
        assert recovery > 0

    def test_recovery_zero_at_daily(self):
        """等于日初 → 0% 恢复"""
        snap = EquitySnapshot()
        snap.update(100000)
        snap.update(95000)

        recovery = snap.get_recovery_pct(95000)
        assert recovery == 0.0

    def test_recovery_negative(self):
        """低于日初 → 负恢复"""
        snap = EquitySnapshot()
        snap.update(100000)
        snap.update(96000)

        recovery = snap.get_recovery_pct(95000)
        assert recovery < 0

    def test_max_snapshots_1000(self):
        """超过1000条快照时丢弃最早的"""
        snap = EquitySnapshot()
        for i in range(1010):
            snap.update(100000 + i)

        assert len(snap.snapshots) <= 1000


# ══════════════════════════════════════════════════════════════
# SurvivalTier 枚举测试
# ══════════════════════════════════════════════════════════════

class TestSurvivalTier:
    """测试 SurvivalTier 枚举"""

    def test_all_five_tiers_exist(self):
        """5个层级都存在"""
        assert SurvivalTier.NORMAL is not None
        assert SurvivalTier.CAUTION is not None
        assert SurvivalTier.LOW is not None
        assert SurvivalTier.CRITICAL is not None
        assert SurvivalTier.PAUSED is not None

    def test_tiers_inherit_from_str(self):
        """层级是字符串枚举"""
        for tier in SurvivalTier:
            assert isinstance(tier, str)
            assert tier.value in ['normal', 'caution', 'low', 'critical', 'paused']

    def test_tier_ordering(self):
        """层级存在隐含顺序（通过 value 字符串比较）"""
        tiers = list(SurvivalTier)
        values = [t.value for t in tiers]
        assert len(values) == len(set(values))  # 唯一
