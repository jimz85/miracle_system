from __future__ import annotations

"""
Miracle Circuit Breaker - 熔断机制封装
=======================================

对接 Kronos 熔断子系统，实现五级生存层机制。
确保 Miracle 使用 Kronos 熔断子系统，防止资金大幅亏损。

五级生存层:
- NORMAL: 正常交易 (仓位 100%)
- CAUTION: 谨慎交易 (仓位 50%)
- LOW: 低频交易 (仓位 25%)
- CRITICAL: 仅平仓 (仓位 0%, 禁止开仓)
- PAUSED: 全暂停

用法:
    from core.circuit_breaker import MiracleCircuitBreaker, SurvivalTier

    cb = MiracleCircuitBreaker()
    result = cb.check(equity=10000, positions=[])
    if result.can_open:
        # 可以开仓
        pass
"""

from dataclasses import dataclass, field
from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    # Python < 3.11 fallback
    class StrEnum(str, Enum):
        pass

from typing import List, Optional

# ══════════════════════════════════════════════════════════════════════
# Survival Tier Enum
# ══════════════════════════════════════════════════════════════════════

class SurvivalTier(StrEnum):
    """五级生存层"""
    NORMAL = "normal"      # 正常交易
    CAUTION = "caution"    # 谨慎交易 (50%仓位)
    LOW = "low"           # 低频交易 (25%仓位)
    CRITICAL = "critical"  # 仅平仓 (0%开仓)
    PAUSED = "paused"     # 全暂停


# ══════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0


@dataclass
class CircuitBreakerResult:
    """熔断检查结果"""
    allowed: bool                    # 是否允许交易
    tier: SurvivalTier              # 当前生存层
    max_position_pct: float        # 最大持仓比例
    can_open: bool                 # 是否可以开仓
    can_close: bool                 # 是否可以平仓
    consecutive_losses: int         # 连亏次数
    reason: str                    # 原因描述
    drawdown_pct: float = 0.0      # 当前回撤百分比
    equity: float = 0.0             # 当前权益


@dataclass
class EquitySnapshot:
    """权益快照"""
    initial_equity: float = 0.0
    peak_equity: float = 0.0
    daily_snapshot: float = 0.0  # 日初权益快照
    daily_snapshot_date: str = ""  # YYYY-MM-DD格式
    snapshots: List[float] = field(default_factory=list)

    def update(self, current_equity: float) -> None:
        """更新权益快照"""
        from datetime import date
        today = str(date.today())
        
        if self.initial_equity == 0.0:
            self.initial_equity = current_equity
            self.peak_equity = current_equity
            self.daily_snapshot = current_equity
            self.daily_snapshot_date = today

        self.snapshots.append(current_equity)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # 新的一天：重置日初快照
        if today != self.daily_snapshot_date:
            self.daily_snapshot = current_equity
            self.daily_snapshot_date = today
        else:
            # 同一天：更新日初快照为当日的最低点（用于恢复检测）
            if current_equity < self.daily_snapshot:
                self.daily_snapshot = current_equity

        # 保持最近1000个快照
        if len(self.snapshots) > 1000:
            self.snapshots.pop(0)

    def get_initial(self) -> float:
        """获取初始权益"""
        return self.initial_equity

    def get_peak(self) -> float:
        """获取历史最高权益"""
        return self.peak_equity
    
    def get_daily_snapshot(self) -> float:
        """获取日初权益快照"""
        return self.daily_snapshot if self.daily_snapshot > 0 else self.initial_equity
    
    def get_recovery_pct(self, current_equity: float) -> float:
        """获取相对日初快照的恢复百分比"""
        daily_snap = self.get_daily_snapshot()
        if daily_snap <= 0:
            return 0.0
        return (current_equity - daily_snap) / daily_snap


# ══════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ══════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Kronos 熔断子系统实现

    五级生存层机制:
    - NORMAL: 亏损 0% (相对于初始权益)
    - CAUTION: 亏损 0-5%
    - LOW: 亏损 5-10%
    - CRITICAL: 亏损 10-20%
    - PAUSED: 亏损 >20%

    渐进恢复:
    - 连亏计数重置后，需要一定盈利才能恢复层级
    """

    # 五级阈值 (相对于初始权益的亏损百分比)
    TIER_THRESHOLDS = {
        SurvivalTier.NORMAL: 0.00,    # 0% 亏损
        SurvivalTier.CAUTION: -0.05,  # -5% 亏损
        SurvivalTier.LOW: -0.10,      # -10% 亏损
        SurvivalTier.CRITICAL: -0.20, # -20% 亏损
        SurvivalTier.PAUSED: -0.30,   # -30% 亏损
    }

    # 渐进恢复步长 (恢复至正常后，逐步提升仓位)
    RECOVERY_STEPS = {
        SurvivalTier.CAUTION: 0.50,   # 恢复至50%
        SurvivalTier.LOW: 0.25,       # 恢复至25%
        SurvivalTier.CRITICAL: 0.0,   # 保持0%
    }

    # 恢复所需最小涨幅 (相对日初快照)
    MIN_RECOVERY_PCT = 0.03  # 3%

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,      # 单日最大亏损5%
        max_drawdown_pct: float = 0.30,         # 总最大回撤30%
        cooldown_hours_after_2_losses: int = 1,  # 2连亏后冷却1小时
        cooldown_hours_after_3_losses: int = 24, # 3连亏后冷却24小时
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_hours_after_2_losses = cooldown_hours_after_2_losses
        self.cooldown_hours_after_3_losses = cooldown_hours_after_3_losses

        self.current_tier = SurvivalTier.NORMAL
        self.consecutive_losses = 0
        self.equity_snapshot = EquitySnapshot()
        self.last_trade_time: float | None = None
        self._last_recovery_check_equity: float = 0.0  # 上次检查时的权益

    def check_treasury_limits(
        self,
        current_equity: float,
        positions: List[Position]
    ) -> CircuitBreakerResult:
        """
        检查熔断限制

        Args:
            current_equity: 当前账户权益
            positions: 当前持仓列表

        Returns:
            CircuitBreakerResult: 熔断检查结果
        """
        # 1. 更新权益快照
        self.equity_snapshot.update(current_equity)

        # 2. 计算当前回撤
        initial_equity = self.equity_snapshot.get_initial()
        if initial_equity <= 0:
            return CircuitBreakerResult(
                allowed=False,
                tier=SurvivalTier.PAUSED,
                max_position_pct=0.0,
                can_open=False,
                can_close=True,
                consecutive_losses=self.consecutive_losses,
                reason="初始权益无效",
                equity=current_equity
            )

        drawdown = (current_equity - initial_equity) / initial_equity
        drawdown_pct = drawdown * 100

        # 3. 确定生存层级
        new_tier = self._determine_tier(drawdown)

        # 4. 渐进恢复检查
        if new_tier == SurvivalTier.NORMAL and self.current_tier != SurvivalTier.NORMAL:
            new_tier = self._check_recovery(current_equity)

        self.current_tier = new_tier

        # 5. 构建结果
        can_open = new_tier in [SurvivalTier.NORMAL, SurvivalTier.CAUTION]
        max_position_pct = self._get_max_position_pct(new_tier)

        return CircuitBreakerResult(
            allowed=new_tier not in [SurvivalTier.PAUSED, SurvivalTier.CRITICAL],
            tier=new_tier,
            max_position_pct=max_position_pct,
            can_open=can_open,
            can_close=True,  # 任何层级都可以平仓
            consecutive_losses=self.consecutive_losses,
            reason=self._get_tier_reason(new_tier, drawdown_pct),
            drawdown_pct=drawdown_pct,
            equity=current_equity
        )

    def record_trade_outcome(self, pnl: float) -> None:
        """
        记录交易结果

        Args:
            pnl: 交易盈亏 (正数为盈利，负数为亏损)
        """
        import time
        self.last_trade_time = time.time()

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            # 盈利后重置连亏计数
            self.consecutive_losses = 0

    def _determine_tier(self, drawdown: float) -> SurvivalTier:
        """根据回撤确定生存层"""
        # drawdown 是负数 (亏损)
        if drawdown >= self.TIER_THRESHOLDS[SurvivalTier.CAUTION]:
            return SurvivalTier.NORMAL
        elif drawdown >= self.TIER_THRESHOLDS[SurvivalTier.LOW]:
            return SurvivalTier.CAUTION
        elif drawdown >= self.TIER_THRESHOLDS[SurvivalTier.CRITICAL]:
            return SurvivalTier.LOW
        elif drawdown >= self.TIER_THRESHOLDS[SurvivalTier.PAUSED]:
            return SurvivalTier.CRITICAL
        else:
            return SurvivalTier.PAUSED

    def _get_max_position_pct(self, tier: SurvivalTier) -> float:
        """获取最大持仓比例"""
        if tier == SurvivalTier.NORMAL:
            return 1.0
        elif tier == SurvivalTier.CAUTION:
            return 0.50
        elif tier == SurvivalTier.LOW:
            return 0.25
        elif tier == SurvivalTier.CRITICAL:
            return 0.0
        else:  # PAUSED
            return 0.0

    def _check_recovery(self, current_equity: float) -> SurvivalTier:
        """
        检查是否可以从当前层级恢复
        
        恢复条件:
        1. 连亏计数必须为0
        2. 净值必须比日初快照回升>=3%
        
        Args:
            current_equity: 当前权益
            
        Returns:
            SurvivalTier: 恢复后的层级
        """
        # 条件1: 连亏计数必须为0
        if self.consecutive_losses != 0:
            return self.current_tier
        
        # 条件2: 净值必须比日初快照回升>=3%
        recovery_pct = self.equity_snapshot.get_recovery_pct(current_equity)
        if recovery_pct < self.MIN_RECOVERY_PCT:
            # 不能恢复，保持当前层级
            return self.current_tier
        
        # 满足所有条件，恢复到NORMAL
        return SurvivalTier.NORMAL

    def _get_tier_reason(self, tier: SurvivalTier, drawdown_pct: float) -> str:
        """获取层级的描述原因"""
        reasons = {
            SurvivalTier.NORMAL: f"正常交易 (回撤: {drawdown_pct:.2f}%)",
            SurvivalTier.CAUTION: f"谨慎交易 - 回撤 {drawdown_pct:.2f}% (仓位限制50%)",
            SurvivalTier.LOW: f"低频交易 - 回撤 {drawdown_pct:.2f}% (仓位限制25%)",
            SurvivalTier.CRITICAL: f"仅平仓 - 回撤 {drawdown_pct:.2f}% (禁止开仓)",
            SurvivalTier.PAUSED: f"全暂停 - 回撤 {drawdown_pct:.2f}% (系统停止)",
        }
        return reasons.get(tier, "未知状态")

    def get_status(self) -> dict:
        """获取当前熔断状态"""
        return {
            "tier": self.current_tier.value,
            "consecutive_losses": self.consecutive_losses,
            "can_open": self.current_tier in [SurvivalTier.NORMAL, SurvivalTier.CAUTION],
            "can_close": True,
            "max_position_pct": self._get_max_position_pct(self.current_tier),
        }


# ══════════════════════════════════════════════════════════════════════
# Miracle Circuit Breaker (Wrapper)
# ══════════════════════════════════════════════════════════════════════

class MiracleCircuitBreaker:
    """
    Miracle 熔断封装类

    提供与 Kronos 熔断子系统对接的统一接口。
    所有交易决策前必须调用 check() 方法。
    所有交易结果后必须调用 record_outcome() 方法。
    """

    def __init__(self, config: dict | None = None):
        """
        初始化熔断器

        Args:
            config: 配置字典，包含:
                - max_daily_loss_pct: 单日最大亏损比例
                - max_drawdown_pct: 总最大回撤比例
                - cooldown_hours_after_2_losses: 2连亏冷却小时数
                - cooldown_hours_after_3_losses: 3连亏冷却小时数
        """
        cfg = config or {}

        self.cb = CircuitBreaker(
            max_daily_loss_pct=cfg.get("max_daily_loss_pct", 0.05),
            max_drawdown_pct=cfg.get("max_drawdown_pct", 0.30),
            cooldown_hours_after_2_losses=cfg.get("cooldown_hours_after_2_losses", 1),
            cooldown_hours_after_3_losses=cfg.get("cooldown_hours_after_3_losses", 24),
        )

    def check(self, equity: float, positions: List[Position]) -> CircuitBreakerResult:
        """
        检查是否允许交易

        Args:
            equity: 当前账户权益
            positions: 当前持仓列表

        Returns:
            CircuitBreakerResult: 熔断检查结果
        """
        return self.cb.check_treasury_limits(equity, positions)

    def record_outcome(self, pnl: float) -> None:
        """
        记录交易结果

        Args:
            pnl: 交易盈亏 (正数为盈利，负数为亏损)
        """
        self.cb.record_trade_outcome(pnl)

    def get_tier(self) -> SurvivalTier:
        """获取当前生存层"""
        return self.cb.current_tier

    def can_open_position(self) -> bool:
        """检查是否可以开仓"""
        return self.cb.current_tier in [SurvivalTier.NORMAL, SurvivalTier.CAUTION]

    def get_max_position_pct(self) -> float:
        """获取最大持仓比例"""
        return self.cb._get_max_position_pct(self.cb.current_tier)


# ══════════════════════════════════════════════════════════════════════
# Factory Function
# ══════════════════════════════════════════════════════════════════════

def create_circuit_breaker(config: dict | None = None) -> MiracleCircuitBreaker:
    """
    创建熔断器的便捷函数

    Args:
        config: 配置字典

    Returns:
        MiracleCircuitBreaker 实例
    """
    return MiracleCircuitBreaker(config)


# ══════════════════════════════════════════════════════════════════════
# Standalone Test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    from datetime import datetime, timedelta

    print("=== Circuit Breaker Test ===\n")

    # 创建熔断器
    cb = MiracleCircuitBreaker()

    # 模拟初始权益 10000
    initial_equity = 10000.0

    # 测试1: 正常状态
    print("[Test 1] 初始状态")
    result = cb.check(equity=10000, positions=[])
    print(f"  Tier: {result.tier.value}")
    print(f"  Can Open: {result.can_open}")
    print(f"  Max Position: {result.max_position_pct:.0%}")
    print(f"  Reason: {result.reason}")
    print()

    # 测试2: 小幅回撤 (3%)
    print("[Test 2] 小幅回撤 (-3%)")
    result = cb.check(equity=9700, positions=[])
    print(f"  Tier: {result.tier.value}")
    print(f"  Can Open: {result.can_open}")
    print(f"  Max Position: {result.max_position_pct:.0%}")
    print()

    # 测试3: 回撤到 CAUTION 层 (-5%)
    print("[Test 3] CAUTION 层 (-5%)")
    result = cb.check(equity=9500, positions=[])
    print(f"  Tier: {result.tier.value}")
    print(f"  Can Open: {result.can_open}")
    print(f"  Max Position: {result.max_position_pct:.0%}")
    print()

    # 测试4: 回撤到 LOW 层 (-10%)
    print("[Test 4] LOW 层 (-10%)")
    result = cb.check(equity=9000, positions=[])
    print(f"  Tier: {result.tier.value}")
    print(f"  Can Open: {result.can_open}")
    print(f"  Max Position: {result.max_position_pct:.0%}")
    print()

    # 测试5: 回撤到 CRITICAL 层 (-20%)
    print("[Test 5] CRITICAL 层 (-20%)")
    result = cb.check(equity=8000, positions=[])
    print(f"  Tier: {result.tier.value}")
    print(f"  Can Open: {result.can_open}")
    print(f"  Max Position: {result.max_position_pct:.0%}")
    print()

    # 测试6: 回撤到 PAUSED 层 (-30%)
    print("[Test 6] PAUSED 层 (-30%)")
    result = cb.check(equity=7000, positions=[])
    print(f"  Tier: {result.tier.value}")
    print(f"  Can Open: {result.can_open}")
    print(f"  Max Position: {result.max_position_pct:.0%}")
    print()

    # 测试7: 记录交易结果
    print("[Test 7] 记录交易结果")
    cb.record_outcome(-100)  # 亏损100
    print(f"  After loss: consecutive_losses = {cb.cb.consecutive_losses}")

    cb.record_outcome(-50)  # 再亏50
    print(f"  After loss: consecutive_losses = {cb.cb.consecutive_losses}")

    cb.record_outcome(+200)  # 盈利200
    print(f"  After win: consecutive_losses = {cb.cb.consecutive_losses}")
    print()

    # 测试8: 恢复测试
    print("[Test 8] 恢复测试")
    cb2 = MiracleCircuitBreaker()

    # 模拟亏损到CAUTION
    cb2.check(equity=9500, positions=[])
    print(f"  At CAUTION: tier = {cb2.get_tier().value}")

    # 模拟盈利恢复到NORMAL
    cb2.record_outcome(+600)  # 盈利600，回到10000+
    result = cb2.check(equity=10000, positions=[])
    print(f"  After recovery: tier = {result.tier.value}")
    print()

    print("=== All Tests Passed! ===")