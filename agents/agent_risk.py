"""
Agent-R: 风险与仓位管理Agent
Miracle 1.0.1 — 高频趋势跟踪+事件驱动混合系统

职责:
1. 接收Agent-S的交易信号
2. 计算最优仓位
3. 计算动态杠杆（趋势强3x，趋势弱1x）
4. 设置止损/止盈
5. 熔断检查（单日亏损/总回撤）
6. 批准/拒绝信号
7. 输出最终执行指令给Agent-E
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Signal:
    """交易信号（来自Agent-S）"""
    symbol: str                    # 交易标的
    direction: str                # "long" 或 "short"
    trend_strength: float         # 趋势强度 0-100
    confidence: float            # 置信度 0-1
    rr_ratio: float              # 信号自带的预期风险回报比
    atr: float                   # ATR波动率
    entry_price: float            # 建议入场价
    event_impact: str = "none"   # 事件影响: "high", "medium", "low", "none"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AccountState:
    """账户状态"""
    balance: float               # 当前余额
    peak_balance: float          # 历史最高余额（用于计算回撤）
    today_pnl: float             # 今日盈亏
    today_trades: int = 0         # 今日交易次数
    loss_streak: int = 0          # 连续亏损次数
    last_trade_time: Optional[datetime] = None  # 最后交易时间


@dataclass
class RiskApproval:
    """风控审批结果"""
    approved: bool
    rejection_reason: Optional[str] = None
    leverage: int = 1
    position_size_pct: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: float = 0.0
    modified_signal: Optional[Signal] = None
    warnings: list = field(default_factory=list)


# ============================================================
# 熔断器
# ============================================================

class CircuitBreaker:
    """
    熔断机制：防止连续亏损和过大回撤
    - 单日亏损超过阈值 → 停止交易
    - 总回撤超过阈值 → 停止交易
    - 连续亏损触发冷却期
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.max_daily_loss_pct: float = cfg.get("max_daily_loss_pct", 5.0)      # 单日亏损5%停止
        self.max_drawdown_pct: float = cfg.get("max_drawdown_pct", 20.0)         # 总回撤20%停止
        self.cooldown_2_losses_hours: int = cfg.get("cooldown_2_losses_hours", 1) # 2连亏暂停1小时
        self.cooldown_3_losses_hours: int = cfg.get("cooldown_3_losses_hours", 24) # 3连亏暂停1天
        self.min_trend_strength: float = cfg.get("min_trend_strength", 30.0)      # 最低趋势强度阈值

    def check_daily_loss(self, today_pnl: float, account_balance: float) -> Tuple[bool, Optional[str]]:
        """
        检查单日亏损是否触发熔断
        Returns: (passed, reason_if_failed)
        """
        if account_balance <= 0:
            return False, "账户余额为0"

        loss_pct = abs(today_pnl) / account_balance * 100

        if today_pnl < 0 and loss_pct >= self.max_daily_loss_pct:
            return False, f"单日亏损 {loss_pct:.2f}% 超过阈值 {self.max_daily_loss_pct}%，熔断触发"

        return True, None

    def check_drawdown(self, peak_balance: float, current_balance: float) -> Tuple[bool, Optional[str]]:
        """
        检查总回撤是否触发熔断
        Returns: (passed, reason_if_failed)
        """
        if peak_balance <= 0:
            return False, "历史最高余额无效"

        drawdown_pct = (peak_balance - current_balance) / peak_balance * 100

        if drawdown_pct >= self.max_drawdown_pct:
            return False, f"总回撤 {drawdown_pct:.2f}% 超过阈值 {self.max_drawdown_pct}%，熔断触发"

        return True, None

    def check_consecutive_losses(self, loss_streak: int, last_trade_time: Optional[datetime]) -> Tuple[bool, Optional[str], Optional[datetime]]:
        """
        检查连续亏损是否触发冷却期
        Returns: (can_trade, reason_if_blocked, resume_time)
        """
        now = datetime.now()

        if loss_streak >= 3:
            # 3连亏 → 暂停1天
            resume_time = last_trade_time + timedelta(hours=self.cooldown_3_losses_hours) if last_trade_time else now + timedelta(hours=self.cooldown_3_losses_hours)
            if now < resume_time:
                return False, f"连续亏损 {loss_streak} 笔，暂停交易至 {resume_time.strftime('%H:%M')}", resume_time

        elif loss_streak >= 2:
            # 2连亏 → 暂停1小时
            resume_time = last_trade_time + timedelta(hours=self.cooldown_2_losses_hours) if last_trade_time else now + timedelta(hours=self.cooldown_2_losses_hours)
            if now < resume_time:
                return False, f"连续亏损 {loss_streak} 笔，暂停交易至 {resume_time.strftime('%H:%M')}", resume_time

        return True, None, None

    def get_position_recovery_multiplier(self, loss_streak: int, time_since_last_trade: float) -> float:
        """
        获取渐进式仓位恢复系数

        机制：
        - 连亏后不立即恢复全仓位
        - 而是逐步恢复：25% → 50% → 75% → 100%
        - 恢复速度取决于冷却时间是否已过

        Args:
            loss_streak: 连续亏损次数
            time_since_last_trade: 距离上次交易的小时数

        Returns:
            仓位恢复系数 (0.0 - 1.0)
        """
        # 无亏损，全仓位
        if loss_streak == 0:
            return 1.0

        # 单亏：直接100%仓位，不需要冷却
        if loss_streak == 1:
            return 1.0

        # 2连亏：冷却1小时，分阶段恢复
        if loss_streak == 2:
            if time_since_last_trade < self.cooldown_2_losses_hours:
                return 0.0  # 冷却中
            elif time_since_last_trade < self.cooldown_2_losses_hours * 2:
                return 0.5  # 恢复50%
            else:
                return 1.0  # 完全恢复

        # 3连亏：冷却24小时，分阶段恢复
        if loss_streak == 3:
            if time_since_last_trade < self.cooldown_3_losses_hours:
                return 0.0  # 冷却中
            elif time_since_last_trade < self.cooldown_3_losses_hours * 1.5:
                return 0.25  # 0-36小时：25%
            elif time_since_last_trade < self.cooldown_3_losses_hours * 2:
                return 0.5  # 36-48小时：50%
            elif time_since_last_trade < self.cooldown_3_losses_hours * 3:
                return 0.75  # 48-72小时：75%
            else:
                return 1.0  # 完全恢复

        # 4连亏及以上：最多恢复75%
        if loss_streak >= 4:
            if time_since_last_trade < self.cooldown_3_losses_hours:
                return 0.0  # 冷却中
            elif time_since_last_trade < self.cooldown_3_losses_hours * 2:
                return 0.25
            elif time_since_last_trade < self.cooldown_3_losses_hours * 3:
                return 0.5
            else:
                return 0.75  # 最多恢复75%

        return 1.0


# ============================================================
# Agent-R: 风控核心
# ============================================================

class AgentRisk:
    """
    风险管理Agent
    完整风控流程：熔断 → 趋势/置信度检查 → 杠杆 → 仓位 → 止损 → 止盈 → 审批
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.circuit_breaker = CircuitBreaker(self.config.get("circuit_breaker"))

        # 仓位配置
        self.base_position_pct: float = self.config.get("base_position_pct", 2.0)  # 基础仓位2%
        self.max_position_pct: float = self.config.get("max_position_pct", 15.0)   # 最大仓位15%

        # 止损配置
        self.min_stop_loss_pct: float = self.config.get("min_stop_loss_pct", 0.5)   # 最小止损0.5%
        self.max_stop_loss_pct: float = self.config.get("max_stop_loss_pct", 5.0)  # 最大止损5%
        self.atr_multiplier: int = self.config.get("atr_multiplier", 3)             # ATR倍数
        self.account_risk_pct: float = self.config.get("account_risk_pct", 1.0)    # 账户1%风险

        # 止盈配置
        self.min_risk_reward: float = self.config.get("min_risk_reward", 2.0)      # 最小RR=2.0

        # 趋势/置信度阈值（高频模式调低）
        self.min_confidence: float = self.config.get("min_confidence", 0.15)       # 最低置信度 0.15
        self.min_trend_strength: float = 15.0  # 最低趋势强度阈值（高频模式）

    # --------------------------------------------------------
    # 1. 杠杆计算
    # --------------------------------------------------------

    def calc_leverage(self, signal: Signal) -> Tuple[int, str]:
        """
        根据趋势强度和置信度计算杠杆
        规则:
        - 趋势强度 ≥ 60 + 置信度 ≥ 0.7 → 3x杠杆
        - 趋势强度 ≥ 40 + 置信度 ≥ 0.5 → 2x杠杆
        - 趋势强度 ≥ 20 + 置信度 ≥ 0.3 → 1x杠杆
        - 市场不明确 → 禁止开仓 (趋势强度 < 15)
        """
        trend = signal.trend_strength
        conf = signal.confidence

        # 市场不明确，禁止开仓（阈值与min_trend_strength一致）
        if trend < 15:
            return 0, "市场趋势不明确，禁止开仓"

        # 3x杠杆：强趋势 + 高置信度
        if trend >= 60 and conf >= 0.7:
            return 3, "强趋势+高置信度，3x杠杆"

        # 2x杠杆：中等趋势 + 中等置信度
        if trend >= 40 and conf >= 0.5:
            return 2, "中等趋势+中等置信度，2x杠杆"

        # 1x杠杆：弱趋势或低置信度
        return 1, "弱趋势或低置信度，1x杠杆"

    # --------------------------------------------------------
    # 2. 仓位计算
    # --------------------------------------------------------

    def calc_position_size(self, signal: Signal, account_balance: float, leverage: int,
                          recovery_multiplier: float = 1.0) -> float:
        """
        科学计算仓位百分比
        公式: 仓位% = 基础仓位 × 杠杆系数 × 置信度系数 × 恢复系数
        - 基础仓位 = 账户2%
        - 杠杆系数: 1x=1.0, 2x=1.5, 3x=1.8
        - 置信度系数: 0.5-1.0 线性
        - 恢复系数: 连亏后渐进恢复（0.25-1.0）
        - 最大仓位 ≤ 15%单币
        """
        if leverage == 0:
            return 0.0

        # 基础仓位
        base_pos = self.base_position_pct / 100.0  # 2%

        # 杠杆系数
        leverage_factor = {1: 1.0, 2: 1.5, 3: 1.8}.get(leverage, 1.0)

        # 置信度系数 (0.5-1.0 线性)
        confidence_factor = max(0.5, min(1.0, signal.confidence))

        # 计算仓位
        position_pct = base_pos * leverage_factor * confidence_factor * 100.0

        # 应用渐进式恢复系数
        position_pct *= recovery_multiplier

        # 限制最大仓位
        position_pct = min(position_pct, self.max_position_pct)

        return round(position_pct, 2)

    # --------------------------------------------------------
    # 3. 止损计算
    # --------------------------------------------------------

    def calc_stop_loss(self, signal: Signal, entry_price: float) -> float:
        """
        计算止损价
        规则:
        - 做多: 止损 = entry_price × (1 - 止损比例)
        - 做空: 止损 = entry_price × (1 + 止损比例)
        - 止损比例 = max(3xATR, 账户1%风险对应的价格波动)
        - 最小止损: entry_price × 0.5%
        - 最大止损: entry_price × 5%
        """
        if entry_price <= 0 or signal.atr <= 0:
            return 0.0

        # 3xATR 对应的止损比例
        atr_stop_pct = (signal.atr * self.atr_multiplier) / entry_price * 100.0

        # 账户1%风险对应的价格波动比例
        account_risk_pct = self.account_risk_pct

        # 取较大值
        stop_pct = max(atr_stop_pct, account_risk_pct)

        # 限制范围
        stop_pct = max(self.min_stop_loss_pct, min(self.max_stop_loss_pct, stop_pct))

        # 计算止损价
        if signal.direction == "long":
            stop_price = entry_price * (1 - stop_pct / 100.0)
        else:  # short
            stop_price = entry_price * (1 + stop_pct / 100.0)

        return round(stop_price, 6)

    # --------------------------------------------------------
    # 4. 止盈计算
    # --------------------------------------------------------

    def calc_take_profit(self, signal: Signal, entry_price: float, stop_loss: float) -> Tuple[float, float]:
        """
        计算止盈价（赔率优先）
        规则:
        - 最小RR = 2.0（必须保证赔率）
        - 目标RR = max(2.0, signal['rr_ratio'])
        - 做多: TP = entry_price × (1 + 目标RR × 止损比例)
        - 做空: TP = entry_price × (1 - 目标RR × 止损比例)
        返回: (止盈价, 实际RR)
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0, 0.0

        # 计算止损比例
        if signal.direction == "long":
            stop_loss_pct = (entry_price - stop_loss) / entry_price * 100.0
        else:
            stop_loss_pct = (stop_loss - entry_price) / entry_price * 100.0

        if stop_loss_pct <= 0:
            return 0.0, 0.0

        # 目标RR
        target_rr = max(self.min_risk_reward, signal.rr_ratio)

        # 计算止盈价
        if signal.direction == "long":
            tp_price = entry_price * (1 + target_rr * stop_loss_pct / 100.0)
        else:  # short
            tp_price = entry_price * (1 - target_rr * stop_loss_pct / 100.0)

        # 计算实际RR
        actual_rr = (abs(tp_price - entry_price) / abs(stop_loss - entry_price)) if abs(stop_loss - entry_price) > 0 else 0.0

        return round(tp_price, 6), round(actual_rr, 2)

    # --------------------------------------------------------
    # 5. 完整风控流程
    # --------------------------------------------------------

    def process_signal(self, signal: Signal, account_state: AccountState) -> RiskApproval:
        """
        完整风控流程:
        1. 熔断检查（先检查能不能交易）
        2. 趋势强度检查（太弱就拒绝）
        3. 置信度检查（<0.5就拒绝）
        4. 计算杠杆
        5. 计算仓位
        6. 计算止损
        7. 计算止盈（保证最小RR=2.0）
        8. 汇总审批结果
        """
        warnings = []
        approval = RiskApproval(approved=False)

        # ---- 1. 熔断检查 ----
        # 1.1 单日亏损检查
        passed, reason = self.circuit_breaker.check_daily_loss(
            account_state.today_pnl, account_state.balance
        )
        if not passed:
            approval.rejection_reason = f"[熔断-单日亏损] {reason}"
            return approval

        # 1.2 总回撤检查
        passed, reason = self.circuit_breaker.check_drawdown(
            account_state.peak_balance, account_state.balance
        )
        if not passed:
            approval.rejection_reason = f"[熔断-总回撤] {reason}"
            return approval

        # 1.3 连续亏损检查
        passed, reason, resume_time = self.circuit_breaker.check_consecutive_losses(
            account_state.loss_streak, account_state.last_trade_time
        )
        if not passed:
            approval.rejection_reason = f"[熔断-连亏] {reason}"
            return approval

        # ---- 2. 趋势强度检查 ----
        if signal.trend_strength < self.min_trend_strength:
            approval.rejection_reason = f"趋势强度 {signal.trend_strength} 过低（<{self.min_trend_strength}），市场不明确"
            return approval

        if signal.trend_strength < 30:
            warnings.append(f"趋势强度偏弱: {signal.trend_strength}")

        # ---- 3. 置信度检查 ----
        if signal.confidence < self.min_confidence:
            approval.rejection_reason = f"置信度 {signal.confidence:.2f} 低于阈值 {self.min_confidence}"
            return approval

        # ---- 4. 计算杠杆 ----
        leverage, leverage_reason = self.calc_leverage(signal)

        if leverage == 0:
            approval.rejection_reason = f"[杠杆计算] {leverage_reason}"
            return approval

        warnings.append(f"杠杆决策: {leverage}x - {leverage_reason}")

        # ---- 5. 计算仓位（带渐进式恢复）----
        # 计算距离上次交易的时间
        time_since_last = 0.0
        if account_state.last_trade_time:
            time_since_last = (datetime.now() - account_state.last_trade_time).total_seconds() / 3600

        # 获取渐进式仓位恢复系数
        recovery_multiplier = self.circuit_breaker.get_position_recovery_multiplier(
            account_state.loss_streak, time_since_last
        )

        if recovery_multiplier < 1.0:
            warnings.append(f"渐进式仓位恢复: {recovery_multiplier*100:.0f}% (连亏{account_state.loss_streak}笔)")

        position_pct = self.calc_position_size(
            signal, account_state.balance, leverage, recovery_multiplier
        )

        if position_pct <= 0:
            approval.rejection_reason = "仓位计算结果为0，禁止开仓"
            return approval

        # ---- 6. 计算止损 ----
        stop_loss = self.calc_stop_loss(signal, signal.entry_price)

        if stop_loss <= 0:
            approval.rejection_reason = "止损价计算失败"
            return approval

        # ---- 7. 计算止盈 ----
        take_profit, actual_rr = self.calc_take_profit(signal, signal.entry_price, stop_loss)

        if take_profit <= 0:
            approval.rejection_reason = "止盈价计算失败"
            return approval

        # ---- 8. 汇总审批结果 ----
        approval.approved = True
        approval.leverage = leverage
        approval.position_size_pct = position_pct
        approval.stop_loss = stop_loss
        approval.take_profit = take_profit
        approval.risk_reward = actual_rr
        approval.warnings = warnings
        approval.modified_signal = Signal(
            symbol=signal.symbol,
            direction=signal.direction,
            trend_strength=signal.trend_strength,
            confidence=signal.confidence,
            rr_ratio=signal.rr_ratio,
            atr=signal.atr,
            entry_price=signal.entry_price,
            event_impact=signal.event_impact,
            timestamp=signal.timestamp
        )

        return approval

    # --------------------------------------------------------
    # 便捷方法
    # --------------------------------------------------------

    def approve_signal(self, signal: Signal, account_state: AccountState) -> RiskApproval:
        """最终审批交易信号的入口方法"""
        return self.process_signal(signal, account_state)

    def get_risk_metrics(self, entry_price: float, stop_loss: float, position_pct: float, account_balance: float) -> Dict[str, float]:
        """
        计算风险指标（用于日志和监控）
        """
        position_value = account_balance * (position_pct / 100.0)
        risk_amount = abs(entry_price - stop_loss) * (position_value / entry_price)
        risk_pct = (risk_amount / account_balance) * 100.0 if account_balance > 0 else 0.0

        return {
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_pct": round(risk_pct, 2),
            "account_balance": round(account_balance, 2),
            "position_pct": position_pct
        }


# ============================================================
# 模拟测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Agent-R: 风险与仓位管理Agent - 单元测试")
    print("=" * 60)

    # 初始化Agent
    agent = AgentRisk()

    # 模拟账户状态
    account = AccountState(
        balance=100000.0,
        peak_balance=110000.0,
        today_pnl=-2000.0,
        today_trades=3,
        loss_streak=0,
        last_trade_time=None
    )

    # 模拟信号1: 强趋势信号
    print("\n[测试1] 强趋势信号")
    signal1 = Signal(
        symbol="BTC",
        direction="long",
        trend_strength=85,
        confidence=0.85,
        rr_ratio=2.5,
        atr=150.0,
        entry_price=65000.0
    )
    result1 = agent.process_signal(signal1, account)
    print(f"  审批结果: {'✅ 批准' if result1.approved else '❌ 拒绝'}")
    if result1.approved:
        print(f"  杠杆: {result1.leverage}x")
        print(f"  仓位: {result1.position_size_pct}%")
        print(f"  止损: ${result1.stop_loss:,.2f}")
        print(f"  止盈: ${result1.take_profit:,.2f}")
        print(f"  风险回报: {result1.risk_reward}")
        print(f"  警告: {result1.warnings}")
    else:
        print(f"  拒绝原因: {result1.rejection_reason}")

    # 模拟信号2: 中等趋势信号
    print("\n[测试2] 中等趋势信号")
    signal2 = Signal(
        symbol="ETH",
        direction="short",
        trend_strength=55,
        confidence=0.65,
        rr_ratio=2.0,
        atr=50.0,
        entry_price=3500.0
    )
    result2 = agent.process_signal(signal2, account)
    print(f"  审批结果: {'✅ 批准' if result2.approved else '❌ 拒绝'}")
    if result2.approved:
        print(f"  杠杆: {result2.leverage}x")
        print(f"  仓位: {result2.position_size_pct}%")
        print(f"  止损: ${result2.stop_loss:,.2f}")
        print(f"  止盈: ${result2.take_profit:,.2f}")
        print(f"  风险回报: {result2.risk_reward}")
    else:
        print(f"  拒绝原因: {result2.rejection_reason}")

    # 模拟信号3: 弱趋势信号（应被拒绝）
    print("\n[测试3] 弱趋势信号（应被拒绝）")
    signal3 = Signal(
        symbol="SOL",
        direction="long",
        trend_strength=25,
        confidence=0.4,
        rr_ratio=1.5,
        atr=10.0,
        entry_price=150.0
    )
    result3 = agent.process_signal(signal3, account)
    print(f"  审批结果: {'✅ 批准' if result3.approved else '❌ 拒绝'}")
    if not result3.approved:
        print(f"  拒绝原因: {result3.rejection_reason}")

    # 模拟熔断测试: 单日亏损过大
    print("\n[测试4] 熔断-单日亏损过大")
    account_high_loss = AccountState(
        balance=100000.0,
        peak_balance=110000.0,
        today_pnl=-6000.0,  # 6%亏损
        today_trades=5,
        loss_streak=0
    )
    result4 = agent.process_signal(signal1, account_high_loss)
    print(f"  审批结果: {'✅ 批准' if result4.approved else '❌ 拒绝'}")
    if not result4.approved:
        print(f"  拒绝原因: {result4.rejection_reason}")

    # 熔断测试: 3连亏
    print("\n[测试5] 熔断-3连亏")
    account_3loss = AccountState(
        balance=100000.0,
        peak_balance=110000.0,
        today_pnl=-1000.0,
        today_trades=3,
        loss_streak=3,
        last_trade_time=datetime.now() - timedelta(minutes=30)
    )
    result5 = agent.process_signal(signal1, account_3loss)
    print(f"  审批结果: {'✅ 批准' if result5.approved else '❌ 拒绝'}")
    if not result5.approved:
        print(f"  拒绝原因: {result5.rejection_reason}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
