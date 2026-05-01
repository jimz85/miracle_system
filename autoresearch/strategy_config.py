"""
Strategy Config - 所有可参数量化策略实现

每个实验可以修改这些参数，backtest_engine根据参数执行回测。
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class StrategyConfig:
    """策略参数配置 - Agent可以修改的所有参数"""

    # ===== 信号生成层 =====
    # RSI 条件
    rsi_period: int = 14
    rsi_oversold: float = 35.0  # 做多阈值
    rsi_overbought: float = 65.0  # 做空阈值
    rsi_filter_enabled: bool = True  # 是否启用RSI过滤

    # ADX 条件
    adx_period: int = 14
    adx_threshold: float = 15.0  # ADX > 此值认为有趋势
    adx_filter_enabled: bool = True

    # 布林带
    bb_period: int = 20
    bb_std: float = 2.0
    bb_position_oversold: float = 20.0  # 布林位置 < 此值认为超卖
    bb_position_overbought: float = 80.0  # 布林位置 > 此值认为超买
    bb_filter_enabled: bool = False

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_filter_enabled: bool = False

    # 成交量
    vol_ratio_threshold: float = 1.2  # 放量阈值
    vol_filter_enabled: bool = True

    # EMA 趋势
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    ema_trend_enabled: bool = True  # 启用EMA趋势过滤

    # ===== 仓位管理 =====
    position_size_pct: float = 0.10  # 每笔仓位占总资金比例 (10%)
    max_positions: int = 3  # 最大同时持仓数
    allow_short: bool = True  # 允许做空

    # ===== 止损止盈 =====
    # ATR倍数模式 (优先使用)
    sl_atr_mult: float = 1.5  # 止损 ATR倍数
    tp1_atr_mult: float = 3.0  # TP1 ATR倍数 (平50%)
    tp2_atr_mult: float = 6.0  # TP2 ATR倍数 (平50%)
    use_atr_stops: bool = True

    # 固定百分比模式
    sl_pct: float = 0.02  # 2% 固定止损
    tp1_pct: float = 0.04  # 4% TP1 (平50%)
    tp2_pct: float = 0.08  # 8% TP2 (平50%)

    # TP1后移动止损到成本+0.5%
    move_sl_to_cost_on_tp1: bool = True

    # ===== 过滤器 =====
    # 趋势过滤器: 基于BTC市场环境
    btc_trend_filter: bool = False  # 是否启用BTC趋势过滤
    btc_lookback: int = 20  # BTC均线周期

    # 波动率过滤器
    vol_filter_mode: str = "ratio"  # "ratio" | "absolute"
    vol_max_atr_pct: float = 0.05  # ATR超过行情的5%认为波动异常

    # ===== 入场确认 =====
    require_volume_confirm: bool = True  # 需要成交量确认
    min_vol_ratio: float = 0.8  # 最小成交量比

    # ===== 新增过滤器 ======
    # ATR Percentile (避免高波动入场)
    atr_percentile_max: float = 80.0  # 只在ATR百分位 < 此值时入场 (0-100)
    atr_percentile_period: int = 50  # ATR百分位计算周期
    atr_filter_enabled: bool = False  # 是否启用ATR百分位过滤

    # Momentum Score (RSI+MACD+布林组合评分)
    min_momentum_score: float = 0.0  # 入场最低动量评分 (0-100)
    momentum_filter_enabled: bool = False  # 是否启用动量过滤

    # 时段过滤 (避开低流动性时段 UTC)
    hour_filter_enabled: bool = False  # 是否启用时段过滤
    trading_hour_start: int = 7  # 允许交易的开始小时 (UTC)
    trading_hour_end: int = 23  # 允许交易的结束小时 (UTC)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyConfig":
        # 只保留已知的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def baseline(cls) -> "StrategyConfig":
        """标准基线参数（参考历史验证的最优参数）"""
        return cls()

    @classmethod
    def varied(cls, seed: int) -> "StrategyConfig":
        """低频高确定性 + 中频稳健双分支变异 — 2026-04-24注入
        40% 算力 → 极端趋势跟踪（tp2=4~8，探索大行情）
        60% 算力 → 中频稳健区（tp2=2.2~3.5，每日1~3次高质量信号）
        """
        rng = np.random.RandomState(seed)

        # === 40% 极端趋势跟踪分支 ===
        # 目标：捕捉单边大行情，容忍极低频
        if rng.random() < 0.4:
            c = cls.baseline()
            c.rsi_oversold     = round(min(55, 35 + rng.uniform(8, 15)), 1)
            c.rsi_overbought   = round(max(40, 65 - rng.uniform(8, 15)), 1)
            c.adx_threshold    = round(min(40, 15 + rng.uniform(10, 20)), 1)
            c.bb_position_oversold  = round(min(30, 20 + rng.uniform(-5, 5)), 1)
            c.bb_position_overbought = round(max(80, 80 - rng.uniform(-5, 5)), 1)
            c.position_size_pct = round(max(0.05, min(0.15, 0.10 + rng.uniform(-0.03, 0.03))), 3)
            c.sl_atr_mult      = round(max(2.0, 1.5 + rng.uniform(0.5, 1.5)), 1)
            c.tp1_atr_mult     = round(max(3.0, 2.0 + rng.uniform(0.5, 2.0)), 1)
            c.tp2_atr_mult     = round(max(4.0, 2.0 + rng.uniform(2.0, 4.0)), 1)   # 4~8x ATR
            c.vol_ratio_threshold = round(max(1.2, 0.8 + rng.uniform(0.3, 0.8)), 1)
            return c

        # === 60% 中频稳健分支 ===
        # 目标：每日1~3次信号，tp2=2.2~3.5，稳定波段
        c = cls.baseline()
        # RSI：中频区间（避免极端位置导致无信号）
        c.rsi_oversold   = round(max(30, min(45, 35 + rng.uniform(-3, 8))), 1)
        c.rsi_overbought = round(max(55, min(70, 65 + rng.uniform(-8, 3))), 1)
        # ADX：中频趋势门槛（不过高导致无信号，不过低导致噪音）
        c.adx_threshold  = round(max(15, min(28, 15 + rng.uniform(2, 10))), 1)
        # 布林带：中频位置（放宽避免无信号）
        c.bb_position_oversold   = round(max(15, min(35, 20 + rng.uniform(-5, 10))), 1)
        c.bb_position_overbought = round(max(65, min(88, 80 + rng.uniform(-10, 5))), 1)
        # 仓位：中频上限（不冒险）
        c.position_size_pct = round(max(0.05, min(0.12, 0.10 + rng.uniform(-0.03, 0.01))), 3)
        # 止损：中频（不给太大空间也不扫止损）
        c.sl_atr_mult  = round(max(1.5, min(3.0, 1.5 + rng.uniform(-0.2, 0.8))), 1)
        # 止盈：中频核心区 2.2~3.5
        c.tp1_atr_mult = round(max(2.0, min(3.5, 2.0 + rng.uniform(0.0, 1.5))), 1)
        c.tp2_atr_mult = round(max(2.2, min(3.5, 2.0 + rng.uniform(0.2, 1.5))), 1)  # 核心约束
        # 波动率：中等门槛
        c.vol_ratio_threshold = round(max(0.9, min(1.5, 0.8 + rng.uniform(-0.1, 0.5))), 1)
        return c


@dataclass
class Trade:
    """交易记录"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: Direction
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    exit_reason: str = ""  # "sl", "tp1", "tp2", "signal", "end"
    hold_bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time) if self.exit_time else None,
            "direction": self.direction.name,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "commission": self.commission,
            "exit_reason": self.exit_reason,
            "hold_bars": self.hold_bars,
        }


@dataclass
class BacktestResult:
    """回测结果"""
    config: StrategyConfig
    symbol: str
    total_return: float = 0.0  # 总收益率
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0  # 最大回撤
    win_rate: float = 0.0  # 胜率
    win_loss_ratio: float = 0.0  # 盈亏比
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_hold_bars: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "win_loss_ratio": self.win_loss_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_hold_bars": self.avg_hold_bars,
            "config": self.config.to_dict(),
        }


def compute_equity_curve(trades: List[Trade], initial_capital: float = 100000.0) -> List[float]:
    """从交易列表计算权益曲线"""
    equity = [initial_capital]
    for trade in trades:
        equity.append(equity[-1] + trade.pnl - trade.commission)
    return equity


def compute_sharpe(equity: List[float], risk_free_rate: float = 0.0) -> float:
    """
    计算年化Sharpe Ratio
    使用日收益重采样，避免小时级别噪声和年化假设错误
    """
    if len(equity) < 10:
        return float('nan')

    # 计算日收益
    import pandas as pd
    eq_series = pd.Series(equity)
    # 重采样到日线 (每 ~2000个点 ≈ 1H bars，2000点 resample 到约 83天日线)
    # 简化处理：直接按比例估算
    # 1H数据约2000 bars = 83天，每天24个bar
    bars_per_day = 24
    n_days = max(1, len(equity) // bars_per_day)

    # 生成日线权益
    daily_equity = []
    for d in range(n_days):
        bar_idx = min((d + 1) * bars_per_day, len(equity))
        daily_equity.append(equity[bar_idx - 1])
    daily_equity = daily_equity[:n_days]

    if len(daily_equity) < 5:
        return float('nan')

    returns = np.diff(daily_equity) / np.array(daily_equity[:-1])
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return float('nan')
    std_val = np.std(returns)
    if std_val == 0 or np.isnan(std_val):
        return float('nan')

    # 年化 (252 交易日)
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    return round(sharpe, 3)


def compute_max_drawdown(equity: List[float]) -> float:
    """计算最大回撤"""
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)
