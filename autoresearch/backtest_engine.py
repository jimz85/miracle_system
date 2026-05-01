"""
Backtest Engine - 事件驱动回测引擎

逐根K线处理，支持多头、空头、止损止盈追踪。
每次实验运行一个币种在一个Walk-Forward窗口上的回测。
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import sys
import os

from autoresearch.strategy_config import (
    StrategyConfig, BacktestResult, Trade, Direction,
    compute_equity_curve, compute_sharpe, compute_max_drawdown
)


# 手续费 (maker)
COMMISSION_RATE = 0.0004  # 0.04%


@dataclass
class Position:
    """持仓状态"""
    direction: Direction
    entry_price: float
    quantity: float
    entry_bar: int
    entry_time: pd.Timestamp  # 存储入场时间，避免回查df
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp1_triggered: bool = False
    tp2_triggered: bool = False
    sl_triggered: bool = False
    entry_commission: float = 0.0  # 入场手续费（开仓时扣除）

    def update_stop_loss(self, new_sl: float):
        """只允许更新到更好的价格"""
        if self.direction == Direction.LONG:
            if new_sl > self.stop_loss:
                self.stop_loss = new_sl
        else:
            if new_sl < self.stop_loss:
                self.stop_loss = new_sl


class BacktestEngine:
    """
    事件驱动回测引擎
    逐根K线处理，每个实验独立运行
    """

    def __init__(self, config: StrategyConfig, commission: float = COMMISSION_RATE):
        self.config = config
        self.commission = commission
        self._reset()

    def _reset(self):
        """重置回测状态"""
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [1.0]  # 从1开始，pnl百分比
        self.cumulative_return = 0.0
        self.current_bar = 0
        self._atr_percentile_cache: Dict[int, Any] = {}  # 按需ATR百分位缓存

    def run(self, df: pd.DataFrame, initial_capital: float = 100000.0) -> BacktestResult:
        """
        运行回测
        df: 包含 OHLCV + 指标的数据
        返回: BacktestResult
        """
        self._reset()
        self.initial_capital = initial_capital
        self._df = df  # 保存引用供ATR百分位按需计算
        c = self.config

        # 预热期 (指标计算需要前N根)
        warmup = max(c.rsi_period, c.adx_period, c.bb_period, 50)

        i = warmup
        while i < len(df):
            row = df.iloc[i]
            self.current_bar = i

            # 如果有持仓，先检查止损止盈
            if self.position is not None:
                self._check_exits(row)

            # 如果无持仓，检查入场信号
            if self.position is None:
                signal = self._check_entry(row)
                if signal != Direction.FLAT:
                    self._open_position(signal, row, df)

            # 记录权益
            self._record_equity(row)

            i += 1

        # 平仓最后持仓
        if self.position is not None:
            row = df.iloc[-1]
            self._close_position(row, "end")

        return self._build_result()

    def _check_exits(self, row: pd.Series):
        """检查止损止盈"""
        pos = self.position
        price = row["close"]
        high = row["high"]
        low = row["low"]

        if pos.direction == Direction.LONG:
            # 止损检查
            if price <= pos.stop_loss or low <= pos.stop_loss:
                pos.sl_triggered = True
                # 使用触发价格成交
                exit_price = min(pos.stop_loss, high if low <= pos.stop_loss else price)
                self._close_position(row, "sl", exit_price)
                return

            # TP1检查 (50%)
            if not pos.tp1_triggered and price >= pos.tp1:
                pos.tp1_triggered = True
                exit_price = pos.tp1
                self._close_position(row, "tp1", exit_price, quantity_ratio=0.5)

                # TP1后移动止损到成本+0.5%
                if self.config.move_sl_to_cost_on_tp1:
                    new_sl = pos.entry_price * (1 + 0.005)
                    pos.update_stop_loss(new_sl)
                return

            # TP2检查 (剩余50%)
            if pos.tp1_triggered and not pos.tp2_triggered and price >= pos.tp2:
                pos.tp2_triggered = True
                exit_price = pos.tp2
                self._close_position(row, "tp2", exit_price)
                return

        elif pos.direction == Direction.SHORT:
            # 止损
            if price >= pos.stop_loss or high >= pos.stop_loss:
                pos.sl_triggered = True
                exit_price = max(pos.stop_loss, low if high >= pos.stop_loss else price)
                self._close_position(row, "sl", exit_price)
                return

            # TP1
            if not pos.tp1_triggered and price <= pos.tp1:
                pos.tp1_triggered = True
                exit_price = pos.tp1
                self._close_position(row, "tp1", exit_price, quantity_ratio=0.5)

                if self.config.move_sl_to_cost_on_tp1:
                    new_sl = pos.entry_price * (1 - 0.005)
                    pos.update_stop_loss(new_sl)
                return

            # TP2
            if pos.tp1_triggered and not pos.tp2_triggered and price <= pos.tp2:
                pos.tp2_triggered = True
                exit_price = pos.tp2
                self._close_position(row, "tp2", exit_price)
                return

    def _check_entry(self, row: pd.Series) -> Direction:
        """检查入场信号"""
        c = self.config

        # RSI信号
        rsi = row.get("rsi_14", 50)
        if c.rsi_filter_enabled:
            if rsi < c.rsi_oversold:
                direction = Direction.LONG
            elif rsi > c.rsi_overbought:
                direction = Direction.SHORT if c.allow_short else Direction.FLAT
            else:
                return Direction.FLAT
        else:
            # 无RSI过滤时，用RSI超买超卖判断方向
            if rsi < c.rsi_oversold:
                direction = Direction.LONG
            elif rsi > c.rsi_overbought:
                direction = Direction.SHORT if c.allow_short else Direction.FLAT
            else:
                return Direction.FLAT

        # ADX趋势确认
        if c.adx_filter_enabled:
            adx = row.get("adx_14", 0)
            if adx < c.adx_threshold:
                return Direction.FLAT

        # 布林带过滤
        if c.bb_filter_enabled:
            bb_pos = row.get("bollinger_position", 50)
            if direction == Direction.LONG and bb_pos > c.bb_position_overbought:
                return Direction.FLAT
            if direction == Direction.SHORT and bb_pos < c.bb_position_oversold:
                return Direction.FLAT

        # MACD过滤
        if c.macd_filter_enabled:
            macd_hist = row.get("macd_histogram", 0)
            if direction == Direction.LONG and macd_hist < 0:
                return Direction.FLAT
            if direction == Direction.SHORT and macd_hist > 0:
                return Direction.FLAT

        # EMA趋势过滤
        if c.ema_trend_enabled:
            ema_fast = row.get("ema_9", 0)
            ema_slow = row.get("ema_21", 0)
            if direction == Direction.LONG and ema_fast < ema_slow:
                return Direction.FLAT
            if direction == Direction.SHORT and ema_fast > ema_slow:
                return Direction.FLAT

        # 成交量确认
        if c.require_volume_confirm:
            vol_ratio = row.get("volume_ratio", 1.0)
            if vol_ratio < c.min_vol_ratio:
                return Direction.FLAT

        # ATR Percentile 过滤 (避免高波动入场)
        if c.atr_filter_enabled:
            atr_pct_period = c.atr_percentile_period
            col = f"atr_pct_{atr_pct_period}"
            if col not in row.index:
                # 按需计算ATR百分位
                period = atr_pct_period
                atr_vals = self._atr_percentile_cache.get(period, None)
                if atr_vals is None:
                    # 计算全局ATR百分位序列（首次）
                    close = self._df["close"]
                    high = self._df["high"]
                    low = self._df["low"]
                    tr1 = high - low
                    tr2 = (high - close.shift(1)).abs()
                    tr3 = (low - close.shift(1)).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean()
                    atr_pct_series = atr.rolling(period).apply(
                        lambda x: (x < x[-1]).sum() / len(x) * 100, raw=True
                    )
                    self._atr_percentile_cache[period] = atr_pct_series
                    atr_vals = atr_pct_series
                cur_bar = self.current_bar
                atr_pct = atr_vals.iloc[cur_bar] if cur_bar < len(atr_vals) else 50
            else:
                atr_pct = row.get(col, 50)
            if atr_pct > c.atr_percentile_max:
                return Direction.FLAT

        # Momentum Score 过滤
        if c.momentum_filter_enabled:
            mom_score = row.get("momentum_score", 50)
            if mom_score < c.min_momentum_score:
                return Direction.FLAT

        # 时段过滤 (避开低流动性时段)
        if c.hour_filter_enabled:
            hour = row.get("hour", -1)
            if hour >= 0:
                if not (c.trading_hour_start <= hour <= c.trading_hour_end):
                    return Direction.FLAT

        return direction

    def _open_position(self, direction: Direction, row: pd.Series, df: pd.DataFrame):
        """开仓"""
        entry_price = row["close"]
        atr = row.get("atr_14", entry_price * 0.01)

        # 计算仓位数量
        risk_amount = self.initial_capital * self.config.position_size_pct
        if self.config.use_atr_stops:
            sl_dist = atr * self.config.sl_atr_mult
            sl_dist = max(sl_dist, entry_price * 0.005)  # 最小0.5%
            quantity = risk_amount / sl_dist
        else:
            sl_dist = entry_price * self.config.sl_pct
            quantity = risk_amount / sl_dist

        # 计算SL/TP
        if direction == Direction.LONG:
            sl = entry_price - sl_dist
            tp1 = entry_price + atr * self.config.tp1_atr_mult
            tp2 = entry_price + atr * self.config.tp2_atr_mult
        else:
            sl = entry_price + sl_dist
            tp1 = entry_price - atr * self.config.tp1_atr_mult
            tp2 = entry_price - atr * self.config.tp2_atr_mult

        # 估算手续费
        notional = entry_price * quantity
        commission_est = notional * self.commission

        self.position = Position(
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            entry_bar=self.current_bar,
            entry_time=row["datetime_utc"],
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            entry_commission=commission_est,  # ✅ 记录入场手续费
        )

    def _close_position(self, row: pd.Series, reason: str,
                         exit_price: Optional[float] = None, quantity_ratio: float = 1.0):
        """平仓"""
        if self.position is None:
            return

        pos = self.position
        price = exit_price if exit_price is not None else row["close"]
        qty = pos.quantity * quantity_ratio

        if pos.direction == Direction.LONG:
            pnl = (price - pos.entry_price) * qty
        else:
            pnl = (pos.entry_price - price) * qty

        commission = price * qty * self.commission
        entry_comm_due = pos.entry_commission * quantity_ratio  # 入场手续费按本次平仓比例分摊
        net_pnl = pnl - commission - entry_comm_due

        # ✅ 部分平仓后，剩余未平仓部分的入场手续费按比例减少
        if quantity_ratio < 1.0:
            pos.entry_commission = pos.entry_commission * (1.0 - quantity_ratio)

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=row["datetime_utc"],
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=qty,
            pnl=net_pnl,
            pnl_pct=net_pnl / (pos.entry_price * qty),
            commission=commission,
            exit_reason=reason,
            hold_bars=self.current_bar - pos.entry_bar,
        )
        self.trades.append(trade)

        # 更新权益
        self.cumulative_return += net_pnl / self.initial_capital

        if quantity_ratio >= 1.0:
            self.position = None
        else:
            # 部分平仓，剩余数量减少
            pos.quantity *= (1 - quantity_ratio)

    def _record_equity(self, row: pd.Series):
        """记录权益曲线点"""
        equity = 1.0 + self.cumulative_return
        if self.position is not None:
            # 标记为未实现盈亏
            pos = self.position
            if pos.direction == Direction.LONG:
                unrealized = (row["close"] - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - row["close"]) * pos.quantity
            equity += unrealized / self.initial_capital
        self.equity_curve.append(equity)

    def _build_result(self) -> BacktestResult:
        """构建回测结果"""
        trades = self.trades
        # === 最小交易笔数约束 (2026-04-24) ===
        # 样本量不足30笔时，Sharpe无统计显著性 → 强制NaN
        # 但仍需计算真实return，避免丢弃有效数据
        MIN_TRADE_COUNT = 30
        total_trades_count = len(trades)
        insufficient_trades = total_trades_count < MIN_TRADE_COUNT

        # 无交易时返回零值结果
        if not trades:
            return BacktestResult(
                config=self.config,
                symbol="",
                total_return=0.0,
                sharpe_ratio=float('nan'),
                max_drawdown=0.0,
                win_rate=0.0,
                win_loss_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                equity_curve=self.equity_curve,
                trades=[],
            )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_return = self.equity_curve[-1] - 1.0
        sharpe = float('nan') if insufficient_trades else compute_sharpe(self.equity_curve)
        max_dd = compute_max_drawdown(self.equity_curve)

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1
        wlr = avg_win / avg_loss if avg_loss > 0 else 0

        return BacktestResult(
            config=self.config,
            symbol="",
            total_return=round(total_return * 100, 2),  # 百分比
            sharpe_ratio=sharpe,
            max_drawdown=round(max_dd * 100, 2),  # 百分比
            win_rate=round(win_rate * 100, 1),
            win_loss_ratio=round(wlr, 2),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            avg_hold_bars=round(np.mean([t.hold_bars for t in trades]), 1),
            equity_curve=self.equity_curve,
            trades=trades,
        )


def run_walkforward(
    df: pd.DataFrame,
    config: StrategyConfig,
    n_windows: int = 8,
    train_ratio: float = 0.7,
    initial_capital: float = 100000.0,
    wf_mode: str = "expanding",
) -> BacktestResult:
    """
    Walk-Forward回测：在多个窗口上运行回测，取平均结果

    wf_mode:
      - "expanding": Expanding窗口，每个窗口从数据开头训练，测试集逐步前推（默认）
      - "rolling": 等宽窗口滑动
      - "rolling_recent": 只取最近n_windows个窗口
    """
    from data_loader import walkforward_split

    splits = walkforward_split(df, n_windows=n_windows, train_ratio=train_ratio, mode=wf_mode)
    results = []

    for train_df, test_df in splits:
        if len(test_df) < 50:
            continue

        # 在test window上回测
        engine = BacktestEngine(config)
        result = engine.run(test_df, initial_capital)
        results.append(result)

    if not results:
        return BacktestResult(config=config, symbol="", total_trades=0)

    # 汇总平均
    avg_return = np.mean([r.total_return for r in results])
    avg_sharpe = np.mean([r.sharpe_ratio for r in results])
    avg_dd = np.mean([r.max_drawdown for r in results])
    avg_wr = np.mean([r.win_rate for r in results])
    avg_wlr = np.mean([r.win_loss_ratio for r in results])
    total_trades = sum(r.total_trades for r in results)
    total_wins = sum(r.winning_trades for r in results)

    # 用第一个结果的config，但汇总指标
    summary = BacktestResult(
        config=config,
        symbol="",
        total_return=round(avg_return, 2),
        sharpe_ratio=round(avg_sharpe, 3),
        max_drawdown=round(avg_dd, 2),
        win_rate=round(avg_wr, 1),
        win_loss_ratio=round(avg_wlr, 2),
        total_trades=total_trades,
        winning_trades=total_wins,
        losing_trades=total_trades - total_wins,
    )

    return summary


def run_single_backtest(
    df: pd.DataFrame,
    config: StrategyConfig,
    initial_capital: float = 100000.0,
) -> BacktestResult:
    """单次回测"""
    engine = BacktestEngine(config)
    return engine.run(df, initial_capital)


if __name__ == "__main__":
    # 快速测试
    from data_loader import load_timeframe_data, compute_indicators

    print("=== Backtest Engine Smoke Test ===")
    df = load_timeframe_data("BTC", "1h")
    if df is not None:
        df = compute_indicators(df)
        print(f"Data: {len(df)} rows, {df['datetime_utc'].min()} ~ {df['datetime_utc'].max()}")

        config = StrategyConfig.baseline()
        result = run_single_backtest(df, config)
        print(f"\nResult:")
        print(f"  Total Return: {result.total_return:.2f}%")
        print(f"  Sharpe: {result.sharpe_ratio:.3f}")
        print(f"  Max DD: {result.max_drawdown:.2f}%")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  WLR: {result.win_loss_ratio:.2f}")
        print(f"  Total Trades: {result.total_trades}")
        if result.trades:
            print(f"\nLast 3 trades:")
            for t in result.trades[-3:]:
                print(f"  {t.direction.name}: entry={t.entry_price:.2f} exit={t.exit_price:.2f} pnl={t.pnl:.2f} ({t.pnl_pct*100:.1f}%) reason={t.exit_reason}")
