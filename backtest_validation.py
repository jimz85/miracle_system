#!/usr/bin/env python3
"""
Miracle 1.0.1 - 回测验证脚本
==============================
P0测试: DOGE/ETH/BTC 4H RSI均值回归策略
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Add project root to path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from autoresearch.data_loader import load_timeframe_data
from backtest.engine import BacktestEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("validation")


# ============================================================
# Signal function: RSI Mean Reversion
# ============================================================
def rsi_mean_reversion(params: dict):
    """Factory for RSI mean reversion signal function.

    params:
        rsi_period: int (default 14)
        oversold: float (default 30)
        overbought: float (default 70)
        sl_pct: float (default 0.02)
        tp_pct: float (default 0.06)
        leverage: float (default 2)
        direction: str ('both', 'long_only', 'short_only') (default 'both')
    """
    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    sl_pct = params.get("sl_pct", 0.02)
    tp_pct = params.get("tp_pct", 0.06)
    leverage = params.get("leverage", 2.0)
    direction = params.get("direction", "both")

    def signal_func(prices, highs, lows, idx):
        if len(prices) < rsi_period + 1:
            return None

        # Compute RSI
        recent = prices[-(rsi_period + 1):]
        gains = []
        losses = []
        for i in range(1, len(recent)):
            diff = recent[i] - recent[i - 1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-diff)

        avg_gain = sum(gains) / rsi_period
        avg_loss = sum(losses) / rsi_period
        if avg_loss < 1e-10:
            rsi = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        entry_price = prices[-1]

        # Long signal: RSI oversold
        if rsi < oversold and direction in ("both", "long_only"):
            return {
                "direction": "long",
                "entry_price": entry_price,
                "stop_loss": entry_price * (1 - sl_pct),
                "take_profit": entry_price * (1 + tp_pct),
                "leverage": leverage,
                "factors": {"rsi": rsi},
                "confidence": max(0, (oversold - rsi) / oversold),
            }

        # Short signal: RSI overbought
        if rsi > overbought and direction in ("both", "short_only"):
            return {
                "direction": "short",
                "entry_price": entry_price,
                "stop_loss": entry_price * (1 + sl_pct),
                "take_profit": entry_price * (1 - tp_pct),
                "leverage": leverage,
                "factors": {"rsi": rsi},
                "confidence": max(0, (rsi - overbought) / (100 - overbought)),
            }

        return None

    return signal_func


# ============================================================
# Run single backtest
# ============================================================
def run_backtest(symbol: str, df: pd.DataFrame, params: dict,
                 engine_config: dict = None) -> dict:
    """Run backtest on dataframe using RSI mean reversion."""
    if engine_config is None:
        engine_config = {
            "initial_balance": 100000,
            "taker_commission_rate": 0.0005,
            "maker_commission_rate": 0.0002,
            "slippage_rate": 0.0002,
            "funding_rate": 0.0,
        }

    # Convert DataFrame to klines list
    klines = []
    for _, row in df.iterrows():
        ts = row["datetime_utc"]
        if isinstance(ts, pd.Timestamp):
            ts_ms = int(ts.timestamp() * 1000)
        elif isinstance(ts, (int, float)):
            ts_ms = int(ts)
        else:
            ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)

        klines.append({
            "timestamp": ts_ms,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        })

    # Create engine
    engine = BacktestEngine(engine_config)
    engine.load_klines(symbol, klines)

    # Get signal function
    signal_func = rsi_mean_reversion(params)

    # Run
    success, result = engine.run(signal_func, min_trades=5)

    if not success:
        return {
            "success": False,
            "error": result.get("error", "Unknown error"),
            "trades": result.get("trades", []),
        }

    stats = result["stats"]
    trades = result["trades"]

    # Compute additional metrics for trade splits
    long_trades = [t for t in trades if t["direction"] == "long"]
    short_trades = [t for t in trades if t["direction"] == "short"]

    long_sharpe = compute_trade_sharpe(long_trades)
    short_sharpe = compute_trade_sharpe(short_trades)

    return {
        "success": True,
        "stats": {
            "total_trades": stats["total_trades"],
            "winning_trades": stats["winning_trades"],
            "losing_trades": stats["losing_trades"],
            "win_rate": stats["win_rate"],
            "total_pnl_pct": stats["total_pnl_pct"],
            "sharpe_ratio": stats["sharpe_ratio"],
            "sortino_ratio": stats["sortino_ratio"],
            "max_drawdown_pct": stats["max_drawdown_pct"],
            "avg_hold_hours": stats["avg_hold_hours"],
            "avg_trades_per_day": stats["avg_trades_per_day"],
            "total_pnl": stats["total_pnl"],
        },
        "long_sharpe": long_sharpe,
        "short_sharpe": short_sharpe,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "equity_curve": result["equity_curve"],
    }


def compute_trade_sharpe(trades: list) -> float:
    """Compute Sharpe ratio from a list of trade pnl_pct values."""
    if len(trades) < 3:
        return 0.0
    pnl_pcts = np.array([t["pnl_pct"] for t in trades]) / 100.0  # convert % to ratio
    mean_pnl = pnl_pcts.mean()
    std_pnl = pnl_pcts.std()
    if std_pnl < 1e-10:
        return 0.0
    # Annualize approx: treat avg trade return as per-bar
    # For 4H bars: ~2190 bars/year (4H * 6/day * 365)
    bars_per_year = 2190  # 4H bars per year
    return (mean_pnl / std_pnl) * np.sqrt(bars_per_year / len(pnl_pcts))


def format_pct(val: float) -> str:
    return f"{val:+.2f}%" if abs(val) < 1000 else f"{val:+.0f}%"


def run_validation(coin: str, df: pd.DataFrame):
    """Run all P0 tests for a single coin."""
    # Time range info
    start_dt = df["datetime_utc"].min()
    end_dt = df["datetime_utc"].max()
    if isinstance(start_dt, pd.Timestamp):
        start_str = start_dt.strftime("%Y-%m")
        end_str = end_dt.strftime("%Y-%m")
    else:
        start_str = str(start_dt)[:7]
        end_str = str(end_dt)[:7]

    print(f"\n=== {coin} 4H RSI均值回归 ===")
    print(f"数据: {start_str} 到 {end_str} ({len(df):,}根K线)")

    # --- P0_a: Both directions (standard) ---
    result = run_backtest(coin, df, {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "sl_pct": 0.02,
        "tp_pct": 0.06,
        "leverage": 2.0,
        "direction": "both",
    })

    if not result["success"]:
        print(f"  回测失败: {result.get('error', 'unknown')}")
        return

    s = result["stats"]
    print(f"  总交易数: {s['total_trades']} | 胜率: {s['win_rate']*100:.1f}% | Sharpe: {s['sharpe_ratio']:.2f}")
    print(f"  总收益: {format_pct(s['total_pnl_pct'])} | 最大回撤: {format_pct(-s['max_drawdown_pct'])}")

    # --- P0_c: Long-only and Short-only ---
    long_result = run_backtest(coin, df, {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "sl_pct": 0.02,
        "tp_pct": 0.06,
        "leverage": 2.0,
        "direction": "long_only",
    })

    short_result = run_backtest(coin, df, {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "sl_pct": 0.02,
        "tp_pct": 0.06,
        "leverage": 2.0,
        "direction": "short_only",
    })

    long_sharpe = long_result["stats"]["sharpe_ratio"] if long_result["success"] else 0.0
    short_sharpe = short_result["stats"]["sharpe_ratio"] if short_result["success"] else 0.0

    print(f"  LONG单独: Sharpe {long_sharpe:.2f} ({result['n_long']}笔) | SHORT单独: Sharpe {short_sharpe:.2f} ({result['n_short']}笔)")

    # Detailed trade breakdown
    long_win_rate = long_result["stats"]["win_rate"] * 100 if long_result["success"] else 0
    short_win_rate = short_result["stats"]["win_rate"] * 100 if short_result["success"] else 0
    long_pnl = long_result["stats"]["total_pnl_pct"] if long_result["success"] else 0
    short_pnl = short_result["stats"]["total_pnl_pct"] if short_result["success"] else 0

    print(f"  LONG: WinRate {long_win_rate:.1f}% | PnL {format_pct(long_pnl)}")
    print(f"  SHORT: WinRate {short_win_rate:.1f}% | PnL {format_pct(short_pnl)}")

    return result


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Miracle 回测验证 | RSI均值回归(4H)")
    print("参数: RSI(14)<30做多/>70做空, 2%SL, 6%TP, 2x杠杆")
    print("=" * 60)

    coins = ["DOGE", "ETH", "BTC"]
    all_results = {}

    for coin in coins:
        print(f"\n{'─' * 50}")
        print(f"加载 {coin} 4H数据...")
        df = load_timeframe_data(coin, "4h")

        if df is None or len(df) < 200:
            print(f"  ⚠  {coin} 数据加载失败或不足 ({len(df) if df is not None else 0}行)")
            continue

        print(f"  ✓ 加载完成: {len(df):,} 根K线 ({df['datetime_utc'].min()} ~ {df['datetime_utc'].max()})")
        result = run_validation(coin, df)
        if result:
            all_results[coin] = result

    # Summary table
    print(f"\n\n{'=' * 60}")
    print("汇总")
    print(f"{'=' * 60}")
    print(f"{'币种':<6} {'总交易':<8} {'胜率':<8} {'Sharpe':<8} {'收益%':<10} {'最大回撤':<10} {'L_Sharpe':<8} {'S_Sharpe':<8}")
    print(f"{'─' * 60}")
    for coin in coins:
        if coin not in all_results:
            print(f"{coin:<6} {'FAIL':<8}")
            continue
        r = all_results[coin]
        if not r["success"]:
            print(f"{coin:<6} {'FAIL':<8}")
            continue
        s = r["stats"]
        print(f"{coin:<6} {s['total_trades']:<8} {s['win_rate']*100:<8.1f} {s['sharpe_ratio']:<8.2f} "
              f"{s['total_pnl_pct']:<+10.2f} {s['max_drawdown_pct']:<10.2f} "
              f"{r['long_sharpe']:<8.2f} {r['short_sharpe']:<8.2f}")

    print(f"\n{'=' * 60}")
    print("回测验证完成")


if __name__ == "__main__":
    main()
