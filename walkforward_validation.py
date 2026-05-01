#!/usr/bin/env python3
"""
Walk-Forward Validation — 验证TOP策略候选在样本外是否有效
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))
OUTPUT_DIR = PROJECT_DIR / "scan_results"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('wf_validation')

# ── 候选列表（从108币种扫描结果中提取） ──
# (coin, strategy, sharpe, wr)
CANDIDATES = [
    ("FIL", "TREND_RSI", 4.99, 0.57),
    ("AVAX", "TREND_RSI", 4.21, 0.55),
    ("CITY", "ADX_RSI", 3.79, 0.80),
    ("BREV", "EMA_CROSS", 3.63, 0.62),
    ("BNB", "TREND_RSI", 2.96, 0.70),
    ("AERGO", "ADX_RSI", 2.91, 0.47),
    ("ELF", "ADX_RSI", 2.58, 0.60),
    ("GRT", "TREND_RSI", 2.58, 0.29),
    ("DASH", "EMA_CROSS", 2.52, 0.70),
    ("APT", "TREND_RSI", 1.92, 0.60),
]

# ── 策略函数（与strategy_scanner.py一致） ──
def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = np.diff(closes[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains[-period:])
    avg_l = np.mean(losses[-period:])
    if avg_l == 0: return 100.0
    return 100.0 - (100.0 / (1.0 + avg_g / avg_l))

def calc_adx_full(highs, lows, closes, period=14):
    # 转换为numpy数组
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    if len(highs) < period * 2 + 1: return 20.0, 20.0, 20.0, 0
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    up = highs[1:] - highs[:-1]
    dn = lows[:-1] - lows[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0)
    atr = np.mean(tr[-period:])
    if atr == 0: return 20.0, 20.0, 20.0, 0
    pdi = 100 * np.mean(pdm[-period:]) / atr
    ndi = 100 * np.mean(ndm[-period:]) / atr
    dx = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0
    return dx, pdi, ndi, atr

def calc_ema(closes, period):
    if len(closes) < period: return closes[-1] if closes else 0
    arr = np.array(closes[-period*3:], dtype=float)
    alpha = 2.0 / (period + 1)
    ema = arr[0]
    for v in arr[1:]: ema = v * alpha + ema * (1 - alpha)
    return ema

# ── TREND_RSI 信号（Sharpe=4.99的关键策略） ──
def trend_rsi_signal(prices, highs, lows):
    """趋势跟踪+RSI回调入场"""
    if len(prices) < 250: return None, None  # direction, reason

    adx, pdi, ndi, atr = calc_adx_full(highs, lows, prices, 14)
    rsi = calc_rsi(prices, 14)
    cp = prices[-1]
    ema50 = calc_ema(prices, 50)
    ema200 = calc_ema(prices, 200) if len(prices) >= 200 else ema50

    # 多头: 价格>EMA200 + DI+主导 + RSI回调到超卖区
    if cp > ema200 and pdi > ndi and rsi < 45:
        return "long", f"TREND_LONG RSI={rsi:.0f} ADX={adx:.0f}"
    # 空头: 价格<EMA200 + DI-主导 + RSI反弹到超买区
    if cp < ema200 and ndi > pdi and rsi > 55:
        return "short", f"TREND_SHORT RSI={rsi:.0f} ADX={adx:.0f}"
    return None, None

# ── ADX_RSI 信号 ──
def adx_rsi_signal(prices, highs, lows):
    if len(prices) < 50: return None, None
    adx, pdi, ndi, atr = calc_adx_full(highs, lows, prices, 14)
    if adx < 20: return None, None
    rsi = calc_rsi(prices, 14)
    cp = prices[-1]
    if rsi < 30 and pdi > ndi:
        return "long", f"ADX+RSI LONG RSI={rsi:.0f}"
    if rsi > 70 and ndi > pdi:
        return "short", f"ADX+RSI SHORT RSI={rsi:.0f}"
    return None, None

# ── EMA_CROSS 信号 ──
def ema_cross_signal(prices, highs, lows):
    if len(prices) < 60: return None, None
    fast = calc_ema(prices, 10)
    slow = calc_ema(prices, 30)
    if len(prices) > 70:
        prev_f = calc_ema(prices[:-1], 10)
        prev_s = calc_ema(prices[:-1], 30)
        if prev_f <= prev_s and fast > slow:
            return "long", "EMA_GOLDEN_CROSS"
        if prev_f >= prev_s and fast < slow:
            return "short", "EMA_DEATH_CROSS"
    return None, None

SIGNAL_FUNCS = {
    "TREND_RSI": trend_rsi_signal,
    "ADX_RSI": adx_rsi_signal,
    "EMA_CROSS": ema_cross_signal,
}

# ── Walk-Forward 回测（精简版） ──
def wf_backtest(klines, signal_func, sl_pct=0.03, tp_pct=0.10, leverage=2):
    """简化的回测，返回交易列表"""
    trades = []
    balance = 100000
    peaks = [balance]

    n = len(klines)
    position = None

    for i in range(250, n):
        cp = klines[i]['close']
        ts = klines[i]['timestamp']

        # Exit check
        if position:
            p_entry = position['entry']
            p_dir = position['direction']
            p_sl = position['sl']
            p_tp = position['tp']
            p_entry_idx = position['entry_idx']
            p_entry_ts = position['entry_ts']

            # SL check
            if p_dir == "long" and klines[i]['low'] <= p_sl:
                exit_px = p_sl
                exit_reason = "SL"
            elif p_dir == "short" and klines[i]['high'] >= p_sl:
                exit_px = p_sl
                exit_reason = "SL"
            # TP check
            elif p_dir == "long" and klines[i]['high'] >= p_tp:
                exit_px = p_tp
                exit_reason = "TP"
            elif p_dir == "short" and klines[i]['low'] <= p_tp:
                exit_px = p_tp
                exit_reason = "TP"
            else:
                exit_px = None

            if exit_px is not None:
                if p_dir == "long":
                    pnl_pct = (exit_px - p_entry) / p_entry
                else:
                    pnl_pct = (p_entry - exit_px) / p_entry

                # 手续费
                commission = balance * 0.0005 * 2  # 开+平
                pnl = balance * pnl_pct * leverage - commission
                balance += pnl
                peaks.append(balance)

                trades.append({
                    'entry_time': p_entry_ts,
                    'exit_time': ts,
                    'direction': p_dir,
                    'entry_px': p_entry,
                    'exit_px': exit_px,
                    'pnl_pct': pnl_pct,
                    'pnl_abs': pnl,
                    'reason': exit_reason,
                    'open_reason': position.get('open_reason', ''),
                    'hold_hours': (ts - p_entry_ts) / 3600000,
                })
                position = None

        # Entry check
        if not position:
            prices = [k['close'] for k in klines[:i+1]]
            highs = [k['high'] for k in klines[:i+1]]
            lows = [k['low'] for k in klines[:i+1]]
            direction, reason = signal_func(prices, highs, lows)

            if direction:
                entry_px = cp
                if direction == "long":
                    sl = entry_px * (1 - sl_pct)
                    tp = entry_px * (1 + tp_pct)
                else:
                    sl = entry_px * (1 + sl_pct)
                    tp = entry_px * (1 - tp_pct)

                position = {
                    'entry': entry_px, 'direction': direction,
                    'sl': sl, 'tp': tp,
                    'entry_idx': i, 'entry_ts': ts,
                    'open_reason': reason,
                }

    # Compute stats
    if len(trades) < 5:
        return None

    pnls = [t['pnl_pct'] for t in trades]
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1)
    sharpe = (avg_pnl / std_pnl * np.sqrt(365.25 * 4 / 1)) if std_pnl > 0 else 0  # 4H bars
    wins = sum(1 for p in pnls if p > 0)
    total_ret = (balance - 100000) / 100000
    dd = max(peaks) - min(peaks)
    dd_pct = dd / max(peaks) * 100 if max(peaks) > 0 else 0

    return {
        'trades': len(trades),
        'win_rate': wins / len(pnls),
        'sharpe': sharpe,
        'total_return_pct': total_ret,
        'max_dd_pct': dd_pct,
        'trades': trades,
    }

# ── 主函数 ──
def run_walkforward():
    from strategy_scanner import load_coin_klines_4h

    results_summary = []

    for coin, strategy, orig_sharpe, orig_wr in CANDIDATES:
        print(f"\n{'='*60}")
        print(f"  {coin} | {strategy} (原: S={orig_sharpe:.2f} WR={orig_wr:.0%})")
        print(f"{'='*60}")

        klines = load_coin_klines_4h(coin)
        if len(klines) < 500:
            print(f"  ⏭️  数据不足: {len(klines)} bars")
            continue

        signal_func = SIGNAL_FUNCS.get(strategy)
        if not signal_func:
            print(f"  ⏭️  无信号函数: {strategy}")
            continue

        # 分两段验证: 80%训练/20%测试
        split = int(len(klines) * 0.8)
        train_data = klines[:split]
        test_data = klines[split:]

        print(f"  数据: {len(klines)} bars | 训练: {len(train_data)} | 测试: {len(test_data)}")

        # 训练集回测
        train_result = wf_backtest(train_data, signal_func)
        # 测试集回测
        test_result = wf_backtest(test_data, signal_func)

        if train_result and test_result:
            train_s = train_result['sharpe']
            test_s = test_result['sharpe']
            train_wr = train_result['win_rate']
            test_wr = test_result['win_rate']

            # Walk-Forward通过条件：测试集Sharpe>0且不过度退化
            wf_pass = test_s > 0 and test_s > train_s * 0.3

            print(f"  训练集: S={train_s:.2f} WR={train_wr:.0%} N={train_result['trades']}")
            print(f"  测试集: S={test_s:.2f} WR={test_wr:.0%} N={test_result['trades']}")
            print(f"  {'✅ WF通过' if wf_pass else '❌ WF未通过'} (退化比: {test_s/train_s if train_s != 0 else 0:.2f})")

            results_summary.append({
                'coin': coin, 'strategy': strategy,
                'train_sharpe': train_s, 'test_sharpe': test_s,
                'train_wr': train_wr, 'test_wr': test_wr,
                'wf_pass': wf_pass,
            })
        else:
            train_n = train_result['trades'] if train_result else 0
            test_n = test_result['trades'] if test_result else 0
            print(f"  ❌ 交易不足: 训练={train_n} 测试={test_n}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"  Walk-Forward 验证汇总")
    print(f"{'='*60}")
    print(f"{'币种':>8} {'策略':>12} {'训练S':>8} {'测试S':>8} {'训练WR':>8} {'测试WR':>8} {'结果':>8}")
    print('-' * 60)
    passed = 0
    for r in results_summary:
        status = '✅' if r['wf_pass'] else '❌'
        if r['wf_pass']: passed += 1
        print(f"{r['coin']:>8} {r['strategy']:>12} {r['train_sharpe']:>8.2f} {r['test_sharpe']:>8.2f} {r['train_wr']:>7.0%} {r['test_wr']:>7.0%} {status:>8}")
    print('-' * 60)
    print(f"  Walk-Forward通过率: {passed}/{len(results_summary)}")

    return results_summary


if __name__ == '__main__':
    t0 = time.time()
    results = run_walkforward()
    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed/60:.1f} 分钟")
