"""
Technical Indicators Module
==========================
Extracted from miracle_kronos.py (Lines 182-298, 1153-1172)
Pure calculation functions - no side effects, no file I/O.
"""
from __future__ import annotations

import numpy as np


def calc_rsi(prices, period=14):
    """RSI with Wilder's Smoothing (same as core/factor_calculations.py)"""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    alpha = 1.0 / period
    for i in range(period, len(gains)):
        avg_gain = avg_gain + alpha * (gains[i] - avg_gain)
        avg_loss = avg_loss + alpha * (losses[i] - avg_loss)
    if avg_loss == 0:
        if avg_gain == 0:
            return 50.0  # 无涨跌 = 中性
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_adx(highs, lows, closes, period=14):
    """
    Calculate ADX using Wilder's smoothing (recursive EMA equivalent).
    Unlike simple SMA which re-averages each window, Wilder's smoothing
    uses: smoothed = prev * (period-1)/period + current/period

    This produces smoother, less reactive ADX values.
    """
    if len(closes) < period * 2 + 1:
        return 20.0, 20.0, 20.0

    # Step 1: Calculate raw TR and DM values
    trs = []
    dm_plus = []
    dm_minus = []
    for i in range(1, len(closes)):
        h, lo = highs[i], lows[i]
        prev_c = closes[i - 1]
        tr = max(h - lo, abs(h - prev_c), abs(lo - prev_c))
        trs.append(tr)
        dm_plus.append(max(h - highs[i - 1], 0))
        dm_minus.append(max(lows[i - 1] - lo, 0))

    if len(trs) < period:
        return 20.0, 20.0, 20.0

    # Step 2: Initialize smoothed ATR, DI+, DI- with SMA of first 'period' values
    atr = sum(trs[:period]) / period
    if atr == 0:
        return 20.0, 20.0, 20.0
    di_plus = sum(dm_plus[:period]) / atr
    di_minus = sum(dm_minus[:period]) / atr

    # Step 3: Apply Wilder's smoothing for remaining bars
    dx_values = []
    for i in range(period, len(trs)):
        tr = trs[i]
        dmp = dm_plus[i]
        dmm = dm_minus[i]

        # Wilder's smoothing: new = prev * (period-1)/period + current/period
        atr = (atr * (period - 1) + tr) / period
        di_plus = (di_plus * (period - 1) + dmp / atr * 100) / period if atr > 0 else 0
        di_minus = (di_minus * (period - 1) + dmm / atr * 100) / period if atr > 0 else 0

        if (di_plus + di_minus) > 0:
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        else:
            dx = 0
        dx_values.append(dx)

    if len(dx_values) < period:
        return 20.0, 20.0, 20.0

    # Step 4: ADX is Wilder smoothed DX
    adx = sum(dx_values[:period]) / period
    if len(dx_values) > period:
        for dx in dx_values[period:]:
            adx = (adx * (period - 1) + dx) / period

    # Return DI+, DI-, ADX
    return di_plus, di_minus, adx

def calc_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0
    # 正向计算EMA (从旧到新)
    alpha_f = 2 / (fast + 1)
    alpha_s = 2 / (slow + 1)
    alpha_sig = 2 / (signal + 1)
    ema_f = prices[0]
    ema_s = prices[0]
    ema_macd = 0.0
    macd_values = []
    for i, p in enumerate(prices):
        ema_f = p * alpha_f + ema_f * (1 - alpha_f)
        ema_s = p * alpha_s + ema_s * (1 - alpha_s)
        m = ema_f - ema_s
        macd_values.append(m)
        if i == 0:
            ema_macd = m
        else:
            ema_macd = m * alpha_sig + ema_macd * (1 - alpha_sig)
    macd = macd_values[-1]
    signal_line = ema_macd
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calc_bollinger(prices, period=20, mult=2):
    if len(prices) < period:
        return 50.0, 50.0, 50.0
    recent = prices[-period:]
    sma = sum(recent) / period
    std = (sum((p - sma) ** 2 for p in recent) / period) ** 0.5
    upper = sma + mult * std
    lower = sma - mult * std
    pos = (prices[-1] - lower) / (upper - lower) * 100 if (upper - lower) > 0 else 50.0
    return upper, lower, pos


def calc_atr(highs, lows, closes, period=14):
    """计算ATR (Average True Range)"""
    if len(closes) < period * 2 + 1:
        return 0.0

    trs = []
    for i in range(1, len(closes)):
        h, lo = highs[i], lows[i]
        prev_c = closes[i - 1]
        tr = max(h - lo, abs(h - prev_c), abs(lo - prev_c))
        trs.append(tr)

    if len(trs) < period:
        return 0.0

    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr

