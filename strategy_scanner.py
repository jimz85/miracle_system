#!/usr/bin/env python3
"""
Strategy Scanner — 108币种自动策略发现
纯本地计算，不消耗API token
结果输出到 CSV，方便分析

Stage 1: 快速扫描所有币种 (4H, 5种基础策略)
Stage 2: 深度扫描优胜币种 (多时间框架+参数)
Stage 3: Walk-Forward验证
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 项目路径 ──
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('strategy_scanner')

# ── 桌面数据路径 ──
DATA_DIR = Path.home() / "Desktop" / "crypto_data_Pre5m"

# ── 输出路径 ──
OUTPUT_DIR = PROJECT_DIR / "scan_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 指标计算 ──
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calc_adx(highs, lows, closes, period=14):
    if len(highs) < period * 2 + 1:
        return 20.0
    h = np.array(highs[-(period*2+1):])
    l = np.array(lows[-(period*2+1):])
    c = np.array(closes[-(period*2+1):])
    tr = np.maximum(h[1:] - l[1:], 
                    np.maximum(np.abs(h[1:] - c[:-1]), 
                               np.abs(l[1:] - c[:-1])))
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    if atr == 0:
        return 20.0, 20.0, 20.0, atr
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    adx = np.mean([dx] * period)  # simplified smoothing
    return adx, plus_di, minus_di, atr

def calc_ema(closes, period):
    if len(closes) < period:
        return closes[-1] if closes else 0
    arr = np.array(closes[-period*3:], dtype=float)
    alpha = 2.0 / (period + 1)
    ema = arr[0]
    for v in arr[1:]:
        ema = v * alpha + ema * (1 - alpha)
    return ema

def calc_bb(closes, period=20, std=2.0):
    if len(closes) < period:
        return 0, 0, 0  # upper, middle, lower all 0
    arr = np.array(closes[-period:])
    mid = np.mean(arr)
    sd = np.std(arr, ddof=1)
    return mid + std * sd, mid, mid - std * sd

# ── 数据加载 ──
def list_available_coins() -> List[str]:
    """列出桌面数据目录中所有可用币种"""
    coins = []
    if not DATA_DIR.exists():
        logger.error(f"数据目录不存在: {DATA_DIR}")
        return coins
    for f in sorted(DATA_DIR.glob("*_USDT_5m_from_*.csv")):
        coin = f.name.split("_")[0]
        coins.append(coin)
    return coins

def load_coin_klines_4h(coin: str) -> List[Dict]:
    """加载币种5m数据并重采样到4H"""
    # 找文件
    files = list(DATA_DIR.glob(f"{coin}_USDT_5m_from_*.csv"))
    if not files:
        logger.warning(f"{coin}: 数据文件不存在")
        return []
    filepath = files[0]
    
    # 尝试缓存（pickle，不依赖pyarrow）
    cache_dir = PROJECT_DIR / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{coin}_4h.pkl"
    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                records = pickle.load(f)
            logger.debug(f"{coin}: 从缓存加载 {len(records)} 根4H K线")
            return records
        except Exception:
            pass
    
    # 加载CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"{coin}: 加载CSV失败 {e}")
        return []
    
    # 检查必要的列
    required = ['timestamp', 'open', 'high', 'low', 'close']
    if not all(c in df.columns for c in required):
        logger.warning(f"{coin}: 缺少必要列，现有列: {list(df.columns)}")
        return []
    
    # 时间戳处理
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('datetime')
    
    # 重命名vol→volume
    if 'vol' in df.columns:
        df = df.rename(columns={'vol': 'volume'})
    
    # 重采样到4H
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'
    use_cols = list(agg_dict.keys())
    df_4h = df[use_cols].resample('4h').agg(agg_dict)
    df_4h = df_4h.dropna()
    
    # 添加timestamp字段（ms毫秒）
    # NOTE: pandas DatetimeIndex.asi8 返回毫秒（新版pandas行为）
    # 用 view(np.int64) 获取同一单位
    df_4h['timestamp'] = df_4h.index.view(np.int64).copy()
    df_4h = df_4h.reset_index(drop=True)
    
    records = df_4h.to_dict('records')
    
    # 保存缓存
    try:
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(records, f)
    except Exception:
        pass
    
    logger.debug(f"{coin}: 从CSV加载 {len(records)} 根4H K线")
    return records

# ── 信号函数 ──
def make_rsi_mr_signal(rsi_period=14, rsi_oversold=30, rsi_overbought=70, 
                         sl_pct=0.02, tp_pct=0.06, leverage=2):
    """RSI均值回归信号"""
    def signal_func(prices, highs, lows, index):
        if len(prices) < rsi_period + 5:
            return None
        rsi = calc_rsi(prices, rsi_period)
        current_price = prices[-1]
        
        if rsi < rsi_oversold:
            return {
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': current_price * (1 - sl_pct),
                'take_profit': current_price * (1 + tp_pct),
                'leverage': leverage,
                'factors': {'rsi': rsi},
            }
        elif rsi > rsi_overbought:
            return {
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': current_price * (1 + sl_pct),
                'take_profit': current_price * (1 - tp_pct),
                'leverage': leverage,
                'factors': {'rsi': rsi},
            }
        return None
    return signal_func

def make_adx_rsi_signal(adx_min=20, rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                          sl_pct=0.03, tp_pct=0.08, leverage=2):
    """ADX趋势过滤+RSI入场"""
    def signal_func(prices, highs, lows, index):
        if len(prices) < max(adx_min * 2 + 1, rsi_period + 5):
            return None
        adx, plus_di, minus_di, atr = calc_adx(highs, lows, prices, 14)
        if adx < adx_min:
            return None
        rsi = calc_rsi(prices, rsi_period)
        current_price = prices[-1]
        
        # 趋势方向由DI决定
        trend_long = plus_di > minus_di
        trend_short = minus_di > plus_di
        
        if rsi < rsi_oversold and trend_long:
            return {
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': current_price * (1 - sl_pct),
                'take_profit': current_price * (1 + tp_pct),
                'leverage': leverage,
                'factors': {'adx': adx, 'rsi': rsi, 'di_plus': plus_di, 'di_minus': minus_di},
            }
        elif rsi > rsi_overbought and trend_short:
            return {
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': current_price * (1 + sl_pct),
                'take_profit': current_price * (1 - tp_pct),
                'leverage': leverage,
                'factors': {'adx': adx, 'rsi': rsi, 'di_plus': plus_di, 'di_minus': minus_di},
            }
        return None
    return signal_func

def make_ema_cross_signal(fast=5, slow=20, sl_pct=0.03, tp_pct=0.08, leverage=2):
    """EMA金叉/死叉趋势跟随"""
    def signal_func(prices, highs, lows, index):
        if len(prices) < slow + 10:
            return None
        ema_fast = calc_ema(prices, fast)
        ema_slow = calc_ema(prices, slow)
        current_price = prices[-1]
        
        # 用前一根确认交叉
        prev_fast = calc_ema(prices[:-1], fast) if len(prices) > slow + 5 else ema_fast
        prev_slow = calc_ema(prices[:-1], slow) if len(prices) > slow + 5 else ema_slow
        
        # 金叉：快线上穿慢线
        if prev_fast <= prev_slow and ema_fast > ema_slow:
            return {
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': current_price * (1 - sl_pct),
                'take_profit': current_price * (1 + tp_pct),
                'leverage': leverage,
                'factors': {'ema_fast': ema_fast, 'ema_slow': ema_slow},
            }
        # 死叉：快线下穿慢线
        elif prev_fast >= prev_slow and ema_fast < ema_slow:
            return {
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': current_price * (1 + sl_pct),
                'take_profit': current_price * (1 - tp_pct),
                'leverage': leverage,
                'factors': {'ema_fast': ema_fast, 'ema_slow': ema_slow},
            }
        return None
    return signal_func

def make_bb_mr_signal(bb_period=20, bb_std=2.0, sl_pct=0.02, tp_pct=0.06, leverage=2):
    """布林带均值回归"""
    def signal_func(prices, highs, lows, index):
        if len(prices) < bb_period + 5:
            return None
        upper, mid, lower = calc_bb(prices, bb_period, bb_std)
        current_price = prices[-1]
        
        if current_price <= lower:
            return {
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': current_price * (1 - sl_pct),
                'take_profit': current_price * (1 + tp_pct),
                'leverage': leverage,
                'factors': {'bb_upper': upper, 'bb_mid': mid, 'bb_lower': lower},
            }
        elif current_price >= upper:
            return {
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': current_price * (1 + sl_pct),
                'take_profit': current_price * (1 - tp_pct),
                'leverage': leverage,
                'factors': {'bb_upper': upper, 'bb_mid': mid, 'bb_lower': lower},
            }
        return None
    return signal_func

def make_trend_rsi_signal(adx_min=20, rsi_period=14, sl_pct=0.03, tp_pct=0.10, leverage=2):
    """趋势跟踪 + RSI回调入场（专业交易员常用）"""
    def signal_func(prices, highs, lows, index):
        if len(prices) < max(50, rsi_period + 5):
            return None
        adx, plus_di, minus_di, atr = calc_adx(highs, lows, prices, 14)
        if adx < adx_min:
            return None
        rsi = calc_rsi(prices, rsi_period)
        current_price = prices[-1]
        
        # EMA200作为长期趋势
        ema50 = calc_ema(prices, 50)
        ema200 = calc_ema(prices, 200) if len(prices) >= 200 else ema50
        
        # 多头趋势：价格在EMA200之上 + DI+主导
        if current_price > ema200 and plus_di > minus_di and rsi < 45:
            # 回调到趋势线附近做多
            return {
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': current_price * (1 - sl_pct),
                'take_profit': current_price * (1 + tp_pct),
                'leverage': leverage,
                'factors': {'adx': adx, 'rsi': rsi, 'ema50': ema50, 'ema200': ema200},
            }
        # 空头趋势：价格在EMA200之下 + DI-主导
        elif current_price < ema200 and minus_di > plus_di and rsi > 55:
            return {
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': current_price * (1 + sl_pct),
                'take_profit': current_price * (1 - tp_pct),
                'leverage': leverage,
                'factors': {'adx': adx, 'rsi': rsi, 'ema50': ema50, 'ema200': ema200},
            }
        return None
    return signal_func

# ── 策略扫描配置 ──
STRATEGIES = [
    # (name, make_func, params_variants)
    ("RSI_MR", make_rsi_mr_signal, [
        {"rsi_period": 7, "sl_pct": 0.02, "tp_pct": 0.06},
        {"rsi_period": 14, "sl_pct": 0.02, "tp_pct": 0.06},
        {"rsi_period": 14, "sl_pct": 0.03, "tp_pct": 0.09},
        {"rsi_period": 14, "sl_pct": 0.05, "tp_pct": 0.10},
        {"rsi_period": 21, "sl_pct": 0.02, "tp_pct": 0.06},
    ]),
    ("ADX_RSI", make_adx_rsi_signal, [
        {"adx_min": 20, "rsi_period": 14, "sl_pct": 0.03, "tp_pct": 0.08},
        {"adx_min": 25, "rsi_period": 14, "sl_pct": 0.03, "tp_pct": 0.08},
        {"adx_min": 20, "rsi_period": 7, "sl_pct": 0.02, "tp_pct": 0.06},
    ]),
    ("EMA_CROSS", make_ema_cross_signal, [
        {"fast": 5, "slow": 20, "sl_pct": 0.03, "tp_pct": 0.08},
        {"fast": 10, "slow": 30, "sl_pct": 0.03, "tp_pct": 0.08},
        {"fast": 10, "slow": 50, "sl_pct": 0.05, "tp_pct": 0.12},
    ]),
    ("BB_MR", make_bb_mr_signal, [
        {"bb_period": 20, "bb_std": 2.0, "sl_pct": 0.02, "tp_pct": 0.06},
        {"bb_period": 20, "bb_std": 2.5, "sl_pct": 0.03, "tp_pct": 0.09},
    ]),
    ("TREND_RSI", make_trend_rsi_signal, [
        {"adx_min": 20, "rsi_period": 14, "sl_pct": 0.03, "tp_pct": 0.10},
        {"adx_min": 25, "rsi_period": 14, "sl_pct": 0.03, "tp_pct": 0.12},
    ]),
]

# ── 回测引擎（精简版） ──
def run_backtest(klines, signal_func, name: str, params: dict) -> Dict:
    """运行一次回测"""
    from backtest.engine import BacktestEngine
    
    config = {
        'initial_balance': 100000,
        'taker_commission_rate': 0.0005,
        'maker_commission_rate': 0.0002,
        'slippage_rate': 0.0002,
    }
    engine = BacktestEngine(config)
    engine.load_klines(name, klines)
    success, result = engine.run(signal_func, min_trades=5)
    
    if not success:
        return None
    
    stats = result.get('stats', {})
    trades = result.get('trades', [])
    
    # 检查是否有足够的交易
    if len(trades) < 10:
        return None
    
    # 提取关键指标（注意stats字典的字段名）
    return {
        'trades': len(trades),
        'win_rate': stats.get('win_rate', 0),
        'sharpe': stats.get('sharpe_ratio', 0),
        'total_return_pct': stats.get('total_pnl_pct', 0),
        'max_dd_pct': stats.get('max_drawdown_pct', 0),
        'avg_hold_hours': stats.get('avg_hold_hours', 0),
        'profit_factor': stats.get('avg_rr', 0),
    }

# ── 主逻辑 ──
def scan_all_coins():
    """Stage 1: 扫描所有币种"""
    log_path = OUTPUT_DIR / "scan_progress.log"
    log_f = open(log_path, 'w', buffering=1)
    
    def plog(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()
    
    plog("=" * 70)
    plog(f"  Strategy Scanner — Stage 1: 全币种快速扫描")
    plog(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    plog("=" * 70)
    
    coins = list_available_coins()
    plog(f"  发现 {len(coins)} 个可用币种")
    
    all_results = []
    
    for idx, coin in enumerate(coins):
        plog(f"\n  [{idx+1}/{len(coins)}] {coin}...")
        
        klines = load_coin_klines_4h(coin)
        if len(klines) < 100:
            plog(f"⏭️  数据不足 ({len(klines)})")
            continue
        
        plog(f"({len(klines)} bars)")
        
        best_sharpe = -999
        best_result = None
        
        for strat_name, make_func, param_list in STRATEGIES:
            for params in param_list:
                try:
                    signal_func = make_func(**{k: v for k, v in params.items() if k not in ('name',)})
                    result = run_backtest(klines, signal_func, coin, params)
                    if result and result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_result = {
                            'coin': coin,
                            'bars': len(klines),
                            'strategy': strat_name,
                            **params,
                            **result,
                        }
                except Exception as e:
                    pass
        
        if best_result and best_result['sharpe'] > -999:
            all_results.append(best_result)
            sharpe_str = f"S={best_result['sharpe']:.2f}" if best_result['sharpe'] > 0 else f"s={best_result['sharpe']:.2f}"
            wr_str = f"WR={best_result['win_rate']:.0%}"
            plog(f"🏆 {best_result['strategy']} {sharpe_str} {wr_str}")
        else:
            plog("❌ 无有效策略")
    
    # 排序并保存
    all_results.sort(key=lambda r: r.get('sharpe', -999), reverse=True)
    
    # CSV输出（统一字段名，避免策略参数不匹配）
    csv_path = OUTPUT_DIR / "stage1_all_coins.csv"
    if all_results:
        # 标准字段名（所有策略共有的字段）
        std_fields = ['coin', 'bars', 'strategy', 'trades', 'win_rate', 'sharpe',
                       'total_return_pct', 'max_dd_pct', 'avg_hold_hours', 'profit_factor']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=std_fields, extrasaction='ignore')
            writer.writeheader()
            for r in all_results:
                r_display = dict(r)
                if isinstance(r_display.get('win_rate'), float):
                    r_display['win_rate'] = f"{r_display['win_rate']:.1%}"
                if isinstance(r_display.get('total_return_pct'), float):
                    r_display['total_return_pct'] = f"{r_display['total_return_pct']:.1f}%"
                if isinstance(r_display.get('max_dd_pct'), float):
                    r_display['max_dd_pct'] = f"{r_display['max_dd_pct']:.1f}%"
                writer.writerow(r_display)
    
    # 打印TOP 20
    plog(f"\n{'='*70}")
    plog(f"  TOP 20 最佳策略")
    plog(f"{'='*70}")
    plog(f"{'排名':>4} {'币种':>8} {'策略':>12} {'Sharpe':>8} {'胜率':>8} {'收益%':>8} {'交易数':>6} {'回撤%':>8}")
    plog(f"{'-'*65}")
    
    top = all_results[:20]
    for i, r in enumerate(top, 1):
        sharpe = r.get('sharpe', 0)
        wr = r.get('win_rate', 0)
        ret = r.get('total_return_pct', 0)
        trades = r.get('trades', 0)
        dd = r.get('max_dd_pct', 0)
        plog(f"{i:>4} {r['coin']:>8} {r['strategy']:>12} {sharpe:>8.2f} {wr:>7.1%} {ret:>+7.1%} {trades:>6} {dd:>7.1%}")
    
    plog(f"\n  结果已保存: {csv_path}")
    plog(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f.close()
    
    return all_results


if __name__ == '__main__':
    t0 = time.time()
    results = scan_all_coins()
    elapsed = time.time() - t0
    # 最后的结果写入日志文件
    with open(OUTPUT_DIR / "scan_final.log", 'w') as f:
        f.write(f"\n  总耗时: {elapsed/60:.1f} 分钟 ({elapsed:.0f} 秒)\n")
        f.write(f"  有效策略数: {len(results)}\n")
