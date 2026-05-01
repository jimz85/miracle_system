#!/usr/bin/env python3
"""
全币种Sharpe验证 — 桌面5m数据聚合到4H，新滑点下重跑
"""
import sys, json, logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

DATA_DIR = Path.home() / "Desktop" / "crypto_data_Pre5m"
COIN_PARAMS = json.load(open(Path(__file__).parent.parent / "coin_params.json"))
COINS_MAP = {c["symbol"]: c for c in COIN_PARAMS["coins"]}
CONFIG = json.load(open(Path(__file__).parent.parent / "miracle_config.json"))
PER_COIN_SL = CONFIG["fee"]["per_coin_slippage"]

# 补上启用但漏配的币种默认滑点
PER_COIN_SL.setdefault("BNB", 0.0010)
PER_COIN_SL.setdefault("SOL", 0.0015)
PER_COIN_SL.setdefault("XRP", 0.0010)
PER_COIN_SL.setdefault("LINK", 0.0015)

# 只测enabled=true的币种 + 补测有数据但禁用的(寻找被忽略的alpha)
TARGETS = sorted(set(
    [s.upper() for s in PER_COIN_SL.keys()] +
    ["AVAX", "DOT", "BCH", "ETC", "ALGO", "CRO", "BNT", "CVC", "GAS"]
))

TEST_SETS = {
    "优化窗口": ("2025-08-01", "2026-04-28"),
    "样本外2024": ("2024-01-01", "2024-12-31"),
    "中间段2025H1": ("2025-01-01", "2025-07-31"),
}

def load(symbol):
    csv = DATA_DIR / f"{symbol}_USDT_5m_from_20180101.csv"
    if not csv.exists() and symbol == "BTC":
        csv = DATA_DIR / f"BTC_USDT_5m_from_20200101.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, dtype={"open": float, "high": float, "low": float, "close": float, "vol": float})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    ohlc = df["close"].resample("4h").ohlc()
    ohlc["volume"] = df.get("vol", df.get("volume", pd.Series(0, index=df.index))).resample("4h").sum()
    return ohlc.dropna()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def backtest_rsi_mr(df, oversold=30, overbought=70, slippage=0.0002):
    df = df.copy()
    df["rsi"] = calc_rsi(df["close"])
    df = df.dropna()
    
    pos, entry = 0, 0; trades = []
    for i in range(1, len(df)):
        p, r = df.iloc[i]["close"], df.iloc[i]["rsi"]
        if pos == 0 and r < oversold:
            pos, entry = 1, p
            trades.append({"entry": p})
        elif pos == 1 and r > overbought:
            fee = slippage + 0.0005
            ex = p * (1 - fee)
            trades[-1].update({"exit": ex, "pnl": (ex - entry) / entry * 100})
            pos = 0
    
    pnl = [t["pnl"] for t in trades if "pnl" in t]
    if len(pnl) < 5: return {"trades": len(pnl), "sharpe": 0, "winrate": 0, "return": 0}
    wins = sum(1 for p in pnl if p > 0)
    sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(365*6/4)) if np.std(pnl) > 0 else 0
    return {"trades": len(trades), "sharpe": round(sharpe, 2), "winrate": f"{wins}/{len(pnl)}", "return": f"{sum(pnl):.1f}%"}

# 跑所有币种
results = []
for sym in TARGETS:
    params = COINS_MAP.get(sym)
    if not params or not params.get("enabled", True):
        continue
    
    df = load(sym)
    if df is None or len(df) < 500:
        print(f"  {sym}: 数据不存在，跳过")
        continue
    
    new_sl = PER_COIN_SL.get(sym, 0.0002)
    params = COINS_MAP.get(sym, {})
    os = params.get("signal_params", {}).get("rsi_oversold", 30)
    ob = params.get("signal_params", {}).get("rsi_overbought", 70)
    
    row = {"币种": sym, "滑点": f"{new_sl*100:.2f}%"}
    for set_name, (start, end) in TEST_SETS.items():
        sub = df[(df.index >= start) & (df.index <= end)]
        if len(sub) < 100:
            row[set_name] = "数据不足"
            continue
        r = backtest_rsi_mr(sub, os, ob, new_sl)
        row[set_name] = f"S={r['sharpe']} T={r['trades']} W={r['winrate']} R={r['return']}"
        row[f"{set_name}_S"] = r["sharpe"]
    results.append(row)
    print(f"  {sym}: 加载完成 ({len(df)} candles)")

# 表格输出
print(f"\n{'='*85}")
print(f"  全币种RSI均值回归验证 — 新滑点")
print(f"{'='*85}")
print(f"  {'币种':>5} {'滑点':>7} {'优化窗口(S/R)':>22} {'2024样本外(S/R)':>22} {'2025H1(S/R)':>22}")
print(f"  {'-'*5} {'-'*7} {'-'*22} {'-'*22} {'-'*22}")
for r in results:
    o = r.get("优化窗口", "N/A").split(" R=")[0] if " R=" in r.get("优化窗口","") else r.get("优化窗口","N/A")
    o_r = r.get("优化窗口", "N/A").split(" R=")[-1] if " R=" in r.get("优化窗口","") else ""
    s24 = r.get("样本外2024", "N/A").split(" R=")[0] if " R=" in r.get("样本外2024","") else r.get("样本外2024","N/A")
    s24_r = r.get("样本外2024", "N/A").split(" R=")[-1] if " R=" in r.get("样本外2024","") else ""
    s25 = r.get("中间段2025H1", "N/A").split(" R=")[0] if " R=" in r.get("中间段2025H1","") else r.get("中间段2025H1","N/A")
    s25_r = r.get("中间段2025H1", "N/A").split(" R=")[-1] if " R=" in r.get("中间段2025H1","") else ""
    print(f"  {r['币种']:>5} {r['滑点']:>7} {o:>18} {s24:>18} {s25:>18}")

# 结论
print(f"\n{'='*85}")
stable = [r for r in results if r.get("样本外2024_S", -999) > 1.0]
print(f"  样本外Sharpe>1.0的币种: {[r['币种'] for r in stable] if stable else '无'}")
if not stable:
    print(f"  结论: 当前策略在所有币种上都没有可复现的alpha")
    print(f"  建议: 设CRYPTOPANIC_TOKEN→跑模拟盘积累真实数据→等待市场结构变化")
print(f"{'='*85}")
