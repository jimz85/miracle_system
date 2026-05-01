#!/usr/bin/env python3
"""
DOGE/ADA Sharpe验证 v2 — 使用桌面5m数据聚合到4H，新滑点下重跑
"""
import sys, os, json, logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

# 配置
DATA_DIR = Path.home() / "Desktop" / "crypto_data_Pre5m"
COIN_PARAMS = json.load(open(Path(__file__).parent.parent / "coin_params.json"))
COINS_MAP = {c["symbol"]: c for c in COIN_PARAMS["coins"]}
CONFIG = json.load(open(Path(__file__).parent.parent / "miracle_config.json"))
PER_COIN_SL = CONFIG["fee"]["per_coin_slippage"]

# 测试集定义
TEST_SETS = {
    "Set1_优化窗口": ("2025-08-01", "2026-04-28"),
    "Set2_样本外2024": ("2024-01-01", "2024-12-31"),
    "Set3_中间段2025H1": ("2025-01-01", "2025-07-31"),
}

def load_and_resample(symbol: str) -> pd.DataFrame:
    """加载5m数据，聚合到4H"""
    csv_path = DATA_DIR / f"{symbol}_USDT_5m_from_20180101.csv"
    if not csv_path.exists():
        print(f"❌ 数据不存在: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path, dtype={"open": float, "high": float, "low": float, "close": float, "vol": float})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    
    # 聚合到4H
    ohlc = df["close"].resample("4H").ohlc()
    ohlc["volume"] = df["vol"].resample("4H").sum()
    ohlc = ohlc.dropna()
    
    print(f"  {symbol}: {len(df)} 行5m → {len(ohlc)} 行4H ({ohlc.index[0].date()} ~ {ohlc.index[-1].date()})")
    return ohlc

def run_simple_backtest(df: pd.DataFrame, symbol: str, params: dict, slippage: float) -> dict:
    """简单的RSI均值回归回测"""
    rsi_oversold = params["signal_params"]["rsi_oversold"]
    rsi_overbought = params["signal_params"]["rsi_overbought"]
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df = df.dropna()
    
    # 交易
    position = 0  # 0:无, 1:多
    entry_price = 0
    trades = []
    
    for i in range(1, len(df)):
        price = df.iloc[i]["close"]
        rsi = df.iloc[i]["rsi"]
        
        if position == 0:
            if rsi < rsi_oversold:
                position = 1
                entry_price = price
                trades.append({"entry": price, "entry_idx": i, "dir": "long"})
        elif position == 1:
            if rsi > rsi_overbought:
                exit_price = price * (1 - slippage) * (1 - 0.0005)  # 滑点+手续费
                trades[-1].update({"exit": exit_price, "exit_idx": i, "pnl_pct": (exit_price - entry_price) / entry_price * 100})
                position = 0
    
    if not trades:
        return {"trades": 0, "sharpe": 0, "winrate": 0, "return_pct": 0}
    
    # 计算指标
    pnl = [t["pnl_pct"] for t in trades if "pnl_pct" in t]
    if not pnl:
        return {"trades": 0, "sharpe": 0, "winrate": 0, "return_pct": 0}
    
    wins = sum(1 for p in pnl if p > 0)
    sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(365*6/4) if np.std(pnl) > 0 else 0  # 4H数据年化
    total_return = sum(pnl)
    
    return {
        "trades": len(trades),
        "sharpe": round(sharpe, 2),
        "winrate": f"{wins/len(pnl)*100:.1f}%",
        "return_pct": f"{total_return:.1f}%",
        "avg_pnl": f"{np.mean(pnl):.2f}%",
    }

# 主循环
for symbol in ["DOGE", "ADA"]:
    params = COINS_MAP[symbol]
    tf = params["timeframe"]
    
    df = load_and_resample(symbol)
    if df is None:
        continue
    
    print(f"\n{'='*55}")
    print(f"  {symbol} — 新滑点 {PER_COIN_SL[symbol]*100:.2f}%")
    print(f"{'='*55}")
    
    for set_name, (start_date, end_date) in TEST_SETS.items():
        mask = (df.index >= start_date) & (df.index <= end_date)
        subset = df[mask].copy()
        
        if len(subset) < 50:
            print(f"  {set_name}: 数据不足 ({len(subset)} 行, 跳过)")
            continue
        
        old_sl = 0.0002
        new_sl = PER_COIN_SL[symbol]
        
        old_result = run_simple_backtest(subset, symbol, params, old_sl)
        new_result = run_simple_backtest(subset, symbol, params, new_sl)
        
        print(f"\n  {set_name} ({start_date}~{end_date}, {len(subset)} candles):")
        print(f"    旧滑点(0.02%): Sharpe={old_result['sharpe']} 交易={old_result['trades']} 胜率={old_result['winrate']} 收益={old_result['return_pct']}")
        print(f"    新滑点({new_sl*100:.2f}%): Sharpe={new_result['sharpe']} 交易={new_result['trades']} 胜率={new_result['winrate']} 收益={new_result['return_pct']}")

print(f"\n{'='*55}")
print("  结论: Set2(2024样本外) Sharpe<1.5 = 过拟合")
print(f"{'='*55}")
