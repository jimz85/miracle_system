#!/usr/bin/env python3
"""
DOGE/ADA Sharpe验证 — 在新滑点下重跑回测
测试集: 2025-08-22 至今（本地数据可用范围）
"""
import sys, os, json, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

# 加载coin_params
cp = json.load(open(Path(__file__).parent.parent / "coin_params.json"))
coins_map = {c["symbol"]: c for c in cp["coins"]}

# 加载per_coin_slippage
cfg = json.load(open(Path(__file__).parent.parent / "miracle_config.json"))
per_coin_sl = cfg["fee"]["per_coin_slippage"]

from backtest.backtest import BacktestRunner

for symbol in ["DOGE", "ADA"]:
    params = coins_map[symbol]
    tf = params["timeframe"].lower()
    csv_path = Path.home() / ".hermes" / "cron" / "output" / f"klines_{symbol}_{tf}.csv"
    
    if not csv_path.exists():
        print(f"\n❌ {symbol}: 数据文件不存在 {csv_path}")
        continue
    
    # 读取CSV
    import csv as csv_mod
    klines = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            klines.append({
                "timestamp": int(int(row["ts"]) / 1000),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
    
    print(f"\n{'='*50}")
    print(f"{symbol} ({tf}) — {len(klines)} 根K线")
    
    # 滑点: 旧的统一值 vs 新的币种专属值
    for label, sl in [("旧(统一0.02%)", 0.0002), ("新(币种专属)", per_coin_sl[symbol])]:
        config_override = {
            "slippage_rate": sl,
            "per_coin_slippage": {symbol: sl},
        }
        runner = BacktestRunner(config_override)
        runner.load_data(symbol, klines)
        result = runner.run_walkforward(symbol, strategy="both")
        
        pf = result.performance if hasattr(result, 'performance') else {}
        sharpe = getattr(result, 'best_sharpe', None) or pf.get('sharpe', 'N/A')
        ret = getattr(result, 'best_return', None) or pf.get('return', 'N/A')
        
        print(f"  {label} 滑点={sl:.4f}")
        print(f"    策略: {result.best_strategy}")
        print(f"    Sharpe: {sharpe}")
        print(f"    收益: {ret}")
        
        if hasattr(result, 'mean_reversion') and result.mean_reversion:
            mr = result.mean_reversion
            print(f"    均值回归: Sharpe={mr.get('sharpe','N/A')} Return={mr.get('return','N/A')}%")
        if hasattr(result, 'trend_following') and result.trend_following:
            tf_r = result.trend_following
            print(f"    趋势跟踪: Sharpe={tf_r.get('sharpe','N/A')} Return={tf_r.get('return','N/A')}%")

print(f"\n{'='*50}")
print("注意: 只有2025年8月至今的数据，无法做2024年样本外验证")
print("如需完整样本外测试，需先从交易所下载2024年历史数据")
