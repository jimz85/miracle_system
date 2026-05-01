# Phase 3: Integration Plan — Inject Backtest Results into miracle_kronos.py

## Goal
Based on backtest results from Phase 1-2, inject per-coin optimal parameters into miracle_kronos.py.

## What Needs to Change

### 1. Per-Coin Strategy Config
Add a new config section in miracle_kronos.py:
```python
# Per-coin optimized strategies (from 8-year backtest)
PER_COIN_STRATEGY = {
    'BTC': {
        'best_tf': '4H',           # Best timeframe
        'preferred_side': 'long',   # or 'short'/'both'
        'adx_thresh': 25,          # RSI trend threshold
        'rsi_ob': 75,              # overbought for trend
        'rsi_os': 25,              # oversold for trend
        'rsi_ob_range': 70,        # overbought for range
        'rsi_os_range': 30,        # oversold for range
        'atr_sl_mult': 2.0,        # SL multiplier
        'atr_tp_mult': 8.0,        # TP multiplier (RR=1:4)
        'min_confidence': 0.30,    # Min confidence for entry
        'factor_weights': {        # Override default weights
            'RSI': 0.12, 'ADX': 0.12, 'Bollinger': 0.24,
            'Vol': 0.08, 'MACD': 0.20, 'BTC': 0.13, 'Gemma': 0.11
        }
    },
    # ... per coin
}
```

### 2. Strategy Selection in voting_vote()
Modify `voting_vote()` to accept coin-specific parameters.

### 3. Factor Weights
Update `factor_weights.json` with backtest-optimized weights.

## Verification
1. Run `pilot --full` with new config → verify signals
2. Compare signal direction with historical backtest predictions
3. Monitor first 10-20 trades → compare actual PnL with backtest expectations
