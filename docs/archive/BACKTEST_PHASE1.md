# Phase 1A: Backtest Engine Specification

## Goal
Build a complete backtest engine (`~/Desktop/bt/backtest_engine.py`) that:
1. Loads 5m CSV data from `~/Desktop/crypto_data_Pre5m/`
2. Resamples to 1H/4H/1D
3. Runs miracle_kronos.py's exact vote-based signal logic
4. Outputs per-coin, per-strategy, per-timeframe performance metrics

## Data Format

CSV files: `{COIN}_USDT_5m_from_20180101.csv`
Header: timestamp,open,high,low,close,vol,volCcy,volCcyQuote,confirm,datetime_utc
- timestamp: ms since epoch
- 5-minute bars, 858K+ rows per coin

## Output Directory
All files go to `~/Desktop/bt/`

## Signal Logic (exact replica of miracle_kronos.py v6.5)

### 1. Indicators (exact copies from miracle_kronos.py)

#### calc_rsi(prices: list, period=14) -> float
Wilder's smoothing RSI:
- Deltas = [p[i]-p[i-1] for i in range(1,len(p))]
- Gains = [d if d>0 else 0], Losses = [-d if d<0 else 0]
- avg_gain = SMA of first period gains, then Wilder's: avg_gain = (prev_avg_gain * 13 + current_gain) / 14
- Same for avg_loss
- RS = avg_gain/avg_loss; RSI = 100 - 100/(1+RS)
- Return None if not enough data

#### calc_adx(highs, lows, closes, period=14) -> dict
Returns: {'adx': float, 'di_plus': float, 'di_minus': float}
- TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
- +DM = high - prev_high (if > prev_low-low and > 0 else 0)
- -DM = prev_low - low (if > high-prev_high and > 0 else 0)
- Wilder's smoothing on TR, +DM, -DM
- +DI = 100 * smoothed_DM+ / smoothed_TR
- -DI = 100 * smoothed_DM- / smoothed_TR
- DX = 100 * abs(+DI - -DI) / (+DI + -DI)
- ADX = Wilder's smoothing of DX
- Return None if not enough data

#### calc_bollinger(closes, period=20, mult=2) -> dict
Returns: {'upper': float, 'middle': float, 'lower': float, 'bb_pos': float}
- Middle = SMA of period closes
- Std = sqrt(sum((close-ma)^2)/period)
- Upper = ma + mult*std, Lower = ma - mult*std
- bb_pos = 100 * (close - lower) / (upper - lower)
- Return None if not enough data

#### calc_macd(closes, fast=12, slow=26, signal=9) -> dict
Returns: {'macd': float, 'signal': float, 'histogram': float}
- EMA(fast), EMA(slow), MACD = EMA_fast - EMA_slow
- Signal = EMA(macd, signal)
- Histogram = MACD - Signal
- Return None if not enough data

### 2. Voting Logic (exact replica of miracle_kronos.py voting_vote)

```python
def compute_vote(factors: dict) -> dict:
    """
    factors = {
        'rsi': float,
        'adx': float,
        'di_plus': float,
        'di_minus': float,
        'bb_pos': float (0-100),
        'macd_hist': float,
        'vol_ratio': float (current_vol / avg_vol),
        'btc_trend': 'bull'|'bear'|'neutral'  # optional, default 'neutral'
    }
    
    Returns: {
        'score': float,          # weighted sum -1.0 to +1.0
        'direction': str,        # 'long'|'short'|'wait'
        'confidence': float,     # 0.0 to 1.0
        'votes': dict,           # per-factor votes
    }
    """
```

#### Voting Rules:
**RSI (weight=0.18):**
- ADX>25 (strong trend): RSI<25→+1, RSI>75→-1, RSI<40→+0.5, RSI>60→-0.5
- ADX<=25 (weak/range): RSI<30→+1, RSI>70→-1, RSI<40→+0.5, RSI>60→-0.5

**ADX (weight=0.14):**
- ADX>30→+1, ADX>22→+0.5, ADX<15→0

**Bollinger (weight=0.14):**
- bb_pos<20→+1, bb_pos>80→-1, bb_pos<35→+0.5, bb_pos>65→-0.5

**Vol (weight=0.10):**
- vol_ratio>1.5→+0.5, vol_ratio<0.7→-0.5

**MACD (weight=0.12):**
- macd_hist>0.01→+1, macd_hist<-0.01→-1

**Weight total should be 1.0 across all factors.**

#### Score Calculation:
```python
votes = {'RSI': rsi_vote, 'ADX': adx_vote, 'Bollinger': bb_vote, 
         'Vol': vol_vote, 'MACD': macd_vote}
weights = {'RSI': 0.18, 'ADX': 0.14, 'Bollinger': 0.14, 
           'Vol': 0.10, 'MACD': 0.12}

# Normalize weights to sum to 1.0
total_w = sum(weights.values())
weights = {k: v/total_w for k,v in weights.items()}

score = sum(weights[k] * votes[k] for k in votes)
direction = 'long' if score > 0.05 else ('short' if score < -0.05 else 'wait')

# Confidence
confidence = min(abs(score) / 2.0, 1.0)
```

### 3. 4H Direction Confirmation
- Compute indicators on 4H-resampled data
- ADX<20 → neutral (no override)
- ADX>=20: direction=bull if di_plus>di_minus else bear
- Strength = min(adx/50, 1.0)
- If 1H direction conflicts with 4H direction AND 4H strength>0.5:
  confidence *= 0.30 (severe penalty)

### 4. Backtest Metrics
For each backtest run, compute:
- total_return_pct = (final_equity - 1.0) * 100
- sharpe_ratio = mean(pnl_per_trade) / std(pnl_per_trade) * sqrt(365*24/tf_hours) [annualized]
- max_drawdown_pct = max peak-to-trough equity decline
- win_rate = wins / total_trades
- profit_factor = gross_profit / gross_loss
- total_trades
- avg_hold_hours

### 5. Run Loop
```python
def backtest_coin(coin, timeframe='1H', start_date=None, end_date=None):
    """Load data, compute indicators on rolling basis, generate signals, track PnL"""
    # 1. Load CSV
    # 2. Resample to timeframe
    # 3. For each bar (starting after warmup):
    #    a. Compute indicators on lookback window
    #    b. Compute vote -> signal
    #    c. If signal != 0 and no position: enter with SL=ATR*2, TP=ATR*8, RR=1:4
    #    d. If in position: check SL/TP hits
    # 4. Return metrics dict
```

### 6. Test Matrix
```python
COINS = ['BTC', 'ETH', 'DOGE', 'BNB', 'AVAX']
TIMEFRAMES = ['1H', '4H']
```

## Output Files
1. `backtest_results.json` — all results with metrics
2. `per_coin_results.json` — grouped by coin
3. `equity_curves/{coin}_{timeframe}.json` — equity over time for best params

## Success Criteria
- [ ] All 5 coins load and resample correctly
- [ ] Signal generates trades on all coins (not all-wait)
- [ ] Return metrics are reasonable (not NaN/Inf)
- [ ] Win rate and Sharpe consistent with real trading
- [ ] Results saved to JSON for Phase 2 analysis
