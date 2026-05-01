## 持仓全链路审查报告

### 扫描周期
```
每3-15分钟运行一次（cron调度）
  │
  ├─ run_scan()
  │    ├─ 并发扫描4个币种(每个调用scan_coin)
  │    │    └─ scan_coin(): 计算指标 + _gemma_vote_cached → voting_vote
  │    ├─ OKX实时持仓检查(SL/TP触发/ATR移动止损/时间止损)
  │    ├─ 本地持仓追踪(24h+ATR停留/peak追踪/trailing SL)
  │    └─ 选最优候选 → 开新仓
  │
  ├─ run_position_management()  ⭐ 关键
  │    ├─ 强制平仓(亏损>3%+持仓>3h / 持仓>6h无盈利)
  │    ├─ 部分止盈(盈利>12% → 平50%)
  │    ├─ SL上移成本(盈利>2%)
  │    ├─ 反向信号覆盖(亏损>3%+RSI反向)
  │    └─ ⚠️ 只用RSI/ADX，没有gemma
  │
  └─ main() → 执行所有decisions(close_position/place_oco)
```

### ✅ 已存在的保护机制
| 机制 | 触发条件 | 执行 |
|------|---------|------|
| SL触发 | 价格 < SL | close_position |
| TP触发(弱趋势) | 价格 > TP + ADX<25 | close_position |
| 24h时间止损 | 24h+ATR震荡 | close_position |
| 移动止损 | ATR×2 trailing | close_position |
| 强制平仓 | 亏3%+持3h / 持6h无盈利 | close_position |
| 部分止盈 | 盈利>12% | close 50% |
| SL上移 | 盈利>2% | OCO cancel+replace |
| 反向信号 | RSI>65做多/RSI<35做空 | close_position |

### ❌ 真实 Gap：Gemma4 不参与持仓管理

`run_position_management()` 只用 RSI/ADX 硬规则决策：
```python
# Line 2228: 只用RSI
if direction in ('LONG') and rsi_val > 65:  ...
elif direction in ('SHORT') and rsi_val < 35:  ...
```

**gemma4 仅在开仓时调用**（在 `scan_coin()` 中的 `_gemma_vote_cached`）。持仓期间 gemma4 完全不参与。

### 修复方案

在 `run_position_management()` 中加入 Gemma 持仓评估：

```python
# 持仓期间调用gemma评估（gemma_vote=0-1, >0.6=继续, <0.4=退出）
gemma_vote = _gemma_vote_cached(coin, rsi_val, adx_val, bb_pos, current_price, di_plus, di_minus)
# 转换为-1到+1
gemma_signal = (gemma_vote - 0.5) * 2

# 如果gemma强烈看空持仓方向且亏损中
if direction == 'long' and gemma_signal < -0.5 and pnl_pct < 0:
    decisions.append({'action': 'force_close', 'reason': f'Gemma看空+亏损', ...})
elif direction == 'short' and gemma_signal > 0.5 and pnl_pct < 0:
    decisions.append({'action': 'force_close', 'reason': f'Gemma看多+亏损', ...})
```
