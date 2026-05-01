# Miracle与Kronos IC动态因子权重系统对比

## 概述

本文档对比Kronos `voting_system.py` 和 Miracle `ic_weights.py` 的IC动态因子权重实现，为Miracle添加动态权重调整机制提供参考。

## 核心架构对比

| 特性 | Kronos | Miracle |
|------|--------|---------|
| IC计算方法 | Spearman秩相关系数 | Spearman秩相关系数 |
| 权重更新公式 | `W_new = 0.7*W_old + 0.3*IC_last` | `W_new = 0.7*W_old + 0.3*IC_last` |
| 滚动窗口 | 90天（3个月） | 90天（3个月） |
| IC历史存储 | `~/.hermes/kronos_ic_weights.json` | `~/.hermes/miracle_ic_weights.json` |
| BTC因子权重上限 | 15% | 15% |
| 其他因子权重上限 | 20% | 25% |

## Kronos IC权重实现

### ICTracker类（Kronos）

```python
class ICTracker:
    CACHE_FILE = '~/.hermes/kronos_ic_weights.json'
    WINDOW_DAYS = 90
    
    # 核心方法
    def record_ic(factor_name, ic_value): ...
    def compute_weights() -> Dict[str, float]: ...
    def get_weight(factor_name) -> float: ...
    
    # Spearman IC计算
    @staticmethod
    def spearman_ic(signal_values, future_returns) -> float:
        # 归一化到[-1, 1]
        def rank(x):
            order = np.argsort(np.argsort(x))
            return order / (len(x) - 1) * 2 - 1
        rank_sig = rank(sv)
        rank_ret = rank(fr)
        return np.corrcoef(rank_sig, rank_ret)[0, 1]
```

### Kronos因子列表

| 因子 | 基准IC | 权重上限 | 说明 |
|------|--------|----------|------|
| RSI | 0.08 | 20% | 相对强弱指数 |
| ADX | 0.05 | 20% | 趋势强度指标 |
| Bollinger | 0.06 | 20% | 布林带位置 |
| Vol | 0.07 | 20% | 成交量比率 |
| MACD | 0.05 | 20% | MACD直方图 |
| BTC | 0.04 | 15% | BTC方向过滤器 |
| Gemma | 0.10 | 20% | AI情绪分析 |

### Kronos权重计算算法

```
1. 对每个因子，计算指数衰减加权平均IC
   decay_weights = [0.7^i for i in range(len(recent_ics))]
   weighted_ic = sum(ic * w for ic, w in zip(ics, decay_weights))

2. 新权重 = 0.7*旧权重 + 0.3*最新IC
   new_ic_weight = 0.7 * old_weight + 0.3 * max(0, latest_ic)

3. IC为负的因子权重置0

4. 归一化使权重和=1

5. 应用权重上限（BTC 15%, 其他 20%）
   - 超出上限的权重释放给未达上限的因子
   - 按"可吸收量比例"分配
```

## Miracle IC权重实现

### MiracleICTracker类

位于: `miracle_system/core/ic_weights.py`

```python
class MiracleICTracker:
    CACHE_FILE = '~/.hermes/miracle_ic_weights.json'
    WINDOW_DAYS = 90
    
    # 可选：与Kronos IC系统同步
    def _sync_from_kronos(self): ...
    
    # 核心方法（与Kronos一致）
    def record_ic(factor_name, ic_value): ...
    def compute_weights() -> Dict[str, float]: ...
    def get_weight(factor_name) -> float: ...
    def get_ic(factor_name) -> float: ...
```

### Miracle因子列表（扩展）

| 因子 | 基准IC | 权重上限 | 说明 |
|------|--------|----------|------|
| RSI | 0.08 | 25% | 相对强弱指数 |
| ADX | 0.05 | 25% | 趋势强度指标 |
| MACD | 0.05 | 25% | MACD直方图 |
| Bollinger | 0.06 | 25% | 布林带位置 |
| Vol | 0.07 | 25% | 成交量比率 |
| BTC | 0.04 | 15% | BTC方向过滤器 |
| Momentum | 0.06 | 25% | 价格动量 |
| Trend | 0.05 | 25% | 趋势方向 |
| News | 0.03 | 20% | 新闻情绪 |
| Onchain | 0.02 | 25% | 链上指标 |
| Wallet | 0.02 | 25% | 钱包分布 |

### Miracle权重计算算法

与Kronos完全一致，额外特性：

1. **与Kronos同步**: `sync_with_kronos=True` 时自动从Kronos加载IC历史
2. **因子名映射**: Kronos的Gemma → Miracle的News
3. **权重上限差异化**: News因子20%，其他25%（高于Kronos的20%）

## 关键差异

### 1. 因子粒度

**Kronos**: 7个因子（技术指标为主）
**Miracle**: 11个因子（包含News/Onchain/Wallet）

### 2. 权重上限

| 因子类型 | Kronos上限 | Miracle上限 |
|----------|------------|-------------|
| BTC | 15% | 15% |
| 新闻/情绪 | 20% | 20% |
| 其他技术因子 | 20% | 25% |

Miracle对技术因子更宽松，允许更高权重。

### 3. 数据同步

Miracle支持与Kronos IC系统同步：
```python
tracker = MiracleICTracker(sync_with_kronos=True)
# 自动从 ~/.hermes/kronos_ic_weights.json 加载IC历史
```

## 集成到Miracle

### 使用示例

```python
from core.ic_weights import MiracleICTracker, compute_factor_ic_for_miracle

# 初始化追踪器（自动从Kronos同步）
tracker = MiracleICTracker()

# 获取当前权重
weights = tracker.get_all_weights()
# {'RSI': 0.15, 'ADX': 0.10, 'MACD': 0.12, ...}

# 获取某因子IC
rsi_ic = tracker.get_ic('RSI')  # 0.08

# 记录新IC
tracker.record_ic('RSI', 0.10)

# 重新计算权重
new_weights = tracker.compute_weights()

# 计算当前K线的因子IC
ics = compute_factor_ic_for_miracle(closes, highs, lows, volumes)
# {'RSI': 0.08, 'ADX': 0.05, ...}
```

### 与现有代码集成

现有代码使用固定权重：
```python
# miracle_core.py
combined = (
    price_score * weights["price_momentum"] +
    news_score * weights["news_sentiment"] +
    onchain_score * weights["onchain"] +
    wallet_score * weights["wallet"]
)
```

集成IC权重后：
```python
from core.ic_weights import MiracleICTracker

tracker = MiracleICTracker()

# 获取IC调整后的权重
ic_weights = tracker.get_all_weights()

# 归一化到price/news/onchain/wallet结构
normalized = {
    'price_momentum': ic_weights.get('RSI', 0) + ic_weights.get('ADX', 0) + 
                      ic_weights.get('MACD', 0) + ic_weights.get('Bollinger', 0) +
                      ic_weights.get('Momentum', 0) + ic_weights.get('Trend', 0),
    'news_sentiment': ic_weights.get('News', 0),
    'onchain': ic_weights.get('Onchain', 0),
    'wallet': ic_weights.get('Wallet', 0),
}

combined = (
    price_score * normalized["price_momentum"] +
    news_score * normalized["news_sentiment"] +
    onchain_score * normalized["onchain"] +
    wallet_score * normalized["wallet"]
)
```

## 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `miracle_system/core/ic_weights.py` | ICTracker实现 |
| `miracle_system/docs/IC_WEIGHT_COMPARISON.md` | 本文档 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `miracle_system/miracle_core.py` | 集成ICTracker，支持IC动态权重 |

## 下一步

1. 将IC权重集成到 `agent_signal.py` 的 `SignalGenerator`
2. 创建定时任务（如每小时）计算最新IC并更新权重
3. 考虑添加IC衰减监控和告警
4. 与Kronos共享IC历史数据（symlink或配置）