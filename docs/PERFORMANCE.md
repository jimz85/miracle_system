# Performance Optimization Guide

## 性能目标

| 操作 | 目标延迟 | 最大延迟 |
|------|---------|---------|
| 信号生成 | < 100ms | 500ms |
| 风险检查 | < 10ms | 50ms |
| 数据获取 | < 200ms | 1s |
| LLM调用 | < 2s | 10s |
| 订单执行 | < 500ms | 3s |

---

## 关键路径优化

### 1. 数据获取优化

```python
# ❌ 慢: 每次都请求API
def get_price(symbol):
    return requests.get(f"https://api.okx.com/price/{symbol}").json()

# ✅ 快: 缓存 + 增量更新
class PriceCache:
    def __init__(self, ttl=1.0):
        self._cache = {}
        self._ttl = ttl
    
    def get(self, symbol):
        now = time.time()
        if symbol in self._cache:
            price, timestamp = self._cache[symbol]
            if now - timestamp < self._ttl:
                return price
        # 过期,重新获取
        price = fetch_price(symbol)
        self._cache[symbol] = (price, now)
        return price
```

### 2. 向量检索优化

```python
# ❌ 慢: 每次都检索
results = collection.query(query_texts=["..."])

# ✅ 快: 批量查询 + 缓存
class VectorCache:
    def __init__(self, cache_size=1000):
        self._cache = LRUCache(cache_size)
    
    def query(self, collection, text, n_results=5):
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        results = collection.query(query_texts=[text], n_results=n_results)
        self._cache[cache_key] = results
        return results
```

### 3. 指标计算优化

```python
# ❌ 慢: 每次重新计算
def calculate_rsi(prices, period=14):
    # 每次从头计算
    ...

# ✅ 快: 增量更新
class IncrementalRSI:
    def __init__(self, period=14):
        self.period = period
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
    
    def update(self, price, prev_price):
        change = price - prev_price
        self.gains.append(max(change, 0))
        self.losses.append(max(-change, 0))
        
        avg_gain = sum(self.gains) / self.period
        avg_loss = sum(self.losses) / self.period
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
```

### 4. 并行回测优化

```python
# ✅ 使用多进程
from multiprocessing import Pool

def run_parallel_backtest(params_list):
    with Pool(processes=4) as pool:
        results = pool.map(run_single_backtest, params_list)
    return aggregate_results(results)
```

### 5. LLM调用优化

```python
# ✅ 缓存重复请求
class LLMCallCache:
    def __init__(self):
        self._cache = {}
    
    def chat(self, prompt, provider="claude"):
        # 检查缓存
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        
        # 调用LLM
        response = call_llm(prompt, provider)
        self._cache[key] = response
        return response
```

---

## 延迟分析

```python
import cProfile
import pstats

# 分析热点
profiler = cProfile.Profile()
profiler.enable()

# 执行交易逻辑
run_trading_loop()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20热点
```

---

## 内存优化

### 1. 使用生成器处理大数据

```python
# ❌ 慢: 一次性加载所有数据
data = load_all_candles(symbol, years=5)  # 内存爆炸

# ✅ 快: 流式处理
def candle_generator(symbol, years=5):
    for batch in load_candles_batched(symbol, years, batch_size=1000):
        for candle in batch:
            yield candle

for candle in candle_generator("BTC-USDT", years=5):
    process(candle)
```

### 2. 使用适当的数据结构

```python
# ❌ 使用list存储固定大小数据
prices = []  # 不断append
for _ in range(1000000):
    prices.append(new_price)

# ✅ 使用deque固定大小
from collections import deque
prices = deque(maxlen=1000000)  # 自动清理旧数据
```

---

## 监控指标

```python
# 关键性能指标
metrics = {
    "signal_latency_p50": 0.05,   # 50分位延迟
    "signal_latency_p99": 0.25,   # 99分位延迟
    "llm_cache_hit_rate": 0.35,   # 缓存命中率
    "api_error_rate": 0.001,       # API错误率
    "memory_usage_mb": 512,        # 内存使用
}
```

---

## Checklist

- [ ] 数据缓存TTL < 5秒
- [ ] 向量检索结果缓存
- [ ] LLM请求去重
- [ ] 使用增量指标计算
- [ ] 并行回测 > 2进程
- [ ] 延迟监控 P50/P95/P99
- [ ] 内存使用 < 1GB
- [ ] CPU使用 < 80%
