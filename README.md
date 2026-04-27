# Miracle 1.0.1

**高频趋势跟踪 + 事件驱动混合交易系统**

赔率优先：赢了要赢很多，输了只输一点。

---

## 系统架构

```
Agent-M (市场情报) → Agent-S (信号生成) → Agent-R (风险管理) → Agent-E (执行) → Agent-L (学习)
```

| Agent | 职责 | 数据源 |
|-------|------|--------|
| **Agent-M** | 新闻情感、链上数据、钱包分布 | OKX + yfinance (免费) |
| **Agent-S** | 多因子融合、趋势检测、信号生成 | 内部计算 |
| **Agent-R** | 仓位计算、杠杆管理、熔断机制 | 内部计算 |
| **Agent-E** | 交易所下单、持仓监控 | 需要 OKX API Key |
| **Agent-L** | 自适应学习、参数优化 | 历史交易数据 |

---

## 每币种参数优化 (Per-Coin Parameter Optimization)

参考 Kronos `coin_strategy_map.json` 实现，支持每个币种独立的最优策略参数。

### 核心特性

- **独立参数**: 每个币种有自己专属的 RSI、ADX、ATR 等指标参数
- **策略选择**: 根据币种特性选择最优策略 (RSI_MR / RSI_EMAn / VOL_BRK 等)
- **Kronos 对比**: 自动对比 Kronos 的 coin_strategy_map.json 配置
- **Walk-Forward 验证**: 确保参数在历史数据上的有效性

### 相关文件

| 文件 | 说明 |
|------|------|
| `coin_params.json` | 币种参数配置 (灵感来自 Kronos coin_strategy_map.json) |
| `coin_optimizer.py` | 参数优化器模块 |

### 使用示例

```python
from core import get_coin_optimizer, get_coin_signal_generator

# 获取优化器
optimizer = get_coin_optimizer()

# 获取币种参数
btc_params = optimizer.get_coin_params("BTC")
print(f"BTC 策略: {btc_params.optimal_strategy}")

# 获取信号参数
signal_params = optimizer.get_signal_params("BTC")
print(f"RSI 超卖阈值: {signal_params['rsi_oversold']}")

# 获取已启用的币种列表
enabled_coins = optimizer.get_enabled_coins()

# 对比 Kronos
comparison = optimizer.compare_with_kronos("ETH")
print(comparison)

# 生成优化报告
print(optimizer.generate_optimization_report())
```

### 策略类型

| 策略 | 说明 | 适用币种 |
|------|------|----------|
| RSI_MR | RSI 均值回归 | BTC, ETH, SOL, DOGE, ADA |
| RSI_EMAn | RSI + EMA 共振 | BNB, XRP |
| RSI_VOL | RSI + 成交量混合 | DOT |
| VOL_BRK | 成交量突破 | 熊市/高波动环境 |
| BB_TREND | 布林带趋势跟踪 | ADA |

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 查看系统信息
python miracle.py --info

# 扫描所有币种
python miracle.py

# 扫描指定币种
python miracle.py --symbol BTC

# 持续运行模式 (每30分钟扫描)
python miracle.py --daemon --interval 30
```

---

## 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 最小RR | 2.0 | 输了亏1%，赢了至少2% |
| 最大杠杆 | 3x | 强趋势时使用 |
| 最大仓位 | 15% | 单币种最大暴露 |
| 日交易上限 | 5笔 | 每交易日最多5笔开仓 |
| 日亏熔断 | 5% | 日亏损超5%停止交易 |
| 回撤熔断 | 20% | 总回撤超20%停止交易 |

---

## 数据源

| 类型 | 来源 | 状态 |
|------|------|------|
| 价格数据 | OKX (实时) | ✅ 可用 |
| K线数据 | OKX + yfinance | ✅ 可用 |
| 新闻情感 | CryptoCompare + 价格动量代理 | ✅ 可用 |
| 链上数据 | 模拟数据 | ⚠️ 需Glassnode API |
| 交易所下单 | OKX | ⚠️ 需API Key |

---

## 文件结构

```
miracle-1.0.1/
├── miracle.py              # 主入口
├── miracle_core.py         # 核心计算函数
├── miracle_config.json     # 配置文件
├── adaptive_learner.py     # 自适应学习模块
├── requirements.txt        # 依赖
├── agents/
│   ├── agent_market_intel.py   # 市场情报
│   ├── agent_signal.py         # 信号生成
│   ├── agent_risk.py           # 风险管理
│   ├── agent_executor.py       # 执行引擎
│   └── agent_learner.py        # 学习迭代
├── core/
│   └── data_fetcher.py     # 统一数据源 (OKX/yfinance)
├── data/                   # 数据目录
└── logs/                   # 日志目录
```

---

## Pilot驾驶舱

Miracle Pilot驾驶舱提供系统状态监控面板，对比Kronos的`kronos_pilot.py`实现。

```bash
# 完整日报（包含所有监控面板）
python miracle_pilot.py --full

# 状态摘要
python miracle_pilot.py --status

# 持仓监控
python miracle_pilot.py --positions

# 信号列表
python miracle_pilot.py --signals

# 风险仪表盘
python miracle_pilot.py --risk

# 查看日志
python miracle_pilot.py --log 50
```

### 驾驶舱功能

| 功能 | 说明 |
|------|------|
| **状态摘要** | 系统模式、账户信息、熔断器状态 |
| **持仓监控** | 活跃持仓、未实现盈亏、持仓时间 |
| **信号列表** | 实时信号、置信度、趋势强度、质量评分 |
| **风险仪表盘** | VaR、CVaR、最大回撤、Sharpe/Sortino比率、IC权重 |

---

## 状态说明

- ✅ 可用：已实现并测试通过
- ⚠️ 需配置：需要API Key或其他配置
- ❌ 未实现：功能未完成
