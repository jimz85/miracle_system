# Miracle 2.0 架构设计详细文档

本文档包含系统架构的详细设计说明。

## 目录

1. [完整架构图](#完整架构图)
2. [Orchestrator LLM降级机制](#orchestrator-llm降级机制)
3. [Memory过期清理与遗忘机制](#memory过期清理与遗忘机制)
4. [Autoresearch 循环](#autoresearch-循环)
5. [LLM Provider 支持](#llm-provider-支持)
6. [Memory System](#memory-system)

---

## 完整架构图

### ASCII架构图 (纯文本版)

```
┌──────────────────────────────────────────────────────────────────┐
│                    Miracle 2.0 自主学习系统                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │              Orchestrator (LLM大脑)                       │   │
│   │         任务分解 + 结果聚合 + 自我反思                    │   │
│   └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐            │
│          ▼                   ▼                   ▼                │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│   │  Agent-M    │   │  Agent-S    │   │  Agent-L    │        │
│   │  市场情报    │──▶│  信号生成    │──▶│  学习迭代    │        │
│   │  (LLM增强)  │   │  (LLM增强)  │   │  (核心)     │        │
│   └─────────────┘   └─────────────┘   └─────────────┘        │
│          │                   │                   │                │
│          │                   ▼                   │                │
│          │           ┌─────────────┐            │                │
│          └──────────▶│  Agent-R    │◀───────────┘                │
│                      │  风险管理   │                             │
│                      └─────────────┘                             │
│                              │                                   │
│                              ▼                                   │
│                      ┌─────────────┐                            │
│                      │  Agent-E    │                            │
│                      │  执行引擎   │                            │
│                      └─────────────┘                             │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │              Memory System (记忆系统)                     │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│   │  │向量记忆 │  │结构化  │  │示范库  │  │规则库  │   │   │
│   │  │(Chroma)│  │经验库  │  │(Few-shot)│  │(Policy)│   │   │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │              Autoresearch Loop (自主研究循环)              │   │
│   │   数据收集 → 假设生成 → 回测验证 → 反思改进 → 持续进化     │   │
│   └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Orchestrator LLM降级机制

当LLM连续失败时，系统自动切换到规则引擎：

```python
# 配置降级参数
config = {
    "llm_failure_threshold": 3,      # 连续失败3次后降级
    "llm_recovery_interval": 300,    # 每5分钟尝试恢复LLM
    "rule_engine_fallback": True,    # 启用规则引擎
}

# 检查降级状态
orchestrator = Orchestrator(config)
status = orchestrator.get_degradation_status()
# {
#     "llm_available": True,
#     "llm_degraded": False,
#     "llm_failures": 0,
#     "total_llm_fallbacks": 0
# }
```

---

## Memory过期清理与遗忘机制

```python
from core.memory import get_memory_system

memory = get_memory_system()

# 健康报告
health = memory.get_memory_health_report()

# 遗忘低价值记忆（成功率<30%且应用次数>=5）
result = memory.forget_low_value_memories(
    min_success_rate=0.3,
    min_applied=5,
    dry_run=False  # 设为False执行遗忘
)

# 完整维护（清理旧数据）
result = memory.run_memory_maintenance(
    trade_days=90,         # 保留90天交易
    vector_age_days=30,    # 30天以上的向量记忆
    lesson_success_rate=0.3,
    lesson_min_applied=5,
    dry_run=True  # 先预览
)
```

---

## Autoresearch 循环

```python
class AutoresearchLoop:
    """
    自主研究循环：Keep/Discard策略淘汰机制
    """

    def run(self, experiments=50):
        """
        运行自主研究循环

        1. 数据收集 - 多源市场数据
        2. 假设生成 - 随机/趋势外推/聚焦优化
        3. 回测验证 - Walk-Forward多窗口
        4. 反思改进 - IC权重反馈闭环
        """
        for i in range(experiments):
            # 生成新策略假设
            hypothesis = self.generate_hypothesis()

            # 回测验证
            result = self.backtest(hypothesis)

            # 反思改进
            insight = self.reflect(hypothesis, result)

            # 更新记忆
            self.update_memory(insight)

            # 演化策略
            self.evolve_strategy(insight)
```

---

## LLM Provider 支持

| Provider | 模型 | 用途 |
|----------|------|------|
| Claude | Sonnet 4 | 主力推理 |
| GPT-4o | GPT-4o | 备用 |
| Gemini | Flash 2.0 | 成本优化 |
| DeepSeek | Chat | 研究循环 |

---

## Memory System

- **ChromaDB**: 向量记忆，语义检索历史经验，支持过期时间(TTL)
- **SQLite**: 结构化记忆，交易记录、因子表现、策略参数
- **遗忘机制**: 自动清理低价值教训（成功率低+应用次数少）
- **过期清理**: 自动清理过期数据和旧交易记录
- **Few-shot**: 示范库，成功/失败模式

---

## Agent详细职责

### Agent-M (市场情报)

负责从多个数据源收集市场信息：
- 价格数据 (OKX/Binance)
- 链上数据 (on-chain metrics)
- 新闻/社交媒体情绪
- 钱包数据

文件: `agents/agent_market_intel.py`, `agents/agent_market_intel_llm.py`

### Agent-S (信号生成)

负责生成交易信号：
- 技术指标计算 (RSI, ADX, MACD, Bollinger)
- 因子融合
- 信号置信度评估

文件: `agents/agent_signal.py`

### Agent-R (风险管理)

负责风险控制：
- 熔断机制 (5级生存层)
- 仓位计算
-止损/止盈管理

文件: `agents/agent_risk.py`, `core/circuit_breaker.py`

### Agent-E (执行引擎)

负责交易所下单和持仓监控：
- OKX/Binance API对接
- OCO订单管理
- 滑点监控

文件: `agents/agent_executor.py`

### Agent-L (学习迭代)

负责策略学习和演化：
- IC权重更新
- 模式发现
- 策略淘汰

文件: `agents/agent_learner.py`, `adaptive_learner.py`

---

## 熔断五级生存层

| 层级 | 回撤 | 最大仓位 | 允许开仓 |
|------|------|----------|----------|
| NORMAL | 0% | 100% | ✅ |
| CAUTION | 0~10% | 50% | ✅ |
| LOW | 10~20% | 25% | ❌ |
| CRITICAL | 20~30% | 0% | ❌ |
| PAUSED | >30% | 0% | ❌ |

---

## IC因子权重系统

动态权重根据IC (Information Coefficient) 自动调整：

| 因子 | 权重上限 |
|------|----------|
| BTC | 15% |
| Gemma | 16% |
| RSI | 19.9% |
| Bollinger | 20% |
| MACD | 17.5% |

最小权重: 5% (防止负IC因子完全被忽略)

文件: `core/ic_weights.py`
