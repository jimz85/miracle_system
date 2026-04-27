# Miracle 2.0 — 自主学习量化交易系统

**版本**: 2.0  
**定位**: 大语言模型 + 多Agent协同的自主学习交易系统  
**核心理念**: 赔率优先 + LLM驱动的自适应学习 + Autoresearch持续进化

---

## 系统架构

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

## 核心能力

### 1. 自主学习 (Agent-L)

| 能力 | 1.0 | 2.0 |
|------|------|------|
| 学习触发 | 固定周期 | 每笔交易后即时学习 |
| 因子调整 | IC阈值规则 | LLM分析因果 |
| 模式发现 | 简单统计 | LLM语义聚类 |
| 策略演化 | 月度淘汰 | 持续进化 |
| 知识积累 | 规则存储 | 向量记忆检索 |

### 2. Autoresearch 循环

```python
# 完整闭环
while True:
    data = collect_market_data()        # 1. 收集数据
    hypothesis = generate_hypothesis(data) # 2. 生成假设
    result = backtest(hypothesis)          # 3. 回测验证
    insight = reflect(result)              # 4. 反思改进
    update_memory(insight)                # 5. 更新记忆
    evolve_strategy(insight)              # 6. 演化策略
```

### 3. 多Agent协同

| Agent | 职责 | LLM增强 |
|-------|------|---------|
| **Orchestrator** | 全局规划、决策 | LLM推理 + 反思 |
| **Agent-M** | 市场情报、情感分析 | LLM深度理解 |
| **Agent-S** | 信号生成、因子融合 | LLM动态权重 |
| **Agent-R** | 风险管理、熔断 | LLM风险评估 |
| **Agent-L** | 学习迭代、策略演化 | LLM自我反思 |
| **Agent-E** | 交易所执行 | - |

---

## 快速开始

### 安装依赖

```bash
cd ~/miracle_system
pip install -r requirements.txt

# 可选：安装ChromaDB（向量记忆）
pip install chromadb
```

### 基本使用

```python
from miracle_autonomous import MiracleAutonomous

# 初始化系统
system = MiracleAutonomous(
    symbols=["BTC", "ETH", "SOL", "DOGE"],
    mode="simulation"
)

# 运行自主研究循环
system.run_autonomous_cycle(experiments=50)

# 做交易决策
decision = system.make_decision(market_data)
```

### 命令行使用

```bash
# 运行完整自主研究
python miracle_autonomous.py --experiments 50 --coins BTC,ETH,SOL,DOGE

# 仅做决策
python miracle.py --symbol BTC

# 查看Pilot驾驶舱
python miracle_pilot.py --full
```

---

## 目录结构

```
miracle_system/
├── miracle.py                    # 1.0主入口（兼容）
├── miracle_autonomous.py         # 2.0主入口（自主学习）
├── miracle_core.py               # 核心计算
├── miracle_pilot.py              # 驾驶舱
├── miracle_kronos.py             # Kronos兼容
├── backtest.py                  # 回测引擎
├── adaptive_learner.py           # 自适应学习
├── coin_optimizer.py            # 每币种参数优化
│
├── core/
│   ├── orchestrator.py          # 协调器（LLM大脑）
│   ├── llm_provider.py          # LLM接口（Claude/GPT/Gemini/DeepSeek）
│   ├── ic_weights.py            # IC动态权重
│   ├── regime_classifier.py     # 市场状态分类
│   ├── state_reconciler.py      # OKX状态同步
│   ├── feishu_notifier.py       # 飞书通知
│   ├── data_fetcher.py          # 数据获取
│   └── memory/
│       ├── vector_memory.py      # ChromaDB向量记忆
│       ├── structured_memory.py  # SQLite结构化记忆
│       └── system.py            # 记忆系统
│
├── agents/
│   ├── agent_market_intel.py     # 市场情报
│   ├── agent_signal.py           # 信号生成
│   ├── agent_risk.py             # 风险管理
│   ├── agent_executor.py         # 执行引擎
│   └── agent_learner.py          # 学习迭代
│
├── data/
│   ├── decision_journal/         # 决策日志
│   └── cache/                   # 数据缓存
│
├── docs/
│   ├── MIRACLE_LLM_ARCHITECTURE.md  # 详细架构设计
│   ├── IC_WEIGHT_COMPARISON.md      # IC权重对比
│   └── REGIME_CLASSIFIER_COMPARISON.md
│
├── tests/                       # 测试
├── requirements.txt
└── README.md
```

---

## 关键特性

### Orchestrator LLM降级机制

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

### Memory过期清理与遗忘机制

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

### Autoresearch 循环

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
        for i in range(eximents):
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

### LLM Provider 支持

| Provider | 模型 | 用途 |
|----------|------|------|
| Claude | Sonnet 4 | 主力推理 |
| GPT-4o | GPT-4o | 备用 |
| Gemini | Flash 2.0 | 成本优化 |
| DeepSeek | Chat | 研究循环 |

### Memory System

- **ChromaDB**: 向量记忆，语义检索历史经验，支持过期时间(TTL)
- **SQLite**: 结构化记忆，交易记录、因子表现、策略参数
- **遗忘机制**: 自动清理低价值教训（成功率低+应用次数少）
- **过期清理**: 自动清理过期数据和旧交易记录
- **Few-shot**: 示范库，成功/失败模式

---

## 配置

### 环境变量

```bash
# LLM API Keys
export ANTHROPIC_API_KEY=sk-xxx        # Claude
export OPENAI_API_KEY=sk-xxx           # GPT
export GOOGLE_API_KEY=xxx               # Gemini
export DEEPSEEK_API_KEY=sk-xxx         # DeepSeek

# OKX API
export OKX_API_KEY=xxx
export OKX_SECRET_KEY=xxx
export OKX_PASSPHRASE=xxx

# 飞书通知
export FEISHU_APP_ID=xxx
export FEISHU_APP_SECRET=xxx
export FEISHU_CHAT_ID=oc_xxx
```

### 配置文件

```json
{
  "symbols": ["BTC", "ETH", "SOL", "DOGE", "BNB", "XRP", "ADA", "AVAX", "DOT", "LINK"],
  "min_rr": 2.0,
  "min_confidence": 0.6,
  "max_trades_per_day": 5,
  "max_position_pct": 15,
  "leverage": 3,
  "llm_provider": "claude",
  "enable_autoresearch": true,
  "enable_memory": true
}
```

---

## 与Kronos对比

| 维度 | Kronos | Miracle 2.0 |
|------|--------|-------------|
| **学习方式** | IC权重规则 | LLM驱动的自主学习 |
| **信号生成** | 固定公式 | LLM推理 + 动态权重 |
| **知识积累** | JSON文件 | ChromaDB向量记忆 |
| **策略演化** | 定期淘汰 | Autoresearch持续进化 |
| **反思能力** | 有限 | 每笔交易即时反思 |
| **多Agent** | 5 Agent | Orchestrator + 5 Agent |

---

## 开发指南

### 添加新的Agent

```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    async def execute(self, context):
        # 执行逻辑
        return {"result": "..."}
```

### 添加新的工具

```python
from core.tools import register_tool

@register_tool
async def my_tool(param1, param2):
    """工具描述"""
    return await do_something(param1, param2)
```

---

## 状态说明

- ✅ 可用：已实现并测试通过
- ⚠️ 需配置：需要API Key或其他配置
- 🔄 开发中：正在实现

| 功能 | 状态 |
|------|------|
| Orchestrator | ✅ |
| LLM Provider | ✅ |
| Memory System | ✅ |
| Agent-M | ✅ |
| Agent-S | ✅ |
| Agent-R | ✅ |
| Agent-E | ✅ |
| Agent-L | ✅ |
| Autoresearch Loop | ✅ |
| OKX集成 | ✅ |
| 飞书通知 | ✅ |

---

## 许可证

MIT License

---

**赔率优先，永不妥协。**
