# Miracle 2.0 — 自主学习系统架构设计

**版本：** 2.0  
**定位：** 大语言模型 + 多Agent协同的自主学习交易系统  
**核心理念：** 赔率优先 + LLM驱动的自适应学习

---

## 一、设计目标

| 维度 | 1.0 (当前) | 2.0 (目标) |
|------|------------|------------|
| 学习方式 | 规则驱动（IC权重调整） | LLM驱动的自主学习 |
| 信号生成 | 固定因子公式 | LLM推理 + 因子融合 |
| 决策链路 | 规则匹配 | ReAct推理 + 规划 |
| 知识积累 | 简单数据库 | 向量记忆 + 结构化经验 |
| Agent能力 | 被动响应 | 主动思考 + 工具调用 |
| 策略演化 | 月度淘汰 | 持续反思 + 自我改进 |

---

## 二、系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Miracle 2.0 自主学习系统                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    Orchestrator (协调器)                      │     │
│   │              LLM驱动的主控制Agent，负责全局规划                  │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│          ┌───────────────────┼───────────────────┐                   │
│          ▼                   ▼                   ▼                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│   │  Agent-M    │     │  Agent-S    │     │  Agent-L    │            │
│   │  市场情报   │────▶│  信号生成   │────▶│  学习迭代   │            │
│   │  (LLM增强)  │     │  (LLM增强)  │     │  (核心)     │            │
│   └─────────────┘     └─────────────┘     └─────────────┘            │
│          │                   │                   │                    │
│          │                   ▼                   │                    │
│          │           ┌─────────────┐            │                    │
│          │           │  Agent-R    │            │                    │
│          └──────────▶│  风险管理   │◀───────────┘                    │
│                      └─────────────┘                                  │
│                              │                                          │
│                              ▼                                          │
│                      ┌─────────────┐                                    │
│                      │  Agent-E    │                                    │
│                      │  执行引擎   │                                    │
│                      └─────────────┘                                    │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    Memory System (记忆系统)                   │     │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │     │
│   │  │向量记忆 │  │结构化  │  │示范库  │  │规则库  │         │     │
│   │  │(Chroma)│  │经验库  │  │(Few-shot)│  │(Policy)│         │     │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    Tool Ecosystem (工具生态)                   │     │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │     │
│   │  │数据获取│  │技术指标│  │交易所  │  │MCP工具  │         │     │
│   │  │OKX/Binance│ │RSI/ADX │  │API      │  │外部能力 │         │     │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心组件设计

### 3.1 Orchestrator (协调器) — LLM大脑

```
Orchestrator
├── LLM Provider (可切换: Claude/GPT/Gemini/DeepSeek)
├── System Prompt (角色: 交易主管)
├── Planning Module (任务分解)
├── Reflection Module (自我反思)
└── Memory Interface (记忆读写)
```

**职责：**
- 接收市场状态，输出交易决策
- 分解复杂任务给子Agent
- 聚合子Agent结果，形成最终决策
- 反思决策质量，持续优化

**Prompt模板：**
```markdown
你是Miracle交易系统的首席交易员。

## 你的职责
1. 分析市场情报（Agent-M产出）
2. 评估交易信号（Agent-S产出）
3. 审核风险管理（Agent-R反馈）
4. 决定是否执行
5. 从每笔交易中学习

## 赔率优先原则
- 永远选择RR≥2.0的机会
- 输了只亏1%，赢了要赚2%以上
- 高置信度机会可适当加大仓位

## 决策流程
1. 理解当前市场状态
2. 评估各币种机会
3. 考虑风险约束
4. 选择最优策略
5. 明确执行计划

## 输出格式
```json
{
  "decision": "EXECUTE/SKIP/WAIT",
  "symbol": "BTC",
  "direction": "LONG/SHORT",
  "reasoning": "为什么这样做",
  "confidence": 0.85,
  "lessons": "从这次决策学到了什么"
}
```
```

---

### 3.2 Agent-M (市场情报) — LLM增强

```
Agent-M
├── Data Collector (数据获取)
├── Sentiment Analyzer (LLM情感分析)
├── On-chain Analyzer (LLM链上分析)
└── Context Builder (上下文构建)
```

**能力升级：**

| 能力 | 1.0 | 2.0 |
|------|-----|-----|
| 新闻处理 | 关键词匹配 | LLM深度理解 + 情感量化 |
| 链上数据 | 固定阈值 | LLM解释异常 + 趋势判断 |
| 钱包分布 | 集中度指标 | LLM识别聪明钱动向 |
| 上下文 | 原始数据 | LLM生成的叙事摘要 |

**输出示例：**
```json
{
  "symbol": "BTC",
  "market_narrative": "美联储偏鹰但BTC展现韧性，机构逢低买入明显",
  "sentiment_score": 0.65,
  "sentiment_drivers": [
    {"factor": "宏观", "impact": "负", "reason": "CPI数据超预期"},
    {"factor": "机构", "impact": "正", "reason": "Coinbase净流入增加"}
  ],
  "onchain_insights": "大额钱包未出现抛售信号，反而在54000下方持续吸筹",
  "confidence": 0.7
}
```

---

### 3.3 Agent-S (信号生成) — LLM推理

```
Agent-S
├── Factor Aggregator (因子聚合)
├── Strategy Selector (LLM策略选择)
├── Signal Generator (LLM推理信号)
└── Confidence Estimatior (置信度估计)
```

**能力升级：**

| 能力 | 1.0 | 2.0 |
|------|-----|-----|
| 因子融合 | 固定权重公式 | LLM动态权重推理 |
| 策略选择 | 枚举选择 | LLM根据情境选择+组合 |
| 信号生成 | 阈值触发 | LLM综合判断 |
| 置信度 | 统计估计 | LLM考虑多维度不确定性 |

**LLM信号推理Prompt：**
```markdown
## 任务
给定以下市场数据，判断是否应入场交易。

## 输入
- 价格动量指标：RSI=38, ADX=65, MACD=金叉
- 新闻情感：0.65（偏利好）
- 链上数据：交易所净流出，大户持仓增加
- 当前趋势：上涨趋势(ADX>50)

## 判断标准
1. 入场条件：多因子共振方向一致
2. 赔率要求：RR≥2.0
3. 置信度要求：≥0.6

## 输出
分析每个因子的信号强度，权衡利弊，给出最终判断和理由。
```

---

### 3.4 Agent-R (风险管理) — LLM辅助

```
Agent-R
├── Position Calculator (仓位计算)
├── Risk Assessor (LLM风险评估)
├── Circuit Breaker (熔断管理)
└── Exposure Monitor (暴露监控)
```

**能力升级：**

| 能力 | 1.0 | 2.0 |
|------|-----|-----|
| 仓位计算 | 固定公式 | LLM考虑情境调整 |
| 风险评估 | 阈值规则 | LLM识别复合风险 |
| 熔断判断 | 固定阈值 | LLM理解因果链 |
| 组合优化 | 简单分散 | LLM优化相关性 |

---

### 3.5 Agent-L (学习迭代) — 核心升级

```
Agent-L
├── Trade Recorder (交易记录)
├── Pattern Miner (LLM模式挖掘)
├── Strategy Evolver (LLM策略演化)
├── Memory Manager (记忆管理)
└── Self-Reflector (自我反思)
```

**能力升级（核心）：**

| 能力 | 1.0 | 2.0 |
|------|-----|-----|
| 学习触发 | 固定周期 | 每笔交易后即时学习 |
| 因子调整 | IC阈值规则 | LLM分析因果 |
| 模式发现 | 简单统计 | LLM语义聚类 |
| 策略演化 | 月度淘汰 | 持续进化 |
| 知识积累 | 规则存储 | 向量记忆检索 |

**Agent-L详细流程：**

```python
class AgentLearner:
    """
    自主学习核心：
    1. 记录交易 → 2. 分析结果 → 3. 提取模式 → 4. 更新记忆 → 5. 演化策略
    """
    
    async def learn_from_trade(self, trade):
        # Step 1: 记录交易
        await self.record_trade(trade)
        
        # Step 2: LLM分析
        analysis = await self.llm.analyze([
            f"这笔{trade.direction}交易，入场{trade.entry_price}，"
            f"出场{trade.exit_price}，盈亏{trade.pnl_pct}%。"
            f"持仓时长{trade.duration}小时。"
            f"当时的市场状态：{trade.market_context}"
            f"当时的信号：{trade.signals}"
            f"请分析：为什么赚钱/亏损？下次如何改进？"
        ])
        
        # Step 3: 提取可操作洞察
        insights = await self.extract_actionable_insights(analysis)
        
        # Step 4: 更新记忆系统
        if insights:
            await self.memory.add(insights, type="lesson")
        
        # Step 5: 检查是否需要策略调整
        if self.should_evolve_strategy(insights):
            await self.evolve_strategy(insights)
    
    async def extract_actionable_insights(self, analysis):
        """LLM从分析中提取可操作的洞察"""
        prompt = f"""
        从以下交易分析中，提取可操作的洞察（不是泛泛而谈，而是具体可执行的改进）：

        {analysis}

        格式：
        - 什么情况下应该多做/少做？
        - 哪些因子组合有效/无效？
        - 仓位应该如何调整？
        - 入场/出场时机如何优化？
        """
        return await self.llm.extract_structured(prompt)
```

---

### 3.6 Memory System (记忆系统)

```
Memory Architecture
├── Vector Memory (ChromaDB)
│   ├── 交易经验 embeddings
│   ├── 市场状态 embeddings
│   └── 策略模式 embeddings
├── Structured Memory (SQLite)
│   ├── trade_records
│   ├── factor_performance
│   └── strategy_params
├── Few-shot Examples
│   ├── profitable_patterns
│   └── loss_patterns
└── Policy Memory (规则库)
    ├── hard_rules (不可违反)
    └── soft_rules (可调整)
```

**记忆检索：**
```python
class MemorySystem:
    async def retrieve_relevant(self, query, k=5):
        """检索最相关的历史经验"""
        # 向量相似度检索
        similar = await self.vector_db.similarity_search(query, k)
        
        # LLM从中选择最相关的
        selected = await self.llm.select_relevant(query, similar)
        
        return selected
    
    async def add_experience(self, experience):
        """新增经验到记忆"""
        # 存储向量
        await self.vector_db.add(experience)
        
        # 提取关键词更新结构化索引
        keywords = await self.llm.extract_keywords(experience)
        await self.structured.add(experience, keywords)
```

---

### 3.7 Tool Ecosystem (工具生态)

```
Tools
├── Data Tools
│   ├── get_price_data()      # OKX/Binance价格
│   ├── get_news()            # 新闻获取
│   └── get_onchain()         # 链上数据
├── Indicator Tools
│   ├── calc_rsi()
│   ├── calc_adx()
│   ├── calc_macd()
│   └── calc_atr()
├── Execution Tools
│   ├── place_order()
│   ├── close_position()
│   └── get_positions()
├── Analysis Tools (MCP)
│   └── (可扩展外部能力)
└── LLM Tools
    ├── reason()
    ├── reflect()
    └── plan()
```

---

## 四、LLM驱动的工作流

### 4.1 每日交易循环

```python
async def daily_trading_cycle():
    """
    每日交易循环：观察 → 思考 → 行动 → 学习
    """
    # Phase 1: 观察 (Scanning)
    market_data = await agent_m.scan_all_coins()
    
    # Phase 2: 思考 (Reasoning)
    opportunities = await orchestrator.evaluate(market_data)
    
    # Phase 3: 行动 (Acting)
    for opp in opportunities:
        signal = await agent_s.generate_signal(opp)
        risk_check = await agent_r.assess(signal)
        
        if risk_check.approved:
            execution = await agent_e.execute(signal)
            
            # Phase 4: 学习 (Learning)
            await agent_l.learn_from_trade(execution)
    
    # Phase 5: 反思 (Reflection)
    await orchestrator.reflect_on_day()
```

### 4.2 ReAct决策模式

```
ReAct = Reasoning + Acting

┌─────────────────────────────────────────┐
│           Thought → Action → Observe     │
├─────────────────────────────────────────┤
│                                         │
│  Thought: "BTC RSI=38，显示超卖...       │
│           新闻偏利好，可能反弹"            │
│                     ↓                    │
│  Action: 做多 BTC @ 54000               │
│                     ↓                    │
│  Observe: "持仓2小时后，盈利2%...        │
│           但ADX开始下降，趋势可能减弱"     │
│                     ↓                    │
│  Thought: "是否应该止盈？RR已达2.0..."   │
│                     ↓                    │
│  Action: 移动止损到保本                  │
│                                         │
└─────────────────────────────────────────┘
```

---

## 五、与1.0的平滑迁移

### 迁移策略：渐进式升级

```
Phase 1: 基础设施
├── 集成LLM Provider (Claude/GPT)
├── 搭建Memory System
└── 保留所有1.0规则

Phase 2: 增强模式
├── Agent-M LLM增强
├── Agent-S LLM辅助
└── Agent-L 初步学习

Phase 3: 主导模式
├── Orchestrator LLM决策
├── 1.0规则作为约束检查
└── 自主策略演化启用

Phase 4: 完全自主
├── 规则系统完全由LLM管理
├── 持续自我改进
└── 无人干预
```

### 向后兼容

```python
class HybridMode:
    """混合模式：1.0规则 + 2.0 LLM"""
    
    def decide(self, market_data):
        # 1.0 规则信号
        rule_signal = self.rule_based.decide(market_data)
        
        # 2.0 LLM信号
        llm_signal = await self.llm.decide(market_data)
        
        # 置信度融合
        if llm_signal.confidence > 0.8:
            return llm_signal  # LLM高置信时信任LLM
        elif abs(rule_signal.direction) != abs(llm_signal.direction):
            return rule_signal  # 方向矛盾时用规则
        else:
            # 加权融合
            return self.blend(rule_signal, llm_signal)
```

---

## 六、技术栈

| 组件 | 技术选型 |
|------|---------|
| LLM Provider | Claude (主力), GPT-4o (备用), DeepSeek (成本优化) |
| 向量数据库 | ChromaDB (本地) |
| 结构化存储 | SQLite (交易记录), JSON (配置) |
| Agent框架 | 自研 (基于Golemancy理念) |
| 数据源 | OKX API, Binance API, News API |
| MCP协议 | 原生支持 (扩展外部能力) |
| 执行 | OKX Futures API |

---

## 七、文件结构

```
miracle_system_v2/
├── README.md
├── ARCHITECTURE.md              ← 本文件
│
├── core/
│   ├── orchestrator.py          ← 协调器 (LLM大脑)
│   ├── memory_system.py          ← 记忆系统
│   ├── llm_provider.py          ← LLM统一接口
│   ├── tools/
│   │   ├── data_tools.py        ← 数据获取
│   │   ├── indicator_tools.py   ← 技术指标
│   │   └── execution_tools.py   ← 交易所执行
│   └── utils.py
│
├── agents/
│   ├── agent_market.py          ← Agent-M 市场情报
│   ├── agent_signal.py          ← Agent-S 信号生成
│   ├── agent_risk.py            ← Agent-R 风险管理
│   ├── agent_executor.py        ← Agent-E 执行引擎
│   └── agent_learner.py         ← Agent-L 学习迭代
│
├── memory/
│   ├── vector_store.py          ← ChromaDB向量存储
│   ├── structured_store.py      ← SQLite结构化存储
│   ├── few_shot_examples.py     ← Few-shot示例库
│   └── policy_store.py          ← 规则策略库
│
├── prompts/
│   ├── orchestrator_prompt.md
│   ├── agent_m_prompt.md
│   ├── agent_s_prompt.md
│   ├── agent_r_prompt.md
│   └── agent_l_prompt.md
│
├── config/
│   ├── llm_config.json          ← LLM配置
│   ├── agent_config.json        ← Agent配置
│   └── memory_config.json       ← 记忆配置
│
├── data/
│   ├── trades.db                ← 交易记录
│   └── memories/                ← 记忆存储
│
└── scripts/
    ├── migrate_v1_to_v2.py       ← 迁移脚本
    └── evaluate.py              ← 评估脚本
```

---

## 八、关键设计原则

### 8.1 赔率永不妥协
```python
# 所有决策必须满足最小RR=2.0
class RiskConstraint:
    MIN_RR = 2.0
    
    def validate(self, signal):
        if signal.rr_ratio < self.MIN_RR:
            return False, f"RR={signal.rr_ratio} < {self.MIN_RR}"
        return True, "OK"
```

### 8.2 LLM作为推理引擎，而非规则替代
- LLM负责推理和决策
- 硬性约束（如止损、熔断）仍用规则
- 规则可以随着学习被调整，但不能被完全绕过

### 8.3 记忆驱动学习
- 每笔交易都产生学习素材
- 重要经验永久存储
- 遗忘机制防止记忆饱和

### 8.4 渐进式自主
- Phase 1: LLM辅助人类决策
- Phase 2: LLM主导，人类监督
- Phase 3: LLM完全自主
- 每个阶段都可回退

---

## 九、风险控制

| 风险类型 | 控制措施 |
|---------|---------|
| LLM幻觉 | 规则约束兜底，多LLM交叉验证 |
| 过拟合 | 向量化泛化，定期压力测试 |
| 模型切换 | 主备LLM自动切换 |
| 记忆污染 | 经验分级，高置信才入主记忆 |
| 过度交易 | 熔断规则 + 交易间隔限制 |

---

## 十、路线图

| 阶段 | 时间 | 目标 |
|------|------|------|
| Phase 1 | Week 1-2 | LLM基础设施 + Agent-M增强 |
| Phase 2 | Week 3-4 | Orchestrator + Agent-S |
| Phase 3 | Week 5-6 | Agent-L学习循环 |
| Phase 4 | Week 7-8 | 记忆系统 + 策略演化 |
| Phase 5 | Week 9-10 | 全系统集成 + 回测 |
| Phase 6 | Week 11-12 | 模拟盘验证 |

---

**文档版本：** 1.0  
**创建日期：** 2026-04-27  
**状态：** 设计待实现
