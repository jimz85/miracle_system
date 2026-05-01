# Miracle+TradingAgents 融合架构设计

**文档版本:** v1.0  
**生成时间:** 2026-04-27  
**架构师:** AI System Architect

---

## 一、架构设计原则

| 原则 | 说明 |
|------|------|
| **Kronos为骨架** | OKX封装、熔断、IC投票、Cron调度全部复用 |
| **TradingAgents为智能层** | 多Agent辩论、Structured Output、快慢思考 |
| **保留技术面因子系统** | IC权重、熔断、五级生存层 |
| **新增多空辩论层** | Bull/Bear Researcher辩论机制 |
| **新增Structured Output** | Pydantic决策格式统一输出 |
| **新增Memory Log** | 交易记忆与反思机制 |
| **快慢思考分离** | deep_think_llm + quick_think_llm |

---

## 二、系统整体架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Cron Scheduler (Hermes)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │              Miracle+TradingAgents Fusion Layer                          │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                          │ │
│  │   [每3分钟]  fusion_scanner.py  ← 主扫描循环                            │ │
│  │        │                                                                  │ │
│  │        ├── 读取 factor_context.json (MiniMax小时层输出)                   │ │
│  │        ├── OKX实时行情 + 本地CSV历史数据                                 │ │
│  │        ├── 因子计算 (RSI/ADX/Bollinger/MACD/Vol)                       │ │
│  │        ├── IC权重投票 (voting_system.py)                                  │ │
│  │        │                                                                  │ │
│  │        ├── [新增] Multi-Agent Debate Layer                               │ │
│  │        │    ├── Technical Analyst (快速技术分析)                         │ │
│  │        │    ├── Market Intel Agent (市场情报)                            │ │
│  │        │    ├── Bull Researcher (多头论点)                              │ │
│  │        │    ├── Bear Researcher (空头论点)                              │ │
│  │        │    └── Debate Judge (辩论裁决)                                  │ │
│  │        │                                                                  │ │
│  │        ├── [新增] Structured Output决策格式                              │ │
│  │        │    └── FusionDecision schema (统一决策输出)                     │ │
│  │        │                                                                  │ │
│  │        ├── [保留] 熔断检查 (check_treasury_limits)                       │ │
│  │        └── 执行 or 等待 ← 决策结果                                       │ │
│  │                                                                          │ │
│  │   [每5分钟]  position_monitor.py  ← 持仓监控                            │ │
│  │        │                                                                  │ │
│  │        ├── 检查所有持仓的SL/TP触发                                       │ │
│  │        ├── 更新positions.json                                           │ │
│  │        ├── 权益快照更新                                                 │ │
│  │        └── 熔断层级检查                                                 │ │
│  │                                                                          │ │
│  │   [每小时]   heartbeat_learning.py  ← 复盘+学习                         │ │
│  │        │                                                                  │ │
│  │        ├── 持仓复盘                                                     │ │
│  │        ├── IC权重更新                                                   │ │
│  │        ├── [新增] Memory Log反馈更新                                     │ │
│  │        └── 白名单/黑名单更新                                             │ │
│  │                                                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、新增模块详细设计

### 3.1 Multi-Agent Debate Layer (多空辩论层)

#### 模块职责
- 整合TradingAgents的多Agent辩论机制到Kronos扫描循环
- 支持Bull/Bear双视角研究分析
- 通过结构化辩论识别市场矛盾点

#### 输入输出

```python
# 输入
class DebateInput:
    ticker: str                      # 交易标的
    price_data: PriceData            # 价格数据
    factor_context: FactorContext    # 因子上下文 (RSI, ADX, MACD, BB)
    market_intel: MarketIntel        # 市场情报
    ic_weights: ICWeights            # IC权重

# 输出
class DebateOutput:
    bull_case: str                  # 多头论点
    bear_case: str                  # 空头论点
    debate_verdict: DebateVerdict   # BUY / HOLD / SELL
    confidence: float               # 置信度 0-1
    key_insights: List[str]         # 关键洞察
```

#### 内部Agent结构

```
Bull Researcher (quick_think_llm)
    │
    ├── 分析factor_context中的做多信号
    ├── 识别支撑位、趋势确认
    └── 输出: bull_case, bull_evidence[]

Bear Researcher (quick_think_llm)
    │
    ├── 分析factor_context中的做空信号
    ├── 识别阻力位、趋势反转
    └── 输出: bear_case, bear_evidence[]

Debate Judge (deep_think_llm)
    │
    ├── 综合bull/bear论点
    ├── 评估证据权重
    └── 输出: debate_verdict, confidence, key_insights
```

#### 调用关系

```python
# fusion_debate.py
class FusionDebateLayer:
    def __init__(self, llm_clients):
        self.bull_researcher = BullResearcher(quick_llm)
        self.bear_researcher = BearResearcher(quick_llm)
        self.judge = DebateJudge(deep_llm)
    
    def run_debate(self, input: DebateInput) -> DebateOutput:
        # 1. 并行运行多空研究
        bull_future = self.bull_researcher.analyze(input)
        bear_future = self.bear_researcher.analyze(input)
        
        bull_result = bull_future.result()
        bear_result = bear_future.result()
        
        # 2. 裁决辩论
        verdict = self.judge.arbitrate(bull_result, bear_result)
        
        return DebateOutput(
            bull_case=bull_result.case,
            bear_case=bear_result.case,
            debate_verdict=verdict.decision,
            confidence=verdict.confidence,
            key_insights=verdict.insights
        )
```

---

### 3.2 Structured Output决策格式

#### 模块职责
- 统一TradingAgents的Pydantic决策schema到融合系统
- 确保决策输出的确定性、可解析性
- 支撑下游执行层的原子化操作

#### 决策Schema定义

```python
# fusion_decision.py
from pydantic import BaseModel, Field
from enum import Enum

class TradeAction(str, Enum):
    """交易动作枚举"""
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    WAIT = "Wait"  # 新增: 观望

class PositionRating(str, Enum):
    """仓位评级枚举"""
    STRONG_BUY = "StrongBuy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "StrongSell"

class FusionDecision(BaseModel):
    """融合系统统一决策格式"""
    
    # 核心决策
    action: TradeAction = Field(
        description="交易动作: Buy / Hold / Sell / Wait"
    )
    rating: PositionRating = Field(
        description="仓位评级: StrongBuy / Buy / Hold / Sell / StrongSell"
    )
    
    # 置信度与理由
    confidence: float = Field(
        description="置信度 0.0-1.0",
        ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="决策理由，2-4句话总结核心逻辑"
    )
    
    # 执行参数
    entry_price: float | None = Field(
        default=None,
        description="推荐入场价格"
    )
    stop_loss: float | None = Field(
        default=None,
        description="止损价格"
    )
    take_profit: float | None = Field(
        default=None,
        description="止盈价格"
    )
    position_size: str | None = Field(
        default=None,
        description="仓位大小，如 '5% of portfolio'"
    )
    
    # 辩论层输出
    bull_evidence: list[str] = Field(
        default_factory=list,
        description="多头证据列表"
    )
    bear_evidence: list[str] = Field(
        default_factory=list,
        description="空头证据列表"
    )
    debate_insights: list[str] = Field(
        default_factory=list,
        description="辩论关键洞察"
    )
    
    # IC与因子信息
    ic_weights: dict[str, float] = Field(
        default_factory=dict,
        description="因子IC权重"
    )
    factor_signals: dict[str, float] = Field(
        default_factory=dict,
        description="各因子信号值"
    )
    
    # 风控信息
    risk_level: str = Field(
        description="风险等级: low / medium / high / critical"
    )
    survival_tier: str = Field(
        description="生存层级: normal / caution / low / critical / paused"
    )

class FusionDecisionRenderer:
    """将FusionDecision渲染为Markdown格式"""
    
    @staticmethod
    def render(decision: FusionDecision) -> str:
        parts = [
            f"**Action**: {decision.action.value}",
            f"**Rating**: {decision.rating.value}",
            f"**Confidence**: {decision.confidence:.1%}",
            "",
            f"**Reasoning**: {decision.reasoning}",
        ]
        
        if decision.entry_price:
            parts.extend(["", f"**Entry Price**: {decision.entry_price}"])
        if decision.stop_loss:
            parts.extend(["", f"**Stop Loss**: {decision.stop_loss}"])
        if decision.take_profit:
            parts.extend(["", f"**Take Profit**: {decision.take_profit}"])
        if decision.position_size:
            parts.extend(["", f"**Position Size**: {decision.position_size}"])
        
        parts.extend(["", "**Bull Evidence**:"])
        for evidence in decision.bull_evidence:
            parts.append(f"- {evidence}")
        
        parts.append("**Bear Evidence**:")
        for evidence in decision.bear_evidence:
            parts.append(f"- {evidence}")
        
        parts.extend(["", f"**Risk Level**: {decision.risk_level}"])
        parts.extend(["", f"**Survival Tier**: {decision.survival_tier}"])
        
        return "\n".join(parts)
```

---

### 3.3 Memory Log记忆层

#### 模块职责
- 持久化交易决策与结果
- 支持跨周期学习的反思机制
- 注入历史上下文到Agent prompt

#### 交互接口

```python
# fusion_memory.py
from pathlib import Path
from typing import List, Optional
import re

class FusionMemoryLog:
    """融合系统的交易记忆日志"""
    
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    
    def __init__(self, config: dict):
        self._log_path = Path(config.get(
            "memory_log_path", 
            "~/.miracle_trading/memory/trading_memory.md"
        )).expanduser()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = config.get("memory_log_max_entries", 1000)
    
    # ========== 写入接口 ==========
    
    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        decision: FusionDecision,
        outcome: str = "pending"
    ) -> None:
        """存储交易决策（Phase A: 决策时）"""
        entry = self._build_entry(ticker, trade_date, decision, outcome)
        self._append_entry(entry)
    
    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float,
        holding_days: int,
        reflection: str
    ) -> None:
        """更新决策结果与反思（Phase B: 结算时）"""
        # 原子性写入：读->修改->写
        text = self._read_log()
        blocks = text.split(self._SEPARATOR)
        
        updated_blocks = []
        for block in blocks:
            if self._match_pending_entry(block, ticker, trade_date):
                new_block = self._build_resolved_block(
                    block, raw_return, alpha_return, holding_days, reflection
                )
                updated_blocks.append(new_block)
            else:
                updated_blocks.append(block)
        
        # 实施rotation
        updated_blocks = self._apply_rotation(updated_blocks)
        self._write_log(self._SEPARATOR.join(updated_blocks))
    
    # ========== 读取接口 ==========
    
    def get_past_context(
        self, 
        ticker: str, 
        n_same: int = 5, 
        n_cross: int = 3
    ) -> str:
        """获取历史上下文用于prompt注入"""
        entries = self.load_resolved_entries()
        if not entries:
            return ""
        
        same_ticker, cross_ticker = [], []
        for e in reversed(entries):
            if len(same_ticker) >= n_same and len(cross_ticker) >= n_cross:
                break
            if e["ticker"] == ticker and len(same_ticker) < n_same:
                same_ticker.append(e)
            elif e["ticker"] != ticker and len(cross_ticker) < n_cross:
                cross_ticker.append(e)
        
        parts = []
        if same_ticker:
            parts.append(f"=== Past analyses of {ticker} (most recent first) ===")
            for e in same_ticker:
                parts.append(self._format_full_entry(e))
        
        if cross_ticker:
            parts.append("=== Recent cross-ticker lessons ===")
            for e in cross_ticker:
                parts.append(self._format_reflection_only(e))
        
        return "\n\n".join(parts)
    
    def get_ic_feedback(self, ticker: str) -> dict:
        """获取IC反馈用于权重更新"""
        entries = self.load_resolved_entries()
        ticker_entries = [e for e in entries if e["ticker"] == ticker]
        
        return {
            "sample_size": len(ticker_entries),
            "avg_alpha": sum(e.get("alpha", 0) for e in ticker_entries) / max(len(ticker_entries), 1),
            "win_rate": sum(1 for e in ticker_entries if e.get("alpha", 0) > 0) / max(len(ticker_entries), 1)
        }
    
    # ========== 内部方法 ==========
    
    def _build_entry(
        self, 
        ticker: str, 
        trade_date: str, 
        decision: FusionDecision,
        outcome: str
    ) -> str:
        tag = f"[{trade_date} | {ticker} | {decision.action.value} | {outcome}]"
        body = f"""DECISION:
{FusionDecisionRenderer.render(decision)}"""
        return f"{tag}\n\n{body}{self._SEPARATOR}"
    
    def _append_entry(self, entry: str) -> None:
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(entry)
    
    def _apply_rotation(self, blocks: List[str]) -> List[str]:
        """当resolved条目超过_max_entries时，丢弃最旧的"""
        # 实现rotation逻辑
        pass
    
    def load_resolved_entries(self) -> List[dict]:
        """加载所有已结算的条目"""
        pass
    
    def _parse_entry(self, raw: str) -> Optional[dict]:
        """解析单个条目"""
        pass
```

---

### 3.4 快慢思考模型分离

#### 模块职责
- 区分快速决策与深度思考场景
- 优化LLM调用成本
- 保证决策质量与响应速度平衡

#### 模型分配策略

```python
# model_router.py
from enum import Enum

class ThinkMode(str, Enum):
    FAST = "fast"      # 快速思考
    DEEP = "deep"      # 深度思考

class ModelRouter:
    """快慢思考模型路由器"""
    
    # 模型分配映射
    MODEL_ASSIGNMENTS = {
        # 快速思考场景 (quick_think_llm)
        ThinkMode.FAST: {
            "market_analyst": "gpt-5.4-mini",
            "technical_analyst": "gpt-5.4-mini",
            "bull_researcher": "gpt-5.4-mini",
            "bear_researcher": "gpt-5.4-mini",
            "signal_processor": "gpt-5.4-mini",
        },
        # 深度思考场景 (deep_think_llm)
        ThinkMode.DEEP: {
            "research_manager": "gpt-5.4",
            "debate_judge": "gpt-5.4",
            "portfolio_manager": "gpt-5.4",
            "risk_analyst": "gpt-5.4",
            "strategy_formulator": "gpt-5.4",
        }
    }
    
    def __init__(self, llm_clients: dict):
        self.clients = llm_clients
    
    def get_llm(self, task: str, mode: ThinkMode) -> Any:
        """根据任务类型和思考模式获取合适的LLM"""
        model_name = self.MODEL_ASSIGNMENTS[mode].get(task)
        if not model_name:
            # 默认使用快速模型
            model_name = "gpt-5.4-mini"
        
        return self.clients.get(model_name)
    
    def should_use_deep(self, context: dict) -> bool:
        """判断是否需要深度思考"""
        # 场景1: IC权重更新 (高风险)
        if context.get("task") == "ic_weight_update":
            return True
        
        # 场景2: 多因子冲突时
        signals = context.get("factor_signals", {})
        if len(signals) >= 3:
            # 检查是否有明显冲突
            buy_signals = sum(1 for v in signals.values() if v > 0.5)
            sell_signals = sum(1 for v in signals.values() if v < -0.5)
            if buy_signals > 0 and sell_signals > 0:
                return True
        
        # 场景3: 置信度低于阈值
        if context.get("confidence", 1.0) < 0.4:
            return True
        
        return False
```

---

### 3.5 技术面因子子系统 (保留增强)

#### 模块职责
- 保留Kronos的IC权重投票系统
- 增强ATR、布林带等因子计算
- 集成到辩论层作为证据输入

#### 接口定义

```python
# factor_subsystem.py
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class FactorContext:
    """因子上下文"""
    ticker: str
    timestamp: str
    
    # 价格数据
    closes: List[float]
    highs: List[float]
    lows: List[float]
    volumes: List[float]
    
    # 计算因子
    rsi: float
    adx: float
    macd: float
    macd_signal: float
    macd_hist: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_position: float  # 0-1, 价格在布林带位置
    atr: float
    
    # IC权重
    ic_weights: Dict[str, float]
    
    # 市场情报
    funding_rate: float
    market_sentiment: str  # bullish / bearish / neutral

class FactorSubsystem:
    """技术面因子子系统"""
    
    def __init__(self, config: dict):
        self.ic_calculator = ICWeightCalculator()
        self.atr_calculator = ATRCalculator()
        self.confidence_normalizer = ConfidenceNormalizer()
    
    def calculate_factors(
        self, 
        ticker: str, 
        klines: List[KLine]
    ) -> FactorContext:
        """计算所有技术因子"""
        closes = [k.close for k in klines]
        highs = [k.high for k in klines]
        lows = [k.low for k in klines]
        volumes = [k.volume for k in klines]
        
        # RSI (Wilder平滑)
        rsi = self._calc_rsi(closes)
        
        # ADX (Wilder平滑)
        adx = self._calc_adx(highs, lows, closes)
        
        # MACD
        macd, signal, hist = self._calc_macd(closes)
        
        #布林带
        bb_upper, bb_lower, bb_pos = self._calc_bollinger(closes)
        
        # ATR (正确Wilder实现)
        atr = self.atr_calculator.calculate(highs, lows, closes)
        
        # IC权重
        ic_weights = self.ic_calculator.get_weights(ticker)
        
        return FactorContext(
            ticker=ticker,
            timestamp=klines[-1].timestamp,
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            rsi=rsi,
            adx=adx,
            macd=macd,
            macd_signal=signal,
            macd_hist=hist,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            bollinger_position=bb_pos,
            atr=atr,
            ic_weights=ic_weights,
            funding_rate=0.0,
            market_sentiment="neutral"
        )
    
    def build_debate_evidence(
        self, 
        factors: FactorContext
    ) -> Dict[str, List[str]]:
        """从因子构建辩论证据"""
        evidence = {
            "bull_evidence": [],
            "bear_evidence": []
        }
        
        # RSI证据
        if factors.rsi < 30:
            evidence["bull_evidence"].append(
                f"RSI={factors.rsi:.1f} < 30 (超卖区域)"
            )
        elif factors.rsi > 70:
            evidence["bear_evidence"].append(
                f"RSI={factors.rsi:.1f} > 70 (超买区域)"
            )
        
        # ADX趋势强度证据
        if factors.adx > 25:
            if factors.macd_hist > 0:
                evidence["bull_evidence"].append(
                    f"ADX={factors.adx:.1f} > 25 + MACD histogram > 0 (趋势确认)"
                )
            else:
                evidence["bear_evidence"].append(
                    f"ADX={factors.adx:.1f} > 25 + MACD histogram < 0 (下降趋势)"
                )
        
        # 布林带位置证据
        if factors.bollinger_position < 0.2:
            evidence["bull_evidence"].append(
                f"价格在布林带{factors.bollinger_position:.1%}位置 (接近下轨)"
            )
        elif factors.bollinger_position > 0.8:
            evidence["bear_evidence"].append(
                f"价格在布林带{factors.bollinger_position:.1%}位置 (接近上轨)"
            )
        
        # IC权重加权信号
        weighted_signal = sum(
            factors.ic_weights.get(k, 0) * v 
            for k, v in factors.factor_signals.items()
        )
        
        return evidence
```

---

### 3.6 熔断子系统 (保留增强)

#### 模块职责
- 保留Kronos五级生存层机制
- 集成到决策流程作为否决权
- 渐进恢复机制

#### 接口定义

```python
# circuit_breaker.py
from enum import Enum

class SurvivalTier(str, Enum):
    """五级生存层"""
    NORMAL = "normal"          # 正常交易
    CAUTION = "caution"        # 谨慎交易 (50%仓位)
    LOW = "low"               # 低频交易 (25%仓位)
    CRITICAL = "critical"      # 仅平仓 (0%开仓)
    PAUSED = "paused"         # 全暂停

class CircuitBreaker:
    """熔断子系统"""
    
    # 五级阈值
    TIER_THRESHOLDS = {
        SurvivalTier.NORMAL: 0.0,      # 无损失
        SurvivalTier.CAUTION: -0.05,   # 亏损5%
        SurvivalTier.LOW: -0.10,       # 亏损10%
        SurvivalTier.CRITICAL: -0.20, # 亏损20%
        SurvivalTier.PAUSED: -0.30,    # 亏损30%
    }
    
    # 渐进恢复步长
    RECOVERY_STEPS = {
        SurvivalTier.CAUTION: 0.25,   # 恢复至25%
        SurvivalTier.LOW: 0.50,       # 恢复至50%
        SurvivalTier.CRITICAL: 0.75,  # 恢复至75%
    }
    
    def __init__(self, config: dict):
        self.current_tier = SurvivalTier.NORMAL
        self.consecutive_losses = 0
        self.equity_snapshot = EquitySnapshot()
    
    def check_treasury_limits(
        self, 
        current_equity: float,
        positions: List[Position]
    ) -> CircuitBreakerResult:
        """检查熔断限制"""
        
        # 1. 更新权益快照
        self.equity_snapshot.update(current_equity)
        
        # 2. 计算当前亏损
        initial_equity = self.equity_snapshot.get_initial()
        drawdown = (current_equity - initial_equity) / initial_equity
        
        # 3. 确定生存层级
        new_tier = self._determine_tier(drawdown)
        
        # 4. 渐进恢复检查
        if new_tier == SurvivalTier.NORMAL and self.current_tier != SurvivalTier.NORMAL:
            new_tier = self._check_recovery()
        
        self.current_tier = new_tier
        
        # 5. 构建结果
        return CircuitBreakerResult(
            allowed= new_tier not in [SurvivalTier.PAUSED, SurvivalTier.CRITICAL],
            tier=new_tier,
            max_position_pct=self._get_max_position_pct(new_tier),
            can_open=new_tier in [SurvivalTier.NORMAL, SurvivalTier.CAUTION],
            can_close=True,  # 任何层级都可平仓
            reason=self._get_tier_reason(new_tier, drawdown)
        )
    
    def _determine_tier(self, drawdown: float) -> SurvivalTier:
        """根据回撤确定生存层"""
        if drawdown <= self.TIER_THRESHOLDS[SurvivalTier.CAUTION]:
            return SurvivalTier.NORMAL
        elif drawdown <= self.TIER_THRESHOLDS[SurvivalTier.LOW]:
            return SurvivalTier.CAUTION
        elif drawdown <= self.TIER_THRESHOLDS[SurvivalTier.CRITICAL]:
            return SurvivalTier.LOW
        elif drawdown <= self.TIER_THRESHOLDS[SurvivalTier.PAUSED]:
            return SurvivalTier.CRITICAL
        else:
            return SurvivalTier.PAUSED
    
    def _get_max_position_pct(self, tier: SurvivalTier) -> float:
        """获取最大持仓比例"""
        if tier == SurvivalTier.NORMAL:
            return 1.0
        elif tier == SurvivalTier.CAUTION:
            return 0.5
        elif tier == SurvivalTier.LOW:
            return 0.25
        elif tier == SurvivalTier.CRITICAL:
            return 0.0
        else:
            return 0.0
    
    def record_trade_outcome(self, pnl: float):
        """记录交易结果用于连亏计数"""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def _check_recovery(self) -> SurvivalTier:
        """检查是否可以从当前层级恢复"""
        # 连亏次数清零后需要一定盈利才能恢复
        if self.consecutive_losses == 0:
            return SurvivalTier.NORMAL
        return self.current_tier

@dataclass
class CircuitBreakerResult:
    allowed: bool
    tier: SurvivalTier
    max_position_pct: float
    can_open: bool
    can_close: bool
    reason: str
```

---

## 四、主流程融合设计

### 4.1 Fusion Scanner (主扫描循环)

```python
# fusion_scanner.py
class FusionScanner:
    """融合系统主扫描器"""
    
    def __init__(self, config: dict):
        # 初始化各子系统
        self.factor_subsystem = FactorSubsystem(config)
        self.debate_layer = FusionDebateLayer(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.memory_log = FusionMemoryLog(config)
        self.model_router = ModelRouter(config)
        
        # OKX接口
        self.okx_client = OKXClient(config)
        
        # IC权重
        self.voting_system = VotingSystem(config)
    
    def scan(self, ticker: str) -> Optional[FusionDecision]:
        """主扫描流程"""
        
        # Step 1: 获取市场数据
        klines = self.okx_client.get_klines(ticker, "1h", limit=100)
        price_data = self.okx_client.get_ticker_price(ticker)
        
        # Step 2: 计算技术因子
        factors = self.factor_subsystem.calculate_factors(ticker, klines)
        
        # Step 3: 获取市场情报
        market_intel = self._fetch_market_intel(ticker)
        
        # Step 4: IC权重投票
        ic_vote = self.voting_system.vote(factors)
        
        # Step 5: 多空辩论
        debate_input = DebateInput(
            ticker=ticker,
            price_data=price_data,
            factor_context=factors,
            market_intel=market_intel,
            ic_weights=factors.ic_weights
        )
        debate_output = self.debate_layer.run_debate(debate_input)
        
        # Step 6: 构建初步决策
        initial_decision = self._build_initial_decision(
            debate_output, ic_vote, factors
        )
        
        # Step 7: 熔断检查
        current_equity = self.okx_client.get_account_equity()
        positions = self.okx_client.get_positions()
        
        cb_result = self.circuit_breaker.check_treasury_limits(
            current_equity, positions
        )
        
        # Step 8: 应用熔断限制
        final_decision = self._apply_circuit_breaker(
            initial_decision, cb_result
        )
        
        # Step 9: 存储决策
        self.memory_log.store_decision(
            ticker=ticker,
            trade_date=self._get_current_date(),
            decision=final_decision
        )
        
        # Step 10: 执行或等待
        if final_decision.action in [TradeAction.BUY, TradeAction.SELL]:
            return self._execute_decision(final_decision)
        
        return None
    
    def _build_initial_decision(
        self, 
        debate: DebateOutput, 
        ic_vote: ICVote,
        factors: FactorContext
    ) -> FusionDecision:
        """构建初始决策（熔断前）"""
        
        # IC权重加权
        weighted_confidence = (
            debate.confidence * 0.4 + 
            ic_vote.confidence * 0.6
        )
        
        # 综合理由
        reasoning = (
            f"辩论结论: {debate.debate_verdict} "
            f"(置信度{debate.confidence:.0%}), "
            f"IC投票: {ic_vote.direction} "
            f"(权重{ic_vote.weight:.2f})"
        )
        
        return FusionDecision(
            action=TradeAction(debate.debate_verdict.value),
            rating=self._compute_rating(debate, ic_vote),
            confidence=weighted_confidence,
            reasoning=reasoning,
            entry_price=factors.closes[-1],
            stop_loss=self._compute_stop_loss(factors),
            take_profit=self._compute_take_profit(factors),
            bull_evidence=debate.bull_case.split("\n"),
            bear_evidence=debate.bear_case.split("\n"),
            debate_insights=debate.key_insights,
            ic_weights=factors.ic_weights,
            factor_signals=self._extract_factor_signals(factors),
            risk_level=self._compute_risk_level(factors),
            survival_tier="normal"
        )
    
    def _apply_circuit_breaker(
        self, 
        decision: FusionDecision, 
        cb: CircuitBreakerResult
    ) -> FusionDecision:
        """应用熔断限制"""
        
        # 如果禁止开仓，改为Hold
        if not cb.can_open and decision.action in [TradeAction.BUY, TradeAction.SELL]:
            return FusionDecision(
                action=TradeAction.HOLD,
                rating=decision.rating,
                confidence=decision.confidence * 0.5,  # 降低置信度
                reasoning=f"{decision.reasoning}\n[熔断限制: {cb.reason}]",
                # ... 其他字段复制
                survival_tier=cb.tier.value
            )
        
        return FusionDecision(
            survival_tier=cb.tier.value,
            # ... 其他字段复制
        )
```

---

## 五、模块调用关系图

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           FusionScanner.scan()                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                      │
│  │FactorSubsys │────>│ DebateLayer │────>│VoteSystem   │                      │
│  │ .calculate  │     │ .run_debate │     │ .vote       │                      │
│  └─────────────┘     └─────────────┘     └─────────────┘                      │
│         │                   │                   │                              │
│         │                   │                   │                              │
│         v                   v                   v                              │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │              DecisionBuilder._build_initial                  │              │
│  └─────────────────────────────────────────────────────────────┘              │
│                              │                                               │
│                              v                                               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                      │
│  │CircuitBreaker│────>│MemoryLog    │────>│Executor     │                      │
│  │ .check      │     │ .store      │     │ .execute    │                      │
│  └─────────────┘     └─────────────┘     └─────────────┘                      │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 六、数据流图

```
OKX API/REST
    │
    ├──> kronos_utils.okx_req() ─ 统一API封装
    │         │
    │         v
    │    klines 数据
    │         │
    ├──> FactorSubsystem.calculate_factors()
    │    ├── RSI (Wilder平滑)
    │    ├── ADX (Wilder平滑)
    │    ├── MACD
    │    ├── Bollinger Bands
    │    └── ATR (Wilder实现)
    │         │
    │         v
    ├──> VotingSystem.vote() ─ IC权重投票
    │         │
    │         v
    ├──> FusionDebateLayer.run_debate()
    │    ├── BullResearcher.analyze()
    │    ├── BearResearcher.analyze()
    │    └── Judge.arbitrate()
    │         │
    │         v
    ├──> FusionDecision (Structured Output)
    │         │
    │         v
    ├──> CircuitBreaker.check_treasury_limits()
    │         │
    │         v
    ├──> MemoryLog.store_decision()
    │         │
    │         v
    └──> Executor.open_position() / close_position()
              │
              └──> OKX OCO条件单 (SL + TP)
```

---

## 七、文件结构

```
~/miracle_trading/
├── fusion_scanner.py          # 主扫描入口
├── position_monitor.py        # 持仓监控 (5分钟)
├── heartbeat_learning.py      # 心跳学习 (小时级)
│
├── core/
│   ├── __init__.py
│   ├── factor_subsystem.py    # 技术因子计算
│   ├── circuit_breaker.py      # 熔断子系统
│   ├── voting_system.py       # IC权重投票 (复用Kronos)
│   └── okx_client.py         # OKX封装 (复用Kronos)
│
├── agents/
│   ├── __init__.py
│   ├── debate/
│   │   ├── __init__.py
│   │   ├── bull_researcher.py  # 多头研究Agent
│   │   ├── bear_researcher.py # 空头研究Agent
│   │   └── debate_judge.py    # 辩论裁决Agent
│   ├── market_intel.py        # 市场情报Agent
│   └── signal_processor.py    # 信号处理Agent
│
├── memory/
│   ├── __init__.py
│   ├── fusion_memory.py       # 记忆日志
│   └── reflection.py          # 反思生成
│
├── models/
│   ├── __init__.py
│   ├── fusion_decision.py     # Structured Output Schema
│   └── model_router.py        # 快慢思考路由
│
├── strategies/
│   ├── mean_reversion.py      # RSI均值回归
│   ├── trend_following.py     # EMA+ADX趋势跟踪
│   └── bollinger_break.py     # 布林带突破
│
├── config/
│   └── fusion_config.yaml     # 融合系统配置
│
└── memory/
    └── trading_memory.md      # 决策日志
```

---

## 八、接口汇总表

| 模块 | 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| **FactorSubsystem** | calculate_factors | ticker, klines | FactorContext | 计算所有技术因子 |
| **FactorSubsystem** | build_debate_evidence | FactorContext | Dict[bull/bear_evidence] | 构建辩论证据 |
| **FusionDebateLayer** | run_debate | DebateInput | DebateOutput | 执行多空辩论 |
| **FusionDebateLayer** | analyze_bull | DebateInput | BullResult | 多头分析 |
| **FusionDebateLayer** | analyze_bear | DebateInput | BearResult | 空头分析 |
| **FusionDebateLayer** | arbitrate | BullResult, BearResult | Verdict | 裁决辩论 |
| **CircuitBreaker** | check_treasury_limits | equity, positions | CircuitBreakerResult | 熔断检查 |
| **CircuitBreaker** | record_trade_outcome | pnl | None | 记录交易结果 |
| **FusionMemoryLog** | store_decision | ticker, date, decision | None | 存储决策 |
| **FusionMemoryLog** | update_with_outcome | ... | None | 更新结果与反思 |
| **FusionMemoryLog** | get_past_context | ticker, n_same, n_cross | str | 获取历史上下文 |
| **FusionMemoryLog** | get_ic_feedback | ticker | dict | 获取IC反馈 |
| **ModelRouter** | get_llm | task, mode | LLMClient | 获取合适LLM |
| **ModelRouter** | should_use_deep | context | bool | 判断是否深度思考 |
| **FusionDecision** | render | FusionDecision | str | 渲染为Markdown |

---

## 九、配置项

```yaml
# fusion_config.yaml
system:
  name: "Miracle+TradingAgents Fusion"
  version: "v1.0"
  
trading:
  default_holding_days: 24      # 默认持仓周期(小时)
  max_position_pct: 0.20       # 单币最大持仓比例
  stop_loss_atr_multiplier: 2.0 # ATR止损倍数
  take_profit_atr_multiplier: 3.0 # ATR止盈倍数

llm:
  provider: "openai"            # LLM提供商
  deep_think_llm: "gpt-5.4"    # 深度思考模型
  quick_think_llm: "gpt-5.4-mini" # 快速思考模型
  
debate:
  max_rounds: 2                # 最大辩论轮数
  confidence_threshold: 0.5    # 置信度阈值
  
memory:
  log_path: "~/.miracle_trading/memory/trading_memory.md"
  max_entries: 1000           # 最大记忆条目
  n_same_ticker: 5            # 同标的记忆数量
  n_cross_ticker: 3           # 跨标的记忆数量

circuit_breaker:
  enabled: true
  tiers:
    normal: 0.0               # 亏损0%
    caution: -0.05            # 亏损5%
    low: -0.10                # 亏损10%
    critical: -0.20            # 亏损20%
    paused: -0.30             # 亏损30%

ic_weights:
  decay_factor: 0.7           # 指数平滑因子
  min_samples: 10             # 最小样本数
  update_interval: 3600       # 更新间隔(秒)
```

---

## 十、验证计划

1. **单元测试** — 各模块独立验证
2. **集成测试** — 完整流程端到端测试
3. **回测验证** — 对比Kronos原始系统
4. **Paper交易** — 30天模拟盘
5. **实盘监控** — 7×24全天候

---

*文档结束*
