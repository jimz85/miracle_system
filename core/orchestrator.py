#!/usr/bin/env python3
"""
Miracle 2.0 - Orchestrator 协调器
=================================
LLM驱动的大脑，负责任务分解、结果聚合、自我反思

简化版本 - 保证可用
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_provider import get_llm_provider, LLMResponse

logger = logging.getLogger(__name__)

# ==================== 配置 ====================

DEFAULT_CONFIG = {
    "symbols": ["BTC", "ETH", "SOL", "AVAX", "DOGE", "DOT"],
    "min_rr": 2.0,
    "min_confidence": 0.6,
    "max_trades_per_day": 5,
    "llm_provider": "claude",
    "temperature": 0.7,
    "enable_reflection": True,
    "enable_memory": True,
}

# ==================== 数据模型 ====================

class DecisionType(Enum):
    EXECUTE = "EXECUTE"
    SKIP = "SKIP"
    WAIT = "WAIT"

@dataclass
class TradingDecision:
    decision: DecisionType
    symbol: str
    direction: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = 0.0
    leverage: int = 1
    rr_ratio: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    lessons: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size_pct": self.position_size_pct,
            "leverage": self.leverage,
            "rr_ratio": self.rr_ratio,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "lessons": self.lessons,
        }

@dataclass
class OrchestratorState:
    current_cycle: int = 0
    decisions_today: int = 0
    last_decision_time: Optional[str] = None
    consecutive_waits: int = 0
    total_reflections: int = 0

# ==================== 系统提示词 ====================

SYSTEM_PROMPT = """你是Miracle交易系统的首席交易员。

## 你的职责
1. 分析市场情报
2. 评估交易信号
3. 审核风险管理
4. 决定是否执行
5. 从每笔交易中学习

## 赔率优先原则
- 永远选择RR>=2.0的机会
- 输了只亏1%，赢了要赚2%以上
- 高置信度机会可适当加大仓位

## 决策流程
1. 理解当前市场状态
2. 评估各币种机会
3. 考虑风险约束
4. 选择最优策略
5. 明确执行计划

## 输出格式
返回JSON格式：
{
    "decision": "EXECUTE/SKIP/WAIT",
    "symbol": "BTC",
    "direction": "LONG/SHORT",
    "reasoning": "为什么这样做",
    "confidence": 0.85,
    "entry_price": 54000.0,
    "stop_loss": 53000.0,
    "take_profit": 56000.0,
    "lessons": "从这次决策学到了什么"
}
"""

# ==================== Orchestrator ====================

class Orchestrator:
    """
    LLM驱动的大脑协调器
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.state = OrchestratorState()
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """初始化LLM"""
        try:
            self.llm = get_llm_provider(
                provider=self.config.get("llm_provider", "claude"),
                model=self.config.get("model"),
                temperature=self.config.get("temperature", 0.7)
            )
            logger.info(f"Orchestrator: LLM initialized with {self.config.get('llm_provider', 'claude')}")
        except Exception as e:
            logger.warning(f"Orchestrator: LLM init failed, using rule-based fallback: {e}")
            self.llm = None

    async def decide(self, market_data: Dict[str, Any]) -> TradingDecision:
        """
        做交易决策
        """
        self.state.current_cycle += 1

        # 检查是否应该跳过
        if self.state.decisions_today >= self.config["max_trades_per_day"]:
            return TradingDecision(
                decision=DecisionType.SKIP,
                symbol="",
                direction="",
                reasoning="今日交易次数已达上限",
                confidence=0.0
            )

        # 如果有LLM，使用LLM决策
        if self.llm:
            return await self._llm_decide(market_data)

        # 否则使用规则决策
        return self._rule_based_decide(market_data)

    async def _llm_decide(self, market_data: Dict[str, Any]) -> TradingDecision:
        """使用LLM做决策"""
        prompt = f"""当前市场数据:
{json.dumps(market_data, ensure_ascii=False, indent=2)}

请做出交易决策。"""

        try:
            response = await self.llm.chat_simple(
                prompt,
                system=SYSTEM_PROMPT
            )

            if not response.error:
                content = response.content.strip()
                # 提取JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content)

                return TradingDecision(
                    decision=DecisionType(data.get("decision", "WAIT")),
                    symbol=data.get("symbol", ""),
                    direction=data.get("direction", ""),
                    entry_price=data.get("entry_price"),
                    stop_loss=data.get("stop_loss"),
                    take_profit=data.get("take_profit"),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    lessons=data.get("lessons", "")
                )
        except Exception as e:
            logger.error(f"LLM决策失败: {e}")

        return self._rule_based_decide(market_data)

    def _rule_based_decide(self, market_data: Dict[str, Any]) -> TradingDecision:
        """基于规则的决策（备用）"""
        signal = market_data.get("signal", {})

        rsi = signal.get("rsi", 50)
        trend = signal.get("trend", "neutral")
        confidence = signal.get("confidence", 0.5)

        # 简单规则
        if rsi < 35 and trend == "bull":
            return TradingDecision(
                decision=DecisionType.EXECUTE,
                symbol=market_data.get("symbol", "BTC"),
                direction="LONG",
                entry_price=market_data.get("price"),
                confidence=confidence,
                reasoning=f"RSI超卖({rsi}), 趋势向上"
            )
        elif rsi > 65 and trend == "bear":
            return TradingDecision(
                decision=DecisionType.EXECUTE,
                symbol=market_data.get("symbol", "BTC"),
                direction="SHORT",
                entry_price=market_data.get("price"),
                confidence=confidence,
                reasoning=f"RSI超买({rsi}), 趋势向下"
            )

        return TradingDecision(
            decision=DecisionType.WAIT,
            symbol=market_data.get("symbol", ""),
            direction="",
            reasoning="无明确信号",
            confidence=0.0
        )

    async def reflect(self, trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        反思交易结果
        """
        self.state.total_reflections += 1

        pnl = trade_result.get("pnl", 0)
        outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

        logger.info(f"Orchestrator: 反思 {outcome}, PnL={pnl}%")

        # 如果有LLM且启用了反思
        if self.llm and self.config.get("enable_reflection"):
            try:
                prompt = f"""分析以下交易:

交易结果: {outcome}
盈亏: {pnl}%
方向: {trade_result.get('direction')}
入场: {trade_result.get('entry_price')}
出场: {trade_result.get('exit_price')}

请分析: 为什么赚钱/亏损? 下次如何改进?
"""
                response = await self.llm.chat_simple(prompt, system="你是交易分析师。")
                if not response.error:
                    return {"analysis": response.content, "outcome": outcome}
            except Exception as e:
                logger.error(f"反思失败: {e}")

        return {"analysis": "反思功能暂不可用", "outcome": outcome}

    def get_state(self) -> Dict[str, Any]:
        """获取状态"""
        return self.state.__dict__

    async def run_cycle(self, market_data: Dict[str, Any]) -> TradingDecision:
        """
        运行完整决策循环
        """
        decision = await self.decide(market_data)

        if decision.decision == DecisionType.EXECUTE:
            self.state.decisions_today += 1
            self.state.consecutive_waits = 0
        else:
            self.state.consecutive_waits += 1

        self.state.last_decision_time = datetime.now().isoformat()

        return decision

# ==================== 便捷函数 ====================

def get_orchestrator(config: Optional[Dict] = None) -> Orchestrator:
    """获取Orchestrator实例"""
    return Orchestrator(config)
