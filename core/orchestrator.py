#!/usr/bin/env python3
"""
Miracle 2.0 - Orchestrator 协调器
=================================
LLM驱动的大脑，负责任务分解、结果聚合、自我反思

增强版本 - 带LLM降级机制
"""

import json
import logging
import time
import uuid
import asyncio
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
    # LLM降级机制配置
    "llm_failure_threshold": 3,      # 连续失败N次后降级到规则引擎
    "llm_recovery_interval": 300,    # 每5分钟尝试恢复LLM
    "rule_engine_fallback": True,    # 启用规则引擎降级
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
    # LLM降级机制状态
    llm_failures: int = 0              # 连续LLM失败次数
    llm_degraded: bool = False         # 是否已降级到规则引擎
    last_llm_retry: Optional[float] = None  # 上次LLM重试时间戳
    total_llm_fallbacks: int = 0       # 总降级次数

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
    
    增强功能:
    - LLM降级机制: 连续失败后自动切换到规则引擎
    - 自动恢复: 定期尝试恢复LLM
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

    def _should_try_llm_recovery(self) -> bool:
        """检查是否应该尝试恢复LLM"""
        if self.llm is not None:
            return False
        
        current_time = time.time()
        recovery_interval = self.config.get("llm_recovery_interval", 300)
        
        if self.state.last_llm_retry is None:
            return True
        
        return (current_time - self.state.last_llm_retry) >= recovery_interval

    def _try_llm_recovery(self) -> bool:
        """尝试恢复LLM"""
        if not self._should_try_llm_recovery():
            return False
        
        self.state.last_llm_retry = time.time()
        
        try:
            self.llm = get_llm_provider(
                provider=self.config.get("llm_provider", "claude"),
                model=self.config.get("model"),
                temperature=self.config.get("temperature", 0.7)
            )
            # 重置失败计数
            self.state.llm_failures = 0
            self.state.llm_degraded = False
            logger.info("Orchestrator: LLM recovered successfully")
            return True
        except Exception as e:
            logger.warning(f"Orchestrator: LLM recovery failed: {e}")
            self.llm = None
            return False

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

        # 如果已降级到规则引擎，先尝试恢复LLM
        if self.state.llm_degraded:
            if self._try_llm_recovery():
                # LLM恢复成功，使用LLM决策
                return await self._llm_decide(market_data)
            elif self.config.get("rule_engine_fallback", True):
                # 使用规则引擎
                return self._rule_based_decide(market_data)
            else:
                # 不允许降级，等待LLM恢复
                return TradingDecision(
                    decision=DecisionType.WAIT,
                    symbol="",
                    direction="",
                    reasoning="LLM不可用，等待恢复",
                    confidence=0.0
                )

        # 如果有LLM，使用LLM决策
        if self.llm:
            return await self._llm_decide(market_data)

        # 没有LLM，尝试初始化
        if self._try_llm_recovery():
            return await self._llm_decide(market_data)
        
        # 规则引擎备用
        if self.config.get("rule_engine_fallback", True):
            return self._rule_based_decide(market_data)

        return TradingDecision(
            decision=DecisionType.WAIT,
            symbol="",
            direction="",
            reasoning="LLM不可用，规则引擎未启用",
            confidence=0.0
        )

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
                # LLM成功，重置失败计数
                if self.state.llm_failures > 0:
                    logger.info(f"Orchestrator: LLM recovered after {self.state.llm_failures} failures")
                self.state.llm_failures = 0
                
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

        # LLM失败
        self.state.llm_failures += 1
        failure_threshold = self.config.get("llm_failure_threshold", 3)
        
        if self.state.llm_failures >= failure_threshold and not self.state.llm_degraded:
            self.state.llm_degraded = True
            self.state.total_llm_fallbacks += 1
            logger.warning(
                f"Orchestrator: Degraded to rule engine after {self.state.llm_failures} "
                f"consecutive LLM failures (total fallbacks: {self.state.total_llm_fallbacks})"
            )

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
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """获取LLM降级机制状态"""
        return {
            "llm_available": self.llm is not None,
            "llm_degraded": self.state.llm_degraded,
            "llm_failures": self.state.llm_failures,
            "total_llm_fallbacks": self.state.total_llm_fallbacks,
            "failure_threshold": self.config.get("llm_failure_threshold", 3),
            "recovery_interval_seconds": self.config.get("llm_recovery_interval", 300),
        }

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
