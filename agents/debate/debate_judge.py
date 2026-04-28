from __future__ import annotations

"""
Debate Judge Agent — 辩论裁决

职责:
    - 综合bull/bear论点
    - 评估证据权重
    - 输出debate_verdict, confidence, key_insights

模型: deep_think_llm (深度思考模型，用于综合裁决)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from .bull_researcher import ResearchResult

logger = logging.getLogger(__name__)


class DebateVerdict(Enum):
    """辩论裁决结果"""
    BUY = "Buy"       # 强烈建议买入
    SELL = "Sell"     # 强烈建议卖出
    HOLD = "Hold"     # 建议持有/观望


@dataclass
class VerdictResult:
    """裁决结果"""
    decision: DebateVerdict           # 裁决: Buy/Sell/Hold
    confidence: float                  # 置信度 0-1
    insights: List[str]               # 关键洞察列表
    reasoning: str                    # 裁决理由
    bull_weight: float = 0.5          # 多头权重
    bear_weight: float = 0.5          # 空头权重


class DebateJudge:
    """
    Debate Judge — 综合评估多空论点并输出裁决

    输入:
        - bull_result: BullResearcher的分析结果
        - bear_result: BearResearcher的分析结果

    输出:
        - VerdictResult: 包含decision, confidence, insights
    """

    SYSTEM_PROMPT = """你是一位专业的中短期加密货币交易裁决者。

你的任务:
1. 权衡多头和空头论点
2. 评估证据的可信度和权重
3. 给出最终裁决和置信度

裁决原则:
- 多头信号强于空头 → BUY
- 空头信号强于多头 → SELL
- 多空势均力敌或信号模糊 → HOLD

需要考虑的因素:
- RSI: 超买(>70)倾向于SELL，超卖(<30)倾向于BUY
- ADX: >25表示趋势明确，<20表示震荡
- MACD: 金叉倾向于BUY，死叉倾向于SELL
- 成交量: 放量确认趋势
- 市场情报: 情感极端时需谨慎

输出格式:
- 裁决: BUY / SELL / HOLD
- 置信度: 0.0-1.0
- 关键洞察: 3-5条
- 裁决理由: 100字以内总结

你是严谨的，不确定性高时会选择HOLD。"""

    def __init__(self, llm_manager=None):
        """
        初始化Debate Judge

        Args:
            llm_manager: LLMProviderManager实例，用于深度思考模型调用
        """
        self.llm_manager = llm_manager
        self.model_type = "deep"  # 深度思考模型

    async def arbitrate(self, bull_result: ResearchResult,
                       bear_result: ResearchResult) -> VerdictResult:
        """
        裁决多空辩论

        Args:
            bull_result: 多头研究结果
            bear_result: 空头研究结果

        Returns:
            VerdictResult: 包含decision, confidence, insights
        """
        logger.info(f"[DebateJudge] 开始裁决: bull_conf={bull_result.confidence:.2f}, "
                    f"bear_conf={bear_result.confidence:.2f}")

        # 构建裁决prompt
        prompt = self._build_verdict_prompt(bull_result, bear_result)

        # 使用深度思考模型
        if self.llm_manager:
            try:
                response = await self.llm_manager.chat_simple(
                    prompt=prompt,
                    system=self.SYSTEM_PROMPT
                )
                verdict_text = response.content
                logger.info(f"[DebateJudge] LLM裁决完成，延迟: {response.latency_ms:.0f}ms")
            except Exception as e:
                logger.warning(f"[DebateJudge] LLM调用失败: {e}，使用规则裁决")
                verdict_text = self._rule_based_verdict(bull_result, bear_result)
        else:
            verdict_text = self._rule_based_verdict(bull_result, bear_result)

        # 解析裁决结果
        result = self._parse_verdict(verdict_text, bull_result, bear_result)

        logger.info(f"[DebateJudge] 裁决完成: {result.decision.value}, "
                    f"置信度: {result.confidence:.2f}")

        return result

    def _build_verdict_prompt(self, bull_result: ResearchResult,
                              bear_result: ResearchResult) -> str:
        """构建裁决prompt"""
        prompt = f"""## 多空辩论裁决

### 多头论点 (Bull Case)
{bull_result.case}

**多头证据:**
{chr(10).join(['- ' + e for e in bull_result.evidence])}

**支撑位:** {bull_result.support_levels}
**多头置信度:** {bull_result.confidence:.2f}

---

### 空头论点 (Bear Case)
{bear_result.case}

**空头证据:**
{chr(10).join(['- ' + e for e in bear_result.evidence])}

**阻力位:** {bear_result.support_levels}
**空头置信度:** {bear_result.confidence:.2f}

---

### 裁决任务
请综合以上多空论点，给出最终裁决。

输出格式:
裁决: [BUY/SELL/HOLD]
置信度: [0.0-1.0]
关键洞察: [3-5条洞察]
裁决理由: [100字以内]
"""
        return prompt

    def _rule_based_verdict(self, bull_result: ResearchResult,
                            bear_result: ResearchResult) -> str:
        """无LLM时的规则基础裁决"""
        lines = []

        # 计算权重
        bull_weight = bull_result.confidence
        bear_weight = bear_result.confidence

        # 信号计数
        bull_signals = len(bull_result.evidence)
        bear_signals = len(bear_result.evidence)

        # 技术因子决策
        bull_signals_data = bull_result.signals or {}

        rsi = bull_signals_data.get('rsi', 50)
        adx = bull_signals_data.get('adx', 0)

        # 裁决逻辑
        verdict = "HOLD"
        confidence = 0.5
        insights = []

        # RSI极值判断
        if rsi < 30:
            insights.append(f"RSI超卖({rsi:.1f})，反弹概率高")
            bull_weight += 0.2
        elif rsi > 70:
            insights.append(f"RSI超买({rsi:.1f})，回落风险大")
            bear_weight += 0.2

        # ADX趋势强度
        if adx > 25:
            insights.append(f"ADX={adx:.1f}，趋势明确")
            if bull_weight > bear_weight:
                bull_weight += 0.1
            else:
                bear_weight += 0.1

        # 多空信号对比
        if bull_signals > bear_signals + 1:
            insights.append(f"多头信号更多({bull_signals}>{bear_signals})")
            bull_weight += 0.1
        elif bear_signals > bull_signals + 1:
            insights.append(f"空头信号更多({bear_signals}>{bull_signals})")
            bear_weight += 0.1

        # 归一化权重
        total_weight = bull_weight + bear_weight
        if total_weight > 0:
            bull_weight /= total_weight
            bear_weight /= total_weight

        # 裁决
        if bull_weight > bear_weight + 0.15:
            verdict = "BUY"
            confidence = min(0.9, 0.5 + bull_weight - bear_weight)
        elif bear_weight > bull_weight + 0.15:
            verdict = "SELL"
            confidence = min(0.9, 0.5 + bear_weight - bull_weight)
        else:
            verdict = "HOLD"
            confidence = 0.5 + abs(bull_weight - bear_weight)

        lines.append(f"裁决: {verdict}")
        lines.append(f"置信度: {confidence:.2f}")
        lines.append("关键洞察:")
        for insight in insights:
            lines.append(f"- {insight}")
        lines.append(f"裁决理由: 多头权重{bull_weight:.2f}，空头权重{bear_weight:.2f}，{verdict}信号更强")

        return "\n".join(lines)

    def _parse_verdict(self, verdict_text: str, bull_result: ResearchResult,
                       bear_result: ResearchResult) -> VerdictResult:
        """解析LLM输出为VerdictResult"""
        lines = verdict_text.strip().split('\n')

        decision = DebateVerdict.HOLD
        confidence = 0.5
        insights = []
        reasoning = ""
        bull_weight = 0.5
        bear_weight = 0.5

        for line in lines:
            line = line.strip()

            if line.startswith('裁决:') or line.startswith('Decision:'):
                if 'BUY' in line.upper():
                    decision = DebateVerdict.BUY
                elif 'SELL' in line.upper():
                    decision = DebateVerdict.SELL
                else:
                    decision = DebateVerdict.HOLD

            elif line.startswith('置信度:') or line.startswith('Confidence:'):
                import re
                match = re.search(r'0?\.\d+', line)
                if match:
                    confidence = float(match.group())

            elif line.startswith('关键洞察:') or line.startswith('Insights:'):
                continue  # 后续行是洞察

            elif line.startswith('-') and len(line) > 2:
                insights.append(line[1:].strip())

            elif '裁决理由' in line or 'Reasoning' in line:
                reasoning = line.split(':', 1)[-1].strip() if ':' in line else ""

        # 如果没有解析到洞察，使用默认
        if not insights:
            insights = [
                f"多头置信度: {bull_result.confidence:.2f}",
                f"空头置信度: {bear_result.confidence:.2f}",
                f"多头证据数: {len(bull_result.evidence)}",
                f"空头证据数: {len(bear_result.evidence)}"
            ]

        # 计算权重
        total = bull_result.confidence + bear_result.confidence
        if total > 0:
            bull_weight = bull_result.confidence / total
            bear_weight = bear_result.confidence / total

        # 如果没有裁决理由
        if not reasoning:
            reasoning = f"综合多头({bull_weight:.0%})和空头({bear_weight:.0%})权重，裁决{decision.value}"

        return VerdictResult(
            decision=decision,
            confidence=confidence,
            insights=insights,
            reasoning=reasoning,
            bull_weight=bull_weight,
            bear_weight=bear_weight
        )