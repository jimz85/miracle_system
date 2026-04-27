"""
Fusion Debate Layer — 多空辩论层编排器

职责:
    - 整合Bull/Bear Researcher + Debate Judge
    - 支持并行多空研究分析
    - 输出统一的DebateOutput

参考: TradingAgents agents/researchers/ + Fusion Architecture §3.1
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .bull_researcher import BullResearcher, ResearchResult
from .bear_researcher import BearResearcher
from .debate_judge import DebateJudge, VerdictResult, DebateVerdict

logger = logging.getLogger(__name__)


@dataclass
class DebateInput:
    """辩论层输入"""
    ticker: str                          # 交易标的
    price_data: Dict[str, Any]          # 价格数据 (close, high_24h, low_24h, volume_24h)
    factor_context: Dict[str, Any]       # 因子上下文 (RSI, ADX, MACD, BB等)
    market_intel: Dict[str, Any]        # 市场情报 (sentiment_score, whale_signal等)
    ic_weights: Dict[str, float] = field(default_factory=dict)  # IC权重


@dataclass
class DebateOutput:
    """辩论层输出"""
    bull_case: str                      # 多头论点
    bear_case: str                      # 空头论点
    debate_verdict: DebateVerdict        # 裁决: BUY/SELL/HOLD
    confidence: float                     # 置信度 0-1
    key_insights: List[str]             # 关键洞察列表
    bull_evidence: List[str] = field(default_factory=list)   # 多头证据
    bear_evidence: List[str] = field(default_factory=list)   # 空头证据
    support_levels: List[float] = field(default_factory=list)  # 支撑位
    resistance_levels: List[float] = field(default_factory=list)  # 阻力位
    latency_ms: float = 0.0             # 总延迟


class FusionDebateLayer:
    """
    Fusion Debate Layer — 多空辩论层编排器

    使用方式:
        layer = FusionDebateLayer(llm_manager)
        result = await layer.run_debate(debate_input)

    子Agent:
        - BullResearcher: 并行分析多头信号 (quick_think_llm)
        - BearResearcher: 并行分析空头信号 (quick_think_llm)
        - DebateJudge: 综合裁决输出 (deep_think_llm)

    性能要求:
        - 单次辩论延迟 < 5秒 (快速模型)
    """

    def __init__(self, llm_manager=None, config: Optional[Dict[str, Any]] = None):
        """
        初始化辩论层

        Args:
            llm_manager: LLMProviderManager实例
            config: 可选配置 {'timeout': 5.0, 'parallel': True}
        """
        self.llm_manager = llm_manager
        self.config = config or {}

        # 初始化子Agent
        self.bull_researcher = BullResearcher(llm_manager)
        self.bear_researcher = BearResearcher(llm_manager)
        self.judge = DebateJudge(llm_manager)

        # 配置
        self.timeout = self.config.get('timeout', 5.0)
        self.parallel = self.config.get('parallel', True)

        logger.info("[FusionDebateLayer] 辩论层初始化完成")

    async def run_debate(self, debate_input: DebateInput) -> DebateOutput:
        """
        执行完整辩论流程

        Args:
            debate_input: DebateInput包含ticker, price_data, factor_context, market_intel

        Returns:
            DebateOutput: 包含bull_case, bear_case, verdict, confidence, insights
        """
        start_time = time.time()
        logger.info(f"[FusionDebateLayer] 开始辩论: {debate_input.ticker}")

        try:
            # 1. 并行运行多空研究
            if self.parallel:
                bull_result, bear_result = await asyncio.gather(
                    self.bull_researcher.analyze(
                        debate_input.ticker,
                        debate_input.price_data,
                        debate_input.factor_context,
                        debate_input.market_intel
                    ),
                    self.bear_researcher.analyze(
                        debate_input.ticker,
                        debate_input.price_data,
                        debate_input.factor_context,
                        debate_input.market_intel
                    )
                )
            else:
                # 串行执行(保留选项)
                bull_result = await self.bull_researcher.analyze(
                    debate_input.ticker,
                    debate_input.price_data,
                    debate_input.factor_context,
                    debate_input.market_intel
                )
                bear_result = await self.bear_researcher.analyze(
                    debate_input.ticker,
                    debate_input.price_data,
                    debate_input.factor_context,
                    debate_input.market_intel
                )

            logger.info(f"[FusionDebateLayer] 多空研究完成: bull_conf={bull_result.confidence:.2f}, "
                        f"bear_conf={bear_result.confidence:.2f}")

            # 2. 裁决辩论
            verdict_result = await self.judge.arbitrate(bull_result, bear_result)

            logger.info(f"[FusionDebateLayer] 裁决完成: {verdict_result.decision.value}, "
                        f"置信度={verdict_result.confidence:.2f}")

            # 3. 构建输出
            latency_ms = (time.time() - start_time) * 1000

            output = DebateOutput(
                bull_case=bull_result.case,
                bear_case=bear_result.case,
                debate_verdict=verdict_result.decision,
                confidence=verdict_result.confidence,
                key_insights=verdict_result.insights,
                bull_evidence=bull_result.evidence,
                bear_evidence=bear_result.evidence,
                support_levels=bull_result.support_levels,
                resistance_levels=bear_result.support_levels,
                latency_ms=latency_ms
            )

            logger.info(f"[FusionDebateLayer] 辩论完成，延迟: {latency_ms:.0f}ms")

            return output

        except asyncio.TimeoutError:
            logger.error(f"[FusionDebateLayer] 辩论超时({self.timeout}s)")
            raise
        except Exception as e:
            logger.error(f"[FusionDebateLayer] 辩论异常: {e}", exc_info=True)
            raise

    async def run_debate_simple(self, ticker: str, price_data: Dict[str, Any],
                                factor_context: Dict[str, Any],
                                market_intel: Dict[str, Any]) -> DebateOutput:
        """
        简化接口 - 便捷方法

        Args:
            ticker: 交易标的
            price_data: 价格数据
            factor_context: 因子上下文
            market_intel: 市场情报

        Returns:
            DebateOutput: 辩论结果
        """
        input_data = DebateInput(
            ticker=ticker,
            price_data=price_data,
            factor_context=factor_context,
            market_intel=market_intel
        )
        return await self.run_debate(input_data)

    def get_verdict_action(self, verdict: DebateVerdict) -> str:
        """
        将裁决转换为交易动作

        Args:
            verdict: DebateVerdict

        Returns:
            str: "buy", "sell", "hold"
        """
        mapping = {
            DebateVerdict.BUY: "buy",
            DebateVerdict.SELL: "sell",
            DebateVerdict.HOLD: "hold"
        }
        return mapping.get(verdict, "hold")

    def format_output_markdown(self, output: DebateOutput) -> str:
        """
        格式化辩论输出为Markdown

        Args:
            output: DebateOutput

        Returns:
            str: Markdown格式字符串
        """
        verdict_action = self.get_verdict_action(output.debate_verdict)

        lines = [
            f"# 辩论裁决报告: {output.bull_case.split(chr(10))[0][:50] if output.bull_case else 'N/A'}...",
            "",
            f"## 裁决结果",
            f"- **决策:** {verdict_action.upper()}",
            f"- **置信度:** {output.confidence:.2f}",
            f"- **延迟:** {output.latency_ms:.0f}ms",
            "",
            f"## 多头论点 (Bull Case)",
            f"{output.bull_case}",
            "",
            f"### 多头证据",
        ]

        for evidence in output.bull_evidence:
            lines.append(f"- {evidence}")

        lines.extend([
            "",
            f"## 空头论点 (Bear Case)",
            f"{output.bear_case}",
            "",
            f"### 空头证据",
        ])

        for evidence in output.bear_evidence:
            lines.append(f"- {evidence}")

        lines.extend([
            "",
            f"## 关键洞察",
        ])

        for insight in output.key_insights:
            lines.append(f"- {insight}")

        lines.extend([
            "",
            f"## 技术位",
            f"- 支撑位: {output.support_levels}",
            f"- 阻力位: {output.resistance_levels}",
        ])

        return "\n".join(lines)