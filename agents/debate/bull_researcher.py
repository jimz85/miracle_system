from __future__ import annotations

"""
Bull Researcher Agent — 多头论点研究

职责:
    - 分析factor_context中的做多信号
    - 识别支撑位、趋势确认
    - 输出bull_case和bull_evidence[]

模型: quick_think_llm (快速思考模型，保证<5秒延迟)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """研究结果"""
    case: str                           # 多头/空头论点文本
    evidence: List[str]                 # 证据列表
    support_levels: List[float]         # 支撑位列表
    confidence: float = 0.5             # 置信度 0-1
    signals: Dict[str, Any] = field(default_factory=dict)  # 触发信号详情


class BullResearcher:
    """
    Bull Researcher — 分析做多信号和支撑位

    输入:
        - price_data: 价格数据
        - factor_context: 因子上下文 (RSI, ADX, MACD, BB等)
        - market_intel: 市场情报

    输出:
        - ResearchResult包含bull_case和bull_evidence[]
    """

    SYSTEM_PROMPT = """你是一位专业的中短期加密货币交易分析师，专注于多头趋势研究。

你的任务:
1. 分析市场数据，识别做多信号
2. 找出关键支撑位
3. 评估趋势强度和持续性

分析框架:
- RSI: 是否处于超卖区域或从低位回升
- ADX: 趋势强度是否足够(>25)
- MACD: 是否形成金叉或底背离
- Bollinger Bands: 价格是否触及下轨获得支撑
- 成交量: 上涨时是否放量

输出要求:
- 给出清晰的多头论点(3-5个核心论据)
- 列出关键支撑位(具体价格)
- 评估置信度(0-1)

保持简洁专业，分析控制在200字以内。"""

    def __init__(self, llm_manager=None):
        """
        初始化Bull Researcher

        Args:
            llm_manager: LLMProviderManager实例，用于快速思考模型调用
        """
        self.llm_manager = llm_manager
        self.model_type = "quick"  # 快速思考模型

    def _build_analysis_prompt(self, ticker: str, price_data: Dict[str, Any],
                               factor_context: Dict[str, Any],
                               market_intel: Dict[str, Any]) -> str:
        """构建分析prompt"""
        prompt = f"""## 交易标的研究: {ticker}

### 当前价格数据
- 当前价格: {price_data.get('close', 'N/A')}
- 24h最高: {price_data.get('high_24h', 'N/A')}
- 24h最低: {price_data.get('low_24h', 'N/A')}
- 24h成交量: {price_data.get('volume_24h', 'N/A')}

### 技术因子
- RSI(14): {factor_context.get('rsi', 'N/A')}
- ADX: {factor_context.get('adx', 'N/A')}
- +DI: {factor_context.get('plus_di', 'N/A')}
- -DI: {factor_context.get('minus_di', 'N/A')}
- MACD: {factor_context.get('macd', 'N/A')}
- MACD Signal: {factor_context.get('macd_signal', 'N/A')}
- MACD Histogram: {factor_context.get('macd_hist', 'N/A')}
- Bollinger Upper: {factor_context.get('bb_upper', 'N/A')}
- Bollinger Lower: {factor_context.get('bb_lower', 'N/A')}
- 波动率: {factor_context.get('volatility', 'N/A')}

### 市场情报
- 情感评分: {market_intel.get('sentiment_score', 'N/A')}
- 情感标签: {market_intel.get('sentiment_label', 'N/A')}
- 鲸鱼信号: {market_intel.get('whale_signal', 'N/A')}

### 研究任务
请分析以上数据，识别做多信号，给出:
1. 3-5个核心多头论据(bullet points)
2. 关键支撑位(具体价格)
3. 置信度评估(0-1)
"""
        return prompt

    async def analyze(self, ticker: str, price_data: Dict[str, Any],
                      factor_context: Dict[str, Any],
                      market_intel: Dict[str, Any]) -> ResearchResult:
        """
        执行多头分析

        Args:
            ticker: 交易标的
            price_data: 价格数据
            factor_context: 因子上下文
            market_intel: 市场情报

        Returns:
            ResearchResult: 包含bull_case和bull_evidence[]
        """
        logger.info(f"[BullResearcher] 开始分析 {ticker} 多头信号")

        prompt = self._build_analysis_prompt(ticker, price_data, factor_context, market_intel)

        # 如果有LLM管理器，使用快速模型
        if self.llm_manager:
            try:
                response = await self.llm_manager.chat_simple(
                    prompt=prompt,
                    system=self.SYSTEM_PROMPT
                )
                analysis_text = response.content
                logger.info(f"[BullResearcher] LLM分析完成，延迟: {response.latency_ms:.0f}ms")
            except Exception as e:
                logger.warning(f"[BullResearcher] LLM调用失败: {e}，使用规则分析")
                analysis_text = self._rule_based_analysis(price_data, factor_context)
        else:
            # 无LLM时使用规则分析
            analysis_text = self._rule_based_analysis(price_data, factor_context)

        # 解析结果
        result = self._parse_analysis(analysis_text, price_data, factor_context)

        logger.info(f"[BullResearcher] 分析完成，置信度: {result.confidence:.2f}, "
                    f"证据数: {len(result.evidence)}")

        return result

    def _rule_based_analysis(self, price_data: Dict[str, Any],
                            factor_context: Dict[str, Any]) -> str:
        """无LLM时的规则基础分析"""
        lines = []
        price = price_data.get('close', 0)

        # RSI分析
        rsi = factor_context.get('rsi', 50)
        if rsi < 30:
            lines.append(f"- RSI超卖({rsi:.1f})，可能反弹")
        elif rsi < 45:
            lines.append(f"- RSI偏低({rsi:.1f})，下跌动能减弱")

        # ADX趋势分析
        adx = factor_context.get('adx', 0)
        plus_di = factor_context.get('plus_di', 0)
        minus_di = factor_context.get('minus_di', 0)
        if plus_di > minus_di and adx > 20:
            lines.append(f"- +DI>{minus_di:.1f}且ADX={adx:.1f}，多头趋势")

        # MACD分析
        macd = factor_context.get('macd', 0)
        macd_signal = factor_context.get('macd_signal', 0)
        if macd > macd_signal:
            lines.append(f"- MACD金叉(MACD={macd:.2f}>Signal={macd_signal:.2f})")

        # 布林带分析
        bb_lower = factor_context.get('bb_lower', 0)
        if price < bb_lower:
            lines.append("- 价格触及布林下轨，可能反弹")

        # 波动率分析
        volatility = factor_context.get('volatility', 0)
        if volatility < 0.3:
            lines.append(f"- 低波动率({volatility:.2f})，可能酝酿突破")

        if not lines:
            lines.append("- 无明显做多信号")

        return "\n".join(lines)

    def _parse_analysis(self, analysis_text: str, price_data: Dict[str, Any],
                       factor_context: Dict[str, Any]) -> ResearchResult:
        """解析LLM输出为ResearchResult"""
        lines = analysis_text.strip().split('\n')
        evidence = []
        support_levels = []
        confidence = 0.5

        price = price_data.get('close', 0)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 收集证据(以-开头的行)
            if line.startswith('-'):
                evidence.append(line[1:].strip())

            # 识别支撑位(包含数字的行)
            import re
            numbers = re.findall(r'\d+\.?\d*', line)
            for num_str in numbers:
                try:
                    num = float(num_str)
                    # 过滤明显不是价格/RSI的值
                    if 10 < num < 200000 or (0 < num < 100 and 'RSI' in line):
                        if num not in support_levels and len(support_levels) < 3:
                            support_levels.append(num)
                except ValueError:
                    continue

        # 评估置信度
        if '超卖' in analysis_text or '金叉' in analysis_text:
            confidence = 0.7
        if '强' in analysis_text and ('趋势' in analysis_text or '突破' in analysis_text):
            confidence = 0.8

        # 如果有明确的多头信号，增加置信度
        if factor_context.get('rsi', 50) < 35:
            confidence = min(0.85, confidence + 0.15)
        if factor_context.get('macd', 0) > factor_context.get('macd_signal', 0):
            confidence = min(0.85, confidence + 0.1)

        return ResearchResult(
            case=analysis_text,
            evidence=evidence,
            support_levels=support_levels,
            confidence=confidence,
            signals={
                "rsi": factor_context.get('rsi', 50),
                "adx": factor_context.get('adx', 0),
                "price": price
            }
        )