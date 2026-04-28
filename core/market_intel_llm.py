"""
Market Intel LLM - LLM驱动的情感分析
=====================================

从 agents/agent_market_intel_llm.py 提取

包含:
- LLMSentimentAnalyzer: LLM驱动的新闻情感分析

依赖:
- core.market_intel_types (基础类型)
- core.llm_provider (LLM接口)

用法:
    from core.market_intel_llm import LLMSentimentAnalyzer
    from agents.agent_market_intel_llm import LLMSentimentAnalyzer  # 向后兼容
"""

import json
import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from core.market_intel_base import (
    API_CONFIG,
    DEFAULT_LLM_PROVIDER,
    LLMSentimentResult,
)

logger = logging.getLogger("MarketIntelLLM")

# 尝试导入LLM Provider
try:
    from core.llm_provider import (
        get_llm_manager,
        LLMProviderManager,
        LLMProviderType,
        Message,
        LLMResponse,
        LLMConfig,
    )
    HAS_LLM = True
except ImportError as e:
    HAS_LLM = False
    logger.warning(f"LLM Provider导入失败: {e}")


class LLMSentimentAnalyzer:
    """
    LLM驱动的情感分析模块
    使用大语言模型进行深度情感分析，替代简单的关键词匹配
    """

    SYSTEM_PROMPT = """你是一位专业的加密货币市场分析师。你的任务是对加密货币相关新闻进行深度情感分析。

分析要求：
1. 评估新闻对市场的潜在影响（短期和中期）
2. 识别关键驱动因素
3. 判断市场情绪的强度
4. 区分"噪音"和真正的信号

输出格式（JSON）：
{
    "score": 分数(-1到+1之间，-1=极度利空，+1=极度利好),
    "label": "看多/看空/中性",
    "reasoning": "分析理由（50-200字）",
    "key_factors": ["关键因素1", "关键因素2"],
    "confidence": 置信度(0到1之间),
    "market_tone": "市场语调描述",
    "affected_factors": ["受影响方面1", "受影响方面2"]
}

请直接输出JSON，不要有其他内容。"""

    def __init__(self, provider_type: str = "auto"):
        self.provider_type = provider_type
        self.llm_manager = None
        if HAS_LLM:
            try:
                if provider_type == "auto":
                    self.llm_manager = get_llm_manager()
                else:
                    self.llm_manager = get_llm_manager()
                    self.llm_manager.set_provider(LLMProviderType(provider_type))
                logger.info(f"LLM情感分析器初始化，使用provider: {self.llm_manager.current_provider}")
            except Exception as e:
                logger.warning(f"LLM Manager初始化失败: {e}")
                self.llm_manager = None

    async def analyze_single_news(self, news_item: Dict) -> Optional[LLMSentimentResult]:
        """使用LLM分析单条新闻"""
        if not self.llm_manager:
            return None

        title = news_item.get('title', '')
        body = news_item.get('body', '')[:1000]  # 限制长度
        source = news_item.get('source', 'Unknown')

        prompt = f"""分析以下加密货币新闻的情感：

来源: {source}
标题: {title}
内容: {body}

请进行深度情感分析："""

        try:
            response = await self.llm_manager.chat_simple(
                prompt=prompt,
                system=self.SYSTEM_PROMPT
            )

            if not response.content:
                return None

            result = self._parse_llm_response(response.content)
            return result
        except Exception as e:
            logger.error(f"LLM情感分析失败: {e}")
            return None

    async def analyze_batch(self, news_items: List[Dict], symbol: str) -> Dict[str, Any]:
        """
        批量分析新闻并进行综合情感判断

        Returns:
            {
                "score": float,
                "label": str,
                "reasoning": str,
                "key_factors": list,
                "confidence": float,
                "market_tone": str,
                "affected_factors": list,
                "details": list  # 每条新闻的分析结果
            }
        """
        if not news_items:
            return self._default_result()

        if not self.llm_manager:
            return await self._keyword_fallback_analysis(news_items, symbol)

        details = []
        total_score = 0.0
        valid_count = 0

        for item in news_items[:10]:
            result = await self.analyze_single_news(item)
            if result:
                details.append({
                    "title": item.get('title', '')[:80],
                    "score": result.score,
                    "label": result.label,
                    "reasoning": result.reasoning,
                    "confidence": result.confidence
                })
                total_score += result.score * result.confidence
                valid_count += result.confidence

        if valid_count == 0:
            return await self._keyword_fallback_analysis(news_items, symbol)

        avg_score = total_score / valid_count if valid_count > 0 else 0.0

        overall_result = await self._synthesize_analysis(
            news_items=news_items,
            avg_score=avg_score,
            details=details,
            symbol=symbol
        )

        return overall_result

    async def _synthesize_analysis(self, news_items: List[Dict], avg_score: float,
                                     details: List[Dict], symbol: str) -> Dict[str, Any]:
        """使用LLM进行综合分析"""
        if not self.llm_manager:
            return self._default_result()

        news_summary = "\n".join([
            f"- {item.get('title', '')[:100]}"
            for item in news_items[:10]
        ])

        prompt = f"""对以下{symbol}新闻进行综合情感分析：

新闻列表：
{news_summary}

基于以上新闻的个别分析（平均分数: {avg_score:.2f}），请给出综合判断：

请输出JSON格式的综合分析结果：
{{
    "score": 综合分数(-1到+1),
    "label": "看多/看空/中性",
    "reasoning": "综合分析理由（100-300字）",
    "key_factors": ["关键驱动因素"],
    "confidence": 置信度(0到1),
    "market_tone": "市场整体语调",
    "affected_factors": ["受影响方面"]
}}"""

        try:
            response = await self.llm_manager.chat_simple(
                prompt=prompt,
                system=self.SYSTEM_PROMPT
            )

            result = self._parse_llm_response(response.content)
            if result:
                result_dict = asdict(result)
                result_dict["details"] = details
                return result_dict
        except Exception as e:
            logger.error(f"LLM综合分析失败: {e}")

        return self._default_result()

    def _parse_llm_response(self, content: str) -> Optional[LLMSentimentResult]:
        """解析LLM的JSON响应"""
        try:
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = content.strip()

            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str)
            json_str = json_str.strip()

            data = json.loads(json_str)

            return LLMSentimentResult(
                score=float(data.get("score", 0)),
                label=data.get("label", "中性"),
                reasoning=data.get("reasoning", ""),
                key_factors=data.get("key_factors", []),
                confidence=float(data.get("confidence", 0.5)),
                market_tone=data.get("market_tone", "中性"),
                affected_factors=data.get("affected_factors", [])
            )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, content: {content[:200]}")
            return None
        except Exception as e:
            logger.warning(f"LLM响应解析异常: {e}")
            return None

    async def _keyword_fallback_analysis(self, news_items: List[Dict], symbol: str) -> Dict[str, Any]:
        """当LLM不可用时的备选分析"""
        bullish_keywords = [
            "暴涨", "突破", "新高", "涨势", "利好", "看涨", "买入", "抄底",
            "飙升", "狂涨", "创新高", "收涨", "大涨", "反弹",
            "surge", "bullish", "breakout", "high", "soar", "rally", "buy",
            "pump", "moon", "all-time", "ATH", "uptrend", "support", "bottom"
        ]
        bearish_keywords = [
            "暴跌", "破发", "新低", "跌势", "利空", "看跌", "卖出", "割肉",
            "闪崩", "狂跌", "腰斩", "创新低", "收跌", "大跌", "回落",
            "crash", "bearish", "breakdown", "low", "plunge", "drop", "sell",
            "dump", "capitulation", "ATL", "downtrend", "resistance", "top"
        ]

        scores = []
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('body', '')}".lower()
            score = 0.0
            for kw in bullish_keywords:
                if kw.lower() in text:
                    score += 0.3
            for kw in bearish_keywords:
                if kw.lower() in text:
                    score -= 0.3
            score = max(-1.0, min(1.0, score))
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "score": round(avg_score, 3),
            "label": "看多" if avg_score > 0.2 else ("看空" if avg_score < -0.2 else "中性"),
            "reasoning": "基于关键词的情感分析（LLM不可用）",
            "key_factors": ["关键词匹配"],
            "confidence": 0.4,
            "market_tone": "中性",
            "affected_factors": ["市场情绪"],
            "details": []
        }

    def _default_result(self) -> Dict[str, Any]:
        """默认分析结果"""
        return {
            "score": 0.0,
            "label": "中性",
            "reasoning": "无法获取有效分析",
            "key_factors": [],
            "confidence": 0.0,
            "market_tone": "未知",
            "affected_factors": [],
            "details": []
        }
