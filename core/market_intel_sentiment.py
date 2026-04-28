"""
Market Intel Sentiment - 情绪数据分析模块
==========================================

包含:
- NewsSentimentAnalyzer: 新闻情感分析器
- KeywordSentimentAnalyzer: 关键词情感分析
- SentimentAggregator: 情感聚合器

依赖:
- core.market_intel_base (基础类型和工具)

用法:
    from core.market_intel_sentiment import NewsSentimentAnalyzer, KeywordSentimentAnalyzer
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from core.market_intel_base import (
    API_CONFIG,
    get_timestamp,
    load_cache,
    save_cache,
    SentimentLabel,
    LLMSentimentResult,
)

logger = logging.getLogger("MarketIntelSentiment")

# 尝试导入LLM Provider
try:
    from core.llm_provider import (
        get_llm_manager,
        LLMProviderType,
    )
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    logger.warning("LLM模块不可用")


# ============================================================
# 关键词情感分析器
# ============================================================

class KeywordSentimentAnalyzer:
    """
    基于关键词的情感分析器
    当LLM不可用时作为备选方案
    """

    BULLISH_KEYWORDS = [
        "暴涨", "突破", "新高", "涨势", "利好", "看涨", "买入", "抄底",
        "飙升", "狂涨", "创新高", "收涨", "大涨", "反弹",
        "surge", "bullish", "breakout", "high", "soar", "rally", "buy",
        "pump", "moon", "all-time", "ATH", "uptrend", "support", "bottom"
    ]

    BEARISH_KEYWORDS = [
        "暴跌", "破发", "新低", "跌势", "利空", "看跌", "卖出", "割肉",
        "闪崩", "狂跌", "腰斩", "创新低", "收跌", "大跌", "回落",
        "crash", "bearish", "breakdown", "low", "plunge", "drop", "sell",
        "dump", "capitulation", "ATL", "downtrend", "resistance", "top"
    ]

    NEUTRAL_KEYWORDS = [
        "横盘", "震荡", "整理", "观望", "等待", "平稳", "持平",
        "sideways", "consolidation", "stable", "unchanged", "flat"
    ]

    def analyze(self, news_items: List[Dict]) -> Dict[str, Any]:
        """
        对新闻列表进行情感分析

        Returns:
            {
                "score": float,      # -1(完全利空) ~ +1(完全利好)
                "label": str,        # "看多"/"看空"/"中性"
                "reasoning": str,
                "confidence": float,
                "details": list       # 每条新闻的情感得分
            }
        """
        if not news_items:
            return {
                "score": 0.0,
                "label": "中性",
                "reasoning": "无新闻数据",
                "confidence": 0.0,
                "details": []
            }

        scores = []
        details = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for item in news_items:
            text = f"{item.get('title', '')} {item.get('body', '')}".lower()
            score = 0.0

            for kw in self.BULLISH_KEYWORDS:
                if kw.lower() in text:
                    score += 0.3
            for kw in self.BEARISH_KEYWORDS:
                if kw.lower() in text:
                    score -= 0.3
            for kw in self.NEUTRAL_KEYWORDS:
                if kw.lower() in text:
                    score *= 0.5  # 中性词降低信号强度

            score = max(-1.0, min(1.0, score))
            scores.append(score)
            details.append({
                "title": item.get("title", "")[:80],
                "score": round(score, 3),
                "source": item.get("source", "")
            })

            if score > 0.2:
                bullish_count += 1
            elif score < -0.2:
                bearish_count += 1
            else:
                neutral_count += 1

        total = len(scores) if scores else 1
        avg_score = sum(scores) / total if scores else 0.0

        bullish_pct = round(bullish_count / total * 100)
        bearish_pct = round(bearish_count / total * 100)
        neutral_pct = 100 - bullish_pct - bearish_pct

        if avg_score > 0.2:
            label = "看多"
        elif avg_score < -0.2:
            label = "看空"
        else:
            label = "中性"

        reasoning = f"分析了{total}条新闻，其中{bullish_pct}%利好，{bearish_pct}%利空，{neutral_pct}%中性"

        return {
            "score": round(avg_score, 3),
            "label": label,
            "reasoning": reasoning,
            "confidence": 0.4,  # 关键词分析置信度较低
            "details": details[:10]
        }


# ============================================================
# 新闻情感分析器 (LLM增强版)
# ============================================================

class NewsSentimentAnalyzer:
    """
    LLM驱动的新闻情感分析模块
    使用大语言模型进行深度情感分析
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
        body = news_item.get('body', '')[:1000]
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

            return self._parse_llm_response(response.content)
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
                "details": list
            }
        """
        if not news_items:
            return self._default_result()

        if not self.llm_manager:
            keyword_analyzer = KeywordSentimentAnalyzer()
            result = keyword_analyzer.analyze(news_items)
            result["key_factors"] = ["关键词匹配"]
            result["market_tone"] = result["label"]
            result["affected_factors"] = ["市场情绪"]
            return result

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
            keyword_analyzer = KeywordSentimentAnalyzer()
            result = keyword_analyzer.analyze(news_items)
            result["key_factors"] = ["关键词匹配"]
            result["market_tone"] = result["label"]
            result["affected_factors"] = ["市场情绪"]
            return result

        avg_score = total_score / valid_count if valid_count > 0 else 0.0

        return await self._synthesize_analysis(
            news_items=news_items,
            avg_score=avg_score,
            details=details,
            symbol=symbol
        )

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
                result_dict = {
                    "score": result.score,
                    "label": result.label,
                    "reasoning": result.reasoning,
                    "key_factors": result.key_factors,
                    "confidence": result.confidence,
                    "market_tone": result.market_tone,
                    "affected_factors": result.affected_factors,
                    "details": details
                }
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


# ============================================================
# 情感聚合器
# ============================================================

class SentimentAggregator:
    """
    多源情感数据聚合器
    将新闻情感、社交媒体情感等聚合为统一信号
    """

    def __init__(self):
        self.news_analyzer = NewsSentimentAnalyzer()
        self.keyword_analyzer = KeywordSentimentAnalyzer()

    async def get_combined_sentiment(
        self,
        news_items: List[Dict],
        symbol: str,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        获取综合情感信号

        Args:
            news_items: 新闻列表
            symbol: 交易对符号
            weights: 各来源权重，如 {"news": 0.7, "keyword": 0.3}

        Returns:
            综合情感分析结果
        """
        if weights is None:
            weights = {"news": 0.6, "keyword": 0.4}

        results = {}

        # LLM新闻分析
        if self.news_analyzer.llm_manager:
            results["news"] = await self.news_analyzer.analyze_batch(news_items, symbol)
        else:
            results["news"] = self.keyword_analyzer.analyze(news_items)

        # 关键词分析作为备选
        results["keyword"] = self.keyword_analyzer.analyze(news_items)

        # 加权平均
        combined_score = (
            weights.get("news", 0.5) * results["news"].get("score", 0) +
            weights.get("keyword", 0.5) * results["keyword"].get("score", 0)
        )

        # 决定最终标签
        if combined_score > 0.2:
            label = "看多"
        elif combined_score < -0.2:
            label = "看空"
        else:
            label = "中性"

        # 取较高的置信度
        confidence = max(
            results["news"].get("confidence", 0),
            results["keyword"].get("confidence", 0)
        )

        return {
            "score": round(combined_score, 3),
            "label": label,
            "reasoning": results["news"].get("reasoning", ""),
            "confidence": confidence,
            "details": {
                "news_analysis": results["news"],
                "keyword_analysis": results["keyword"]
            }
        }
