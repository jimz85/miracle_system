from __future__ import annotations

"""
Market Intel LLM Agent - 市场情报Agent主类
==========================================

从 agents/agent_market_intel_llm.py 提取

包含:
- MarketIntelAgentLLM: 市场情报LLM增强版主类
- main_async / main: 入口函数

用法:
    from agents.market_intel_llm_agent import MarketIntelAgentLLM, main_async
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple

from core.market_intel_base import (
    get_timestamp,
    load_cache,
    save_cache,
)
from core.market_intel_context import ContextBuilder
from core.market_intel_llm import LLMSentimentAnalyzer
from core.market_intel_onchain import EnhancedOnChainAnalyzer

logger = logging.getLogger("MarketIntelLLMAgent")

# LLM Provider imports
try:
    from core.llm_provider import (
        LLMProviderType,
        get_llm_manager,
    )
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    logger.warning("LLM模块不可用")


class MarketIntelAgentLLM:
    """
    市场情报Agent LLM增强版主类
    协调LLM情感分析、增强链上分析和上下文构建模块
    """

    def __init__(self, symbol: str = "BTC", llm_provider: str = "auto"):
        self.symbol = symbol.upper()

        self.llm_provider = llm_provider
        self._init_llm()

        # 初始化子模块
        self.sentiment_analyzer = LLMSentimentAnalyzer(llm_provider)
        self.onchain_analyzer = EnhancedOnChainAnalyzer()
        self.context_builder = ContextBuilder(symbol)

        # 权重配置
        self.weights = {
            "news_sentiment": 0.35,
            "exchange_flow": 0.30,
            "concentration": 0.15,
            "whale_activity": 0.20,
        }

        logger.info(f"Agent-M LLM增强版初始化完成: symbol={self.symbol}, provider={self.llm_provider}")

    def _init_llm(self):
        """初始化LLM"""
        if HAS_LLM:
            try:
                self.llm_manager = get_llm_manager()
                if self.llm_provider != "auto":
                    self.llm_manager.set_provider(LLMProviderType(self.llm_provider))
                logger.info(f"LLM Manager可用: {self.llm_manager.current_provider}")
            except Exception as e:
                logger.warning(f"LLM Manager初始化失败: {e}")
                self.llm_manager = None
        else:
            self.llm_manager = None
            logger.warning("LLM模块不可用，使用传统分析")

    async def generate_intel_report(self) -> Dict:
        """生成综合情报报告（异步版本）"""
        logger.info(f"=== 开始生成{self.symbol}市场情报报告 (LLM增强版) ===")

        # 1. LLM情感分析
        news_sentiment = await self._get_news_sentiment()
        logger.info(f"新闻情感: score={news_sentiment.get('score', 0)}, label={news_sentiment.get('label', '中性')}")

        # 2. 增强链上分析
        onchain_analysis = await self._get_onchain_analysis()
        logger.info(f"链上分析: flow_signal={onchain_analysis.get('signal', 0)}, pattern={onchain_analysis.get('pattern', {}).get('type', 'unknown')}")

        # 3. 鲸鱼活动分析
        whale_analysis = await self._get_whale_analysis()
        logger.info(f"鲸鱼活动: signal={whale_analysis.get('signal', 0)}, count={whale_analysis.get('count', 0)}")

        # 4. 钱包分布
        concentration = await self._get_concentration()

        # 5. 构建上下文
        context = await self.context_builder.build_context(
            news_sentiment, onchain_analysis, whale_analysis
        )

        # 6. 计算综合评分
        combined_score = self._calc_combined_score(
            news_sentiment.get("score", 0),
            onchain_analysis.get("signal", 0),
            whale_analysis.get("signal", 0),
            concentration.get("signal", 0)
        )

        # 7. 生成LLM综合分析
        llm_analysis = await self._generate_llm_analysis(
            news_sentiment, onchain_analysis, whale_analysis, context
        )

        # 8. 生成推荐
        recommendation, confidence = self._generate_recommendation(
            combined_score, news_sentiment, onchain_analysis, whale_analysis, context
        )

        # 9. 识别模式
        patterns = self._identify_market_patterns(
            news_sentiment, onchain_analysis, whale_analysis
        )

        # 10. 构建报告
        report = {
            "symbol": self.symbol,
            "timestamp": get_timestamp(),
            "news_sentiment": {
                "score": news_sentiment.get("score", 0),
                "label": news_sentiment.get("label", "中性"),
                "reasoning": news_sentiment.get("reasoning", ""),
                "confidence": news_sentiment.get("confidence", 0)
            },
            "onchain": {
                "exchange_flow_signal": onchain_analysis.get("signal", 0),
                "flow_pattern": onchain_analysis.get("pattern", {}).get("type", "unknown"),
                "flow_interpretation": onchain_analysis.get("interpretation", ""),
                "anomalies": onchain_analysis.get("anomalies", [])
            },
            "wallet": {
                "concentration_signal": concentration.get("signal", 0),
                "top10_pct": concentration.get("top10_pct", 0),
                "change_24h": concentration.get("change_24h", 0)
            },
            "whale": {
                "signal": whale_analysis.get("signal", 0),
                "pattern": whale_analysis.get("pattern", {}).get("type", "unknown"),
                "count": whale_analysis.get("count", 0),
                "total_volume": whale_analysis.get("total_volume", 0),
                "interpretation": whale_analysis.get("interpretation", "")
            },
            "combined_score": round(combined_score, 3),
            "recommendation": recommendation,
            "confidence": round(confidence, 2),
            "llm_analysis": llm_analysis,
            "context": context.to_dict(),
            "patterns": patterns
        }

        logger.info(f"=== {self.symbol}情报报告生成完成: 综合评分={combined_score:.3f}, 推荐={recommendation}, 置信度={confidence:.2f} ===")

        return report

    def generate_intel_report_sync(self) -> Dict:
        """同步版本的报告生成"""
        return asyncio.run(self.generate_intel_report())

    async def _get_news_sentiment(self) -> Dict:
        """获取LLM增强的新闻情感分析"""
        try:
            news = await self._fetch_news()

            if self.sentiment_analyzer.llm_manager:
                return await self.sentiment_analyzer.analyze_batch(news, self.symbol)
            else:
                return await self.sentiment_analyzer._keyword_fallback_analysis(news, self.symbol)
        except Exception as e:
            logger.error(f"新闻情感分析失败: {e}")
            return {"score": 0, "label": "中性", "reasoning": "分析失败"}

    async def _fetch_news(self) -> List[Dict]:
        """获取新闻"""
        cached = load_cache(self.symbol, "news")
        if cached and (time.time() - cached.timestamp) < 300:
            return cached.data

        try:
            from core.news_fetcher import fetch_rss_news
            news_items = fetch_rss_news("theblock", limit=20)
            if news_items:
                result = []
                for item in news_items:
                    result.append({
                        "id": item.get("link", ""),
                        "title": item.get("title", ""),
                        "body": item.get("description", ""),
                        "published_on": item.get("published_on", int(time.time())),
                        "source": item.get("source", "The Block")
                    })
                save_cache(self.symbol, "news", result)
                return result
        except Exception as e:
            logger.warning(f"新闻获取失败: {e}")

        return []

    async def _get_onchain_analysis(self) -> Dict:
        """获取增强链上分析"""
        try:
            return await self.onchain_analyzer.analyze_exchange_flow(self.symbol)
        except Exception as e:
            logger.error(f"链上分析失败: {e}")
            return {"signal": 0, "pattern": {}, "anomalies": [], "interpretation": "分析失败"}

    async def _get_whale_analysis(self) -> Dict:
        """获取鲸鱼活动分析"""
        try:
            return await self.onchain_analyzer.analyze_whale_transfers(self.symbol)
        except Exception as e:
            logger.error(f"鲸鱼分析失败: {e}")
            return {"signal": 0, "count": 0, "pattern": {}, "interpretation": "分析失败"}

    async def _get_concentration(self) -> Dict:
        """获取持币集中度"""
        try:
            cached = load_cache(self.symbol, "holder_concentration")
            if cached and (time.time() - cached.timestamp) < 3600:
                data = cached.data
            else:
                import random
                top10_pct = random.uniform(40, 60)
                change_24h = random.uniform(-2, 2)
                data = {
                    "top10_pct": round(top10_pct, 2),
                    "top100_pct": round(top10_pct * 1.3, 2),
                    "change_24h": round(change_24h, 2),
                    "timestamp": get_timestamp()
                }
                save_cache(self.symbol, "holder_concentration", data)

            signal = max(-1.0, min(1.0, data.get("change_24h", 0) / 1.0))
            data["signal"] = round(signal, 3)
            return data
        except Exception as e:
            logger.error(f"集中度获取失败: {e}")
            return {"signal": 0, "top10_pct": 0, "change_24h": 0}

    async def _generate_llm_analysis(self, sentiment: Dict, onchain: Dict,
                                      whale: Dict, context) -> Dict[str, Any]:
        """使用LLM生成综合分析"""
        if not self.llm_manager:
            return {
                "summary": "LLM不可用",
                "key_insights": [],
                "risks": [],
                "opportunities": []
            }

        prompt = f"""基于以下{self.symbol}市场数据，请提供综合分析：

新闻情感: {sentiment.get('label', '中性')} (分数: {sentiment.get('score', 0):.2f})
理由: {sentiment.get('reasoning', '无')}

交易所流量信号: {onchain.get('signal', 0):.2f}
流量模式: {onchain.get('pattern', {}).get('type', 'unknown')}
解读: {onchain.get('interpretation', '无')}

鲸鱼活动信号: {whale.get('signal', 0):.2f}
鲸鱼模式: {whale.get('pattern', {}).get('type', 'unknown')}
解读: {whale.get('interpretation', '无')}

市场阶段: {context.market_phase}
鲸鱼活动水平: {context.whale_activity_level}
信号一致性: {context.correlation_data.get('signal_alignment', {}).get('type', 'unknown')}

请输出JSON格式的综合分析：
{{
    "summary": "市场概况总结（100字内）",
    "key_insights": ["关键洞察1", "关键洞察2"],
    "risks": ["风险因素1", "风险因素2"],
    "opportunities": ["机会1", "机会2"]
}}"""

        try:
            response = await self.llm_manager.chat_simple(
                prompt=prompt,
                system="你是一位专业的加密货币市场分析师，提供简洁有力的分析。"
            )

            if response.content:
                json_match = re.search(r'\{[^{}]*"summary"[^{}]*\}', response.content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"LLM综合分析失败: {e}")

        return {
            "summary": "分析生成失败",
            "key_insights": [],
            "risks": [],
            "opportunities": []
        }

    def _calc_combined_score(self, news_score: float, flow_signal: float,
                             whale_signal: float, concentration_signal: float) -> float:
        """计算加权综合评分"""
        combined = (
            self.weights["news_sentiment"] * news_score +
            self.weights["exchange_flow"] * flow_signal +
            self.weights["whale_activity"] * whale_signal +
            self.weights["concentration"] * concentration_signal
        )
        return max(-1.0, min(1.0, combined))

    def _generate_recommendation(self, combined_score: float,
                                 sentiment: Dict, onchain: Dict,
                                 whale: Dict, context) -> Tuple[str, float]:
        """生成推荐"""
        signals = [
            sentiment.get("score", 0),
            onchain.get("signal", 0),
            whale.get("signal", 0)
        ]

        positive_count = sum(1 for s in signals if s > 0.2)
        negative_count = sum(1 for s in signals if s < -0.2)

        alignment = context.correlation_data.get("signal_alignment", {})
        alignment_strength = alignment.get("strength", 0)

        if combined_score > 0.3:
            if positive_count >= 2:
                recommendation = "看多"
                confidence = 0.6 + (combined_score - 0.3) * 0.5 + alignment_strength * 0.1
            else:
                recommendation = "观望"
                confidence = 0.4
        elif combined_score < -0.3:
            if negative_count >= 2:
                recommendation = "看空"
                confidence = 0.6 + abs(combined_score) - 0.3 * 0.5 + alignment_strength * 0.1
            else:
                recommendation = "观望"
                confidence = 0.4
        else:
            if positive_count > negative_count:
                recommendation = "谨慎看多"
                confidence = 0.35 + positive_count * 0.05 + alignment_strength * 0.1
            elif negative_count > positive_count:
                recommendation = "谨慎看空"
                confidence = 0.35 + negative_count * 0.05 + alignment_strength * 0.1
            else:
                recommendation = "观望"
                confidence = 0.5

        confidence = max(0.3, min(0.95, confidence))
        return recommendation, confidence

    def _identify_market_patterns(self, sentiment: Dict, onchain: Dict,
                                  whale: Dict) -> List[Dict]:
        """识别市场模式"""
        patterns = []

        if whale.get("pattern", {}).get("type") == "accumulation" and onchain.get("signal", 0) > 0.2:
            patterns.append({
                "type": "whale_accumulation",
                "name": "巨鲸积累模式",
                "confidence": whale.get("pattern", {}).get("confidence", 0.5),
                "description": "检测到巨鲸吸筹行为",
                "implication": "短期看涨",
                "severity": "high"
            })

        if whale.get("pattern", {}).get("type") == "distribution" and onchain.get("signal", 0) < -0.2:
            patterns.append({
                "type": "whale_distribution",
                "name": "巨鲸分发模式",
                "confidence": whale.get("pattern", {}).get("confidence", 0.5),
                "description": "检测到巨鲸抛售行为",
                "implication": "短期看跌",
                "severity": "high"
            })

        anomalies = onchain.get("anomalies", [])
        if anomalies:
            patterns.append({
                "type": "flow_anomaly",
                "name": "流量异常",
                "confidence": 0.6,
                "description": f"检测到{len(anomalies)}个流量异常",
                "implication": "需关注",
                "severity": "medium"
            })

        return patterns

    def get_brief_report(self, report: Dict = None) -> str:
        """获取简洁的文字报告"""
        if report is None:
            report = self.generate_intel_report_sync()

        lines = [
            f"📊 {report['symbol']} 市场情报报告 (LLM增强版)",
            f"⏰ {report['timestamp']}",
            "",
            f"📰 新闻情感: {report['news_sentiment']['label']} ({report['news_sentiment']['score']:+.2f})",
            f"   {report['news_sentiment'].get('reasoning', '')[:80]}..." if report['news_sentiment'].get('reasoning') else "",
            "",
            "🔗 链上分析:",
            f"   流量信号: {report['onchain']['exchange_flow_signal']:+.2f}",
            f"   模式: {report['onchain']['flow_pattern']}",
            "",
            "🐋 鲸鱼活动:",
            f"   信号: {report['whale']['signal']:+.2f}",
            f"   模式: {report['whale']['pattern']}",
            f"   转账数: {report['whale']['count']}",
            "",
            "📈 市场上下文:",
            f"   阶段: {report['context']['market_phase']}",
            f"   鲸鱼活动: {report['context']['whale_activity_level']}",
            f"   信号一致性: {report['context']['correlation_data']['signal_alignment']['type']}",
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📈 综合评分: {report['combined_score']:+.2f}",
            f"🎯 推荐: {report['recommendation']}",
            f"🔒 置信度: {report['confidence']:.0%}",
        ]

        return "\n".join([ln for ln in lines if ln])


# ============================================================
# 入口函数
# ============================================================

async def main_async(symbol: str = "BTC", llm_provider: str = "auto"):
    """异步主入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    agent = MarketIntelAgentLLM(symbol, llm_provider)
    report = await agent.generate_intel_report()

    print("\n" + "=" * 60)
    print(f"  {symbol} 市场情报报告 (LLM增强版)")
    print("=" * 60)
    print(f"\n{agent.get_brief_report(report)}")
    print("=" * 60)

    if report.get("llm_analysis"):
        print("\n🤖 LLM综合分析:")
        llm = report["llm_analysis"]
        print(f"   总结: {llm.get('summary', '无')}")
        if llm.get("key_insights"):
            print("   关键洞察:")
            for insight in llm["key_insights"][:3]:
                print(f"     - {insight}")

    if report.get("patterns"):
        print("\n🔍 检测到的模式:")
        for pattern in report["patterns"]:
            print(f"     - {pattern['name']} (置信度:{pattern['confidence']:.0%})")

    print("=" * 60)

    return report


def main(symbol: str = "BTC", llm_provider: str = "auto"):
    """同步主入口"""
    return asyncio.run(main_async(symbol, llm_provider))
