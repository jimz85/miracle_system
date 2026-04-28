"""
Market Intel Context - 市场上下文构建
====================================

从 agents/agent_market_intel_llm.py 提取

包含:
- ContextBuilder: 市场上下文构建器

用法:
    from core.market_intel_context import ContextBuilder
"""

import logging
import time
from typing import Any, Dict, List

from core.market_intel_types import MarketContext, get_timestamp

logger = logging.getLogger("MarketIntelContext")


class ContextBuilder:
    """
    市场上下文构建模块
    功能：
    1. 多源数据融合
    2. 历史上下文追踪
    3. 市场阶段识别
    4. 预测性上下文生成
    """

    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol
        self._sentiment_history: List[Dict] = []
        self._max_history = 168  # 7天 * 24小时

    async def build_context(
        self,
        current_sentiment: Dict,
        onchain_data: Dict,
        whale_data: Dict
    ) -> MarketContext:
        """构建丰富的市场上下文"""
        self._sentiment_history.append({
            "timestamp": time.time(),
            "score": current_sentiment.get("score", 0),
            "label": current_sentiment.get("label", "中性")
        })

        if len(self._sentiment_history) > self._max_history:
            self._sentiment_history = self._sentiment_history[-self._max_history:]

        historical_sentiment = self._analyze_sentiment_history()
        market_phase = self._identify_market_phase(
            current_sentiment, onchain_data, whale_data
        )
        whale_level = self._assess_whale_activity(whale_data)
        flow_phase = self._assess_flow_phase(onchain_data)
        retail_sentiment = self._infer_retail_sentiment(
            current_sentiment, onchain_data
        )

        return MarketContext(
            symbol=self.symbol,
            timestamp=get_timestamp(),
            historical_sentiment=historical_sentiment,
            correlation_data=self._build_correlation_data(
                current_sentiment, onchain_data, whale_data
            ),
            market_phase=market_phase,
            whale_activity_level=whale_level,
            exchange_flow_phase=flow_phase,
            retail_sentiment=retail_sentiment
        )

    def _analyze_sentiment_history(self) -> List[Dict]:
        """分析情感历史"""
        if len(self._sentiment_history) < 2:
            return self._sentiment_history.copy()

        result = []
        for i in range(0, len(self._sentiment_history), 24):
            window = self._sentiment_history[i:min(i+24, len(self._sentiment_history))]
            if window:
                avg_score = sum(h["score"] for h in window) / len(window)
                result.append({
                    "timestamp": window[-1]["timestamp"],
                    "score": avg_score,
                    "label": window[-1]["label"]
                })

        return result[-30:]

    def _identify_market_phase(self, sentiment: Dict, onchain: Dict,
                               whale: Dict) -> str:
        """识别市场阶段"""
        sentiment_score = sentiment.get("score", 0)
        flow_signal = onchain.get("signal", 0)
        whale_signal = whale.get("signal", 0)

        avg_signal = (sentiment_score + flow_signal + whale_signal) / 3

        if avg_signal > 0.3:
            if sentiment_score > 0.5:
                return "markup"
            else:
                return "accumulation"
        elif avg_signal < -0.3:
            if sentiment_score < -0.5:
                return "markdown"
            else:
                return "distribution"
        else:
            return "neutral"

    def _assess_whale_activity(self, whale_data: Dict) -> str:
        """评估鲸鱼活动水平"""
        transfer_count = whale_data.get("count", 0)
        total_volume = whale_data.get("total_volume", 0)
        pattern_type = whale_data.get("pattern", {}).get("type", "")

        if transfer_count >= 10 or total_volume > 50_000_000:
            level = "extreme"
        elif transfer_count >= 5 or total_volume > 10_000_000:
            level = "elevated"
        elif transfer_count >= 2 or total_volume > 1_000_000:
            level = "normal"
        else:
            level = "low"

        if pattern_type in ["accumulation", "distribution"]:
            level = "elevated" if level == "normal" else level

        return level

    def _assess_flow_phase(self, onchain_data: Dict) -> str:
        """评估交易所流量阶段"""
        pattern = onchain_data.get("pattern", {})
        trend = onchain_data.get("trend", {})

        if pattern.get("type") in ["inflow_acceleration"]:
            return "inflow"
        elif pattern.get("type") in ["outflow_acceleration"]:
            return "outflow"
        elif trend.get("direction") == "inflow":
            return "inflow"
        elif trend.get("direction") == "outflow":
            return "outflow"
        else:
            return "neutral"

    def _infer_retail_sentiment(self, sentiment: Dict, onchain: Dict) -> str:
        """推断散户情绪"""
        whale_signal = onchain.get("signal", 0)
        sentiment_score = sentiment.get("score", 0)

        if whale_signal < -0.3:
            if sentiment_score > 0.2:
                return "optimistic"
            elif sentiment_score < -0.2:
                return "pessimistic"
            else:
                return "neutral"
        elif whale_signal > 0.3:
            if sentiment_score < -0.2:
                return "pessimistic"
            elif sentiment_score > 0.2:
                return "optimistic"
            else:
                return "neutral"
        else:
            return "neutral"

    def _build_correlation_data(self, sentiment: Dict, onchain: Dict,
                                 whale: Dict) -> Dict[str, Any]:
        """构建相关性数据"""
        return {
            "sentiment_vs_flow": self._correlation(
                sentiment.get("score", 0),
                onchain.get("signal", 0)
            ),
            "sentiment_vs_whale": self._correlation(
                sentiment.get("score", 0),
                whale.get("signal", 0)
            ),
            "flow_vs_whale": self._correlation(
                onchain.get("signal", 0),
                whale.get("signal", 0)
            ),
            "signal_alignment": self._calculate_alignment(
                sentiment, onchain, whale
            )
        }

    def _correlation(self, x: float, y: float) -> str:
        """计算两个信号的相关性"""
        product = x * y
        if product > 0.1:
            return "positive"
        elif product < -0.1:
            return "negative"
        else:
            return "neutral"

    def _calculate_alignment(self, sentiment: Dict, onchain: Dict,
                            whale: Dict) -> Dict[str, Any]:
        """计算信号对齐程度"""
        signals = [
            sentiment.get("score", 0),
            onchain.get("signal", 0),
            whale.get("signal", 0)
        ]

        positive = sum(1 for s in signals if s > 0.2)
        negative = sum(1 for s in signals if s < -0.2)

        if positive >= 2:
            alignment = "bullish_consensus"
        elif negative >= 2:
            alignment = "bearish_consensus"
        elif positive > negative:
            alignment = "mixed_bullish"
        elif negative > positive:
            alignment = "mixed_bearish"
        else:
            alignment = "neutral"

        return {
            "type": alignment,
            "positive_signals": positive,
            "negative_signals": negative,
            "strength": abs(sum(signals)) / 3
        }
