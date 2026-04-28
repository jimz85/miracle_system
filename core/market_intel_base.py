from __future__ import annotations

"""
Market Intel Base - 市场情报基础模块
=====================================

包含:
- 枚举类型 (SentimentLabel, SignalStrength)
- 数据类 (IntelReport, LLMSentimentResult, OnChainPattern, MarketContext, CacheData)
- 配置常量 (API_CONFIG, CACHE_DIR, SYMBOL_MAP)
- 工具函数 (get_timestamp, load_cache, save_cache, api_request)
- ContextBuilder: 市场上下文构建器

用法:
    from core.market_intel_base import (
        IntelReport,
        SentimentLabel,
        SignalStrength,
        MarketContext,
        CacheData,
        load_cache,
        save_cache,
        ContextBuilder,
    )
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# 尝试导入可选依赖
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from websocket import create_connection
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

logger = logging.getLogger("MarketIntelBase")

# ============================================================
# 配置
# ============================================================

API_CONFIG = {
    "cryptocompare": {
        "base_url": "https://min-api.cryptocompare.com/data",
        "api_key": os.getenv("CRYPTOCOMPARE_API_KEY", ""),
    },
    "glassnode": {
        "base_url": "https://api.glassnode.com/v1",
        "api_key": os.getenv("GLASSNODE_API_KEY", ""),
    },
    "whale_alert": {
        "base_url": "https://api.whale-alert.io/v1",
        "api_key": os.getenv("WHALE_ALERT_API_KEY", ""),
    },
}

# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 默认交易对符号映射
SYMBOL_MAP = {
    "BTC": {"cc": "BTC", "glass": "BTC", "coingecko": "bitcoin"},
    "ETH": {"cc": "ETH", "glass": "ETH", "coingecko": "ethereum"},
}

# LLM配置
DEFAULT_LLM_PROVIDER = os.getenv("MARKET_INTEL_LLM", "auto")


# ============================================================
# 枚举类型
# ============================================================

class SentimentLabel(Enum):
    BULLISH = "利好"
    BEARISH = "利空"
    NEUTRAL = "中性"


class SignalStrength(Enum):
    STRONG_BULLISH = "强烈看多"
    BULLISH = "看多"
    SLIGHT_BULLISH = "轻微看多"
    NEUTRAL = "中性"
    SLIGHT_BEARISH = "轻微看空"
    BEARISH = "看空"
    STRONG_BEARISH = "强烈看空"


# ============================================================
# 数据类
# ============================================================

@dataclass
class IntelReport:
    """综合情报报告"""
    symbol: str
    timestamp: str
    news_sentiment: Dict[str, Any]
    onchain: Dict[str, Any]
    wallet: Dict[str, Any]
    combined_score: float
    recommendation: str
    confidence: float
    # LLM增强字段
    llm_analysis: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    patterns: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LLMSentimentResult:
    """LLM情感分析结果"""
    score: float  # -1 ~ +1
    label: str
    reasoning: str
    key_factors: List[str]
    confidence: float  # 0 ~ 1
    market_tone: str
    affected_factors: List[str]


@dataclass
class OnChainPattern:
    """链上模式识别结果"""
    pattern_type: str
    pattern_name: str
    confidence: float
    description: str
    implication: str
    severity: str  # low/medium/high/critical


@dataclass
class MarketContext:
    """市场上下文"""
    symbol: str
    timestamp: str
    historical_sentiment: List[Dict]  # 历史情感趋势
    correlation_data: Dict[str, Any]  # 相关性数据
    market_phase: str  # accumulation/distribution/markup/markdown
    whale_activity_level: str  # low/normal/elevated/extreme
    exchange_flow_phase: str  # inflow/outflow/neutral
    retail_sentiment: str  # optimistic/neutral/pessimistic

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CacheData:
    data: Any
    timestamp: float
    source: str


# ============================================================
# FOMC事件检测
# ============================================================

# FOMC会议日期（2026年 - 约每6-8周一次）
# 格式：(year, month, day)
KNOWN_FOMC_DATES_2026 = [
    # 2026年FOMC会议日期（通常在周二或周三）
    # 这些是估计日期，实际日期以美联储官方发布为准
    (2026, 1, 27),   # 1月会议
    (2026, 3, 17),   # 3月会议
    (2026, 5, 4),    # 5月会议
    (2026, 6, 15),   # 6月会议
    (2026, 7, 27),   # 7月会议
    (2026, 9, 15),   # 9月会议
    (2026, 11, 2),   # 11月会议
    (2026, 12, 14),  # 12月会议
]

# FOMC窗口期（天）- 会议前后各2天为高波动期
FOMC_WINDOW_DAYS = 2


def is_fomc_window(dt: datetime = None) -> bool:
    """
    检测当前是否处于FOMC窗口期
    
    FOMC窗口期定义：会议日前后各2天（共5天窗口）
    在此期间市场波动性增加，置信度应降低50%
    
    Args:
        dt: 要检查的时间，默认为当前时间
        
    Returns:
        bool: 是否处于FOMC窗口期
    """
    from datetime import timedelta
    
    if dt is None:
        dt = datetime.now()
    
    # 检查是否在已知FOMC日期的窗口内
    for fomc_date in KNOWN_FOMC_DATES_2026:
        meeting = datetime(fomc_date[0], fomc_date[1], fomc_date[2])
        window_start = meeting - timedelta(days=FOMC_WINDOW_DAYS)
        window_end = meeting + timedelta(days=FOMC_WINDOW_DAYS + 1)  # +1因为end是不包含的
        
        if window_start <= dt < window_end:
            return True
    
    return False


def get_fomc_confidence_multiplier(confidence: float, dt: datetime = None) -> float:
    """
    根据FOMC窗口期调整置信度
    
    在FOMC窗口期内，置信度降低50%（乘以0.5）
    
    Args:
        confidence: 原始置信度 (0-100)
        dt: 要检查的时间，默认为当前时间
        
    Returns:
        float: 调整后的置信度
    """
    if is_fomc_window(dt):
        return confidence * 0.5
    return confidence


def get_fomc_status(dt: datetime = None) -> Dict[str, Any]:
    """
    获取FOMC状态信息
    
    Args:
        dt: 要检查的时间，默认为当前时间
        
    Returns:
        Dict: 包含FOMC状态的字典
    """
    from datetime import timedelta
    
    if dt is None:
        dt = datetime.now()
    
    in_window = is_fomc_window(dt)
    
    # 找到最近的FOMC会议
    nearest_meeting = None
    days_to_meeting = None
    
    for fomc_date in KNOWN_FOMC_DATES_2026:
        meeting = datetime(fomc_date[0], fomc_date[1], fomc_date[2])
        if meeting >= dt:
            nearest_meeting = meeting
            days_to_meeting = (meeting - dt).days
            break
    
    return {
        "in_fomc_window": in_window,
        "confidence_multiplier": 0.5 if in_window else 1.0,
        "nearest_meeting": nearest_meeting.isoformat() if nearest_meeting else None,
        "days_to_meeting": days_to_meeting,
        "window_days": FOMC_WINDOW_DAYS,
    }


# ============================================================
# 工具函数
# ============================================================

def get_timestamp() -> str:
    """返回当前UTC时间戳字符串"""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cache(symbol: str, data_type: str) -> CacheData | None:
    """从缓存加载数据"""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{data_type}.json")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file) as f:
            cached = json.load(f)
        return CacheData(
            data=cached.get("data"),
            timestamp=cached.get("timestamp", 0),
            source=cached.get("source", "cache")
        )
    except Exception as e:
        logger.warning(f"读取缓存失败: {cache_file}, {e}")
        return None


def save_cache(symbol: str, data_type: str, data: Any):
    """保存数据到缓存"""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{data_type}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump({
                "data": data,
                "timestamp": time.time(),
                "source": "cache"
            }, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"写入缓存失败: {cache_file}, {e}")


def api_request(url: str, params: Dict = None, headers: Dict = None,
                method: str = "GET", timeout: int = 10) -> Dict | None:
    """统一的API请求方法，带超时和错误处理"""
    if not HAS_REQUESTS:
        logger.error("requests库未安装")
        return None
    try:
        if method == "GET":
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        else:
            resp = requests.post(url, json=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"API请求异常: {url}, {e}")
        return None


# ============================================================
# ContextBuilder - 市场上下文构建器
# ============================================================

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


# =============================================================================
# 市场状态检测 (Regime Classifier)
# =============================================================================

def get_market_regime(btc_adx: float, btc_rsi: float = None) -> str:
    """
    基于BTC指标判断市场状态

    Args:
        btc_adx: BTC当前ADX (>25=趋势市场, <20=震荡市场)
        btc_rsi: BTC当前RSI (辅助判断)

    Returns:
        'trend': 趋势市场 - 趋势跟踪策略有效，ADX/MACD权重×1.2
        'range': 震荡市场 - 均值回归策略有效，RSI/布林权重×1.2
        'neutral': 中性市场 - 无明显方向
    """
    if btc_adx >= 25:
        return "trend"
    elif btc_adx < 20:
        return "range"
    else:
        return "neutral"


def get_regime_confidence_multiplier(
    confidence: float,
    regime: str,
    factor_name: str
) -> float:
    """
    根据市场状态调整因子置信度

    Args:
        confidence: 原始置信度 (0-100)
        regime: 市场状态 ('trend' / 'range' / 'neutral')
        factor_name: 因子名 ('adx', 'macd', 'rsi', 'bollinger', 'momentum', 'gemma')

    Returns:
        调整后的置信度
    """
    if regime == "trend":
        # 趋势市场: ADX/MACD更可靠，RSI/布林容易失效
        trend_factors = {"adx", "macd", "momentum"}
        if factor_name in trend_factors:
            return confidence * 1.1  # +10%
        else:
            return confidence * 0.85  # -15%
    elif regime == "range":
        # 震荡市场: RSI/布林更可靠，趋势因子容易假信号
        range_factors = {"rsi", "bollinger"}
        if factor_name in range_factors:
            return confidence * 1.1  # +10%
        else:
            return confidence * 0.85  # -15%
    return confidence  # neutral: 无调整

