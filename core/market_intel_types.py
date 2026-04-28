from __future__ import annotations

"""
Market Intel Types - 市场情报数据类型和工具函数
==============================================

从 agents/agent_market_intel_llm.py 提取

包含:
- 枚举类型 (SentimentLabel, SignalStrength)
- 数据类 (IntelReport, LLMSentimentResult, OnChainPattern, MarketContext, CacheData)
- 配置常量 (API_CONFIG, CACHE_DIR, SYMBOL_MAP)
- 工具函数 (get_timestamp, load_cache, save_cache, api_request)

用法:
    from core.market_intel_types import IntelReport, SentimentLabel, load_cache
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

logger = logging.getLogger("MarketIntelTypes")

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
