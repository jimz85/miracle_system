"""
Agent-M LLM增强版: 市场情报Agent
===================================
Miracle 2.0 — LLM增强市场情报系统

增强功能：
1. LLM情感分析 - 使用大语言模型进行深度情感分析，替代关键词匹配
2. 链上分析增强 - 智能模式识别、异常检测、趋势预测
3. 上下文构建 - 多源数据融合、历史上下文、预测性上下文

职责：
1. 新闻情感分析（LLM驱动）
2. 链上数据监控（智能模式识别）
3. 钱包分布监控（集中度分析）
4. 计算新闻/链上因子值
5. 构建丰富上下文
6. 输出情报报告给Agent-S
"""

import time
import logging
import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import re

# 尝试导入需要的库
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

# 导入LLM Provider
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from core.llm_provider import (
        get_llm_manager, LLMProviderManager, LLMProviderType,
        Message, LLMResponse, LLMConfig
    )
    HAS_LLM = True
except ImportError as e:
    HAS_LLM = False
    logging.warning(f"LLM Provider导入失败: {e}")
except NameError as e:
    # 处理 llm_provider.py 本身的类型错误
    HAS_LLM = False
    logging.warning(f"LLM Provider初始化失败(NameError): {e}")

# ============================================================
# 配置
# ============================================================

logger = logging.getLogger("AgentMarketIntelLLM")

# API配置
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
DEFAULT_LLM_PROVIDER = os.getenv("MARKET_INTEL_LLM", "auto")  # auto/claude/gpt/deepseek/ollama


# ============================================================
# 数据结构
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


def load_cache(symbol: str, data_type: str) -> Optional[CacheData]:
    """从缓存加载数据"""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{data_type}.json")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r") as f:
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
                method: str = "GET", timeout: int = 10) -> Optional[Dict]:
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
    except requests.exceptions.Timeout:
        logger.warning(f"API请求超时: {url}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.warning(f"API请求失败({e.response.status_code}): {url}")
        return None
    except Exception as e:
        logger.warning(f"API请求异常: {url}, {e}")
        return None


# ============================================================
# LLM情感分析模块
# ============================================================

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
            
            # 解析JSON响应
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
            # Fallback到传统方法
            return await self._keyword_fallback_analysis(news_items, symbol)
        
        # 对每条新闻进行LLM分析
        details = []
        total_score = 0.0
        valid_count = 0
        
        for item in news_items[:10]:  # 限制分析数量
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
        
        # 加权平均
        avg_score = total_score / valid_count if valid_count > 0 else 0.0
        
        # 综合分析
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
        
        # 构建摘要
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
                result["details"] = details
                return asdict(result)
        except Exception as e:
            logger.error(f"LLM综合分析失败: {e}")
        
        return self._default_result()

    def _parse_llm_response(self, content: str) -> Optional[LLMSentimentResult]:
        """解析LLM的JSON响应"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                # 尝试整个内容作为JSON解析
                json_str = content.strip()
            
            # 清理可能存在的markdown代码块
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


# ============================================================
# 链上分析增强模块
# ============================================================

class EnhancedOnChainAnalyzer:
    """
    增强的链上分析模块
    功能：
    1. 智能模式识别（大户行为、交易所流量模式）
    2. 异常检测
    3. 趋势预测
    4. 多维度信号融合
    """
    
    # 模式识别规则
    WHALE_PATTERNS = {
        "accumulation": {
            "description": "巨鲸积累模式",
            "indicators": ["流入交易所减少", "持币地址增加", "链上活跃度下降"],
            "implication": "看涨信号"
        },
        "distribution": {
            "description": "巨鲸分发模式",
            "indicators": ["流入交易所增加", "持币地址减少", "链上转移频繁"],
            "implication": "看跌信号"
        },
        "panic_selling": {
            "description": "恐慌抛售模式",
            "indicators": ["大量小额转账", "交易所流入激增", "价格快速下跌"],
            "implication": "短期看跌，可能超卖"
        },
        "whale_accumulation": {
            "description": "巨鲸吸筹模式",
            "indicators": ["大额转账增加", "交易所流出增加", "钱包余额上升"],
            "implication": "看涨信号"
        },
        "institutional_flow": {
            "description": "机构资金流向",
            "indicators": ["稳定币流入", "合约持仓变化", "ETF净流入"],
            "implication": "机构动向信号"
        }
    }
    
    def __init__(self):
        self.glassnode_base = API_CONFIG["glassnode"]["base_url"]
        self.glassnode_key = API_CONFIG["glassnode"]["api_key"]
        self.whale_base = API_CONFIG["whale_alert"]["base_url"]
        self.whale_key = API_CONFIG["whale_alert"]["api_key"]
        
        # 历史数据缓存（用于模式识别）
        self._flow_history: Dict[str, List[Dict]] = defaultdict(list)
        self._transfer_history: Dict[str, List[Dict]] = defaultdict(list)
    
    async def analyze_exchange_flow(self, symbol: str) -> Dict[str, Any]:
        """
        分析交易所流量
        识别资金流向模式和异常
        """
        # 获取基础流量数据
        flow_data = self._get_exchange_flow_data(symbol)
        
        if not flow_data:
            return self._default_flow_analysis()
        
        # 更新历史
        self._flow_history[symbol].append({
            "timestamp": time.time(),
            "flow": flow_data.get("flow", 0),
            "inflow": flow_data.get("inflow", 0),
            "outflow": flow_data.get("outflow", 0)
        })
        
        # 保持历史数据在合理范围
        if len(self._flow_history[symbol]) > 168:  # 7天 * 24小时
            self._flow_history[symbol] = self._flow_history[symbol][-168:]
        
        # 模式识别
        pattern = self._identify_flow_pattern(symbol)
        
        # 异常检测
        anomalies = self._detect_flow_anomalies(symbol)
        
        # 趋势分析
        trend = self._analyze_flow_trend(symbol)
        
        # 综合信号计算
        signal = self._calculate_flow_signal(flow_data, pattern, trend)
        
        return {
            "flow_data": flow_data,
            "pattern": pattern,
            "trend": trend,
            "anomalies": anomalies,
            "signal": signal,
            "interpretation": self._interpret_flow(flow_data, pattern, trend)
        }
    
    def _get_exchange_flow_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取交易所流量数据"""
        cached = load_cache(symbol, "exchange_flow")
        if cached and (time.time() - cached.timestamp) < 300:
            return cached.data
        
        # 尝试从API获取
        if not self.glassnode_key:
            # 使用免费方法
            return self._fetch_free_exchange_flow(symbol)
        
        return None  # 有API Key时的完整实现略过
    
    def _fetch_free_exchange_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        """使用OKX公开API获取流量代理数据"""
        sym_map = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "SOL": "SOL-USDT"}
        okx_sym = sym_map.get(symbol)
        if not okx_sym:
            return None
        
        try:
            url = f"https://www.okx.com/api/v5/market/candles?instId={okx_sym}&bar=1H&limit=24"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                return None
            
            klines = data["data"]
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            if len(closes) < 2:
                return None
            
            price_change = (closes[0] - closes[-1]) / closes[-1]
            total_volume = sum(volumes)
            estimated_flow = total_volume * price_change * closes[0]
            
            return {
                "flow": round(estimated_flow, 2),
                "inflow": round(max(0, estimated_flow), 2),
                "outflow": round(max(0, -estimated_flow), 2),
                "unit": "USD",
                "timestamp": get_timestamp(),
                "_source": "okx_kline_proxy"
            }
        except Exception as e:
            logger.warning(f"OKX流量API失败: {e}")
            return None
    
    def _identify_flow_pattern(self, symbol: str) -> Dict[str, Any]:
        """识别流量模式"""
        history = self._flow_history.get(symbol, [])
        if len(history) < 12:  # 至少12小时数据
            return {"type": "unknown", "confidence": 0, "description": "数据不足"}
        
        recent = history[-12:]  # 最近12小时
        older = history[-24:-12] if len(history) >= 24 else history[:-12]
        
        recent_avg_flow = sum(h["flow"] for h in recent) / len(recent)
        older_avg_flow = sum(h["flow"] for h in older) / len(older) if older else 0
        
        recent_avg_inflow = sum(h["inflow"] for h in recent) / len(recent)
        recent_avg_outflow = sum(h["outflow"] for h in recent) / len(recent)
        
        # 模式判断
        flow_ratio = recent_avg_flow / (abs(recent_avg_inflow) + abs(recent_avg_outflow) + 1)
        
        if recent_avg_inflow > recent_avg_outflow * 1.5 and recent_avg_flow > 0:
            pattern_type = "inflow_acceleration"
            description = "流入加速"
        elif recent_avg_outflow > recent_avg_inflow * 1.5 and recent_avg_flow < 0:
            pattern_type = "outflow_acceleration"
            description = "流出加速"
        elif abs(flow_ratio) < 0.2:
            pattern_type = "neutral"
            description = "中性平衡"
        elif recent_avg_flow > older_avg_flow * 1.2:
            pattern_type = "flow_increasing"
            description = "流量增加"
        else:
            pattern_type = "stable"
            description = "稳定"
        
        return {
            "type": pattern_type,
            "description": description,
            "confidence": min(0.9, len(history) / 72),  # 最多90%置信度
            "recent_flow_avg": recent_avg_flow,
            "flow_ratio": flow_ratio
        }
    
    def _detect_flow_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """检测流量异常"""
        history = self._flow_history.get(symbol, [])
        if len(history) < 6:
            return []
        
        anomalies = []
        
        # 计算统计量
        flows = [h["flow"] for h in history]
        mean_flow = sum(flows) / len(flows)
        
        # 检测异常值（2标准差外）
        import math
        variance = sum((f - mean_flow) ** 2 for f in flows) / len(flows)
        std_dev = math.sqrt(variance)
        
        for i, h in enumerate(history[-6:]):
            z_score = abs(h["flow"] - mean_flow) / (std_dev + 1e-10)
            if z_score > 2:
                anomalies.append({
                    "timestamp": h["timestamp"],
                    "flow": h["flow"],
                    "z_score": z_score,
                    "type": "spike" if h["flow"] > mean_flow else "dip"
                })
        
        return anomalies
    
    def _analyze_flow_trend(self, symbol: str) -> Dict[str, Any]:
        """分析流量趋势"""
        history = self._flow_history.get(symbol, [])
        if len(history) < 24:
            return {"direction": "unknown", "strength": 0}
        
        recent_6h = sum(h["flow"] for h in history[-6:]) / 6
        prev_6h = sum(h["flow"] for h in history[-12:-6]) / 6
        
        if abs(recent_6h) < 1e-10:
            direction = "neutral"
            strength = 0
        elif recent_6h > 0:
            direction = "inflow"
            strength = min(1.0, recent_6h / (abs(prev_6h) + 1e-10))
        else:
            direction = "outflow"
            strength = min(1.0, abs(recent_6h) / (abs(prev_6h) + 1e-10))
        
        return {
            "direction": direction,
            "strength": strength,
            "recent_avg": recent_6h,
            "prev_avg": prev_6h
        }
    
    def _calculate_flow_signal(self, flow_data: Dict, pattern: Dict, 
                               trend: Dict) -> float:
        """计算综合流量信号"""
        base_flow = flow_data.get("flow", 0)
        inflow = flow_data.get("inflow", 0)
        outflow = flow_data.get("outflow", 0)
        
        if inflow + outflow == 0:
            return 0.0
        
        # 基础信号：净流量比例
        net_ratio = base_flow / (inflow + outflow)
        
        # 趋势加权
        trend_weight = 0.3 if trend["direction"] != "unknown" else 0
        if trend["direction"] == "inflow":
            trend_factor = trend["strength"] * 0.3
        elif trend["direction"] == "outflow":
            trend_factor = -trend["strength"] * 0.3
        else:
            trend_factor = 0
        
        # 模式加权
        pattern_weight = 0.2 if pattern["type"] != "unknown" else 0
        pattern_sign = 1 if pattern["type"] in ["inflow_acceleration", "flow_increasing"] else -1
        
        signal = net_ratio * (1 - trend_weight - pattern_weight) + trend_factor + pattern_sign * pattern_weight
        
        return max(-1.0, min(1.0, signal))
    
    def _interpret_flow(self, flow_data: Dict, pattern: Dict, trend: Dict) -> str:
        """解释流量含义"""
        interpretations = []
        
        if pattern["type"] == "inflow_acceleration":
            interpretations.append("资金加速流入交易所")
        elif pattern["type"] == "outflow_acceleration":
            interpretations.append("资金加速流出交易所")
        elif pattern["type"] == "neutral":
            interpretations.append("资金进出平衡")
        
        if trend["direction"] == "inflow":
            interpretations.append(f"短期看涨（强度:{trend['strength']:.2f}）")
        elif trend["direction"] == "outflow":
            interpretations.append(f"短期看跌（强度:{trend['strength']:.2f}）")
        
        return "; ".join(interpretations) if interpretations else "中性信号"
    
    def _default_flow_analysis(self) -> Dict[str, Any]:
        """默认流量分析"""
        return {
            "flow_data": {"flow": 0, "inflow": 0, "outflow": 0},
            "pattern": {"type": "unknown", "confidence": 0},
            "trend": {"direction": "unknown", "strength": 0},
            "anomalies": [],
            "signal": 0.0,
            "interpretation": "数据获取失败"
        }
    
    async def analyze_whale_transfers(self, symbol: str, threshold_usd: float = 1000000) -> Dict[str, Any]:
        """
        分析大额转账（巨鲸活动）
        """
        # 获取转账数据
        transfers = self._get_large_transfers(symbol, threshold_usd)
        
        if not transfers:
            return self._default_whale_analysis()
        
        # 更新历史
        self._transfer_history[symbol].extend(transfers)
        if len(self._transfer_history[symbol]) > 1000:
            self._transfer_history[symbol] = self._transfer_history[symbol][-1000:]
        
        # 模式识别
        pattern = self._identify_whale_pattern(transfers, symbol)
        
        # 异常检测
        anomalies = self._detect_whale_anomalies(symbol)
        
        # 信号计算
        signal = self._calculate_whale_signal(transfers, pattern, anomalies)
        
        return {
            "transfers": transfers,
            "count": len(transfers),
            "total_volume": sum(t.get("amount_usd", 0) for t in transfers),
            "pattern": pattern,
            "anomalies": anomalies,
            "signal": signal,
            "interpretation": self._interpret_whale_activity(transfers, pattern)
        }
    
    def _get_large_transfers(self, symbol: str, threshold_usd: float) -> List[Dict]:
        """获取大额转账数据"""
        cached = load_cache(symbol, "large_transfers")
        if cached and (time.time() - cached.timestamp) < 60:
            return cached.data
        
        if not self.whale_key:
            return self._mock_large_transfers(symbol, threshold_usd)
        
        return []
    
    def _identify_whale_pattern(self, transfers: List[Dict], symbol: str) -> Dict[str, Any]:
        """识别巨鲸模式"""
        if not transfers:
            return {"type": "low_activity", "confidence": 0}
        
        # 分类转账
        exchange_to_wallet = 0
        wallet_to_exchange = 0
        wallet_to_wallet = 0
        
        for t in transfers:
            from_type = t.get("from", {}).get("type", "")
            to_type = t.get("to", {}).get("type", "")
            
            if from_type == "exchange" and to_type == "wallet":
                exchange_to_wallet += 1
            elif from_type == "wallet" and to_type == "exchange":
                wallet_to_exchange += 1
            else:
                wallet_to_wallet += 1
        
        # 模式判断
        if wallet_to_exchange > exchange_to_wallet * 1.5:
            pattern_type = "distribution"
            description = "巨鲸分发（转向交易所）"
        elif exchange_to_wallet > wallet_to_exchange * 1.5:
            pattern_type = "accumulation"
            description = "巨鲸积累（离开交易所）"
        elif wallet_to_wallet > len(transfers) * 0.7:
            pattern_type = "repositioning"
            description = "巨鲸调仓"
        else:
            pattern_type = "mixed"
            description = "混合活动"
        
        return {
            "type": pattern_type,
            "description": description,
            "confidence": min(0.9, len(transfers) / 10),
            "breakdown": {
                "to_exchange": wallet_to_exchange,
                "from_exchange": exchange_to_wallet,
                "wallet_to_wallet": wallet_to_wallet
            }
        }
    
    def _detect_whale_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """检测巨鲸活动异常"""
        history = self._transfer_history.get(symbol, [])
        if len(history) < 5:
            return []
        
        anomalies = []
        
        # 检测超大额转账
        recent = history[-10:]
        amounts = [t.get("amount_usd", 0) for t in recent]
        avg_amount = sum(amounts) / len(amounts)
        
        for t in recent:
            amount = t.get("amount_usd", 0)
            if amount > avg_amount * 5:
                anomalies.append({
                    "type": "large_transfer",
                    "amount_usd": amount,
                    "timestamp": t.get("timestamp"),
                    "severity": "high" if amount > avg_amount * 10 else "medium"
                })
        
        return anomalies
    
    def _calculate_whale_signal(self, transfers: List[Dict], pattern: Dict,
                                 anomalies: List[Dict]) -> float:
        """计算巨鲸信号"""
        if not transfers:
            return 0.0
        
        # 基础信号
        if pattern["type"] == "accumulation":
            base_signal = 0.3
        elif pattern["type"] == "distribution":
            base_signal = -0.3
        elif pattern["type"] == "repositioning":
            base_signal = 0.1
        else:
            base_signal = 0.0
        
        # 异常加权
        anomaly_factor = len(anomalies) * 0.1 * (-1 if pattern["type"] == "distribution" else 1)
        
        # 规模因子
        total_volume = sum(t.get("amount_usd", 0) for t in transfers)
        scale_factor = min(0.2, total_volume / 1e9)  # 最多0.2
        
        signal = base_signal + anomaly_factor + scale_factor
        return max(-1.0, min(1.0, signal))
    
    def _interpret_whale_activity(self, transfers: List[Dict], pattern: Dict) -> str:
        """解释巨鲸活动"""
        interpretations = []
        
        total_volume = sum(t.get("amount_usd", 0) for t in transfers)
        
        if pattern["type"] == "accumulation":
            interpretations.append(f"检测到{len(transfers)}笔巨鲸积累交易，总额${total_volume/1e6:.2f}M")
            interpretations.append("可能预示短期上涨")
        elif pattern["type"] == "distribution":
            interpretations.append(f"检测到{len(transfers)}笔巨鲸分发交易，总额${total_volume/1e6:.2f}M")
            interpretations.append("可能预示短期下跌")
        elif pattern["type"] == "repositioning":
            interpretations.append("巨鲸正在进行资产调仓")
        else:
            interpretations.append("巨鲸活动正常")
        
        return "; ".join(interpretations)
    
    def _mock_large_transfers(self, symbol: str, threshold_usd: float) -> List[Dict]:
        """生成模拟大额转账数据"""
        import random
        count = random.randint(1, 5)
        transfers = []
        for i in range(count):
            amount = random.uniform(threshold_usd, threshold_usd * 10)
            transfers.append({
                "id": f"mock_tx_{i}",
                "from": {"address": f"0x...mock{i}a", "type": random.choice(["exchange", "wallet"])},
                "to": {"address": f"0x...mock{i}b", "type": random.choice(["exchange", "wallet"])},
                "amount_usd": round(amount, 2),
                "timestamp": int(time.time()) - i * 3600,
                "blockchain": "bitcoin" if symbol == "BTC" else "ethereum"
            })
        return transfers
    
    def _default_whale_analysis(self) -> Dict[str, Any]:
        """默认巨鲸分析"""
        return {
            "transfers": [],
            "count": 0,
            "total_volume": 0,
            "pattern": {"type": "unknown", "confidence": 0},
            "anomalies": [],
            "signal": 0.0,
            "interpretation": "数据获取失败"
        }


# ============================================================
# 上下文构建模块
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
    
    async def build_context(self, 
                           current_sentiment: Dict,
                           onchain_data: Dict,
                           whale_data: Dict) -> MarketContext:
        """
        构建丰富的市场上下文
        
        Args:
            current_sentiment: 当前情感分析结果
            onchain_data: 链上数据
            whale_data: 巨鲸数据
        
        Returns:
            MarketContext: 包含丰富上下文信息的市场上下文
        """
        # 更新历史
        self._sentiment_history.append({
            "timestamp": time.time(),
            "score": current_sentiment.get("score", 0),
            "label": current_sentiment.get("label", "中性")
        })
        
        # 保持历史数据在合理范围
        if len(self._sentiment_history) > self._max_history:
            self._sentiment_history = self._sentiment_history[-self._max_history:]
        
        # 分析历史趋势
        historical_sentiment = self._analyze_sentiment_history()
        
        # 识别市场阶段
        market_phase = self._identify_market_phase(
            current_sentiment, onchain_data, whale_data
        )
        
        # 评估鲸鱼活动水平
        whale_level = self._assess_whale_activity(whale_data)
        
        # 判断交易所流量阶段
        flow_phase = self._assess_flow_phase(onchain_data)
        
        # 推断散户情绪
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
        
        # 计算移动平均
        result = []
        for i in range(0, len(self._sentiment_history), 24):  # 每日采样
            window = self._sentiment_history[i:min(i+24, len(self._sentiment_history))]
            if window:
                avg_score = sum(h["score"] for h in window) / len(window)
                result.append({
                    "timestamp": window[-1]["timestamp"],
                    "score": avg_score,
                    "label": window[-1]["label"]
                })
        
        return result[-30:]  # 返回最近30个数据点
    
    def _identify_market_phase(self, sentiment: Dict, onchain: Dict, 
                               whale: Dict) -> str:
        """识别市场阶段"""
        sentiment_score = sentiment.get("score", 0)
        flow_signal = onchain.get("signal", 0)
        whale_signal = whale.get("signal", 0)
        
        avg_signal = (sentiment_score + flow_signal + whale_signal) / 3
        
        # 基于信号判断
        if avg_signal > 0.3:
            if sentiment_score > 0.5:
                return "markup"  # 上涨阶段
            else:
                return "accumulation"  # 积累阶段
        elif avg_signal < -0.3:
            if sentiment_score < -0.5:
                return "markdown"  # 下跌阶段
            else:
                return "distribution"  # 分发阶段
        else:
            return "neutral"  # 中性/横盘
    
    def _assess_whale_activity(self, whale_data: Dict) -> str:
        """评估鲸鱼活动水平"""
        transfer_count = whale_data.get("count", 0)
        total_volume = whale_data.get("total_volume", 0)
        pattern_type = whale_data.get("pattern", {}).get("type", "")
        
        # 基于多个因素评估
        if transfer_count >= 10 or total_volume > 50_000_000:
            level = "extreme"
        elif transfer_count >= 5 or total_volume > 10_000_000:
            level = "elevated"
        elif transfer_count >= 2 or total_volume > 1_000_000:
            level = "normal"
        else:
            level = "low"
        
        # 根据模式调整
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
        # 散户情绪通常与大户反向
        whale_signal = onchain.get("signal", 0)
        sentiment_score = sentiment.get("score", 0)
        
        # 如果鲸鱼在看跌（信号为负），散户可能乐观
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


# ============================================================
# Agent-M LLM主类
# ============================================================

class MarketIntelAgentLLM:
    """
    市场情报Agent LLM增强版主类
    协调LLM情感分析、增强链上分析和上下文构建模块
    """
    
    def __init__(self, symbol: str = "BTC", llm_provider: str = "auto"):
        self.symbol = symbol.upper()
        
        # 初始化LLM管理器
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
        """
        生成综合情报报告（异步版本）
        
        Returns:
            IntelReport格式的字典
        """
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
        
        # 4. 钱包分布（保持原有功能）
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
            # LLM增强字段
            "llm_analysis": llm_analysis,
            "context": context.to_dict(),
            "patterns": patterns
        }
        
        logger.info(f"=== {self.symbol}情报报告生成完成: 综合评分={combined_score:.3f}, 推荐={recommendation}, 置信度={confidence:.2f} ===")
        
        return report

    def generate_intel_report_sync(self) -> Dict:
        """同步版本的报告生成（兼容现有代码）"""
        return asyncio.run(self.generate_intel_report())

    async def _get_news_sentiment(self) -> Dict:
        """获取LLM增强的新闻情感分析"""
        try:
            # 获取新闻
            news = await self._fetch_news()
            
            # 使用LLM分析
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
        
        # 尝试获取新闻
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
                # 模拟数据
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
            
            # 计算信号
            signal = max(-1.0, min(1.0, data.get("change_24h", 0) / 1.0))
            data["signal"] = round(signal, 3)
            return data
        except Exception as e:
            logger.error(f"集中度获取失败: {e}")
            return {"signal": 0, "top10_pct": 0, "change_24h": 0}

    async def _generate_llm_analysis(self, sentiment: Dict, onchain: Dict,
                                      whale: Dict, context: MarketContext) -> Dict[str, Any]:
        """使用LLM生成综合分析"""
        if not self.llm_manager:
            return {
                "summary": "LLM不可用",
                "key_insights": [],
                "risks": [],
                "opportunities": []
            }
        
        prompt = f"""基于以下{symbol}市场数据，请提供综合分析：

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
                import re
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
                                 whale: Dict, context: MarketContext) -> Tuple[str, float]:
        """生成推荐"""
        signals = [
            sentiment.get("score", 0),
            onchain.get("signal", 0),
            whale.get("signal", 0)
        ]
        
        positive_count = sum(1 for s in signals if s > 0.2)
        negative_count = sum(1 for s in signals if s < -0.2)
        
        # 信号一致性调整
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
        
        # 巨鲸积累模式
        if whale.get("pattern", {}).get("type") == "accumulation" and onchain.get("signal", 0) > 0.2:
            patterns.append({
                "type": "whale_accumulation",
                "name": "巨鲸积累模式",
                "confidence": whale.get("pattern", {}).get("confidence", 0.5),
                "description": "检测到巨鲸吸筹行为",
                "implication": "短期看涨",
                "severity": "high"
            })
        
        # 巨鲸分发模式
        if whale.get("pattern", {}).get("type") == "distribution" and onchain.get("signal", 0) < -0.2:
            patterns.append({
                "type": "whale_distribution",
                "name": "巨鲸分发模式",
                "confidence": whale.get("pattern", {}).get("confidence", 0.5),
                "description": "检测到巨鲸抛售行为",
                "implication": "短期看跌",
                "severity": "high"
            })
        
        # 流量异常
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
            f"🔗 链上分析:",
            f"   流量信号: {report['onchain']['exchange_flow_signal']:+.2f}",
            f"   模式: {report['onchain']['flow_pattern']}",
            "",
            f"🐋 鲸鱼活动:",
            f"   信号: {report['whale']['signal']:+.2f}",
            f"   模式: {report['whale']['pattern']}",
            f"   转账数: {report['whale']['count']}",
            "",
            f"📈 市场上下文:",
            f"   阶段: {report['context']['market_phase']}",
            f"   鲸鱼活动: {report['context']['whale_activity_level']}",
            f"   信号一致性: {report['context']['correlation_data']['signal_alignment']['type']}",
            "",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"📈 综合评分: {report['combined_score']:+.2f}",
            f"🎯 推荐: {report['recommendation']}",
            f"🔒 置信度: {report['confidence']:.0%}",
        ]
        
        return "\n".join([l for l in lines if l])


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
    
    # 打印LLM分析
    if report.get("llm_analysis"):
        print("\n🤖 LLM综合分析:")
        llm = report["llm_analysis"]
        print(f"   总结: {llm.get('summary', '无')}")
        if llm.get("key_insights"):
            print("   关键洞察:")
            for insight in llm["key_insights"][:3]:
                print(f"     - {insight}")
    
    # 打印检测到的模式
    if report.get("patterns"):
        print("\n🔍 检测到的模式:")
        for pattern in report["patterns"]:
            print(f"     - {pattern['name']} (置信度:{pattern['confidence']:.0%})")
    
    print("=" * 60)
    
    return report


def main(symbol: str = "BTC", llm_provider: str = "auto"):
    """同步主入口"""
    return asyncio.run(main_async(symbol, llm_provider))


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    provider = sys.argv[2] if len(sys.argv) > 2 else "auto"
    main(symbol, provider)
