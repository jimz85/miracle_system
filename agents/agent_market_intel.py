from __future__ import annotations

"""
Agent-M: 市场情报Agent
Miracle 1.0.1 — 高频趋势跟踪+事件驱动混合系统

职责：
1. 新闻情感分析（利好/利空/中性）
2. 链上数据监控（交易所净流量、大额转账）
3. 钱包分布监控（持币地址集中度）
4. 计算新闻/链上因子值
5. 输出情报报告给Agent-S

LLM增强版：
    如需使用LLM增强的情感分析和链上分析，请使用 agent_market_intel_llm.py
    该版本提供：
    - LLM驱动的深度情感分析
    - 智能链上模式识别
    - 丰富上下文构建
    - 多源数据融合

用法：
    from agent_market_intel_llm import MarketIntelAgentLLM
    agent = MarketIntelAgentLLM(symbol="BTC")
    report = agent.generate_intel_report_sync()
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

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

# ============================================================
# 配置
# ============================================================

logger = logging.getLogger("AgentMarketIntel")

# API配置（需要用户自行填写API Key）
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


# ============================================================
# 数据结构
# ============================================================

class SentimentLabel(Enum):
    BULLISH = "利好"
    BEARISH = "利空"
    NEUTRAL = "中性"


@dataclass
class IntelReport:
    symbol: str
    timestamp: str
    news_sentiment: Dict[str, Any]
    onchain: Dict[str, Any]
    wallet: Dict[str, Any]
    combined_score: float
    recommendation: str
    confidence: float

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
# 新闻情感分析
# ============================================================

class NewsIntel:
    """
    新闻情感分析模块
    数据源：CryptoCompare News API / 搜索引擎
    """

    def __init__(self):
        self.base_url = API_CONFIG["cryptocompare"]["base_url"]
        self.api_key = API_CONFIG["cryptocompare"]["api_key"]

    def fetch_news(self, symbol: str, hours: int = 24) -> List[Dict]:
        """
        抓取近N小时的新闻

        Args:
            symbol: 交易对符号，如BTC、ETH
            hours: 回溯小时数，默认24

        Returns:
            新闻列表，每条包含title、body、published_on、source等字段
        """
        # 尝试从缓存加载
        cached = load_cache(symbol, "news")
        if cached and (time.time() - cached.timestamp) < 300:  # 5分钟内不重复请求
            logger.info(f"使用缓存新闻数据: {symbol}")
            return cached.data

        if not self.api_key:
            # 先尝试CryptoCompare免费访问（无需API Key）
            free_result = self._fetch_free_news_cryptocompare(symbol, hours)
            if free_result:
                save_cache(symbol, "news", free_result)
                return free_result
            logger.warning("CryptoCompare免费访问也失败，使用本地情绪代理数据")
            return self._generate_sentiment_from_price(symbol, hours)

        url = f"{self.base_url}/v2/news/"
        params = {
            "lang": "ZH",
            "categories": symbol,
            "api_key": self.api_key,
        }

        result = api_request(url, params=params)
        if not result or "Data" not in result:
            logger.warning(f"获取新闻失败，使用缓存: {symbol}")
            return cached.data if cached else []

        news = result["Data"]
        save_cache(symbol, "news", news)
        return news

    def analyze_sentiment(self, news_items: List[Dict]) -> Dict[str, Any]:
        """
        对新闻列表进行情感分析

        Args:
            news_items: 新闻列表

        Returns:
            {
                "score": float,      # -1(完全利空) ~ +1(完全利好)
                "labels": list,       # ['利好:60%', '中性:30%', '利空:10%']
                "count": int,         # 分析的新闻数量
                "details": list       # 每条新闻的情感得分
            }
        """
        if not news_items:
            return {
                "score": 0.0,
                "labels": ["利好:0%", "中性:100%", "利空:0%"],
                "count": 0,
                "details": []
            }

        # 关键词情感词典
        bullish_keywords = [
            "暴涨", "突破", "新高", "涨势", "利好", "看涨", "买入", "抄底",
            "飙升", "狂涨", "疯涨", "创新高", "收涨", "大涨", "反弹",
            "surge", "bullish", "breakout", "high", "soar", "rally", "buy",
            "pump", "moon", "all-time", "ATH", "uptrend"
        ]
        bearish_keywords = [
            "暴跌", "破发", "新低", "跌势", "利空", "看跌", "卖出", "割肉",
            "闪崩", "狂跌", "腰斩", "创新低", "收跌", "大跌", "回落",
            "crash", "bearish", "breakdown", "low", "plunge", "drop", "sell",
            "dump", "capitulation", "ATL", "downtrend"
        ]
        neutral_keywords = [
            "横盘", "震荡", "整理", "观望", "等待", "平稳", "持平",
            "sideways", "consolidation", "stable", "unchanged", "flat"
        ]

        scores = []
        details = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for item in news_items:
            text = f"{item.get('title', '')} {item.get('body', '')}".lower()
            score = 0.0

            for kw in bullish_keywords:
                if kw.lower() in text:
                    score += 0.3
            for kw in bearish_keywords:
                if kw.lower() in text:
                    score -= 0.3
            for kw in neutral_keywords:
                if kw.lower() in text:
                    score *= 0.5  # 中性词降低信号强度

            score = max(-1.0, min(1.0, score))  # 限制在[-1, 1]
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
        neutral_pct = round(neutral_count / total * 100)
        bearish_pct = round(bearish_count / total * 100)

        # 确保加起来=100%
        if bullish_pct + neutral_pct + bearish_pct != 100:
            diff = 100 - (bullish_pct + neutral_pct + bearish_pct)
            neutral_pct += diff

        return {
            "score": round(avg_score, 3),
            "labels": [
                f"利好:{bullish_pct}%",
                f"中性:{neutral_pct}%",
                f"利空:{bearish_pct}%"
            ],
            "count": total,
            "details": details[:10]  # 最多返回10条详情
        }

    def get_sentiment_with_fresh_news(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """获取情感分析报告（自动抓取+分析）"""
        news = self.fetch_news(symbol, hours)
        return self.analyze_sentiment(news)

    def _fetch_free_news_cryptocompare(self, symbol: str, hours: int) -> List[Dict]:
        """
        使用免费RSS新闻源获取加密货币新闻（无需API Key）
        数据源: The Block RSS
        """
        try:
            # 导入免费新闻模块
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.news_fetcher import fetch_rss_news

            news_items = fetch_rss_news("theblock", limit=20)
            if news_items:
                # 转换为兼容格式
                result = []
                for item in news_items:
                    result.append({
                        "id": item.get("link", ""),
                        "title": item.get("title", ""),
                        "body": item.get("description", ""),
                        "published_on": self._parse_rss_date(item.get("pubDate", "")),
                        "source": item.get("source", "The Block"),
                        "categories": symbol
                    })
                logger.info(f"RSS新闻获取到{len(result)}条{symbol}新闻")
                return result[:20]
        except Exception as e:
            logger.warning(f"RSS新闻获取失败: {e}")
        return []

    def _parse_rss_date(self, date_str: str) -> int:
        """解析RSS日期为时间戳"""
        try:
            from email.utils import parsedate_to_timestamp
            return int(parsedate_to_timestamp(date_str))
        except Exception:
            return int(time.time())

    def _generate_sentiment_from_price(self, symbol: str, hours: int) -> List[Dict]:
        """
        当所有新闻API都不可用时，用价格动量作为情感代理指标。
        OKX公开API获取近期价格变化，判断市场情绪方向。
        """
        sym_map = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "SOL": "SOL-USDT"}
        okx_sym = sym_map.get(symbol, f"{symbol}-USDT")
        try:
            url = f"https://www.okx.com/api/v5/market/candles?instId={okx_sym}&bar=1H&limit={hours}"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                return self._generate_mock_news(symbol, hours)
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                return self._generate_mock_news(symbol, hours)
            klines = data["data"]  # 最新在前
            closes = [float(k[4]) for k in klines]
            if len(closes) < 2:
                return self._generate_mock_news(symbol, hours)
            change_pct = (closes[0] - closes[-1]) / closes[-1] * 100
            if change_pct > 3:
                sentiment_label, sentiment_score = "看涨情绪浓厚", 0.8
            elif change_pct > 1:
                sentiment_label, sentiment_score = "小幅上涨", 0.6
            elif change_pct < -3:
                sentiment_label, sentiment_score = "看跌情绪浓厚", -0.8
            elif change_pct < -1:
                sentiment_label, sentiment_score = "小幅下跌", -0.6
            else:
                sentiment_label, sentiment_score = "震荡整理", 0.0
            now = int(time.time())
            return [{
                "id": f"price_proxy_{symbol}",
                "title": f"[价格动量代理] {symbol}近{hours}小时{sentiment_label}（涨跌{change_pct:+.2f}%）",
                "body": f"基于OKX公开K线数据，{symbol}过去{hours}小时价格变化{change_pct:+.2f}%。此为算法生成的情绪代理指标，非真实新闻。",
                "published_on": now,
                "source": "OKX K线代理",
                "categories": symbol,
                "url": "",
                "_sentiment_score": sentiment_score,
            }]
        except Exception as e:
            logger.warning(f"价格动量代理也失败: {e}")
            return self._generate_mock_news(symbol, hours)

    def _generate_mock_news(self, symbol: str, hours: int) -> List[Dict]:
        """生成模拟新闻数据（用于测试）"""
        mock_titles = [
            f"{symbol}价格突破关键阻力位，短期看涨情绪浓厚",
            f"{symbol}交易所净流入增加，机构持续买入",
            f"分析师：{symbol}或将迎来新一轮上涨行情",
            f"某巨鲸地址转入大量{symbol}，市场情绪偏向乐观",
        ]
        now = int(time.time())
        return [
            {
                "id": i,
                "title": title,
                "body": f"关于{symbol}的最新市场分析报道...",
                "published_on": now - i * 3600,
                "source": "模拟数据源",
                "categories": symbol,
                "url": ""
            }
            for i, title in enumerate(mock_titles)
        ]


# ============================================================
# 链上数据监控
# ============================================================

class OnChainIntel:
    """
    链上数据监控模块
    数据源：Glassnode API / Whale Alert API
    """

    def __init__(self):
        self.glassnode_base = API_CONFIG["glassnode"]["base_url"]
        self.glassnode_key = API_CONFIG["glassnode"]["api_key"]
        self.whale_base = API_CONFIG["whale_alert"]["base_url"]
        self.whale_key = API_CONFIG["whale_alert"]["api_key"]

    def get_exchange_flow(self, symbol: str) -> Dict[str, Any]:
        """
        获取交易所净流量

        Args:
            symbol: 交易对符号

        Returns:
            {
                "flow": float,           # 净流量（正值=流入，负值=流出）
                "inflow": float,          # 流入量
                "outflow": float,         # 流出量
                "unit": str,              # 单位
                "timestamp": str         # 数据时间戳
            }
        """
        cached = load_cache(symbol, "exchange_flow")
        if cached and (time.time() - cached.timestamp) < 300:
            logger.info(f"使用缓存交易所流量数据: {symbol}")
            return cached.data

        if not self.glassnode_key:
            # 先尝试免费API：币安交易所余额变动（无需API Key）
            free_flow = self._fetch_free_exchange_flow(symbol)
            if free_flow:
                save_cache(symbol, "exchange_flow", free_flow)
                return free_flow
            logger.warning("免费交易所流量也失败，使用模拟数据")
            return self._mock_exchange_flow(symbol)

        # Glassnode API: 交易所流入/流出
        url = f"{self.glassnode_base}/metrics/distribution/exchange_flow"

        headers = {"Authorization": f"Bearer {self.glassnode_key}"}
        params = {
            "asset": symbol,
            "interval": "24h",
            "time": "now"
        }

        result = api_request(url, params=params, headers=headers)
        if not result or "data" not in result:
            logger.warning(f"获取交易所流量失败，使用缓存: {symbol}")
            return cached.data if cached else {"flow": 0, "inflow": 0, "outflow": 0, "unit": "USD", "timestamp": get_timestamp()}

        # 解析数据
        data_points = result["data"]
        if not data_points:
            return {"flow": 0, "inflow": 0, "outflow": 0, "unit": "USD", "timestamp": get_timestamp()}

        latest = data_points[-1]
        inflow = latest.get("i", 0)  # inflow
        outflow = latest.get("o", 0)  # outflow
        flow = inflow - outflow

        flow_data = {
            "flow": round(flow, 2),
            "inflow": round(inflow, 2),
            "outflow": round(outflow, 2),
            "unit": "USD",
            "timestamp": latest.get("t", get_timestamp())
        }

        save_cache(symbol, "exchange_flow", flow_data)
        return flow_data

    def get_large_transfers(self, symbol: str, threshold_usd: float = 1000000) -> List[Dict]:
        """
        监控大额转账（>threshold_usd美元）

        Args:
            symbol: 交易对符号
            threshold_usd: 阈值（美元），默认100万美元

        Returns:
            大额转账列表
        """
        cached = load_cache(symbol, "large_transfers")
        if cached and (time.time() - cached.timestamp) < 60:  # 1分钟内缓存
            return cached.data

        if not self.whale_key:
            logger.warning("Whale Alert API Key未配置，使用模拟数据")
            return self._mock_large_transfers(symbol, threshold_usd)

        url = f"{self.whale_base}/transactions"
        params = {
            "api_key": self.whale_key,
            "min_value": int(threshold_usd),
            "symbol": symbol
        }

        result = api_request(url, params=params)
        if not result:
            return cached.data if cached else []

        transactions = []
        for tx in result.get("transactions", []):
            transactions.append({
                "id": tx.get("id"),
                "from": tx.get("from", {}),
                "to": tx.get("to", {}),
                "amount": tx.get("amount"),
                "amount_usd": tx.get("amount_usd"),
                "timestamp": tx.get("timestamp"),
                "blockchain": tx.get("blockchain")
            })

        save_cache(symbol, "large_transfers", transactions)
        return transactions

    def calc_net_flow_signal(self, symbol: str) -> float:
        """
        计算净流量信号

        Returns:
            float: -1(大量流出) ~ +1(大量流入)
        """
        flow_data = self.get_exchange_flow(symbol)

        flow = flow_data.get("flow", 0)
        inflow = flow_data.get("inflow", 0)
        outflow = flow_data.get("outflow", 0)

        if inflow + outflow == 0:
            return 0.0

        # 信号计算：净流量 / 总流量 的归一化值
        # 净流量占总流量的比例，再映射到[-1, 1]
        net_ratio = flow / (inflow + outflow) if (inflow + outflow) > 0 else 0

        # 考虑绝对规模：流量相对于市值的比例
        # 这里简化为直接使用net_ratio，因为绝对值难以获取
        signal = max(-1.0, min(1.0, net_ratio))

        logger.info(f"{symbol}净流量信号: {signal:.3f} (流入:{inflow:.0f}, 流出:{outflow:.0f})")
        return round(signal, 3)

    def _fetch_free_exchange_flow(self, symbol: str) -> Dict[str, Any] | None:
        """
        使用免费API获取交易所净流量（无需API Key）
        策略：使用OKX公开API获取近期K线，判断主动买卖压力。
        原理：通过价格变化和成交量估算资金流向。
        """
        sym_map = {
            "BTC": "BTC-USDT",
            "ETH": "ETH-USDT",
            "SOL": "SOL-USDT",
        }
        okx_sym = sym_map.get(symbol)
        if not okx_sym:
            return None

        try:
            # 使用OKX公开API获取近期K线
            url = f"https://www.okx.com/api/v5/market/candles?instId={okx_sym}&bar=1H&limit=24"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                return None

            # OKX返回格式: [ts, open, high, low, close, volume, ...]
            klines = data["data"]  # 最新在前
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            if len(closes) < 2:
                return None

            # 计算价格变化
            price_change = (closes[0] - closes[-1]) / closes[-1]  # 24h变化率
            total_volume = sum(volumes)

            # 估算流量方向
            estimated_flow = total_volume * price_change * closes[0]
            inflow = max(0, estimated_flow) if estimated_flow > 0 else 0
            outflow = max(0, -estimated_flow) if estimated_flow < 0 else 0

            return {
                "flow": round(estimated_flow, 2),
                "inflow": round(inflow, 2),
                "outflow": round(outflow, 2),
                "unit": "USD",
                "timestamp": get_timestamp(),
                "_source": "okx_kline_proxy",
            }
        except Exception as e:
            logger.warning(f"OKX交易所流量API失败: {e}")
            return None

    def _mock_exchange_flow(self, symbol: str) -> Dict[str, Any]:
        """生成模拟交易所流量数据"""
        import random
        inflow = random.uniform(10000000, 50000000)
        outflow = random.uniform(10000000, 50000000)
        flow = inflow - outflow
        return {
            "flow": round(flow, 2),
            "inflow": round(inflow, 2),
            "outflow": round(outflow, 2),
            "unit": "USD",
            "timestamp": get_timestamp()
        }

    def _mock_large_transfers(self, symbol: str, threshold_usd: float) -> List[Dict]:
        """生成模拟大额转账数据"""
        import random
        count = random.randint(1, 5)
        transfers = []
        for i in range(count):
            amount = random.uniform(threshold_usd, threshold_usd * 10)
            transfers.append({
                "id": f"mock_tx_{i}",
                "from": {"address": f"0x...mock{i}a", "type": "exchange"},
                "to": {"address": f"0x...mock{i}b", "type": "wallet"},
                "amount_usd": round(amount, 2),
                "timestamp": int(time.time()) - i * 3600,
                "blockchain": "bitcoin" if symbol == "BTC" else "ethereum"
            })
        return transfers


# ============================================================
# 钱包分布监控
# ============================================================

class WalletIntel:
    """
    钱包分布监控模块
    数据源：Glassnode持币分布API / CoinGecko市值数据
    """

    def __init__(self):
        self.glassnode_base = API_CONFIG["glassnode"]["base_url"]
        self.glassnode_key = API_CONFIG["glassnode"]["api_key"]

    def get_holder_concentration(self, symbol: str) -> Dict[str, Any]:
        """
        获取持币地址集中度

        Returns:
            {
                "top10_pct": float,     # Top10地址持有占比
                "top100_pct": float,    # Top100地址持有占比
                "change_24h": float,    # 24小时集中度变化
                "timestamp": str        # 数据时间戳
            }
        """
        cached = load_cache(symbol, "holder_concentration")
        if cached and (time.time() - cached.timestamp) < 3600:  # 1小时缓存
            logger.info(f"使用缓存持币集中度数据: {symbol}")
            return cached.data

        if not self.glassnode_key:
            logger.warning("Glassnode API Key未配置，使用模拟数据")
            return self._mock_holder_concentration(symbol)

        # Glassnode API: 持币分布
        url = f"{self.glassnode_base}/metrics/distribution/balance_supplypercent"

        headers = {"Authorization": f"Bearer {self.glassnode_key}"}
        params = {
            "asset": symbol,
            "interval": "24h",
            "time": "now"
        }

        result = api_request(url, params=params, headers=headers)
        if not result or "data" not in result:
            logger.warning(f"获取持币集中度失败，使用缓存: {symbol}")
            return cached.data if cached else {"top10_pct": 0, "top100_pct": 0, "change_24h": 0, "timestamp": get_timestamp()}

        data_points = result["data"]
        if len(data_points) < 2:
            return {"top10_pct": 0, "top100_pct": 0, "change_24h": 0, "timestamp": get_timestamp()}

        latest = data_points[-1]
        prev = data_points[-2]

        concentration_data = {
            "top10_pct": round(latest.get("v", {}).get("10", 0), 2),
            "top100_pct": round(latest.get("v", {}).get("100", 0), 2),
            "change_24h": round(latest.get("v", {}).get("10", 0) - prev.get("v", {}).get("10", 0), 2),
            "timestamp": latest.get("t", get_timestamp())
        }

        save_cache(symbol, "holder_concentration", concentration_data)
        return concentration_data

    def calc_concentration_signal(self, symbol: str) -> float:
        """
        计算集中度信号

        Returns:
            float: -1(巨鲸抛售) ~ +1(巨鲸吸筹)
        """
        concentration = self.get_holder_concentration(symbol)

        top10_pct = concentration.get("top10_pct", 0)
        change_24h = concentration.get("change_24h", 0)

        # 信号逻辑：
        # - 如果top10占比增加(change_24h > 0)，说明巨鲸在吸筹，信号正向
        # - 如果top10占比减少(change_24h < 0)，说明巨鲸在抛售，信号负向
        # - 信号强度与变化幅度成正比

        if abs(change_24h) < 0.1:
            # 变化很小，中性信号
            signal = 0.0
        else:
            # 变化幅度归一化：假设1%的集中度变化是显著的
            signal = max(-1.0, min(1.0, change_24h / 1.0))

        logger.info(f"{symbol}集中度信号: {signal:.3f} (Top10:{top10_pct}%, 24h变化:{change_24h}%)")
        return round(signal, 3)

    def _mock_holder_concentration(self, symbol: str) -> Dict[str, Any]:
        """生成模拟持币集中度数据"""
        import random
        top10_pct = random.uniform(40, 60)
        change_24h = random.uniform(-2, 2)
        return {
            "top10_pct": round(top10_pct, 2),
            "top100_pct": round(top10_pct * 1.3, 2),
            "change_24h": round(change_24h, 2),
            "timestamp": get_timestamp()
        }


# ============================================================
# Agent-M 主类
# ============================================================

class MarketIntelAgent:
    """
    市场情报Agent主类
    协调NewsIntel、OnChainIntel、WalletIntel模块
    生成综合情报报告
    """

    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol.upper()

        # 初始化子模块
        self.news_intel = NewsIntel()
        self.onchain_intel = OnChainIntel()
        self.wallet_intel = WalletIntel()

        # 权重配置
        self.weights = {
            "news_sentiment": 0.35,      # 新闻情感权重
            "exchange_flow": 0.35,       # 交易所流量权重
            "concentration": 0.30,       # 集中度权重
        }

    def generate_intel_report(self) -> Dict:
        """
        生成综合情报报告

        Returns:
            IntelReport格式的字典，包含：
            - symbol: 交易对
            - timestamp: UTC时间戳
            - news_sentiment: 新闻情感分析结果
            - onchain: 链上数据
            - wallet: 钱包分布数据
            - combined_score: 加权综合评分 [-1, 1]
            - recommendation: 推荐方向（看多/看空/观望）
            - confidence: 置信度 [0, 1]
        """
        logger.info(f"=== 开始生成{self.symbol}市场情报报告 ===")

        # 1. 新闻情感分析
        news_result = self._safe_get_news_sentiment()
        logger.info(f"新闻情感: score={news_result.get('score', 0)}, labels={news_result.get('labels', [])}")

        # 2. 链上数据
        exchange_flow_signal = self._safe_get_exchange_flow_signal()
        large_transfers = self._safe_get_large_transfers()
        logger.info(f"交易所流量信号: {exchange_flow_signal}, 大额转账数: {len(large_transfers)}")

        # 3. 钱包分布
        concentration_signal = self._safe_get_concentration_signal()
        holder_data = self._safe_get_holder_concentration()
        logger.info(f"集中度信号: {concentration_signal}, Top10: {holder_data.get('top10_pct', 0)}%")

        # 4. 计算综合评分
        combined_score = self._calc_combined_score(
            news_score=news_result.get("score", 0),
            flow_signal=exchange_flow_signal,
            concentration_signal=concentration_signal
        )

        # 5. 生成推荐
        recommendation, confidence = self._generate_recommendation(
            combined_score,
            news_result,
            exchange_flow_signal,
            concentration_signal
        )

        # 6. 构建报告
        report = {
            "symbol": self.symbol,
            "timestamp": get_timestamp(),
            "news_sentiment": {
                "score": news_result.get("score", 0),
                "labels": news_result.get("labels", [])
            },
            "onchain": {
                "exchange_flow_signal": exchange_flow_signal,
                "large_transfer_count": len(large_transfers),
                "large_transfers": large_transfers[:5] if large_transfers else []  # 最多5条
            },
            "wallet": {
                "concentration_signal": concentration_signal,
                "top10_pct": holder_data.get("top10_pct", 0),
                "change_24h": holder_data.get("change_24h", 0)
            },
            "combined_score": round(combined_score, 3),
            "recommendation": recommendation,
            "confidence": round(confidence, 2)
        }

        logger.info(f"=== {self.symbol}情报报告生成完成: 综合评分={combined_score:.3f}, 推荐={recommendation}, 置信度={confidence:.2f} ===")

        return report

    def _safe_get_news_sentiment(self) -> Dict:
        """安全获取新闻情感（失败时返回默认值）"""
        try:
            return self.news_intel.get_sentiment_with_fresh_news(self.symbol)
        except Exception as e:
            logger.error(f"新闻情感分析失败: {e}")
            return {"score": 0, "labels": ["利好:0%", "中性:100%", "利空:0%"], "count": 0}

    def _safe_get_exchange_flow_signal(self) -> float:
        """安全获取交易所流量信号"""
        try:
            return self.onchain_intel.calc_net_flow_signal(self.symbol)
        except Exception as e:
            logger.error(f"交易所流量信号获取失败: {e}")
            return 0.0

    def _safe_get_large_transfers(self) -> List:
        """安全获取大额转账"""
        try:
            return self.onchain_intel.get_large_transfers(self.symbol)
        except Exception as e:
            logger.error(f"大额转账获取失败: {e}")
            return []

    def _safe_get_concentration_signal(self) -> float:
        """安全获取集中度信号"""
        try:
            return self.wallet_intel.calc_concentration_signal(self.symbol)
        except Exception as e:
            logger.error(f"集中度信号获取失败: {e}")
            return 0.0

    def _safe_get_holder_concentration(self) -> Dict:
        """安全获取持币集中度"""
        try:
            return self.wallet_intel.get_holder_concentration(self.symbol)
        except Exception as e:
            logger.error(f"持币集中度获取失败: {e}")
            return {"top10_pct": 0, "top100_pct": 0, "change_24h": 0}

    def _calc_combined_score(self, news_score: float,
                             flow_signal: float,
                             concentration_signal: float) -> float:
        """
        计算加权综合评分

        综合评分 = w1*新闻情感 + w2*流量信号 + w3*集中度信号
        """
        combined = (
            self.weights["news_sentiment"] * news_score +
            self.weights["exchange_flow"] * flow_signal +
            self.weights["concentration"] * concentration_signal
        )
        return max(-1.0, min(1.0, combined))

    def _generate_recommendation(self, combined_score: float,
                                  news_result: Dict,
                                  flow_signal: float,
                                  concentration_signal: float) -> tuple:
        """
        生成推荐方向和置信度

        Returns:
            (recommendation: str, confidence: float)
        """
        # 信号一致性分析：三个信号方向是否一致
        signals = [news_result.get("score", 0), flow_signal, concentration_signal]
        positive_count = sum(1 for s in signals if s > 0.2)
        negative_count = sum(1 for s in signals if s < -0.2)

        # 推荐逻辑
        if combined_score > 0.3:
            if positive_count >= 2:
                recommendation = "看多"
                confidence = 0.6 + (combined_score - 0.3) * 0.5
            else:
                recommendation = "观望"
                confidence = 0.4
        elif combined_score < -0.3:
            if negative_count >= 2:
                recommendation = "看空"
                confidence = 0.6 + abs(combined_score) - 0.3 * 0.5
            else:
                recommendation = "观望"
                confidence = 0.4
        else:
            # 混合信号或中性区域
            if positive_count > negative_count:
                recommendation = "谨慎看多"
                confidence = 0.35 + positive_count * 0.05
            elif negative_count > positive_count:
                recommendation = "谨慎看空"
                confidence = 0.35 + negative_count * 0.05
            else:
                recommendation = "观望"
                confidence = 0.5

        # 限制置信度范围
        confidence = max(0.3, min(0.95, confidence))

        return recommendation, confidence

    def get_brief_report(self) -> str:
        """获取简洁的文字报告（用于快速展示）"""
        report = self.generate_intel_report()

        lines = [
            f"📊 {report['symbol']} 市场情报报告",
            f"⏰ {report['timestamp']}",
            "",
            f"📰 新闻情感: {report['news_sentiment']['score']:+.2f}",
            f"   {', '.join(report['news_sentiment']['labels'])}",
            "",
            "🔗 链上数据:",
            f"   交易所流量信号: {report['onchain']['exchange_flow_signal']:+.2f}",
            f"   大额转账数: {report['onchain']['large_transfer_count']}",
            "",
            "💼 钱包分布:",
            f"   集中度信号: {report['wallet']['concentration_signal']:+.2f}",
            f"   Top10占比: {report['wallet']['top10_pct']:.1f}%",
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📈 综合评分: {report['combined_score']:+.2f}",
            f"🎯 推荐: {report['recommendation']}",
            f"🔒 置信度: {report['confidence']:.0%}",
        ]

        return "\n".join(lines)


# ============================================================
# 入口函数（支持直接运行）
# ============================================================

def main(symbol: str = "BTC"):
    """主入口：生成并打印情报报告"""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )

    agent = MarketIntelAgent(symbol)
    report = agent.generate_intel_report()

    print("\n" + "=" * 50)
    print(f"  {symbol} 市场情报报告")
    print("=" * 50)
    print(f"\n{agent.get_brief_report()}")
    print("=" * 50)

    return report


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    main(symbol)
