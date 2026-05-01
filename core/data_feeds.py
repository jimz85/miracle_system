#!/usr/bin/env python3
"""
data_feeds.py — STUB因子真实数据接入
======================================
替换 miracle_core.py 中的3个STUB函数，对接真实数据源。

数据源:
  1. calc_news_sentiment: CoinDesk/CoinTelegraph RSS + KeywordSentimentAnalyzer
  2. calc_onchain_metrics: market_sentiment.json (Kronos定时写入)
  3. calc_wallet_metrics: 暂用持币集中度代理（待完善）

用法:
    from core.data_feeds import calc_news_sentiment, calc_onchain_metrics, calc_wallet_metrics
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("data_feeds")

# ============================================================
# 常量
# ============================================================

# 新闻源 (来自 kronos/gemma4_hourly_review.py 已验证可用)
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
}

# 币种映射: 交易所符号 → 通用名称（用于新闻关键词搜索）
COIN_ALIASES = {
    "BTC": ["bitcoin", "btc", "比特币"],
    "ETH": ["ethereum", "eth", "以太坊"],
    "SOL": ["solana", "sol", "索拉纳"],
    "DOGE": ["dogecoin", "doge", "狗狗币"],
    "ADA": ["cardano", "ada", "卡尔达诺"],
    "XRP": ["xrp", "ripple", "瑞波"],
    "BNB": ["bnb", "binance coin", "币安币"],
    "AVAX": ["avalanche", "avax", "雪崩"],
    "DOT": ["polkadot", "dot", "波卡"],
    "LINK": ["chainlink", "link", "预言机"],
}

# 持币数据缓存时间（秒）
COINGECKO_CACHE_TTL = 3600  # 1小时

# ============================================================
# 新闻情绪因子 (替换 calc_news_sentiment)
# ============================================================

def _fetch_rss_news(coin: str, max_items: int = 5, timeout: int = 10) -> List[Dict]:
    """
    从RSS源获取币种相关新闻。
    使用CoinDesk/CoinTelegraph RSS，通过关键词过滤。
    
    Returns:
        [{title, body, source}, ...] 或 []
    """
    aliases = COIN_ALIASES.get(coin, [coin.lower()])
    keywords = [a.lower() for a in aliases]
    news = []
    seen_titles = set()

    for source_name, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (compatible; MiracleSystem/1.0)"
            })
            if resp.status_code != 200:
                continue

            # 简单XML解析（RSS格式）
            content = resp.text
            # 提取 <item> 块
            items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)
            for item in items:
                title_match = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
                desc_match = re.search(r"<description>(.*?)</description>", item, re.DOTALL)
                if not title_match:
                    continue

                title = re.sub(r"<.*?>", "", title_match.group(1)).strip()
                body = re.sub(r"<.*?>", "", desc_match.group(1)).strip() if desc_match else ""
                text_lower = (title + " " + body).lower()

                # 关键词过滤：是否提到该币种
                if not any(kw in text_lower for kw in keywords):
                    continue
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                news.append({
                    "title": title,
                    "body": body,
                    "source": source_name,
                })
                if len(news) >= max_items:
                    break
        except Exception as e:
            logger.warning("RSS获取失败 [%s]: %s", source_name, e)

    return news


# ============================================================
# CryptoPanic 情绪API (可选: 比RSS关键词分析更可靠)
# ============================================================

CRYPTOPANIC_API = "https://cryptopanic.com/api/v1/posts/"
CRYPTOPANIC_TOKEN = os.environ.get("CRYPTOPANIC_TOKEN", "")

def _fetch_cryptopanic_sentiment(coin: str) -> Optional[float]:
    """
    从CryptoPanic获取bullish/bearish投票情绪。
    
    比RSS关键词分析更直接: 统计社区bullish/bearish投票比例。
    
    Returns:
        -1.0(极端悲观) ~ 1.0(极端乐观), None(API不可用)
    """
    if not CRYPTOPANIC_TOKEN:
        return None
    
    try:
        params = {
            "auth_token": CRYPTOPANIC_TOKEN,
            "currencies": coin.upper(),
            "limit": 20,
        }
        resp = requests.get(CRYPTOPANIC_API, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        posts = data.get("results", [])
        if not posts:
            return None
        
        bull = sum(1 for p in posts if p.get("votes", {}).get("positive", 0) > 0)
        bear = sum(1 for p in posts if p.get("votes", {}).get("negative", 0) > 0)
        total = bull + bear
        
        if total == 0:
            return 0.0
        
        score = (bull - bear) / total
        logger.info(
            "%s CryptoPanic情绪: bull=%d bear=%d score=%.2f",
            coin, bull, bear, score
        )
        return round(max(-1.0, min(1.0, score)), 4)
    
    except Exception as e:
        logger.warning("CryptoPanic API失败 [%s]: %s", coin, e)
        return None


def calc_news_sentiment(coin: str = "BTC") -> float:
    """
    计算指定币种的新闻情绪 (替换STUB)
    
    流程:
      1. 优先尝试 CryptoPanic API (投票制情绪, 更直接)
      2. fallback: RSS抓取 + KeywordSentimentAnalyzer
      3. 返回 -1.0(强烈利空) ~ 1.0(强烈利好)
    
    降级: 无新闻或失败时返回 0.0 (中性)
    """
    # 优先使用 CryptoPanic (vote-based sentiment, 更可靠)
    cp_score = _fetch_cryptopanic_sentiment(coin)
    if cp_score is not None:
        return cp_score
    
    # fallback: RSS关键词分析
    try:
        news_items = _fetch_rss_news(coin, max_items=5)
        if not news_items:
            logger.info("%s: 无相关新闻，返回中性", coin)
            return 0.0

        # 使用已有的 KeywordSentimentAnalyzer
        from core.market_intel_sentiment import KeywordSentimentAnalyzer
        analyzer = KeywordSentimentAnalyzer()
        result = analyzer.analyze(news_items)
        score = result.get("score", 0.0)
        label = result.get("label", "中性")
        confidence = result.get("confidence", 0.5)

        logger.info(
            "%s 新闻情绪: score=%.2f label=%s confidence=%.2f (%d篇)",
            coin, score, label, confidence, len(news_items)
        )
        return round(max(-1.0, min(1.0, score)), 4)

    except Exception as e:
        logger.error("%s calc_news_sentiment 异常: %s", coin, e)
        return 0.0


# ============================================================
# 链上因子 (替换 calc_onchain_metrics)
# ============================================================

def _read_market_sentiment() -> Optional[Dict]:
    """读取Kronos定时写入的market_sentiment.json"""
    paths = [
        Path.home() / ".hermes" / "cron" / "output" / "market_sentiment.json",
        Path.home() / ".hermes" / "market_sentiment.json",
    ]
    for path in paths:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("读取 %s 失败: %s", path, e)
    return None


def calc_onchain_metrics(coin: str = "BTC") -> Dict[str, float]:
    """
    计算链上因子 (替换STUB)
    
    数据源: market_sentiment.json (Kronos写入)
      - exchange_flow: CEX净流入/流出百分比
      - large_transfer: Fear & Greed 指数归一化
    
    降级: 无数据时返回中性值
    """
    try:
        data = _read_market_sentiment()
        if not data:
            logger.info("%s: market_sentiment.json 不可用，返回中性值", coin)
            return {"exchange_flow": 0.0, "large_transfer": 0.0}

        sentiment_data = data.get("data", {})

        # 1. 交易所净流量: 取所有CEX的7天变化均值
        cex_flows = sentiment_data.get("cex_flows", {})
        flow_changes = []
        for _, exchange in cex_flows.items():
            change = exchange.get("change_7d", 0)
            if isinstance(change, (int, float)) and change != 0:
                flow_changes.append(change)

        if flow_changes:
            # 负变化 = 资金流出交易所 = 偏利好(囤积) / 正变化 = 流入 = 偏利空(派发)
            # 归一化到 -1.0 ~ 1.0
            avg_flow = sum(flow_changes) / len(flow_changes)
            # 映射: 均值通常 -5%~5%，饱和截断
            exchange_flow = max(-1.0, min(1.0, avg_flow / 5.0))
        else:
            exchange_flow = 0.0

        # 2. Fear & Greed 指数 (26 = 恐惧 → 可能过度悲观，反弹机会)
        fear_greed = sentiment_data.get("fear_greed", {})
        fg_value = fear_greed.get("value", 50)
        # Fear→偏多(均值回归), Greed→偏空(过热)
        large_transfer = (50 - fg_value) / 50.0  # 0~100映射到-1~1

        result = {
            "exchange_flow": round(exchange_flow, 4),
            "large_transfer": round(large_transfer, 4),
        }
        logger.info(
            "%s 链上因子: exchange_flow=%.2f large_transfer=%.2f",
            coin, result["exchange_flow"], result["large_transfer"]
        )
        return result

    except Exception as e:
        logger.error("%s calc_onchain_metrics 异常: %s", coin, e)
        return {"exchange_flow": 0.0, "large_transfer": 0.0}


# ============================================================
# 钱包因子 (替换 calc_wallet_metrics)
# ============================================================

_COINGECKO_COIN_ID = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "DOGE": "dogecoin", "ADA": "cardano", "XRP": "ripple",
    "BNB": "binancecoin", "AVAX": "avalanche-2",
    "DOT": "polkadot", "LINK": "chainlink",
}

_coingecko_cache: Dict[str, tuple[float, float]] = {}  # coin -> (holder_concentration, cached_at)


def _fetch_holder_data(coin: str) -> Optional[float]:
    """
    估算持币集中度。
    
    方法: 结合 CoinGecko 未流通比例 + 市值排名推算
      - 未流通比高(团队/VC持有多) → 集中
      - 市值排名高(top 10) → 机构化，相对分散
      - 市值排名低(小币种) → 巨鲸主导，可能高度集中
    
    Returns: holder_concentration (0~1)
      0.0 = 完全分散(健康), 0.5 = 中性, 1.0 = 高度集中(高风险)
    """
    coin_id = _COINGECKO_COIN_ID.get(coin)
    if not coin_id:
        return None

    # 缓存检查
    now = time.time()
    if coin in _coingecko_cache:
        cached_val, cached_at = _coingecko_cache[coin]
        if now - cached_at < COINGECKO_CACHE_TTL:
            return cached_val

    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
        }
        resp = requests.get(url, params=params, timeout=10,
                            headers={"User-Agent": "MiracleSystem/1.0"})
        if resp.status_code != 200:
            logger.warning("CoinGecko %s: HTTP %d", coin, resp.status_code)
            return None

        data = resp.json()
        md = data.get("market_data", {})

        # 1. 未流通比例: total_supply / circ_supply
        total_supply = md.get("total_supply")
        circ_supply = md.get("circulating_supply")
        uncirculated_ratio = 0.0

        if total_supply and circ_supply and circ_supply > 0:
            uncirculated_ratio = abs(total_supply - circ_supply) / total_supply

        # 2. 市值排名: 排名高 = 机构化 = 相对分散
        market_cap_rank = data.get("market_cap_rank", 100)
        # 排名1→0.1(低集中), 排名100→0.5(中性), 排名>200→0.8(高集中)
        rank_factor = min(0.8, max(0.1, market_cap_rank / 200))

        # 综合: 未流通比贡献30% + 排名贡献70%
        concentration = uncirculated_ratio * 0.3 + rank_factor * 0.7
        concentration = max(0.05, min(0.95, concentration))

        # 缓存
        _coingecko_cache[coin] = (concentration, now)
        logger.info(
            "%s 持币集中度: %.2f (未流通%.1f%%, 排名#%d, 因子%.2f)",
            coin, concentration, uncirculated_ratio * 100,
            market_cap_rank, rank_factor
        )
        return round(concentration, 4)

    except Exception as e:
        logger.warning("%s CoinGecko请求失败: %s", coin, e)
        return None


def calc_wallet_metrics(coin: str = "BTC") -> Dict[str, float]:
    """
    计算钱包分布因子 (替换STUB)
    
    数据源: CoinGecko免费API
      - holder_concentration: 持币集中度 (0~1)
        * 0.0 = 完全分散(健康/去中心化)
        * 0.5 = 中性
        * 1.0 = 高度集中(巨鲸控制，风险高)
    
    降级: API失败时返回中性值 0.5
    """
    try:
        concentration = _fetch_holder_data(coin)
        if concentration is None:
            logger.info("%s: 钱包数据不可用，返回中性值 0.5", coin)
            return {"holder_concentration": 0.5}

        result = {"holder_concentration": concentration}
        logger.info(
            "%s 钱包因子: holder_concentration=%.2f",
            coin, result["holder_concentration"]
        )
        return result

    except Exception as e:
        logger.error("%s calc_wallet_metrics 异常: %s", coin, e)
        return {"holder_concentration": 0.5}
