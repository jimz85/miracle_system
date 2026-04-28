from __future__ import annotations

"""
免费新闻数据源模块 - Miracle 1.0.1
============================================
使用公开RSS/API，无需API Key

数据源:
1. The Block RSS - 加密货币新闻
2. 价格动量代理 - 当无新闻时使用
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("miracle.news")

# RSS订阅源
RSS_FEEDS = {
    "theblock": {
        "url": "https://theblock.co/rss.xml",
        "name": "The Block",
        "crypto_focus": True
    },
    "coindesk": {
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "name": "CoinDesk",
        "crypto_focus": True
    },
    "cryptoslate": {
        "url": "https://cryptoslate.com/feed/",
        "name": "CryptoSlate",
        "crypto_focus": True
    }
}


def fetch_rss_news(source: str = "theblock", limit: int = 20) -> List[Dict]:
    """
    从RSS获取新闻

    Args:
        source: 来源标识
        limit: 返回数量

    Returns:
        List[{"title": str, "pubDate": str, "link": str, "description": str}]
    """
    feed = RSS_FEEDS.get(source)
    if not feed:
        return []

    try:
        resp = requests.get(feed["url"], timeout=10)
        if resp.status_code != 200:
            logger.warning(f"RSS {source} 返回 {resp.status_code}")
            return []

        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        items = channel.findall("item")[:limit]

        news = []
        for item in items:
            title = item.find("title")
            pubDate = item.find("pubDate")
            link = item.find("link")
            description = item.find("description")

            news.append({
                "title": title.text if title is not None else "",
                "pubDate": pubDate.text if pubDate is not None else "",
                "link": link.text if link is not None else "",
                "description": description.text[:200] if description is not None and description.text else "",
                "source": feed["name"]
            })

        logger.info(f"从 {feed['name']} 获取 {len(news)} 条新闻")
        return news

    except Exception as e:
        logger.warning(f"RSS {source} 获取失败: {e}")
        return []


def analyze_sentiment(news: List[Dict]) -> Dict:
    """
    简单情感分析

    Returns:
        {
            "score": float,      # -1 ~ +1
            "labels": list,
            "count": int,
            "details": list
        }
    """
    if not news:
        return {
            "score": 0.0,
            "labels": ["利好:0%", "中性:100%", "利空:0%"],
            "count": 0,
            "details": []
        }

    bullish_keywords = [
        "涨", "突破", "新高", "暴涨", "飙升", "买入", "看涨", "牛市",
        "bullish", "surge", "rally", "breakout", "high", "soar", "pump", "moon"
    ]
    bearish_keywords = [
        "跌", "破发", "新低", "暴跌", "崩盘", "卖出", "看跌", "熊市",
        "crash", "plunge", "dump", "bearish", "breakdown", "drop", "fall"
    ]

    scores = []
    details = []
    bullish = neutral = bearish = 0

    for item in news:
        text = f"{item.get('title', '')} {item.get('description', '')}".lower()
        score = 0

        for kw in bullish_keywords:
            if kw.lower() in text:
                score += 0.3
        for kw in bearish_keywords:
            if kw.lower() in text:
                score -= 0.3

        score = max(-1.0, min(1.0, score))
        scores.append(score)

        if score > 0.2:
            bullish += 1
        elif score < -0.2:
            bearish += 1
        else:
            neutral += 1

        details.append({
            "title": item.get("title", "")[:60],
            "score": round(score, 2),
            "source": item.get("source", "")
        })

    total = len(scores) or 1
    avg_score = sum(scores) / total

    return {
        "score": round(avg_score, 3),
        "labels": [
            f"利好:{int(bullish/total*100)}%",
            f"中性:{int(neutral/total*100)}%",
            f"利空:{int(bearish/total*100)}%"
        ],
        "count": total,
        "details": details[:10]
    }


def get_news_sentiment(symbol: str = "BTC") -> Dict:
    """
    获取并分析新闻情感

    Args:
        symbol: 交易对符号（目前RSS源是通用加密货币新闻）

    Returns:
        情感分析结果
    """
    # 获取新闻
    news = fetch_rss_news("theblock", limit=20)

    # 按symbol过滤（如果有相关性）
    if symbol and news:
        # 简单过滤：保留所有新闻（RSS源已经是加密货币相关的）
        pass

    # 分析
    return analyze_sentiment(news)


# 测试
if __name__ == "__main__":
    print("=== 免费新闻API测试 ===\n")

    news = fetch_rss_news("theblock", limit=5)
    print(f"获取 {len(news)} 条新闻:\n")
    for n in news:
        print(f"  [{n['source']}] {n['title'][:60]}")

    print("\n--- 情感分析 ---")
    sentiment = analyze_sentiment(news)
    print(f"得分: {sentiment['score']}")
    print(f"标签: {sentiment['labels']}")
