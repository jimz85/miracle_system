from __future__ import annotations

"""
Miracle 1.0.1 - 统一数据源模块
============================================
支持多个交易所的免费公开API，自动降级

数据源优先级:
1. OKX (https://www.okx.com) - 国内可用
2. 币安 (https://binance.com) - 需要代理
3. yfinance (Yahoo Finance) - 备用
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("miracle.data_fetcher")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ============================================================
# 数据源配置
# ============================================================

OKX_BASE = "https://www.okx.com/api/v5"
BINANCE_BASE = "https://api.binance.com/api/v3"

# 缓存（5秒TTL）
_cache: Dict[str, tuple] = {}


def _get_cached(key: str, ttl: float = 5):
    """从缓存获取数据"""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < ttl:
            return data
    return None


def _set_cached(key: str, data: Any):
    """写入缓存"""
    _cache[key] = (data, time.time())


# ============================================================
# OKX 数据源
# ============================================================

class OKXDataSource:
    """OKX交易所数据源（国内可用）"""

    @staticmethod
    def get_ticker(symbol: str) -> Dict | None:
        """
        获取实时行情

        Args:
            symbol: 如 "BTC-USDT", "ETH-USDT"

        Returns:
            {
                "symbol": str,
                "last": float,       # 最新价
                "bid": float,        # 买一价
                "ask": float,        # 卖一价
                "high_24h": float,
                "low_24h": float,
                "volume_24h": float,
                "change_pct": float,  # 24h涨跌%
                "timestamp": int
            }
        """
        cache_key = f"okx_ticker_{symbol}"
        cached = _get_cached(cache_key, ttl=3)
        if cached:
            return cached

        try:
            url = f"{OKX_BASE}/market/ticker?instId={symbol}"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if data.get("code") != "0":
                return None

            t = data["data"][0]
            result = {
                "symbol": symbol,
                "last": float(t.get("last", 0)),
                "bid": float(t.get("bidPx", 0)),
                "ask": float(t.get("askPx", 0)),
                "high_24h": float(t.get("high24h", 0)),
                "low_24h": float(t.get("low24h", 0)),
                "volume_24h": float(t.get("vol24h", 0)),
                "change_pct": float(t.get("sodUtc8", 0)),  # 开盘至今涨跌
                "timestamp": int(t.get("ts", 0)),
                "source": "okx"
            }
            _set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"OKX ticker失败: {e}")
            return None

    @staticmethod
    def get_klines(symbol: str, timeframe: str = "1H", limit: int = 100) -> List[Dict] | None:
        """
        获取K线数据

        Args:
            symbol: 如 "BTC-USDT"
            timeframe: "1m"/"5m"/"1H"/"4H"/"1D"
            limit: K线数量 (max 100)

        Returns:
            List[{
                "timestamp": int,
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float
            }]
        """
        cache_key = f"okx_kline_{symbol}_{timeframe}_{limit}"
        cached = _get_cached(cache_key, ttl=5)
        if cached:
            return cached

        try:
            bar_map = {"1m": "1m", "5m": "5m", "1H": "1H", "4H": "4H", "1D": "1D"}
            bar = bar_map.get(timeframe, "1H")

            url = f"{OKX_BASE}/market/candles?instId={symbol}&bar={bar}&limit={limit}"
            resp = requests.get(url, timeout=5)
            data = resp.json()

            if data.get("code") != "0":
                return None

            # OKX返回格式: [ts, open, high, low, close, volume, ...]
            klines = []
            for item in reversed(data["data"]):  # 从旧到新
                klines.append({
                    "timestamp": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5])
                })

            _set_cached(cache_key, klines)
            return klines
        except Exception as e:
            logger.warning(f"OKX klines失败: {e}")
            return None

    @staticmethod
    def get_orderbook(symbol: str, depth: int = 20) -> Dict | None:
        """获取订单簿（深度）"""
        cache_key = f"okx_ob_{symbol}_{depth}"
        cached = _get_cached(cache_key, ttl=2)
        if cached:
            return cached

        try:
            url = f"{OKX_BASE}/market/books-l2?instId={symbol}&sz={depth}"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if data.get("code") != "0":
                return None

            bids = [[float(p), float(v)] for p, v in data["data"][0]["bids"][:depth]]
            asks = [[float(p), float(v)] for p, v in data["data"][0]["asks"][:depth]]

            result = {"bids": bids, "asks": asks, "source": "okx"}
            _set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"OKX orderbook失败: {e}")
            return None


# ============================================================
# yfinance 备用数据源
# ============================================================

class YFinanceDataSource:
    """Yahoo Finance数据源（备用）"""

    @staticmethod
    def get_klines(symbol: str, period: str = "30d",
                   interval: str = "1h") -> List[Dict] | None:
        """
        获取K线数据

        Args:
            symbol: 如 "BTC-USD" (注意是-不是/)
            period: "7d", "30d", "90d", "1y"
            interval: "1m", "5m", "1h", "1d"
        """
        cache_key = f"yf_kline_{symbol}_{period}_{interval}"
        cached = _get_cached(cache_key, ttl=60)
        if cached:
            return cached

        if not HAS_YF:
            logger.warning("yfinance未安装")
            return None

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            if hist.empty:
                return None

            klines = []
            for ts, row in hist.iterrows():
                klines.append({
                    "timestamp": int(ts.timestamp() * 1000),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"])
                })

            _set_cached(cache_key, klines)
            return klines
        except Exception as e:
            logger.warning(f"yfinance klines失败: {e}")
            return None

    @staticmethod
    def get_ticker(symbol: str) -> Dict | None:
        """获取实时行情"""
        cache_key = f"yf_ticker_{symbol}"
        cached = _get_cached(cache_key, ttl=5)
        if cached:
            return cached

        if not HAS_YF:
            return None

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            result = {
                "symbol": symbol,
                "last": float(info.last_price) if info.last_price else 0,
                "high_24h": float(info.regular_day_high) if hasattr(info, 'regular_day_high') else 0,
                "low_24h": float(info.regular_day_low) if hasattr(info, 'regular_day_low') else 0,
                "volume_24h": float(info.last_volume) if info.last_volume else 0,
                "source": "yfinance"
            }
            _set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"yfinance ticker失败: {e}")
            return None


# ============================================================
# 统一数据获取接口
# ============================================================

class DataFetcher:
    """
    统一数据获取器
    自动选择可用数据源，按优先级:
    1. OKX (价格/K线/订单簿)
    2. yfinance (备用)
    """

    SYMBOL_MAP = {
        "BTC": {"okx": "BTC-USDT", "yf": "BTC-USD"},
        "ETH": {"okx": "ETH-USDT", "yf": "ETH-USD"},
        "SOL": {"okx": "SOL-USDT", "yf": "SOL-USD"},
        "DOGE": {"okx": "DOGE-USDT", "yf": "DOGE-USD"},
        "AVAX": {"okx": "AVAX-USDT", "yf": "AVAX-USD"},
        "DOT": {"okx": "DOT-USDT", "yf": "DOT-USD"},
        "LINK": {"okx": "LINK-USDT", "yf": "LINK-USD"},
        "ADA": {"okx": "ADA-USDT", "yf": "ADA-USD"},
    }

    def __init__(self, prefer_source: str = "okx"):
        self.prefer_source = prefer_source

    def get_ticker(self, symbol: str) -> Dict | None:
        """获取实时行情"""
        okx_sym = self.SYMBOL_MAP.get(symbol, {}).get("okx", f"{symbol}-USDT")

        # 优先OKX
        if self.prefer_source == "okx":
            ticker = OKXDataSource.get_ticker(okx_sym)
            if ticker:
                return ticker
            # 降级到yfinance
            yf_sym = self.SYMBOL_MAP.get(symbol, {}).get("yf", f"{symbol}-USD")
            return YFinanceDataSource.get_ticker(yf_sym)

        # 优先yfinance
        yf_sym = self.SYMBOL_MAP.get(symbol, {}).get("yf", f"{symbol}-USD")
        ticker = YFinanceDataSource.get_ticker(yf_sym)
        if ticker:
            return ticker
        # 降级到OKX
        return OKXDataSource.get_ticker(okx_sym)

    def get_klines(self, symbol: str, timeframe: str = "1H",
                  limit: int = 100) -> List[Dict] | None:
        """获取K线数据"""
        okx_sym = self.SYMBOL_MAP.get(symbol, {}).get("okx", f"{symbol}-USDT")

        # 优先OKX
        if self.prefer_source == "okx":
            klines = OKXDataSource.get_klines(okx_sym, timeframe, limit)
            if klines:
                return klines
            # 降级到yfinance
            period_map = {"1m": "7d", "5m": "30d", "1H": "30d", "4H": "90d", "1D": "1y"}
            period = period_map.get(timeframe, "30d")
            yf_sym = self.SYMBOL_MAP.get(symbol, {}).get("yf", f"{symbol}-USD")
            return YFinanceDataSource.get_klines(yf_sym, period, timeframe)

        # 优先yfinance
        period_map = {"1m": "7d", "5m": "30d", "1H": "30d", "4H": "90d", "1D": "1y"}
        period = period_map.get(timeframe, "30d")
        yf_sym = self.SYMBOL_MAP.get(symbol, {}).get("yf", f"{symbol}-USD")
        klines = YFinanceDataSource.get_klines(yf_sym, period, timeframe)
        if klines:
            return klines
        # 降级到OKX
        return OKXDataSource.get_klines(okx_sym, timeframe, limit)

    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict | None:
        """获取订单簿（仅OKX支持）"""
        okx_sym = self.SYMBOL_MAP.get(symbol, {}).get("okx", f"{symbol}-USDT")
        return OKXDataSource.get_orderbook(okx_sym, depth)


# ============================================================
# 便捷函数
# ============================================================

_fetcher = DataFetcher()


def get_ticker(symbol: str) -> Dict | None:
    """获取实时行情"""
    return _fetcher.get_ticker(symbol)


def get_klines(symbol: str, timeframe: str = "1H", limit: int = 100) -> List[Dict] | None:
    """获取K线数据"""
    return _fetcher.get_klines(symbol, timeframe, limit)


def get_orderbook(symbol: str, depth: int = 20) -> Dict | None:
    """获取订单簿"""
    return _fetcher.get_orderbook(symbol, depth)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=== DataFetcher 测试 ===\n")

    fetcher = DataFetcher()

    # 测试行情
    ticker = fetcher.get_ticker("BTC")
    if ticker:
        print(f"BTC行情: ${ticker['last']:,.2f} (来源:{ticker['source']})")
    else:
        print("BTC行情: ❌ 获取失败")

    # 测试K线
    klines = fetcher.get_klines("BTC", "1H", 20)
    if klines:
        print(f"BTC K线: ✅ 获取{len(klines)}条")
        print(f"  最新: {klines[-1]['close']}")
    else:
        print("BTC K线: ❌ 获取失败")

    # 测试订单簿
    ob = fetcher.get_orderbook("BTC", 5)
    if ob:
        print("BTC订单簿: ✅ 买卖各5档")
        print(f"  买一: {ob['bids'][0][0]}, 卖一: {ob['asks'][0][0]}")
    else:
        print("BTC订单簿: ❌ 获取失败")


# ============================================================
# 补充: 免费区块链数据源 (2026-04-21)
# ============================================================

class BlockstreamDataSource:
    """
    Blockstream API - 比特币链上数据（免费，无需API Key）
    文档: https://blockstream.info/api/
    """

    @staticmethod
    def get_address_stats(address: str) -> Dict | None:
        """获取BTC地址统计"""
        try:
            url = f"https://blockstream.info/api/address/{address}"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json()
                stats = data.get("chain_stats", {})
                return {
                    "funded_txo_sum": stats.get("funded_txo_sum", 0) / 1e8,  # BTC
                    "spent_txo_sum": stats.get("spent_txo_sum", 0) / 1e8,
                    "tx_count": stats.get("tx_count", 0),
                    "address": address
                }
        except Exception as e:
            logger.warning(f"Blockstream address查询失败: {e}")
        return None

    @staticmethod
    def get_latest_block_height() -> int | None:
        """获取最新区块高度"""
        try:
            r = requests.get("https://blockstream.info/api/blocks/tip/height", timeout=5)
            if r.status_code == 200:
                return int(r.text.strip())
        except Exception as e:
            logger.warning(f"Blockstream区块高度查询失败: {e}")
        return None

    @staticmethod
    def get_mempool_stats() -> Dict | None:
        """获取内存池统计"""
        try:
            r = requests.get("https://blockstream.info/api/mempool", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return {
                    "count": data.get("count", 0),
                    "vsize": data.get("vsize", 0),
                    "total_fee": data.get("total_fee", 0)
                }
        except Exception as e:
            logger.warning(f"Blockstream mempool查询失败: {e}")
        return None


class DeFiLlamaDataSource:
    """
    DeFi Llama API - DeFi TVL数据（免费，无需API Key）
    文档: https://defillama.com/docs/api
    """

    @staticmethod
    def get_protocol_tvl(protocol: str) -> float | None:
        """获取协议TVL（美元）"""
        try:
            url = f"https://api.llama.fi/protocol/{protocol}"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json()
                return data.get("tvl", 0)
        except Exception as e:
            logger.warning(f"DeFiLlama {protocol}查询失败: {e}")
        return None

    @staticmethod
    def get_all_protocols() -> List[Dict]:
        """获取所有协议TVL列表"""
        try:
            r = requests.get("https://api.llama.fi/protocols", timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.warning(f"DeFiLlama protocols查询失败: {e}")
        return []


# ============================================================
# 便捷函数
# ============================================================

def get_btc_mempool() -> Dict | None:
    """获取BTC内存池状态"""
    return BlockstreamDataSource.get_mempool_stats()


def get_btc_block_height() -> int | None:
    """获取BTC最新区块高度"""
    return BlockstreamDataSource.get_latest_block_height()


def get_defi_tvl(protocol: str = None) -> Dict | None:
    """获取DeFi TVL数据"""
    if protocol:
        tvl = DeFiLlamaDataSource.get_protocol_tvl(protocol)
        return {"protocol": protocol, "tvl": tvl}
    else:
        protocols = DeFiLlamaDataSource.get_all_protocols()
        # 返回前10大TVL协议
        top10 = sorted(protocols, key=lambda x: x.get("tvl", 0) or 0, reverse=True)[:10]
        return {"top_protocols": top10}
