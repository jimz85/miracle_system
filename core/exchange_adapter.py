"""
Exchange Adapter - 交易所适配层
统一OKX/Binance双交易所接口
"""
import os
import time
import json
import hmac
import hashlib
import requests
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """支持的交易所类型"""
    OKX = "okx"
    BINANCE = "binance"


@dataclass
class Ticker:
    """行情数据"""
    symbol: str
    last_price: float
    high_24h: float
    low_24h: float
    volume_24h: float
    change_24h: float  # 百分比
    timestamp: int


@dataclass
class OrderBook:
    """订单簿"""
    symbol: str
    bids: List[tuple]  # [(price, size), ...]
    asks: List[tuple]
    timestamp: int


@dataclass
class Balance:
    """账户余额"""
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


# ========================
# 交易所适配器基类
# ========================

class ExchangeAdapter:
    """交易所适配器基类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        passphrase: Optional[str] = None,  # OKX需要
        testnet: bool = False
    ):
        self.api_key = api_key or os.getenv(f"{self.name.upper()}_API_KEY")
        self.secret = secret or os.getenv(f"{self.name.upper()}_API_SECRET")
        self.passphrase = passphrase or os.getenv(f"{self.name.upper()}_PASSPHRASE")
        self.testnet = testnet
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "MiracleSystem/2.0"})
    
    @property
    def name(self) -> str:
        raise NotImplementedError
    
    # ========== 行情 ==========
    
    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """获取24小时行情"""
        raise NotImplementedError
    
    def get_candles(
        self,
        symbol: str,
        bar: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取K线数据"""
        raise NotImplementedError
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        """获取订单簿"""
        raise NotImplementedError
    
    # ========== 账户 ==========
    
    def get_balance(self) -> List[Balance]:
        """获取账户余额"""
        raise NotImplementedError
    
    def get_positions(self) -> List[Position]:
        """获取当前持仓"""
        raise NotImplementedError
    
    # ========== 交易 ==========
    
    def place_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        order_type: str,  # "market" or "limit"
        size: float,
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """下单"""
        raise NotImplementedError
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消订单"""
        raise NotImplementedError
    
    def get_orders(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取订单列表"""
        raise NotImplementedError
    
    # ========== 工具 ==========
    
    def normalize_symbol(self, symbol: str) -> str:
        """标准化交易对格式"""
        raise NotImplementedError
    
    def format_symbol(self, symbol: str) -> str:
        """格式化交易对 (交易所格式)"""
        raise NotImplementedError


# ========================
# OKX 适配器
# ========================

class OKXAdapter(ExchangeAdapter):
    """
    OKX交易所适配器
    
    文档: https://www.okx.com/docs-vn/
    """
    
    BASE_URL = "https://www.okx.com"
    TESTNET_URL = "https://www.okx.com"
    
    def __init__(self, api_key=None, secret=None, passphrase=None, testnet=False):
        super().__init__(api_key, secret, passphrase, testnet)
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
    
    @property
    def name(self) -> str:
        return "okx"
    
    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """签名"""
        if not self.secret:
            return ""
        message = timestamp + method + path + body
        mac = hmac.new(
            self.secret.encode(),
            message.encode(),
            hashlib.sha256
        )
        return mac.hexdigest()
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        auth: bool = True
    ) -> Optional[Dict]:
        """发送请求"""
        url = self.base_url + path
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if auth and self.api_key:
            timestamp = str(int(time.time()))
            headers["OK-ACCESS-KEY"] = self.api_key
            headers["OK-ACCESS-TIMESTAMP"] = timestamp
            headers["OK-ACCESS-PASSPHRASE"] = self.passphrase or ""
            
            body = json.dumps(data) if data else ""
            sign = self._sign(timestamp, method, path, body)
            headers["OK-ACCESS-SIGN"] = sign
        
        try:
            if method == "GET":
                resp = self._session.get(url, params=params, headers=headers, timeout=10)
            else:
                resp = self._session.request(method, url, json=data, headers=headers, timeout=10)
            
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"OKX request failed: {e}")
            return None
    
    def normalize_symbol(self, symbol: str) -> str:
        """OKX格式: BTC-USDT"""
        return symbol.upper().replace("-", "-").replace("_", "-")
    
    def format_symbol(self, symbol: str) -> str:
        """交易所格式"""
        return self.normalize_symbol(symbol)
    
    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """获取24小时行情"""
        symbol = self.format_symbol(symbol)
        data = self._request(
            "GET",
            "/api/v5/market/ticker",
            params={"instId": symbol}
        )
        
        if data and data.get("data"):
            t = data["data"][0]
            last = float(t.get("last", 0))
            open_24h = float(t.get("open24h", 0))
            change_24h = last - open_24h  # 24小时价格变化(绝对值)
            return Ticker(
                symbol=symbol,
                last_price=last,
                high_24h=float(t.get("high24h", 0)),
                low_24h=float(t.get("low24h", 0)),
                volume_24h=float(t.get("vol24h", 0)),
                change_24h=change_24h,
                timestamp=int(t.get("ts", 0))
            )
        return None
    
    def get_candles(
        self,
        symbol: str,
        bar: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取K线"""
        symbol = self.format_symbol(symbol)
        
        # 映射时间周期
        bar_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1H", "4h": "4H", "1d": "1D"
        }
        bar = bar_map.get(bar, bar)
        
        data = self._request(
            "GET",
            "/api/v5/market/candles",
            params={"instId": symbol, "bar": bar, "limit": limit}
        )
        
        if data and data.get("data"):
            # OKX返回倒序 [时间, 开, 高, 低, 收, 量]
            candles = []
            for c in reversed(data["data"]):
                candles.append({
                    "timestamp": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5])
                })
            return candles
        return []
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        """获取订单簿"""
        symbol = self.format_symbol(symbol)
        data = self._request(
            "GET",
            "/api/v5/market/books-lite",
            params={"instId": symbol, "sz": limit}
        )
        
        if data and data.get("data"):
            books = data["data"][0]
            return OrderBook(
                symbol=symbol,
                bids=[[float(b[0]), float(b[1])] for b in books.get("bids", [])],
                asks=[[float(a[0]), float(a[1])] for a in books.get("asks", [])],
                timestamp=int(books.get("ts", 0))
            )
        return None
    
    def get_balance(self) -> List[Balance]:
        """获取账户余额"""
        data = self._request("GET", "/api/v5/account/balance", auth=True)
        
        if data and data.get("data"):
            balances = []
            for details in data["data"][0].get("details", []):
                if float(details.get("availBal", 0)) > 0 or float(details.get("bal", 0)) > 0:
                    balances.append(Balance(
                        asset=details.get("ccy", ""),
                        free=float(details.get("availBal", 0)),
                        locked=float(details.get("ordFrozen", 0)),
                        total=float(details.get("bal", 0))
                    ))
            return balances
        return []
    
    def get_positions(self) -> List[Position]:
        """获取持仓"""
        data = self._request("GET", "/api/v5/account/positions", auth=True)
        
        if data and data.get("data"):
            positions = []
            for p in data["data"]:
                if float(p.get("pos", 0)) != 0:
                    positions.append(Position(
                        symbol=p.get("instId", ""),
                        side="long" if float(p.get("pos", 0)) > 0 else "short",
                        size=abs(float(p.get("pos", 0))),
                        entry_price=float(p.get("avgPx", 0)),
                        unrealized_pnl=float(p.get("upl", 0)),
                        leverage=float(p.get("lever", 1))
                    ))
            return positions
        return []
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """下单"""
        symbol = self.format_symbol(symbol)
        
        # OKX需要完整instId格式
        if "USDT" not in symbol and "USD" not in symbol:
            symbol = symbol + "-USDT-SWAP"  # 永续合约
        
        td_mode = "cross"  # 全仓
        
        data = {
            "instId": symbol,
            "tdMode": td_mode,
            "side": side,
            "ordType": "market" if order_type == "market" else "limit",
            "sz": str(size)
        }
        
        if price:
            data["px"] = str(price)
        
        result = self._request("POST", "/api/v5/trade/order", data=data, auth=True)
        
        if result and result.get("data"):
            return result["data"][0]
        return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消订单"""
        data = {
            "instId": self.format_symbol(symbol),
            "ordId": order_id
        }
        result = self._request("POST", "/api/v5/trade/cancel-order", data=data, auth=True)
        return result is not None


# ========================
# Binance 适配器
# ========================

class BinanceAdapter(ExchangeAdapter):
    """
    Binance交易所适配器
    
    文档: https://developers.binance.com/
    """
    
    BASE_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"
    
    def __init__(self, api_key=None, secret=None, passphrase=None, testnet=False):
        super().__init__(api_key, secret, passphrase, testnet)
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
    
    @property
    def name(self) -> str:
        return "binance"
    
    def _sign(self, params: Dict) -> str:
        """签名"""
        if not self.secret:
            return ""
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self.secret.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        auth: bool = True
    ) -> Optional[Dict]:
        """发送请求"""
        url = self.base_url + path
        headers = {"X-MBX-APIKEY": self.api_key} if auth and self.api_key else {}
        
        if auth and self.api_key and self.secret:
            params = params or {}
            params["timestamp"] = str(int(time.time() * 1000))
            params["signature"] = self._sign(params)
        
        try:
            if method == "GET":
                resp = self._session.get(url, params=params, headers=headers, timeout=10)
            else:
                resp = self._session.request(method, url, params=params, headers=headers, timeout=10)
            
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Binance request failed: {e}")
            return None
    
    def normalize_symbol(self, symbol: str) -> str:
        """Binance格式: BTCUSDT"""
        return symbol.upper().replace("-", "").replace("_", "")
    
    def format_symbol(self, symbol: str) -> str:
        """交易所格式"""
        return self.normalize_symbol(symbol)
    
    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """获取24小时行情"""
        symbol = self.format_symbol(symbol)
        data = self._request("GET", "/api/v3/ticker/24hr", params={"symbol": symbol})
        
        if data:
            return Ticker(
                symbol=symbol,
                last_price=float(data.get("lastPrice", 0)),
                high_24h=float(data.get("highPrice", 0)),
                low_24h=float(data.get("lowPrice", 0)),
                volume_24h=float(data.get("volume", 0)),
                change_24h=float(data.get("priceChangePercent", 0)),
                timestamp=int(data.get("closeTime", 0))
            )
        return None
    
    def get_candles(
        self,
        symbol: str,
        bar: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取K线"""
        symbol = self.format_symbol(symbol)
        
        # 映射时间周期
        bar_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }
        interval = bar_map.get(bar, bar)
        
        data = self._request(
            "GET",
            "/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit}
        )
        
        if data:
            # [时间, 开, 高, 低, 收, 量, ...]
            candles = []
            for c in data:
                candles.append({
                    "timestamp": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5])
                })
            return candles
        return []
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        """获取订单簿"""
        symbol = self.format_symbol(symbol)
        data = self._request(
            "GET",
            "/api/v3/depth",
            params={"symbol": symbol, "limit": limit}
        )
        
        if data:
            return OrderBook(
                symbol=symbol,
                bids=[[float(b[0]), float(b[1])] for b in data.get("bids", [])],
                asks=[[float(a[0]), float(a[1])] for a in data.get("asks", [])],
                timestamp=int(time.time() * 1000)
            )
        return None
    
    def get_balance(self) -> List[Balance]:
        """获取账户余额"""
        data = self._request("GET", "/api/v3/account", auth=True)
        
        if data and data.get("balances"):
            balances = []
            for b in data["balances"]:
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                if free > 0 or locked > 0:
                    balances.append(Balance(
                        asset=b.get("asset", ""),
                        free=free,
                        locked=locked,
                        total=free + locked
                    ))
            return balances
        return []
    
    def get_positions(self) -> List[Position]:
        """获取持仓 (USDT合约)"""
        data = self._request(
            "GET",
            "/fapi/v2/positionRisk",
            params={"pair": "BTCUSDT"},  # 需要指定交易对
            auth=True
        )
        
        # 注意: Binance需要遍历所有交易对
        positions = []
        if data:
            for p in data:
                if float(p.get("positionAmt", 0)) != 0:
                    positions.append(Position(
                        symbol=p.get("symbol", ""),
                        side="long" if float(p.get("positionAmt", 0)) > 0 else "short",
                        size=abs(float(p.get("positionAmt", 0))),
                        entry_price=float(p.get("entryPrice", 0)),
                        unrealized_pnl=float(p.get("unrealizedProfit", 0)),
                        leverage=float(p.get("leverage", 1))
                    ))
        return positions
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """下单"""
        symbol = self.format_symbol(symbol)
        
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET" if order_type == "market" else "LIMIT",
            "quantity": size
        }
        
        if price:
            params["price"] = price
            params["timeInForce"] = "GTC"
        
        result = self._request("POST", "/api/v3/order", params=params, auth=True)
        return result
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消订单"""
        params = {
            "symbol": self.format_symbol(symbol),
            "orderId": order_id
        }
        result = self._request("DELETE", "/api/v3/order", params=params, auth=True)
        return result is not None


# ========================
# 工厂函数
# ========================

def create_exchange_adapter(
    exchange_type: ExchangeType,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    passphrase: Optional[str] = None,
    testnet: bool = False
) -> ExchangeAdapter:
    """
    创建交易所适配器
    
    Args:
        exchange_type: 交易所类型 (ExchangeType.OKX 或 ExchangeType.BINANCE)
        api_key: API密钥
        secret: API密钥
        passphrase: 密码 (OKX需要)
        testnet: 是否使用测试网
    
    Returns:
        ExchangeAdapter实例
    """
    if exchange_type == ExchangeType.OKX:
        return OKXAdapter(api_key, secret, passphrase, testnet)
    elif exchange_type == ExchangeType.BINANCE:
        return BinanceAdapter(api_key, secret, passphrase, testnet)
    else:
        raise ValueError(f"Unsupported exchange type: {exchange_type}")


def get_default_exchange() -> ExchangeAdapter:
    """获取默认交易所适配器 (从环境变量)"""
    default = os.getenv("DEFAULT_EXCHANGE", "okx").lower()
    
    if default == "binance":
        return create_exchange_adapter(ExchangeType.BINANCE)
    else:
        return create_exchange_adapter(ExchangeType.OKX)


if __name__ == "__main__":
    import json
    
    print("=== Exchange Adapter Test ===\n")
    
    # 测试OKX
    print("Testing OKX public API...")
    okx = OKXAdapter()
    
    try:
        ticker = okx.get_ticker("BTC-USDT")
        if ticker:
            print(f"  BTC ticker: ${ticker.last_price:,.2f}")
        
        candles = okx.get_candles("BTC-USDT", "1h", 5)
        print(f"  BTC 1h candles: {len(candles)} bars")
        
        orderbook = okx.get_orderbook("BTC-USDT", 5)
        if orderbook:
            print(f"  Order book: {len(orderbook.bids)} bids, {len(orderbook.asks)} asks")
    except Exception as e:
        print(f"  OKX test failed: {e}")
    
    # 测试Binance
    print("\nTesting Binance public API...")
    binance = BinanceAdapter()
    
    try:
        ticker = binance.get_ticker("BTCUSDT")
        if ticker:
            print(f"  BTC ticker: ${ticker.last_price:,.2f}")
        
        candles = binance.get_candles("BTCUSDT", "1h", 5)
        print(f"  BTC 1h candles: {len(candles)} bars")
    except Exception as e:
        print(f"  Binance test failed: {e}")
    
    print("\n=== Test complete ===")
