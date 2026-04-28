"""
Exchange Client - 统一交易所接口
=================================

从 agents/agent_executor.py 提取

包含:
- ExchangeClient: 支持OKX和Binance的统一交易所API客户端

用法:
    from core.exchange_client import ExchangeClient
    from agents.agent_executor import ExchangeClient  # 向后兼容
"""

import base64
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from requests.exceptions import RequestException, Timeout

from core.executor_config import ExecutorConfig

logger = logging.getLogger("ExchangeClient")


class ExchangeClient:
    """
    统一交易所接口，支持OKX和Binance
    """

    def __init__(self, exchange: str = "okx", config: ExecutorConfig = None):
        self.exchange = exchange.lower()
        self.config = config or ExecutorConfig()

        if self.exchange == "okx":
            self._setup_okx()
        elif self.exchange == "binance":
            self._setup_binance()
        else:
            raise ValueError(f"不支持的交易所: {exchange}")

    def _setup_okx(self):
        """配置OKX API"""
        # OKX永续合约没有testnet URL，通过 x-simulated-trading:1 头切换模拟盘
        # 真实交易时 okx_testnet=False（不加该头）
        self.base_url = "https://www.okx.com"
        # 从安全密钥管理器获取API密钥
        api_key, secret_key, passphrase = self.config.get_okx_keys()
        self.api_key = api_key or ""
        self.secret_key = secret_key or ""
        self.passphrase = passphrase or ""

        # OKX API endpoints
        self.endpoints = {
            "balance": "/api/v5/account/balance",
            "positions": "/api/v5/account/positions",
            "ticker": "/api/v5/market/ticker",
            "order": "/api/v5/trade/order",
            "cancel_order": "/api/v5/trade/cancel-order",
            "close_position": "/api/v5/trade/close-position",
            "algo_order": "/api/v5/trade/order-algo",  # OCO/conditional单
            "algo_pending": "/api/v5/trade/orders-algo-pending",  # 查询活跃条件单
            "algo_history": "/api/v5/trade/orders-algo-history",  # 条件单历史
        }

    def _setup_binance(self):
        """配置Binance API"""
        self.base_url = "https://api.binance.com" if not self.config.binance_testnet else "https://testnet.binance.vision"
        # 从安全密钥管理器获取API密钥
        api_key, secret_key = self.config.get_binance_keys()
        self.api_key = api_key or ""
        self.secret_key = secret_key or ""

        # Binance API endpoints
        self.endpoints = {
            "balance": "/api/v3/account",
            "positions": "/api/v3/account",
            "ticker": "/api/v3/ticker/price",
            "order": "/api/v3/order",
            "cancel_order": "/api/v3/order",
            "close_position": "/api/v3/order",
        }

    def _sign_request(self, params: Dict) -> Dict:
        """签名请求 (OKX HMAC SHA256)"""
        import hashlib
        import hmac

        # OKX签名格式: ISO8601带毫秒 (e.g. "2023-12-01T08:01:01.123Z")
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.') + \
                   f"{datetime.utcnow().microsecond // 1000:03d}Z"
        method = "GET" if params.get("_method") == "GET" else "POST"
        path = params.get("_path", "")
        body = params.get("_body", "")

        message = timestamp + method + path + body
        # OKX使用base64编码签名（非hex）
        signature_b64 = base64.b64encode(
            hmac.new(self.secret_key.encode(), message.encode(), hashlib.sha256).digest()
        ).decode()

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature_b64,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        # OKX模拟盘用 x-simulated-trading: 1 头切换（非改变URL）
        if self.config.okx_testnet:
            headers["x-simulated-trading"] = "1"

        return headers

    def _make_request(self, method: str, endpoint: str, params: Dict = None,
                      data: Dict = None, signed: bool = True) -> Dict:
        """发起API请求"""
        url = self.base_url + endpoint

        headers = {}
        if signed and self.exchange == "okx":
            headers = self._sign_request({
                "_method": method,
                "_path": endpoint,
                "_body": json.dumps(data) if data else ""
            })

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=self.config.order_timeout)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=self.config.order_timeout)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, json=data, timeout=self.config.order_timeout)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()
            result = response.json()

            if result.get("code") != "0" and self.exchange == "okx":
                raise Exception(f"OKX API错误: {result.get('msg', 'Unknown error')}")

            return result

        except Timeout:
            raise Exception(f"{self.exchange.upper()} API超时")
        except RequestException as e:
            raise Exception(f"{self.exchange.upper()} API请求失败: {str(e)}")

    def get_balance(self) -> Dict[str, float]:
        """获取账户余额（API失败时返回模拟余额）"""
        try:
            if self.exchange == "okx":
                result = self._make_request("GET", self.endpoints["balance"], signed=True)
                data = result.get("data", [{}])[0]
                total_equity = float(data.get("totalEq", 0))
                return {
                    "total": total_equity,
                    "available": float(data.get("availEq", total_equity)),
                    "currency": "USDT"
                }
            elif self.exchange == "binance":
                result = self._make_request("GET", self.endpoints["balance"], signed=True)
                total = sum(float(a["free"]) for a in result.get("balances", []))
                return {
                    "total": total,
                    "available": total,
                    "currency": "USDT"
                }
        except Exception as e:
            logging.warning(f"获取余额失败: {e}，使用模拟余额")
            return {"total": 100000.0, "available": 100000.0, "currency": "USDT", "simulated": True}

    def get_ticker(self, symbol: str) -> Optional[float]:
        """获取当前价格（API失败时尝试从其他渠道获取）"""
        try:
            if self.exchange == "okx":
                # OKX symbol格式: BTC-USDT-SWAP
                inst_id = symbol.replace("-", "-SWAP-") if "SWAP" not in symbol else symbol.replace("-USDT", "-USDT-SWAP")
                params = {"instId": inst_id}
                result = self._make_request("GET", self.endpoints["ticker"], params=params, signed=False)
                data = result.get("data", [{}])[0]
                return float(data.get("last", 0))
            elif self.exchange == "binance":
                # Binance symbol格式: BTCUSDT
                binance_symbol = symbol.replace("-", "").replace("SWAP", "")
                params = {"symbol": binance_symbol}
                result = self._make_request("GET", self.endpoints["ticker"], params=params, signed=False)
                return float(result.get("price", 0))
        except Exception as e:
            logging.warning(f"获取Ticker失败 [{symbol}]: {e}")
            # 尝试从币安公开API获取价格作为备选
            try:
                binance_symbol = symbol.replace("-USDT-SWAP", "USDT").replace("-USDT", "USDT").replace("-SWAP", "")
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return float(resp.json().get("price", 0))
            except Exception:
                pass
            return None

    def place_order(self, symbol: str, side: str, order_type: str,
                    price: Optional[float] = None, size: float = 0,
                    leverage: int = 1) -> Optional[Dict]:
        """
        下单
        order_type: "market" / "limit"
        side: "buy" / "sell"
        """
        retry_count = 0
        last_error = None

        while retry_count < self.config.max_retry:
            try:
                if self.exchange == "okx":
                    return self._place_order_okx(symbol, side, order_type, price, size, leverage)
                elif self.exchange == "binance":
                    return self._place_order_binance(symbol, side, order_type, price, size, leverage)
            except Exception as e:
                last_error = str(e)
                logging.warning(f"下单失败 [重试 {retry_count + 1}/{self.config.max_retry}]: {e}")
                retry_count += 1
                if retry_count < self.config.max_retry:
                    time.sleep(self.config.retry_interval)

        # 全部重试失败
        logging.error(f"下单最终失败: {last_error}")
        return None

    def _place_order_okx(self, symbol: str, side: str, order_type: str,
                         price: Optional[float], size: float, leverage: int) -> Dict:
        """OKX下单"""
        inst_id = symbol.replace("-USDT", "-USDT-SWAP") if "-SWAP" not in symbol else symbol

        # 设置杠杆
        self._make_request("POST", "/api/v5/account/set-leverage", data={
            "instId": inst_id,
            "lever": str(leverage),
            "mgnMode": "cross"
        }, signed=True)

        # 构造订单
        order_data = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": side.upper(),
            "ordType": "market" if order_type == "market" else "limit",
            "sz": str(size),
        }

        if order_type == "limit" and price:
            order_data["px"] = str(price)

        result = self._make_request("POST", self.endpoints["order"], data=order_data, signed=True)

        # 解析成交价格
        data = result.get("data", [{}])[0]
        fill_price = float(data.get("fillPx", price or 0))
        order_id = data.get("ordId", "")

        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "price": fill_price,
            "size": size,
            "status": "filled",
            "exchange": "okx"
        }

    # ─────────────────────────────────────────────────────────────
    # OCO bracket订单：止损+止盈合并为1个条件单（OKX每持仓只能1个条件单）
    # ─────────────────────────────────────────────────────────────

    def place_oco_order(self, symbol: str, side: str, size: float,
                        entry_price: float, sl_price: float, tp_price: float,
                        leverage: int = 1) -> Optional[Dict]:
        """
        下一个OCO bracket订单（止损+止盈合并单）

        OKX OCO模式：SL和TP作为同一个algo订单的附加条件，互斥触发。
        止损触发时止盈自动取消，止盈触发时止损自动取消。

        Args:
            symbol: 币种符号，如 "BTC-USDT-SWAP"
            side: "buy"(做多) / "sell"(做空)
            size: 合约数量
            entry_price: 入场价格
            sl_price: 止损价格
            tp_price: 止盈价格
            leverage: 杠杆倍数

        Returns:
            订单结果 dict，失败返回 None
        """
        retry_count = 0
        last_error = None

        while retry_count < self.config.max_retry:
            try:
                if self.exchange == "okx":
                    return self._place_oco_okx(symbol, side, size, entry_price, sl_price, tp_price, leverage)
                elif self.exchange == "binance":
                    logging.warning("Binance OCO暂未实现")
                    return None
            except Exception as e:
                last_error = str(e)
                logging.warning(f"OCO下单失败 [重试 {retry_count + 1}/{self.config.max_retry}]: {e}")
                retry_count += 1
                if retry_count < self.config.max_retry:
                    time.sleep(self.config.retry_interval)

        logging.error(f"OCO下单最终失败: {last_error}")
        return None

    def _place_oco_okx(self, symbol: str, side: str, size: float,
                        entry_price: float, sl_price: float, tp_price: float,
                        leverage: int) -> Dict:
        """OKX OCO bracket订单"""
        inst_id = symbol if "-SWAP" in symbol else f"{symbol}-USDT-SWAP"

        # 1. 设置杠杆
        self._make_request("POST", "/api/v5/account/set-leverage", data={
            "instId": inst_id,
            "lever": str(leverage),
            "mgnMode": "cross"
        }, signed=True)

        # 2. 市价开仓
        open_data = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": side.upper(),
            "ordType": "market",
            "sz": str(int(size)),
        }
        open_result = self._make_request("POST", self.endpoints["order"], data=open_data, signed=True)
        open_data_resp = open_result.get("data", [{}])[0]
        ord_id = open_data_resp.get("ordId", "")

        if not ord_id:
            raise RuntimeError(f"开仓失败，无ordId: {open_result}")

        # 3. 挂OCO条件单（止损+止盈）
        sl_trigger_px = str(sl_price)
        tp_trigger_px = str(tp_price)

        oco_data = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": "sell" if side.upper() == "BUY" else "buy",
            "ordType": "oco",
            "sz": str(int(size)),
            "slTriggerPx": sl_trigger_px,
            "slOrdPx": "-1",
            "tpTriggerPx": tp_trigger_px,
            "tpOrdPx": "-1",
        }

        oco_result = self._make_request("POST", self.endpoints["algo_order"], data=oco_data, signed=True)
        oco_data_resp = oco_result.get("data", [{}])[0]
        algo_id = oco_data_resp.get("algoId", "")

        if not algo_id:
            raise RuntimeError(f"OCO条件单失败: {oco_result}")

        return {
            "order_id": ord_id,
            "algo_id": algo_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "size": size,
            "leverage": leverage,
            "status": "open_with_oco",
            "exchange": "okx"
        }

    def get_pending_algo_orders(self, inst_type: str = "SWAP") -> List[Dict]:
        """
        查询所有活跃条件单（OCO + conditional）
        用于检查某持仓是否已有SL/TP保护。
        """
        try:
            result = self._make_request(
                "GET",
                f"{self.endpoints['algo_pending']}?instType={inst_type}&ordType=oco&limit=100",
                signed=True
            )
            oco_orders = result.get("data", [])

            result2 = self._make_request(
                "GET",
                f"{self.endpoints['algo_pending']}?instType={inst_type}&ordType=conditional&limit=100",
                signed=True
            )
            cond_orders = result2.get("data", [])

            all_orders = []
            for o in oco_orders + cond_orders:
                all_orders.append({
                    "instId": o.get("instId", ""),
                    "algoId": o.get("algoId", ""),
                    "slTriggerPx": o.get("slTriggerPx", ""),
                    "tpTriggerPx": o.get("tpTriggerPx", ""),
                    "sz": o.get("sz", ""),
                    "ordType": o.get("ordType", ""),
                    "state": o.get("state", ""),
                })
            return all_orders
        except Exception as e:
            logging.error(f"查询条件单失败: {e}")
            return []

    def cancel_algo_order(self, inst_id: str, algo_id: str) -> bool:
        """取消指定条件单"""
        try:
            result = self._make_request("POST", "/api/v5/trade/cancel-algos", data={
                "instId": inst_id,
                "algoId": algo_id,
            }, signed=True)
            return result.get("code") == "0"
        except Exception as e:
            logging.error(f"取消条件单失败: {e}")
            return False

    def _place_order_binance(self, symbol: str, side: str, order_type: str,
                              price: Optional[float], size: float, leverage: int) -> Dict:
        """Binance下单"""
        binance_symbol = symbol.replace("-", "").replace("SWAP", "")

        # 设置杠杆
        self._make_request("POST", "/api/v5/margin/leverage", data={
            "symbol": binance_symbol,
            "leverage": str(leverage)
        }, signed=True)

        # 构造订单
        order_data = {
            "symbol": binance_symbol,
            "side": side.upper(),
            "type": "MARKET" if order_type == "market" else "LIMIT",
            "quantity": size,
        }

        if order_type == "limit" and price:
            order_data["price"] = str(price)
            order_data["timeInForce"] = "GTC"

        result = self._make_request("POST", self.endpoints["order"], data=order_data, signed=True)

        # 解析成交价格
        fills = result.get("fills", [])
        if fills:
            fill_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / sum(float(f["qty"]) for f in fills)
        else:
            fill_price = price or 0

        return {
            "order_id": str(result.get("orderId", "")),
            "symbol": symbol,
            "side": side,
            "price": fill_price,
            "size": size,
            "status": "filled",
            "exchange": "binance"
        }

    def close_position(self, symbol: str) -> Optional[Dict]:
        """平仓"""
        retry_count = 0
        last_error = None

        while retry_count < self.config.max_retry:
            try:
                if self.exchange == "okx":
                    return self._close_position_okx(symbol)
                elif self.exchange == "binance":
                    return self._close_position_binance(symbol)
            except Exception as e:
                last_error = str(e)
                logging.warning(f"平仓失败 [重试 {retry_count + 1}/{self.config.max_retry}]: {e}")
                retry_count += 1
                if retry_count < self.config.max_retry:
                    time.sleep(self.config.retry_interval)

        logging.error(f"平仓最终失败: {last_error}")
        return None

    def _close_position_okx(self, symbol: str) -> Dict:
        """OKX平仓"""
        inst_id = symbol.replace("-USDT", "-USDT-SWAP") if "-SWAP" not in symbol else symbol

        result = self._make_request("POST", self.endpoints["close_position"], data={
            "instId": inst_id,
            "mgnMode": "cross"
        }, signed=True)

        data = result.get("data", [{}])[0]
        return {
            "order_id": data.get("ordId", ""),
            "symbol": symbol,
            "status": "closed",
            "exchange": "okx"
        }

    def _close_position_binance(self, symbol: str) -> Dict:
        """Binance平仓"""
        binance_symbol = symbol.replace("-", "").replace("SWAP", "")

        # 先获取持仓方向
        positions = self.get_open_positions()
        pos = next((p for p in positions if p["symbol"] == symbol), None)

        if not pos:
            return {"status": "no_position", "symbol": symbol}

        # 反向下单平仓
        close_side = "SELL" if pos["side"] == "long" else "BUY"
        result = self._make_request("POST", self.endpoints["order"], data={
            "symbol": binance_symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": pos["size"]
        }, signed=True)

        return {
            "order_id": str(result.get("orderId", "")),
            "symbol": symbol,
            "status": "closed",
            "exchange": "binance"
        }

    def get_open_positions(self) -> List[Dict]:
        """获取当前持仓"""
        try:
            if self.exchange == "okx":
                return self._get_positions_okx()
            elif self.exchange == "binance":
                return self._get_positions_binance()
        except Exception as e:
            logging.error(f"获取持仓失败: {e}")
            return []

    def _get_positions_okx(self) -> List[Dict]:
        """OKX获取持仓"""
        result = self._make_request("GET", self.endpoints["positions"], signed=True)
        positions = []

        for pos in result.get("data", []):
            inst_id = pos.get("instId", "")
            if "SWAP" in inst_id:
                symbol = inst_id.replace("-SWAP", "").replace("-USDT-SWAP", "-USDT")
            else:
                symbol = inst_id

            positions.append({
                "symbol": symbol,
                "side": "long" if float(pos.get("pos", 0)) > 0 else "short",
                "size": abs(float(pos.get("pos", 0))),
                "entry_price": float(pos.get("avgPx", 0)),
                "unrealized_pnl": float(pos.get("upl", 0)),
                "exchange": "okx"
            })

        return positions

    def _get_positions_binance(self) -> List[Dict]:
        """Binance获取持仓"""
        result = self._make_request("GET", self.endpoints["positions"], signed=True)
        positions = []

        for pos in result.get("positions", []):
            if float(pos.get("positionAmt", 0)) == 0:
                continue

            symbol = pos.get("symbol", "")
            if symbol.endswith("USDT"):
                symbol = symbol[:-4] + "-USDT"

            positions.append({
                "symbol": symbol,
                "side": "long" if float(pos.get("positionAmt", 0)) > 0 else "short",
                "size": abs(float(pos.get("positionAmt", 0))),
                "entry_price": float(pos.get("entryPrice", 0)),
                "unrealized_pnl": float(pos.get("unrealizedProfit", 0)),
                "exchange": "binance"
            })

        return positions
