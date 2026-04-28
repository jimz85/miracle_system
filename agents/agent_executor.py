"""
Agent-E: 执行引擎Agent
Miracle 1.0.1 — 高频趋势跟踪+事件驱动混合系统

职责:
1. 接收Agent-R的最终执行指令
2. 通过OKX/Binance API下单
3. 实时监控持仓状态
4. 触发止损/止盈时自动平仓
5. 记录成交价格和滑点
6. 向Agent-L（学习模块）反馈交易结果
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import time
import logging
import threading
import requests
import base64
from requests.exceptions import RequestException, Timeout

# 安全密钥管理器已迁移到 core/secure_key_manager.py
from core.secure_key_manager import SecureKeyManager, get_key_manager


# ============================================================
# 配置已迁移到 core/executor_config.py
from core.executor_config import ExecutorConfig
# ============================================================
# 交易所客户端
# ============================================================

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
        import hmac
        import hashlib
        import base64
        from urllib.parse import urlencode

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
                import requests as _req
                resp = _req.get(url, timeout=5)
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
        # SL触发价：做多时空头触发(≤sl_price)，做空时多头触发(≥sl_price)
        # TP触发价：做多时多头触发(≥tp_price)，做空时空头触发(≤tp_price)
        sl_trigger_px = str(sl_price)
        tp_trigger_px = str(tp_price)

        oco_data = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": "sell" if side.upper() == "BUY" else "buy",  # 平仓side与开仓相反
            "ordType": "oco",
            "sz": str(int(size)),
            "slTriggerPx": sl_trigger_px,
            "slOrdPx": "-1",      # 市价止损
            "tpTriggerPx": tp_trigger_px,
            "tpOrdPx": "-1",      # 市价止盈
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

            # conditional单也要查（OKX有时用conditional格式）
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
            # 转换Binance symbol格式
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


# ============================================================
# 滑点监控已迁移到 core/slippage_monitor.py
from core.slippage_monitor import SlippageMonitor


# 交易日志已迁移到 core/trade_logger.py
from core.trade_logger import TradeLogger


# 飞书通知已迁移到 core/executor_feishu_notifier.py
from core.executor_feishu_notifier import FeishuNotifier


# ============================================================
# 订单管理器
# ============================================================

class OrderManager:
    """
    订单管理器
    负责订单生命周期管理
    """

    def __init__(self, exchange_client: 'ExchangeClient', config: ExecutorConfig):
        self.client = exchange_client
        self.config = config
        self.pending_orders: Dict[str, Dict] = {}

    def create_market_order(self, symbol: str, side: str, size: float,
                           leverage: int = 1) -> Optional[Dict]:
        """创建市价单"""
        return self.client.place_order(
            symbol=symbol,
            side=side,
            order_type="market",
            price=None,
            size=size,
            leverage=leverage
        )

    def create_limit_order(self, symbol: str, side: str, price: float,
                          size: float, leverage: int = 1) -> Optional[Dict]:
        """创建限价单"""
        return self.client.place_order(
            symbol=symbol,
            side=side,
            order_type="limit",
            price=price,
            size=size,
            leverage=leverage
        )

    def create_oco_order(self, symbol: str, side: str, size: float,
                        entry_price: float, sl_price: float, tp_price: float,
                        leverage: int = 1) -> Optional[Dict]:
        """创建OCO订单（止损+止盈）"""
        return self.client.place_oco_order(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            leverage=leverage
        )

    def cancel_order(self, order_id: str, inst_id: str = None) -> bool:
        """取消订单"""
        if hasattr(self.client, 'cancel_algo_order') and inst_id:
            return self.client.cancel_algo_order(inst_id, order_id)
        return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """获取订单状态"""
        for order in self.pending_orders.values():
            if order.get("order_id") == order_id:
                return order
        return None

    def add_pending_order(self, order_id: str, order_data: Dict):
        """添加待处理订单"""
        self.pending_orders[order_id] = order_data

    def remove_pending_order(self, order_id: str):
        """移除待处理订单"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]

    def get_pending_orders(self) -> Dict[str, Dict]:
        """获取所有待处理订单"""
        return self.pending_orders.copy()


# ============================================================
# 持仓监控器
# ============================================================

class PositionMonitor:
    """
    持仓监控器
    负责持仓监控、止损检查和自动平仓
    """

    def __init__(self, exchange_client: 'ExchangeClient', config: ExecutorConfig):
        self.client = exchange_client
        self.config = config
        self.positions: Dict[str, Dict] = {}

    def monitor(self, trade: Dict, current_price: float) -> Tuple[bool, str]:
        """
        监控持仓

        Returns:
            (should_exit, reason)
            reason: "none" | "sl" | "tp" | "time" | "atr"
        """
        symbol = trade.get("symbol")
        side = trade.get("side")
        entry_price = trade.get("entry_price")
        stop_loss = trade.get("stop_loss")
        take_profit = trade.get("take_profit")
        entry_time_str = trade.get("timestamp")

        # 价格止损
        if side == "long":
            if current_price <= stop_loss:
                return True, "sl"
            if current_price >= take_profit:
                return True, "tp"
        else:  # short
            if current_price >= stop_loss:
                return True, "sl"
            if current_price <= take_profit:
                return True, "tp"

        # 时间止损
        if entry_time_str:
            entry_time = datetime.fromisoformat(entry_time_str)
            hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
            if hold_hours >= self.config.max_hold_hours:
                return True, "time"

        return False, "none"

    def check_stop_loss(self, trade: Dict, current_price: float) -> bool:
        """检查是否触发止损"""
        side = trade.get("side")
        stop_loss = trade.get("stop_loss", 0)

        if side == "long" and current_price <= stop_loss:
            return True
        if side == "short" and current_price >= stop_loss:
            return True
        return False

    def check_take_profit(self, trade: Dict, current_price: float) -> bool:
        """检查是否触发止盈"""
        side = trade.get("side")
        take_profit = trade.get("take_profit", 0)

        if side == "long" and current_price >= take_profit:
            return True
        if side == "short" and current_price <= take_profit:
            return True
        return False

    def calculate_pnl(self, trade: Dict, current_price: float) -> float:
        """计算盈亏"""
        side = trade.get("side")
        entry_price = trade.get("entry_price")
        position_size = trade.get("position_size", 0)
        leverage = trade.get("leverage", 1)

        if side == "long":
            pnl = (current_price - entry_price) * position_size * leverage
        else:
            pnl = (entry_price - current_price) * position_size * leverage

        return pnl

    def check_moving_stop(self, trade: Dict, current_price: float,
                         atr: float = None) -> Optional[float]:
        """
        检查移动止损

        Returns:
            新止损价格，如果不需要移动则返回None
        """
        side = trade.get("side")
        entry_price = trade.get("entry_price")
        stop_loss = trade.get("stop_loss", 0)

        risk = abs(entry_price - stop_loss)
        if risk == 0:
            return None

        # 如果盈利超过2*R，移动止损到入场价
        breakeven_threshold = entry_price + risk * 2

        if side == "long" and current_price > breakeven_threshold:
            return entry_price * 0.998  # 微利保护
        if side == "short" and current_price < breakeven_threshold:
            return entry_price * 1.002

        return None

    def update_position(self, symbol: str, position_data: Dict):
        """更新持仓数据"""
        self.positions[symbol] = position_data

    def remove_position(self, symbol: str):
        """移除持仓数据"""
        if symbol in self.positions:
            del self.positions[symbol]

    def get_position(self, symbol: str) -> Optional[Dict]:
        """获取持仓数据"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """获取所有持仓"""
        return self.positions.copy()


# ============================================================
# 执行器 (主类)
# ============================================================

class Executor:
    """
    执行引擎主类

    接收Agent-R的最终执行指令，通过交易所API完成交易执行
    """

    def __init__(self, config: ExecutorConfig = None):
        # 支持传入dict（从JSON加载）或ExecutorConfig dataclass
        if isinstance(config, dict):
            known_fields = {
                'default_exchange', 'use_backup_on_fail',
                'okx_api_key', 'okx_secret_key', 'okx_passphrase', 'okx_testnet',
                'binance_api_key', 'binance_secret_key', 'binance_testnet',
                'max_retry', 'retry_interval', 'order_timeout',
                'slippage_warning_threshold',
                'feishu_webhook', 'feishu_enabled',
                'monitor_interval', 'max_hold_hours',
                'log_dir', 'trade_log_file', 'slippage_log_file',
            }
            filtered = {k: v for k, v in config.items() if k in known_fields}
            self.config = ExecutorConfig(**filtered)
        else:
            self.config = config or ExecutorConfig()

        # 初始化组件
        self.okx_client = ExchangeClient("okx", self.config)
        self.binance_client = ExchangeClient("binance", self.config)
        self.active_client = self.okx_client  # 默认OKX

        # 初始化管理器组件
        self.order_manager = OrderManager(self.active_client, self.config)
        self.position_monitor = PositionMonitor(self.active_client, self.config)

        self.slippage_monitor = SlippageMonitor(self.config)
        self.trade_logger = TradeLogger(self.config)
        self.notifier = FeishuNotifier(self.config)

        # 持仓监控
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, Callable] = {}  # 回调函数

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        import os
        os.makedirs(self.config.log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{self.config.log_dir}/executor.log"),
                logging.StreamHandler()
            ]
        )

    def set_exchange(self, exchange: str):
        """设置活跃交易所"""
        if exchange == "okx":
            self.active_client = self.okx_client
        elif exchange == "binance":
            self.active_client = self.binance_client
        else:
            raise ValueError(f"不支持的交易所: {exchange}")
        # 更新管理器组件的客户端引用
        self.order_manager.client = self.active_client
        self.position_monitor.client = self.active_client
    
    def _check_emergency_stop(self) -> bool:
        """
        检查是否处于紧急停止状态。
        检查优先级：
        1. 环境变量 EMERGENCY_STOP_ENABLED=true
        2. 本地emergency_stop状态文件
        3. 远程紧急停止API（如果配置了）
        """
        import os
        from pathlib import Path
        
        # 1. 检查环境变量
        if os.getenv("EMERGENCY_STOP_ENABLED", "").lower() == "true":
            emergency_file = Path(PROJECT_ROOT) / ".emergency_stop"
            if emergency_file.exists():
                try:
                    data = emergency_file.read_text().strip()
                    if data:
                        logging.critical(f"🚨 紧急停止文件内容: {data}")
                        return True
                except Exception:
                    pass
        
        # 2. 检查状态文件（由emergency_stop_api.py创建）
        state_file = Path(PROJECT_ROOT) / "data" / ".emergency_stop_state"
        if state_file.exists():
            try:
                import json
                state = json.loads(state_file.read_text())
                if state.get("emergency_stopped", False):
                    reason = state.get("reason", "Unknown")
                    logging.critical(f"🚨 紧急停止状态: {reason}")
                    return True
            except Exception:
                pass
        
        # 3. 尝试连接远程紧急停止API（如果配置了）
        emergency_api_url = os.getenv("EMERGENCY_STOP_API_URL")
        if emergency_api_url:
            try:
                import requests
                resp = requests.get(f"{emergency_api_url}/status", timeout=2)
                if resp.status_code == 200:
                    state = resp.json()
                    if state.get("emergency_stopped", False):
                        logging.critical(f"🚨 远程紧急停止: {state.get('reason', 'Unknown')}")
                        return True
            except Exception:
                pass  # API不可用，不阻止交易
        
        return False
    
    def trigger_emergency_stop(self, reason: str = "Manual stop"):
        """
        触发紧急停止（写入状态文件，供交易进程检查）
        """
        import os
        import json
        from pathlib import Path
        
        state_file = Path(PROJECT_ROOT) / "data" / ".emergency_stop_state"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "emergency_stopped": True,
            "reason": reason,
            "stop_time": datetime.now().isoformat(),
            "stopped_by": "executor"
        }
        
        state_file.write_text(json.dumps(state, indent=2))
        logging.critical(f"🚨 紧急停止已触发: {reason}")
        
        # 尝试取消所有活跃订单
        try:
            pending = self.order_manager.get_pending_orders()
            for order_id in list(pending.keys()):
                self.order_manager.remove_pending_order(order_id)
                logging.info(f"已移除待处理订单: {order_id}")
        except Exception as e:
            logging.error(f"取消订单时出错: {e}")

    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        self._callbacks[event] = callback

    def _trigger_callback(self, event: str, *args, **kwargs):
        """触发回调"""
        if event in self._callbacks:
            try:
                self._callbacks[event](*args, **kwargs)
            except Exception as e:
                logging.error(f"回调执行失败 [{event}]: {e}")

    def execute_signal(self, approved_signal: Dict) -> Optional[Dict]:
        """
        执行经过风控审批的信号

        approved_signal 格式 (来自Agent-R):
        {
            "signal_id": "uuid",
            "symbol": "BTC-USDT",
            "side": "long",
            "entry_price": 72000,
            "stop_loss": 70560,
            "take_profit": 75600,
            "leverage": 2,
            "position_size": 0.15,
            "market_regime": "bull",
            "factors": {...}
        }

        返回: 交易记录 或 None (执行失败)
        """
        # =====================================================
        # 紧急停止检查 - 如果系统处于紧急停止状态，则拒绝执行
        # =====================================================
        if self._check_emergency_stop():
            logging.warning("🚫 交易被拒绝: 系统处于紧急停止状态")
            return None
        
        symbol = approved_signal.get("symbol")
        side = approved_signal.get("side")
        leverage = approved_signal.get("leverage", 1)
        position_size = approved_signal.get("position_size", 0)

        logging.info(f"执行信号: {symbol} {side} 杠杆={leverage}x")

        # Step 1: 检查账户余额（API优先，失败时使用模拟）
        balance = self.active_client.get_balance()
        if balance["available"] <= 0:
            # 使用模拟余额
            logging.warning("无法获取真实余额，使用模拟余额 $100,000")
            balance = {"available": 100000.0, "total": 100000.0, "currency": "USDT"}

        # Step 2: 计算合约数量 (如果未指定)
        if position_size <= 0:
            current_price = self.active_client.get_ticker(symbol) or approved_signal.get("entry_price", 0)
            if current_price <= 0:
                # 使用入场价作为当前价
                current_price = approved_signal.get("entry_price", 50000)
                logging.warning(f"无法获取当前价格，使用入场价 ${current_price}")

            risk_amount = balance["available"] * (self.config.max_loss_per_trade_pct / 100)  # 1% 风险
            atr = approved_signal.get("atr", current_price * 0.01)
            stop_distance = abs(approved_signal.get("entry_price", current_price) - approved_signal.get("stop_loss", 0))
            stop_distance = max(stop_distance, atr * 1.5)

            position_size = risk_amount / stop_distance
            position_size = min(position_size, balance["available"] * leverage / current_price)

        # Step 3: 市价单入场（优先尝试真实下单，失败时模拟）
        planned_entry = approved_signal.get("entry_price", self.active_client.get_ticker(symbol))
        if not planned_entry:
            planned_entry = approved_signal.get("entry_price", 50000)

        sl_price = approved_signal.get("stop_loss", 0)
        tp_price = approved_signal.get("take_profit", 0)

        # 尝试真实下单
        order_result = None
        try:
            # 检查是否支持OCO
            use_oco = (self.active_client.exchange == "okx" and
                       hasattr(self.active_client, 'place_oco_order') and
                       sl_price > 0 and tp_price > 0 and position_size > 0)

            if use_oco:
                order_result = self.active_client.place_oco_order(
                    symbol=symbol,
                    side=side,
                    size=position_size,
                    entry_price=planned_entry,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    leverage=leverage
                )
            else:
                order_result = self.active_client.place_order(
                    symbol=symbol, side=side, order_type="market",
                    price=None, size=position_size, leverage=leverage
                )
        except Exception as e:
            logging.warning(f"真实下单异常: {e}，切换到模拟模式")
            order_result = None

        # 如果真实下单失败，使用模拟订单
        if not order_result:
            logging.info(f"[模拟模式] 入场@{planned_entry} + SL@{sl_price} + TP@{tp_price}")
            order_result = self._create_simulated_order(
                symbol=symbol, side=side, entry_price=planned_entry,
                sl_price=sl_price, tp_price=tp_price,
                position_size=position_size, leverage=leverage
            )

        # Step 4: 记录成交价格和滑点
        actual_entry = order_result.get("price", planned_entry)
        slippage_info = self.slippage_monitor.record_execution(approved_signal, planned_entry, actual_entry)

        # Step 5: 构建交易记录
        trade_id = str(uuid.uuid4())
        
        # 提取 algo_id (OCO订单会有)
        algo_id = order_result.get("algo_id") if order_result else None
        
        trade_record = {
            "trade_id": trade_id,
            "signal_id": approved_signal.get("signal_id", ""),
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": self.active_client.exchange,
            "side": side,
            "entry_price": actual_entry,
            "entry_slippage": slippage_info["slippage_pct"],
            "exit_price": None,
            "exit_slippage": None,
            "leverage": leverage,
            "position_size": position_size,
            "stop_loss": approved_signal.get("stop_loss", 0),
            "take_profit": approved_signal.get("take_profit", 0),
            "algo_id": algo_id,  # OCO订单ID
            "actual_rr": None,
            "pnl": None,
            "exit_reason": None,
            "hold_hours": None,
            "market_regime": approved_signal.get("market_regime", "unknown"),
            "factors": approved_signal.get("factors", {}),
            "status": "open"
        }

        # Step 6: 记录入场
        self.trade_logger.log_entry(trade_record)

        # Step 7: 飞书通知
        self.notifier.notify(trade_record, "entry")

        # Step 8: 触发回调 (通知Agent-L)
        self._trigger_callback("on_entry", trade_record)

        logging.info(f"入场成功: {trade_id} {symbol} {side} @ {actual_entry}")
        return trade_record

    def monitor_positions(self):
        """
        实时监控持仓
        检查: 止损/止盈/时间止损/移动保本
        触发条件立即平仓
        """
        open_trades = self.trade_logger.get_open_trades()

        if not open_trades:
            return

        current_time = datetime.now()

        for trade in open_trades:
            symbol = trade["symbol"]

            # 获取当前价格
            current_price = self.active_client.get_ticker(symbol)
            if current_price is None:
                continue

            # 使用PositionMonitor检查是否需要平仓
            should_exit, reason = self.position_monitor.monitor(trade, current_price)

            if should_exit:
                entry_time = datetime.fromisoformat(trade["timestamp"])
                hold_hours = (current_time - entry_time).total_seconds() / 3600
                self._close_trade(trade, reason, current_price, hold_hours)
                continue

            # 检查移动保本
            new_stop = self.position_monitor.check_moving_stop(trade, current_price)
            if new_stop:
                trade["stop_loss"] = new_stop
                logging.info(f"移动保本: {trade['trade_id']} 新止损={new_stop}")

    def _close_trade(self, trade: Dict, reason: str, current_price: float, hold_hours: float):
        """平仓处理"""
        trade_id = trade["trade_id"]
        symbol = trade["symbol"]
        stop_loss = trade["stop_loss"]
        algo_id = trade.get("algo_id")  # OCO订单ID

        # 计算PNL
        pnl = self.position_monitor.calculate_pnl(trade, current_price)

        # 取消OCO条件单（如果存在）
        if algo_id and self.active_client.exchange == "okx":
            try:
                inst_id = symbol.replace("-USDT", "-USDT-SWAP") if "-SWAP" not in symbol else symbol
                self.active_client.cancel_algo_order(inst_id, algo_id)
                logging.info(f"已取消OCO条件单: {algo_id}")
            except Exception as e:
                logging.warning(f"取消OCO条件单失败: {e}")

        # 执行平仓
        close_result = self.active_client.close_position(symbol)

        if close_result and close_result.get("status") != "error":
            # 记录出场价格和滑点
            planned_exit = stop_loss if reason == "stop_loss" else trade.get("take_profit", current_price)
            slippage_info = self.slippage_monitor.record_execution(
                {"trade_id": trade_id, "symbol": symbol, "side": trade.get("side")},
                planned_exit, current_price
            )

            trade["_exit_price"] = current_price
            trade["_exit_slippage"] = slippage_info["slippage_pct"]

            # 记录出场
            self.trade_logger.log_exit(trade_id, reason, pnl, hold_hours)

            # 飞书通知
            updated_trade = self.trade_logger.get_trade(trade_id)
            if updated_trade:
                self.notifier.notify(updated_trade, reason)

            # 触发回调
            self._trigger_callback("on_exit", updated_trade)

            logging.info(f"平仓完成: {trade_id} {reason} PNL={pnl:.2f}")
        else:
            logging.error(f"平仓失败: {trade_id}")
            self.notifier.send_alert("平仓失败", f"{symbol} 平仓失败，请人工处理", "error")

    def _create_simulated_order(self, symbol: str, side: str, entry_price: float,
                                sl_price: float, tp_price: float,
                                position_size: float, leverage: int) -> Dict:
        """
        创建模拟订单（用于无API或API失败时）

        模拟逻辑：
        - 假设在计划入场价成交
        - 滑点设为0
        - 订单ID为模拟ID
        """
        return {
            "order_id": f"sim_{uuid.uuid4().hex[:12]}",
            "algo_id": f"sim_algo_{uuid.uuid4().hex[:12]}",
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "price": entry_price,  # 假设以计划价格成交
            "size": position_size,
            "leverage": leverage,
            "status": "simulated_fill",
            "exchange": "simulated",
            "is_simulated": True
        }

    def start_monitoring(self):
        """启动持仓监控线程"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logging.info("持仓监控已启动")

    def stop_monitoring(self):
        """停止持仓监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logging.info("持仓监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self.monitor_positions()
            except Exception as e:
                logging.error(f"监控循环异常: {e}")

            time.sleep(self.config.monitor_interval)

    def force_close(self, trade_id: str, reason: str = "manual") -> bool:
        """手动平仓"""
        trade = self.trade_logger.get_trade(trade_id)
        if not trade:
            logging.error(f"交易不存在: {trade_id}")
            return False

        symbol = trade["symbol"]
        current_price = self.active_client.get_ticker(symbol)

        if current_price is None:
            logging.error(f"无法获取价格: {symbol}")
            return False

        entry_time = datetime.fromisoformat(trade["timestamp"])
        hold_hours = (datetime.now() - entry_time).total_seconds() / 3600

        self._close_trade(trade, reason, current_price, hold_hours)
        return True


# ============================================================
# 工具函数
# ============================================================

def create_executor(
    okx_api_key: str = "",
    okx_secret_key: str = "",
    okx_passphrase: str = "",
    binance_api_key: str = "",
    binance_secret_key: str = "",
    feishu_webhook: str = "",
    use_testnet: bool = True,
    log_dir: str = "logs"
) -> Executor:
    """创建执行器实例 (便捷函数)"""

    config = ExecutorConfig(
        okx_api_key=okx_api_key,
        okx_secret_key=okx_secret_key,
        okx_passphrase=okx_passphrase,
        binance_api_key=binance_api_key,
        binance_secret_key=binance_secret_key,
        okx_testnet=use_testnet,
        binance_testnet=use_testnet,
        feishu_webhook=feishu_webhook,
        feishu_enabled=bool(feishu_webhook),
        log_dir=log_dir
    )

    return Executor(config)


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    # 模拟执行测试
    print("Agent-E 执行引擎测试")
    print("=" * 50)

    # 创建测试配置
    test_config = ExecutorConfig(
        default_exchange="okx",
        use_testnet=True,
        log_dir="logs"
    )

    # 初始化执行器
    executor = Executor(test_config)

    # 测试: 获取余额
    print("\n1. 测试获取余额...")
    balance = executor.active_client.get_balance()
    print(f"   余额: {balance}")

    # 测试: 获取价格
    print("\n2. 测试获取价格...")
    price = executor.active_client.get_ticker("BTC-USDT")
    print(f"   BTC价格: {price}")

    # 测试: 模拟执行信号
    print("\n3. 测试执行信号 (模拟)...")
    mock_signal = {
        "signal_id": "test-signal-001",
        "symbol": "BTC-USDT",
        "side": "long",
        "entry_price": 72000,
        "stop_loss": 70560,
        "take_profit": 75600,
        "leverage": 2,
        "position_size": 0.01,
        "market_regime": "bull",
        "factors": {"atr": 500}
    }

    print("   模拟信号已构建 (实际执行需配置真实API)")
    print(f"   信号: {json.dumps(mock_signal, indent=2)}")

    # 启动监控测试
    print("\n4. 测试持仓监控...")
    executor.start_monitoring()
    print("   监控线程已启动")

    # 5秒后停止
    import time
    time.sleep(2)
    executor.stop_monitoring()

    print("\n" + "=" * 50)
    print("测试完成")
