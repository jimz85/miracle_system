from __future__ import annotations

"""
Market Intel OnChain - 链上分析增强模块
==========================================

从 agents/agent_market_intel_llm.py 提取

包含:
- EnhancedOnChainAnalyzer: 链上数据智能分析

依赖:
- core.market_intel_types

用法:
    from core.market_intel_onchain import EnhancedOnChainAnalyzer
"""

import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from core.market_intel_base import (
    API_CONFIG,
    get_timestamp,
    load_cache,
)

logger = logging.getLogger("MarketIntelOnChain")


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
        """分析交易所流量"""
        flow_data = self._get_exchange_flow_data(symbol)

        if not flow_data:
            return self._default_flow_analysis()

        self._flow_history[symbol].append({
            "timestamp": time.time(),
            "flow": flow_data.get("flow", 0),
            "inflow": flow_data.get("inflow", 0),
            "outflow": flow_data.get("outflow", 0)
        })

        if len(self._flow_history[symbol]) > 168:
            self._flow_history[symbol] = self._flow_history[symbol][-168:]

        pattern = self._identify_flow_pattern(symbol)
        anomalies = self._detect_flow_anomalies(symbol)
        trend = self._analyze_flow_trend(symbol)
        signal = self._calculate_flow_signal(flow_data, pattern, trend)

        return {
            "flow_data": flow_data,
            "pattern": pattern,
            "trend": trend,
            "anomalies": anomalies,
            "signal": signal,
            "interpretation": self._interpret_flow(flow_data, pattern, trend)
        }

    def _get_exchange_flow_data(self, symbol: str) -> Dict[str, Any] | None:
        """获取交易所流量数据"""
        cached = load_cache(symbol, "exchange_flow")
        if cached and (time.time() - cached.timestamp) < 300:
            return cached.data

        if not self.glassnode_key:
            return self._fetch_free_exchange_flow(symbol)

        return None

    def _fetch_free_exchange_flow(self, symbol: str) -> Dict[str, Any] | None:
        """使用OKX公开API获取流量代理数据"""
        import requests
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
        if len(history) < 12:
            return {"type": "unknown", "confidence": 0, "description": "数据不足"}

        recent = history[-12:]
        older = history[-24:-12] if len(history) >= 24 else history[:-12]

        recent_avg_flow = sum(h["flow"] for h in recent) / len(recent)
        older_avg_flow = sum(h["flow"] for h in older) / len(older) if older else 0

        recent_avg_inflow = sum(h["inflow"] for h in recent) / len(recent)
        recent_avg_outflow = sum(h["outflow"] for h in recent) / len(recent)

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
            "confidence": min(0.9, len(history) / 72),
            "recent_flow_avg": recent_avg_flow,
            "flow_ratio": flow_ratio
        }

    def _detect_flow_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """检测流量异常"""
        history = self._flow_history.get(symbol, [])
        if len(history) < 6:
            return []

        anomalies = []
        flows = [h["flow"] for h in history]
        mean_flow = sum(flows) / len(flows)
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

        net_ratio = base_flow / (inflow + outflow)

        trend_weight = 0.3 if trend["direction"] != "unknown" else 0
        if trend["direction"] == "inflow":
            trend_factor = trend["strength"] * 0.3
        elif trend["direction"] == "outflow":
            trend_factor = -trend["strength"] * 0.3
        else:
            trend_factor = 0

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
        """分析大额转账（巨鲸活动）"""
        transfers = self._get_large_transfers(symbol, threshold_usd)

        if not transfers:
            return self._default_whale_analysis()

        self._transfer_history[symbol].extend(transfers)
        if len(self._transfer_history[symbol]) > 1000:
            self._transfer_history[symbol] = self._transfer_history[symbol][-1000:]

        pattern = self._identify_whale_pattern(transfers, symbol)
        anomalies = self._detect_whale_anomalies(symbol)
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

        if pattern["type"] == "accumulation":
            base_signal = 0.3
        elif pattern["type"] == "distribution":
            base_signal = -0.3
        elif pattern["type"] == "repositioning":
            base_signal = 0.1
        else:
            base_signal = 0.0

        anomaly_factor = len(anomalies) * 0.1 * (-1 if pattern["type"] == "distribution" else 1)

        total_volume = sum(t.get("amount_usd", 0) for t in transfers)
        scale_factor = min(0.2, total_volume / 1e9)

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
