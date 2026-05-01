from __future__ import annotations

"""
Agent-S: Signal Generation Agent for Miracle 1.0.1
高频趋势跟踪+事件驱动混合系统

职责：
1. 接收Agent-M的市场情报报告
2. 计算价格因子（RSI/ADX/MACD/布林带）
3. 多因子融合，生成综合信号
4. 白名单/黑名单过滤
5. 输出高置信度交易信号给Agent-R

代码拆分说明（2026-04-28）：
- PriceFactors → core/price_factors.py
- TrendDetector / WhitelistFilter / MultiTimeframeFilter → core/signal_filters.py
- SignalGenerator / AgentSignal 保留于本文件
"""

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.ic_weights import get_weights as get_ic_weights
from core.memory import get_memory_system

# Core modules (extracted classes re-exported for backwards compatibility)
from core.price_factors import PriceFactors
from core.regime_classifier import RegimeClassifier
from core.signal_filters import MultiTimeframeFilter, TrendDetector, WhitelistFilter

# ============================================================================
# 保留类：SignalGenerator / AgentSignal
# ============================================================================

# PriceFactors, TrendDetector, WhitelistFilter, MultiTimeframeFilter
# 已迁移至 core/price_factors.py 和 core/signal_filters.py
# 通过上方导入语句保持向后兼容


# ============================================================================
# 4. 多因子融合信号生成器
# ============================================================================

class SignalGenerator:
    """
    多因子融合信号生成器

    权重配置:
    - price_momentum: 0.6 (价格动量权重)
    - news_sentiment: 0.2 (新闻情绪权重)
    - onchain: 0.1 (链上数据权重)
    - wallet: 0.1 (钱包数据权重)
    """

    def __init__(self, config: Dict | None = None):
        self.config = config or {}
        
        # 加载IC权重，替换硬编码的0.6/0.2/0.1/0.1
        # IC权重反映各因子历史预测精度
        self._load_ic_weights()
        
        self.whitelist = WhitelistFilter()

        # 自学习模式数据库
        self.pattern_db: Dict[str, List[Dict]] = defaultdict(list)
        
        # RegimeClassifier实例
        self._regime_classifier = RegimeClassifier()

        # 初始化Memory系统（决策流接入）
        self._memory = get_memory_system()

        # 加载pattern历史数据用于置信度调整
        self._pattern_history = self._load_pattern_history()

    def _load_ic_weights(self) -> None:
        """
        从IC权重系统加载动态权重，替换硬编码权重
        
        IC权重因子: rsi, macd, adx, bollinger, momentum
        信号因子: price_momentum, news_sentiment, onchain, wallet
        
        映射策略:
        - price_momentum: 基于IC权重最高的三个技术因子(rsi, macd, momentum)的平均
        - news_sentiment: 基于IC权重中的macd(技术信号)
        - onchain: 使用固定较低权重(链上数据置信度低)
        - wallet: 使用固定较低权重(钱包数据置信度低)
        """
        try:
            ic_weights = get_ic_weights()
            
            # price_momentum: 综合RSI、MACD、动量的IC权重
            price_ic = (ic_weights.get('rsi', 0.2) + 
                        ic_weights.get('macd', 0.2) + 
                        ic_weights.get('momentum', 0.2)) / 3.0
            
            # news_sentiment: 使用ADX的IC权重(趋势确认类似新闻信号)
            news_ic = ic_weights.get('adx', 0.2)
            
            # onchain和wallet使用较低的固定权重(数据质量和覆盖率问题)
            onchain_ic = 0.1
            wallet_ic = 0.1
            
            # 归一化使总和为1.0
            total = price_ic + news_ic + onchain_ic + wallet_ic
            if total > 0:
                self.weights = {
                    "price_momentum": price_ic / total,
                    "news_sentiment": news_ic / total,
                    "onchain": onchain_ic / total,
                    "wallet": wallet_ic / total
                }
            else:
                # 回退到默认值
                self.weights = {
                    "price_momentum": 0.6,
                    "news_sentiment": 0.2,
                    "onchain": 0.1,
                    "wallet": 0.1
                }
                
            logger = logging.getLogger(__name__)
            logger.info(f"[SignalGenerator] IC权重已加载: {self.weights}")
            
        except Exception as e:
            # 如果IC加载失败，使用硬编码默认值
            self.weights = {
                "price_momentum": 0.6,
                "news_sentiment": 0.2,
                "onchain": 0.1,
                "wallet": 0.1
            }
            logger = logging.getLogger(__name__)
            logger.warning(f"[SignalGenerator] IC权重加载失败，使用默认权重: {e}")

    def _load_pattern_history(self) -> Dict[str, Any]:
        """从pattern_history.json加载历史胜率数据"""
        import os
        import json
        history_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "pattern_history.json"
        )
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    return data.get("patterns", {})
        except Exception:
            pass
        return {}

    def _get_pattern_win_rate(self, pattern_key: str) -> Tuple[float, bool]:
        """
        获取pattern的历史胜率

        Returns:
            (win_rate, has_history): 胜率 和 是否有历史数据
            如果无历史数据，返回 (0.0, False)
        """
        if not pattern_key or pattern_key not in self._pattern_history:
            return 0.0, False

        pattern_data = self._pattern_history[pattern_key]
        entries = pattern_data.get("entries", 0)
        wins = pattern_data.get("wins", 0)

        if entries == 0:
            return 0.0, False

        win_rate = wins / entries
        return win_rate, True

    def calc_price_score(self, factors: Dict) -> float:
        """
        计算价格因子得分 (-1 到 1)
        正分 = 多头信号, 负分 = 空头信号
        """
        score = 0.0
        weights_sum = 0.0

        # RSI 评分 (权重 0.25)
        # 与WhitelistFilter保持一致的阈值设计
        rsi = factors.get("rsi", 50)
        if rsi < 30:
            rsi_score = 1.0  # 超卖 → 强烈买入（最佳赔率点）
        elif rsi < 40:
            rsi_score = 0.6  # 偏低 → 买入
        elif rsi <= 50:
            rsi_score = 0.3  # 中性偏多
        elif rsi <= 60:
            rsi_score = 0.0  # 中性
        elif rsi <= 70:
            rsi_score = -0.6  # 偏高 → 卖出
        else:
            rsi_score = -1.0  # 超买 → 强烈卖出
        score += rsi_score * 0.25
        weights_sum += 0.25

        # MACD 评分 (权重 0.25)
        macd_hist = factors.get("macd_histogram", 0)
        current_price = factors.get("current_price", 100)
        # 正确归一化：MACD直方图 / 价格 = 价格变动百分比
        # 这样可以消除价格量级的影响
        if abs(current_price) > 0:
            macd_normalized = macd_hist / current_price
        else:
            macd_normalized = 0.0
        # 映射到 -1 ~ 1 范围（0.01 = 1%价格变动）
        macd_score = max(-1.0, min(1.0, macd_normalized / 0.01))
        score += macd_score * 0.25
        weights_sum += 0.25

        # 动量评分 (权重 0.25)
        momentum = factors.get("momentum", 0)
        momentum_score = max(min(momentum / 10, 1.0), -1.0)  # ±10% 归一化
        score += momentum_score * 0.25
        weights_sum += 0.25

        # 趋势评分 (权重 0.25)
        trend = factors.get("trend", "range")
        if trend == "bull":
            trend_score = 1.0
        elif trend == "bear":
            trend_score = -1.0
        else:
            trend_score = 0.0
        score += trend_score * 0.25
        weights_sum += 0.25

        return score / weights_sum if weights_sum > 0 else 0.0

    def calc_news_score(self, intel_report: Dict) -> float:
        """
        从情报报告提取新闻情绪得分
        返回: -1 (极度利空) 到 1 (极度利好)

        兼容两种格式：
        - Agent-M 格式: intel_report['news_sentiment']['score']
        - 旧格式: intel_report['sentiment'] + intel_report['sentiment_score']
        """
        # Agent-M 格式优先（兼容新接口）
        news_sentiment = intel_report.get("news_sentiment", {})
        if isinstance(news_sentiment, dict):
            score = news_sentiment.get("score", 0.0)
            # 同时提取 sentiment 字符串用于方向判断
            labels = news_sentiment.get("labels", [])
            sentiment_str = "neutral"
            for label in labels:
                if "利好" in label or "bullish" in label.lower():
                    sentiment_str = "bullish"
                    break
                elif "利空" in label or "bearish" in label.lower():
                    sentiment_str = "bearish"
                    break
        else:
            # 旧格式兼容
            sentiment_str = intel_report.get("sentiment", "neutral")
            score = intel_report.get("sentiment_score", 0.0)

        if sentiment_str == "bullish":
            return min(score, 1.0)
        elif sentiment_str == "bearish":
            return max(score, -1.0)
        else:
            return score  # 中性

    def calc_onchain_score(self, intel_report: Dict) -> float:
        """
        从情报报告提取链上数据得分
        返回: -1 到 1

        兼容 Agent-M 格式: exchange_flow_signal (float)
        兼容旧格式: cvd_change, exchange_flow_ratio, active_address_change
        """
        onchain_data = intel_report.get("onchain", {})

        score = 0.0
        count = 0

        # Agent-M 格式: exchange_flow_signal（-1~1范围）
        if "exchange_flow_signal" in onchain_data:
            flow_signal = onchain_data["exchange_flow_signal"]
            score += flow_signal
            count += 1

        # 旧格式兼容
        if "cvd_change" in onchain_data:
            cvd = onchain_data["cvd_change"]
            score += (1.0 if cvd > 0 else -1.0) * min(abs(cvd) / 1000, 1.0)
            count += 1

        if "exchange_flow_ratio" in onchain_data:
            flow = onchain_data["exchange_flow_ratio"]
            if flow < 0.3:
                score += 0.5
            elif flow > 0.7:
                score -= 0.5
            count += 1

        if "active_address_change" in onchain_data:
            change = onchain_data["active_address_change"]
            score += max(min(change / 20, 1.0), -1.0)
            count += 1

        return score / count if count > 0 else 0.0

    def calc_wallet_score(self, intel_report: Dict) -> float:
        """
        从情报报告提取钱包/机构数据得分
        返回: -1 到 1

        兼容 Agent-M 格式: concentration_signal (float)
        兼容旧格式: institution_holding_change, whale_activity, etf_net_flow
        """
        wallet_data = intel_report.get("wallet", {})

        score = 0.0
        count = 0

        # Agent-M 格式: concentration_signal（-1~1范围）
        if "concentration_signal" in wallet_data:
            conc_signal = wallet_data["concentration_signal"]
            score += conc_signal
            count += 1

        # 旧格式兼容
        if "institution_holding_change" in wallet_data:
            change = wallet_data["institution_holding_change"]
            score += max(min(change / 5, 1.0), -1.0)
            count += 1

        if "whale_activity" in wallet_data:
            activity = wallet_data["whale_activity"]
            if activity == "accumulating":
                score += 0.7
            elif activity == "distributing":
                score -= 0.7
            count += 1

        if "etf_net_flow" in wallet_data:
            flow = wallet_data["etf_net_flow"]
            score += max(min(flow / 500, 1.0), -1.0)
            count += 1

        return score / count if count > 0 else 0.0

    def calc_combined_score(self, price_factors: Dict,
                             intel_report: Dict) -> Dict[str, float]:
        """
        多因子加权融合
        返回各因子得分和综合得分
        """
        price_score = self.calc_price_score(price_factors)
        news_score = self.calc_news_score(intel_report)
        onchain_score = self.calc_onchain_score(intel_report)
        wallet_score = self.calc_wallet_score(intel_report)

        # 归一化：price_score是-1~1，其他也是-1~1，权重和=1.0
        # 但实际calc_price_score返回-1~1，其他也是-1~1，直接加权即可
        # 唯一问题：price_score * 0.6 范围是-0.6~0.6，其他类似
        # 结果范围是-1~1，这是对的
        combined = (
            price_score * self.weights["price_momentum"] +
            news_score * self.weights["news_sentiment"] +
            onchain_score * self.weights["onchain"] +
            wallet_score * self.weights["wallet"]
        )

        # 额外检查：如果因子数据全为0（未接入真实API），降低置信度
        real_data_score = 1.0
        if abs(price_score) < 0.05:
            real_data_score *= 0.5  # 价格因子无效
        if abs(news_score) < 0.05:
            real_data_score *= 0.7  # 新闻因子无效（未接入）
        if abs(onchain_score) < 0.05:
            real_data_score *= 0.8  # 链上因子无效（未接入）
        if abs(wallet_score) < 0.05:
            real_data_score *= 0.9  # 钱包因子无效（未接入）

        return {
            "price_score": price_score,
            "news_score": news_score,
            "onchain_score": onchain_score,
            "wallet_score": wallet_score,
            "combined": combined,
            "weights_used": self.weights,
            "real_data_score": real_data_score  # 真实数据接入程度
        }

    def generate_signal(self, symbol: str, price_data: Dict,
                        intel_report: Dict,
                        price_data_4h: Dict | None = None,
                        override_mt_filter: bool = False) -> Dict:
        """
        生成综合交易信号

        Args:
            symbol: 交易标的 (如 "BTC")
            price_data: 价格数据 {"prices": [], "highs": [], "lows": []}
            intel_report: Agent-M 情报报告
            price_data_4h: 4H价格数据 (可选，用于多周期确认)
            override_mt_filter: 跳过4H确认过滤器（用于特殊场景）

        Returns:
            交易信号字典
        """
        prices = price_data.get("prices", [])
        highs = price_data.get("highs", prices)
        lows = price_data.get("lows", prices)

        if len(prices) < 50:
            return self._wait_signal(symbol, "Insufficient price data")

        # === 1. 计算1H价格因子 ===
        factors = PriceFactors.calc_all(prices, highs, lows, timeframe="1H")

        # === [Memory接入] 检索相关历史经验 ===
        try:
            memory_ctx = self._memory.retrieve_relevant_experiences(
                f"{symbol} {self._get_direction_hint(factors)}", k=3
            )
            if memory_ctx:
                factors["memory_context"] = memory_ctx
                factors["confidence_boost"] = 1.05  # 有相关经验时微提置信度
        except Exception:
            pass  # Memory不可用时不影响决策

        # === 2. 趋势判断 ===
        trend_info = TrendDetector.detect_trend(prices, highs, lows)
        factors["trend"] = trend_info["trend"]
        factors["trend_strength"] = trend_info["strength"]

        # === 2b. 市场状态分类 (Regime Classification) ===
        # 使用RegimeClassifier进行市场状态分类
        regime_result = {"regime": "sideways", "confidence": 0.5, "metrics": None}
        try:
            if len(prices) >= 50:
                # 延迟导入(pandas)避免冷启动耗时 ~3-18s
                import pandas as pd
                # 构建DataFrame供RegimeClassifier使用
                regime_df = pd.DataFrame({
                    'high': highs[-100:] if len(highs) >= 100 else highs,
                    'low': lows[-100:] if len(lows) >= 100 else lows,
                    'close': prices[-100:] if len(prices) >= 100 else prices
                })
                regime, regime_confidence, regime_metrics = self._regime_classifier.classify(regime_df)
                regime_result = {
                    "regime": regime.value if hasattr(regime, 'value') else str(regime),
                    "confidence": regime_confidence,
                    "metrics": {
                        "adx": regime_metrics.adx if regime_metrics else 0,
                        "plus_di": regime_metrics.plus_di if regime_metrics else 0,
                        "minus_di": regime_metrics.minus_di if regime_metrics else 0,
                        "momentum": regime_metrics.momentum if regime_metrics else 0
                    }
                }
                factors["regime"] = regime_result["regime"]
                factors["regime_confidence"] = regime_confidence
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"[SignalGenerator] RegimeClassification失败: {e}")

        # === 3. 成交量过滤 ===
        volumes = price_data.get("volumes", [])
        volume_filter_result = {"volume_ratio": 1.5, "is_confirmed": True, "is_rejected": False, "confidence_penalty": 0.0, "filter_reason": ""}
        if len(volumes) >= 20:
            volume_filter_result = PriceFactors.calc_volume_filter(volumes)
        factors["volume_ratio"] = volume_filter_result["volume_ratio"]
        factors["volume_confirmed"] = volume_filter_result["is_confirmed"]
        factors["volume_rejected"] = volume_filter_result["is_rejected"]
        factors["volume_filter_reason"] = volume_filter_result["filter_reason"]

        # === 4. 多因子融合 ===
        scores = self.calc_combined_score(factors, intel_report)
        combined_score = scores["combined"]
        real_data_score = scores.get("real_data_score", 1.0)

        # === 5. 白名单过滤 ===
        whitelist_result = self.whitelist.check(scores, factors)
        confidence_modifier = whitelist_result["confidence_modifier"]
        pattern_key = whitelist_result.get("pattern_key", "")

        # === 5b. Pattern历史胜率置信度调整 ===
        # 基于该pattern历史胜率调整置信度
        pattern_win_rate, has_pattern_history = self._get_pattern_win_rate(pattern_key)
        signal_score = abs(combined_score)  # 使用综合得分绝对值作为信号强度指标
        if has_pattern_history:
            # 有历史数据：根据胜率调整置信度
            # 胜率映射到 [0.5, 1.5] 范围
            history_confidence_factor = 0.5 + pattern_win_rate
            signal_score = signal_score * history_confidence_factor
        else:
            # 无历史数据：降低置信度
            history_confidence_factor = 0.5
            signal_score = signal_score * 0.5

        # === 6. 计算基础置信度 ===
        # 基础置信度 = 趋势强度 + 信号得分（已根据历史调整）
        base_confidence = (trend_info["strength"] / 100 * 0.4 +
                          signal_score * 0.6)
        # 成交量惩罚（缩量突破降低置信度）
        volume_penalty = volume_filter_result["confidence_penalty"]
        # 真实数据接入程度因子（未接入API时降低置信度）
        confidence = base_confidence * confidence_modifier * real_data_score
        confidence = confidence * (1.0 - volume_penalty)  # 成交量惩罚
        confidence = max(0.0, min(confidence, 1.0))

        # === 6b. 多周期过滤 (1H + 4H) ===
        mt_filter_result = {
            "applied": False,
            "confirmed": True,  # 如果没有4H数据，默认通过
            "confidence_boost": 0.0,
            "confirmations": 0,
            "total_checks": 0,
            "check_details": {}
        }
        factors_4h = None
        
        if price_data_4h is not None and not override_mt_filter:
            prices_4h = price_data_4h.get("prices", [])
            highs_4h = price_data_4h.get("highs", prices_4h)
            lows_4h = price_data_4h.get("lows", prices_4h)
            volumes_4h = price_data_4h.get("volumes", [])
            
            if len(prices_4h) >= 50:
                # 计算4H因子
                factors_4h = PriceFactors.calc_all_4h(prices_4h, highs_4h, lows_4h, volumes_4h)
                
                # 构建临时信号用于确认
                temp_signal = {
                    "direction": "long" if combined_score > 0 else ("short" if combined_score < 0 else "wait")
                }
                
                # 运行多周期确认
                mt_filter_result = MultiTimeframeFilter.confirm(temp_signal, factors, factors_4h)
                
                # 根据确认结果调整置信度
                if mt_filter_result["confirmed"]:
                    # 确认通过：提升置信度
                    confidence_boost = mt_filter_result["confidence_boost"]
                    confidence = confidence * (1.0 + confidence_boost * 0.2)
                    confidence = min(confidence, 1.0)  # 不超过1.0
                else:
                    # 确认失败：降低置信度
                    confidence_boost = mt_filter_result["confidence_boost"]
                    confidence = confidence * confidence_boost  # 按比例降低
                
                mt_filter_result["applied"] = True

        # === 7. 方向判断 ===
        if abs(combined_score) < 0.05:  # 阈值降低到0.05以产生更多信号（高频模式）
            direction = "wait"
            entry_price = None
            stop_loss = None
            take_profit = None
            rr_ratio = 0.0
        else:
            entry_price = prices[-1]

            # 使用ATR计算止损（更科学，考虑波动率）
            # ATR计算：使用布林带中轨作为波动率代理
            pf = PriceFactors()
            atr_value = factors.get("atr", pf.calc_atr(
                factors.get("highs", prices),
                factors.get("lows", prices),
                prices
            ))
            atr_multiplier = 3  # 3倍ATR作为止损

            # 计算风险金额（止损距离）
            stop_distance = atr_value * atr_multiplier

            # 目标RR = 2.5（赔率优先）
            target_rr = 2.5

            if combined_score > 0:
                direction = "long"
                # 止损：入场价 - 3倍ATR
                stop_loss = entry_price - stop_distance
                # 止盈：入场价 + 止损距离 * RR
                take_profit = entry_price + stop_distance * target_rr
            else:
                direction = "short"
                # 止损：入场价 + 3倍ATR
                stop_loss = entry_price + stop_distance
                # 止盈：入场价 - 止损距离 * RR
                take_profit = entry_price - stop_distance * target_rr

            rr_ratio = target_rr

            # 更新factors中的atr值（供风控模块使用）
            factors["atr"] = atr_value

        # === 9. 成交量硬过滤检查 ===
        # 如果成交量 < 30% 均量，直接拒绝信号
        if volume_filter_result.get("is_rejected", False):
            logger = logging.getLogger(__name__)
            logger.info(f"[{symbol}] 成交量过滤拒绝: {volume_filter_result.get('filter_reason')} (ratio={volume_filter_result.get('volume_ratio', 0):.2f})")
            direction = "wait"
            entry_price = None
            stop_loss = None
            take_profit = None
            rr_ratio = 0.0

        # === 10. 多周期确认后的最终方向判断 ===
        # 只有在置信度 >= 0.3 且 (确认通过 或 override) 时才执行
        if direction != "wait":
            if not override_mt_filter and mt_filter_result["applied"]:
                if confidence < 0.3 or not mt_filter_result["confirmed"]:
                    # 多周期确认失败，降低为wait
                    direction = "wait"
                    entry_price = None
                    stop_loss = None
                    take_profit = None
                    rr_ratio = 0.0

        # === 9. 建议杠杆和仓位 ===
        leverage = 1
        if confidence > 0.75 and trend_info["strength"] > 60:
            leverage = 2
        if confidence > 0.85 and trend_info["strength"] > 75 and abs(combined_score) > 0.5:
            leverage = 3

        position_size = confidence * 0.2  # 最大20%仓位

        # === 10. 生成信号理由 ===
        reason = self._generate_reason(direction, scores, factors, whitelist_result)

        # === 11. 组装信号 ===
        signal = {
            "symbol": symbol,
            "direction": direction,
            "signal_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry_price": round(entry_price, 2) if entry_price else None,
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "take_profit": round(take_profit, 2) if take_profit else None,
            "rr_ratio": round(rr_ratio, 2),
            "confidence": round(confidence, 4),
            "trend_strength": round(trend_info["strength"], 2),
            "leverage_recommended": leverage,
            "position_size_pct": round(position_size, 4),
            "factors": {
                "price_score": round(scores["price_score"], 4),
                "news_score": round(scores["news_score"], 4),
                "onchain_score": round(scores["onchain_score"], 4),
                "wallet_score": round(scores["wallet_score"], 4),
                "combined": round(scores["combined"], 4),
                "atr": atr_value if 'atr_value' in dir() else factors.get("atr", 0),
                "rsi": factors.get("rsi", 50),
                "adx": factors.get("adx", 25),
                "trend": factors.get("trend", "range")
            },
            "reason": reason,
            "whitelist_passed": whitelist_result["passed"],
            "pattern_key": whitelist_result.get("pattern_key", ""),
            "trend_info": {
                "trend": trend_info["trend"],
                "ema20": round(trend_info["ema20"], 2),
                "ema50": round(trend_info["ema50"], 2),
                "ema200": round(trend_info["ema200"], 2)
            },
            "regime_info": {
                "regime": regime_result["regime"],
                "confidence": round(regime_result["confidence"], 4),
                "metrics": regime_result["metrics"]
            },
            "volume_info": {
                "volume_ratio": round(volume_filter_result["volume_ratio"], 2),
                "is_confirmed": volume_filter_result["is_confirmed"],
                "is_rejected": volume_filter_result["is_rejected"],
                "filter_reason": volume_filter_result.get("filter_reason", ""),
                "confidence_penalty": round(volume_filter_result["confidence_penalty"], 3),
                "has_data": len(volumes) >= 20
            },
            "multi_timeframe": {
                "applied": mt_filter_result["applied"],
                "confirmed": mt_filter_result["confirmed"],
                "confidence_boost": round(mt_filter_result["confidence_boost"], 4),
                "confirmations": mt_filter_result["confirmations"],
                "total_checks": mt_filter_result["total_checks"],
                "check_details": mt_filter_result["check_details"],
                "4h_regime": MultiTimeframeFilter.get_4h_regime(factors_4h) if factors_4h else None,
                "factors_4h": {
                    "rsi": round(factors_4h["rsi"], 1) if factors_4h else None,
                    "adx": round(factors_4h["adx"], 1) if factors_4h else None,
                    "macd_direction": factors_4h.get("macd_direction") if factors_4h else None,
                    "volume_ratio": round(factors_4h.get("volume_ratio", 1.0), 2) if factors_4h else None
                } if factors_4h else None
            }
        }

        # === [Memory接入] 记录信号到Memory ===
        try:
            if signal["direction"] != "wait":
                self._memory.add_experience(
                    content=f"{symbol} {signal['direction']} signal, confidence={signal['confidence']:.2f}",
                    memory_type="trade",
                    metadata={"symbol": symbol, "direction": signal["direction"], "confidence": signal["confidence"]}
                )
        except Exception:
            pass  # Memory不可用时不影响决策

        return signal

    def _wait_signal(self, symbol: str, reason: str) -> Dict:
        """生成等待信号"""
        return {
            "symbol": symbol,
            "direction": "wait",
            "signal_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "rr_ratio": 0.0,
            "confidence": 0.0,
            "trend_strength": 0.0,
            "leverage_recommended": 1,
            "position_size_pct": 0.0,
            "factors": {
                "price_score": 0.0,
                "news_score": 0.0,
                "onchain_score": 0.0,
                "wallet_score": 0.0,
                "combined": 0.0
            },
            "reason": reason,
            "whitelist_passed": False
        }

    def _get_direction_hint(self, factors: Dict) -> str:
        """根据当前因子推断方向提示，供Memory检索使用"""
        trend = factors.get("trend", "range")
        if trend == "bull":
            return "bullish"
        elif trend == "bear":
            return "bearish"
        rsi = factors.get("rsi", 50)
        if rsi > 65:
            return "bearish"
        elif rsi < 35:
            return "bullish"
        return "neutral"

    def _generate_reason(self, direction: str, scores: Dict,
                          factors: Dict, whitelist: Dict) -> str:
        """生成信号理由文本"""
        reasons = []

        if direction == "wait":
            return "No clear signal - combined score below threshold"

        # 价格因子理由
        if scores["price_score"] > 0.3:
            reasons.append(f"Price momentum positive ({scores['price_score']:.2f})")
        elif scores["price_score"] < -0.3:
            reasons.append(f"Price momentum negative ({scores['price_score']:.2f})")

        # RSI理由
        rsi = factors.get("rsi", 50)
        if rsi < 30:
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # MACD理由
        if factors.get("macd_histogram", 0) > 0:
            reasons.append("MACD bullish")
        else:
            reasons.append("MACD bearish")

        # 趋势理由
        trend = factors.get("trend", "range")
        if trend != "range":
            reasons.append(f"{trend.capitalize()} trend confirmed")

        # 新闻理由
        if abs(scores["news_score"]) > 0.3:
            direction_word = "bullish" if scores["news_score"] > 0 else "bearish"
            reasons.append(f"News sentiment {direction_word}")

        # 白名单理由
        if whitelist["passed"]:
            reasons.append(f"Pattern: {whitelist['reason']}")
        else:
            reasons.append(f"Pattern rejected: {whitelist['reason']}")

        return "; ".join(reasons)

    # =========================================================================
    # 自学习接口
    # =========================================================================

    def update_pattern_db(self, trade_result: Dict):
        """
        交易结束后更新模式数据库
        trade_result: {
            "pattern_key": "long_RSI35-45_ADX30+",
            "actual_rr": 2.3,
            "won": bool
        }
        """
        pattern_key = trade_result.get("pattern_key", "")
        won = trade_result.get("won", False)
        actual_rr = trade_result.get("actual_rr", 0.0)

        self.pattern_db[pattern_key].append({
            "won": won,
            "rr": actual_rr,
            "timestamp": datetime.utcnow().isoformat()
        })

        # 更新白名单过滤器
        self.whitelist.update_pattern_db(pattern_key, won, actual_rr)

    def get_pattern_stats(self, pattern_key: str) -> Dict[str, Any]:
        """
        查询模式历史表现
        返回: {total, wins, win_rate, avg_rr}
        """
        records = self.pattern_db.get(pattern_key, [])
        if not records:
            return self.whitelist.get_pattern_stats(pattern_key)

        total = len(records)
        wins = sum(1 for r in records if r["won"])
        total_rr = sum(r["rr"] for r in records)

        return {
            "total": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "avg_rr": total_rr / total if total > 0 else 0.0,
            "records": records
        }

    def get_all_patterns_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模式的统计"""
        return {
            key: {
                "total": len(records),
                "wins": sum(1 for r in records if r["won"]),
                "win_rate": sum(1 for r in records if r["won"]) / len(records) if records else 0,
                "avg_rr": sum(r["rr"] for r in records) / len(records) if records else 0
            }
            for key, records in self.pattern_db.items()
        }


# ============================================================================
# 5. Agent-S 主类（对接Agent-M和Agent-R）
# ============================================================================

class AgentSignal:
    """
    Agent-S: 信号生成Agent

    对接:
    - 输入: Agent-M 的市场情报报告
    - 输出: 高置信度交易信号给 Agent-R
    """

    def __init__(self, config: Dict | None = None):
        self.config = config or {}
        self.generator = SignalGenerator(self.config)

    def process_intel(self, symbol: str, price_data: Dict,
                       intel_report: Dict,
                       price_data_4h: Dict | None = None) -> Dict:
        """
        处理Agent-M情报，生成交易信号

        Args:
            symbol: 交易标的
            price_data: 价格数据 {"prices": [], "highs": [], "lows": []}
            intel_report: Agent-M情报报告
            price_data_4h: 4H价格数据 (可选，用于多周期确认)

        Returns:
            高置信度交易信号
        """
        signal = self.generator.generate_signal(symbol, price_data, intel_report,
                                               price_data_4h=price_data_4h)
        return signal

    def feedback(self, signal_id: str, trade_result: Dict):
        """
        接收Agent-R的交易结果反馈，用于自学习

        Args:
            signal_id: 信号ID
            trade_result: {pattern_key, actual_rr, won}
        """
        self.generator.update_pattern_db(trade_result)

    def get_stats(self) -> Dict:
        """获取信号统计"""
        return self.generator.get_all_patterns_stats()


# ============================================================================
# 6. 演示/测试代码
# ============================================================================

if __name__ == "__main__":
    import random

    # 模拟价格数据
    base_price = 72000
    prices = [base_price * (1 + random.uniform(-0.02, 0.025)) for _ in range(100)]
    highs = [p * 1.005 for p in prices]
    lows = [p * 0.995 for p in prices]

    price_data = {
        "prices": prices,
        "highs": highs,
        "lows": lows
    }

    # 模拟Agent-M情报报告
    intel_report = {
        "sentiment": "bullish",
        "sentiment_score": 0.6,
        "onchain": {
            "cvd_change": 500,
            "exchange_flow_ratio": 0.25,
            "active_address_change": 15
        },
        "wallet": {
            "institution_holding_change": 3.2,
            "whale_activity": "accumulating",
            "etf_net_flow": 300
        }
    }

    # 生成信号
    agent = AgentSignal()
    signal = agent.process_intel("BTC", price_data, intel_report)

    print("=" * 60)
    print("Agent-S Signal Generation Test")
    print("=" * 60)
    print(f"Symbol: {signal['symbol']}")
    print(f"Direction: {signal['direction']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Trend Strength: {signal['trend_strength']:.1f}")
    print(f"Entry Price: {signal['entry_price']}")
    print(f"Stop Loss: {signal['stop_loss']}")
    print(f"Take Profit: {signal['take_profit']}")
    print(f"RR Ratio: {signal['rr_ratio']}")
    print(f"Leverage: {signal['leverage_recommended']}x")
    print(f"Position Size: {signal['position_size_pct']:.2%}")
    print(f"Whitelist Passed: {signal['whitelist_passed']}")
    print("-" * 60)
    print(f"Reason: {signal['reason']}")
    print("-" * 60)
    print("Factor Scores:")
    for k, v in signal['factors'].items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)
