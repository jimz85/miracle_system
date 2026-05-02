from __future__ import annotations

"""
Miracle 1.0.1 - 核心交易引擎
高频趋势跟踪+事件驱动混合系统
"""

import copy
import json
import logging
import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 因子计算（从 core.factor_calculations 导入，避免与 core/price_factors.py 重复）
from core.factor_calculations import (
    calc_adx,
    calc_atr,
    calc_combined_score,
    calc_macd,
    calc_rsi,
    normalize_macd_histogram,
)

# IC动态权重（可选导入，失败时返回基准权重）
try:
    from core.ic_weights import ICWeightManager as MiracleICTracker
except ImportError:
    MiracleICTracker = None

# ===== 日志配置 =====

logger = logging.getLogger("miracle")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(Path(__file__).parent / "miracle.log")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

# ===== IC动态权重 =====

def get_ic_adjusted_weights() -> Dict[str, float]:
    """
    获取IC动态调整后的因子权重
    
    返回格式与CONFIG["factors"]一致:
    - price_momentum: 价格动量因子权重
    - news_sentiment: 新闻情绪因子权重
    - onchain: 链上因子权重
    - wallet: 钱包因子权重
    
    Returns:
        Dict[str, float]: 归一化后的权重字典
    """
    try:
        tracker = MiracleICTracker(sync_with_kronos=True)
        ic_weights = tracker.get_all_weights()
        
        if not ic_weights:
            # 无IC历史数据，返回基准权重
            return {
                "price_momentum": 0.6,
                "news_sentiment": 0.2,
                "onchain": 0.1,
                "wallet": 0.1
            }
        
        # 将IC权重归一化到price/news/onchain/wallet结构
        # IC权重是针对单个技术指标的（如RSI, ADX等）
        # 我们需要把它们聚合成大类
        
        price_weight = (
            ic_weights.get('RSI', 0) +
            ic_weights.get('ADX', 0) +
            ic_weights.get('MACD', 0) +
            ic_weights.get('Bollinger', 0) +
            ic_weights.get('Momentum', 0) +
            ic_weights.get('Trend', 0) +
            ic_weights.get('Vol', 0)
        )
        news_weight = ic_weights.get('News', 0)
        onchain_weight = ic_weights.get('Onchain', 0)
        wallet_weight = ic_weights.get('Wallet', 0)
        
        total = price_weight + news_weight + onchain_weight + wallet_weight
        if total > 0:
            return {
                "price_momentum": price_weight / total,
                "news_sentiment": news_weight / total,
                "onchain": onchain_weight / total,
                "wallet": wallet_weight / total
            }
        else:
            return {
                "price_momentum": 0.6,
                "news_sentiment": 0.2,
                "onchain": 0.1,
                "wallet": 0.1
            }
            
    except ImportError:
        # IC模块不存在，返回基准权重
        return {
            "price_momentum": 0.6,
            "news_sentiment": 0.2,
            "onchain": 0.1,
            "wallet": 0.1
        }
    except Exception as e:
        logger.warning(f"获取IC权重失败: {e}")
        return {
            "price_momentum": 0.6,
            "news_sentiment": 0.2,
            "onchain": 0.1,
            "wallet": 0.1
        }


def record_ic_for_factor(factor_name: str, ic_value: float):
    """
    记录某因子的IC值（供外部调用）
    
    Args:
        factor_name: 因子名称 (RSI/ADX/MACD/Bollinger/Vol/etc.)
        ic_value: IC值
    """
    try:
        tracker = MiracleICTracker(sync_with_kronos=False)
        tracker.record_ic(factor_name, ic_value)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"记录IC失败: {e}")


# ===== 风险指标计算 =====

class RiskMetrics:
    """
    风险指标计算器

    计算:
    - VaR (Value at Risk)
    - CVaR (Conditional VaR)
    - 最大回撤
    - Sharpe Ratio
    - Sortino Ratio
    """

    @staticmethod
    def calculate_var(returns: List[float], confidence: float = 0.95) -> float:
        """
        计算VaR (Value at Risk)

        Args:
            returns: 收益列表（百分比，如 [0.01, -0.02, 0.03]）
            confidence: 置信度，默认95%

        Returns:
            var: VaR值（负数表示损失）
        """
        try:
            import numpy as np
        except ImportError:
            return 0.0

        if len(returns) < 10:
            return 0.0

        returns_array = np.array(returns)
        var = float(np.percentile(returns_array, (1 - confidence) * 100))
        return var

    @staticmethod
    def calculate_cvar(returns: List[float], confidence: float = 0.95) -> float:
        """
        计算CVaR (Conditional VaR) - 也称为Expected Shortfall

        Args:
            returns: 收益列表
            confidence: 置信度，默认95%

        Returns:
            cvar: CVaR值
        """
        try:
            import numpy as np
        except ImportError:
            return 0.0

        if len(returns) < 10:
            return 0.0

        var = RiskMetrics.calculate_var(returns, confidence)
        returns_array = np.array(returns)
        cvar = returns_array[returns_array <= var].mean()
        return float(cvar) if not np.isnan(cvar) else var

    @staticmethod
    def calculate_max_drawdown(returns: List[float]) -> float:
        """
        计算最大回撤

        Returns:
            max_dd: 最大回撤（正数，如 0.2 表示20%回撤）
        """
        try:
            import numpy as np
        except ImportError:
            return 0.0

        if not returns:
            return 0.0

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(abs(drawdowns.min()))

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        计算Sharpe Ratio

        Args:
            returns: 收益列表
            risk_free_rate: 无风险利率（年化）

        Returns:
            sharpe: Sharpe比率
        """
        try:
            import numpy as np
        except ImportError:
            return 0.0

        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # 日化无风险利率

        mean_return = excess_returns.mean()
        std_return = excess_returns.std()

        if std_return == 0:
            return 0.0

        # 年化Sharpe
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)

    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        计算Sortino Ratio - 只考虑下行波动率

        Returns:
            sortino: Sortino比率
        """
        try:
            import numpy as np
        except ImportError:
            return 0.0

        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252

        mean_return = excess_returns.mean()
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std()
        sortino = (mean_return / downside_std) * np.sqrt(252)
        return float(sortino)

# ===== 配置加载 =====

def load_config(config_path: str = None) -> Dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "miracle_config.json"
    with open(config_path) as f:
        return json.load(f)

CONFIG = load_config()
_config_lock = threading.Lock()

def get_config() -> Dict:
    """线程安全地获取CONFIG"""
    with _config_lock:
        return CONFIG

def update_config(key: str, value: Any) -> None:
    """线程安全地更新CONFIG"""
    with _config_lock:
        CONFIG[key] = value

# ===== 数据结构 =====

class DataFetchError(Exception):
    """数据获取异常，用于API失败时抛出"""
    pass

class APIError(Exception):
    """API调用异常，用于交易所API调用失败时抛出"""
    pass

class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class TradeSignal:
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    position_pct: float
    confidence: float  # 0-100
    factors: Dict[str, float]
    rr_ratio: float
    timestamp: str

@dataclass
class Trade:
    id: str
    direction: Direction
    entry_price: float
    exit_price: float
    position_size: float
    leverage: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    factors: Dict[str, float]
    stop_triggered: str  # "none" | "sl" | "tp" | "time" | "daily_loss"

# calc_rsi / calc_adx / calc_macd / calc_atr / normalize_macd_histogram / calc_combined_score
# 已迁移至 core/factor_calculations.py（calc_adx返回dict格式以与core/price_factors.py统一）
# calc_onchain_metrics / calc_wallet_metrics / calc_news_sentiment 为STUB，保留于本文件


# ---- 导入非STUB因子实现 ----
# 替换原有的 calc_onchain_metrics / calc_wallet_metrics / calc_news_sentiment
from core.data_feeds import (
    calc_onchain_metrics as _real_onchain,
    calc_wallet_metrics as _real_wallet,
    calc_news_sentiment as _real_news,
)
# 订单簿微观结构信号
from core.orderbook_analyzer import compute_orderbook_signal


def calc_onchain_metrics(coin: str = "BTC") -> Dict[str, float]:
    """
    计算链上因子 —— 真实数据实现 (非STUB)
    
    对接 market_sentiment.json (Kronos定时写入):
      - exchange_flow: CEX净流向
      - large_transfer: Fear & Greed归一化
    
    降级: 数据不可用时返回中性值
    """
    return _real_onchain(coin)


def calc_wallet_metrics(coin: str = "BTC") -> Dict[str, float]:
    """
    计算钱包分布因子 —— 真实数据实现 (非STUB)
    
    对接 CoinGecko 免费API:
      - holder_concentration: 持币集中度
    
    降级: API不可用时返回 0.5 (中性)
    """
    return _real_wallet(coin)


def calc_news_sentiment(coin: str = "BTC") -> float:
    """
    计算新闻情绪 —— 真实数据实现 (非STUB)
    
    对接 CoinDesk/CoinTelegraph RSS + KeywordSentimentAnalyzer:
      返回 -1.0(利空) ~ 1.0(利好)
    
    降级: 无相关新闻时返回 0.0 (中性)
    """
    return _real_news(coin)

def calc_factors(price_data: Dict, onchain_data: Dict | None = None, 
                 news_data: Dict | None = None, coin: str = "BTC",
                 orderbook_data: Dict | None = None) -> Dict[str, Any]:
    """
    计算所有因子值
    
    Args:
        price_data: 包含 highs, lows, closes 的字典
        onchain_data: 链上数据（可选）
        news_data: 新闻数据（可选）
        coin: 币种符号
        orderbook_data: 订单簿数据（可选），包含 bids/asks 列表
    
    Returns:
        因子字典
    """
    highs = price_data.get("highs", [])
    lows = price_data.get("lows", [])
    closes = price_data.get("closes", [])
    
    # 价格动量因子 (60%)
    rsi = calc_rsi(closes)
    adx_data = calc_adx(highs, lows, closes)
    adx = adx_data["adx"]
    plus_di = adx_data["plus_di"]
    minus_di = adx_data["minus_di"]
    macd_data = calc_macd(closes)
    signal = macd_data["signal"]
    hist = macd_data["histogram"]
    atr = calc_atr(highs, lows, closes)
    
    # 归一化因子到0-100
    rsi_norm = rsi  # RSI本身0-100
    adx_norm = min(adx, 100)  # ADX 0-100
    # MACD归一化：使用正确的归一化方法，将直方图映射到-1~1
    macd_norm_float = normalize_macd_histogram(hist, closes[-1]) if closes[-1] > 0 else 0.0
    macd_norm = 50 + macd_norm_float * 50  # -1~1 -> 0~100
    
    price_score = (rsi_norm * 0.33 + adx_norm * 0.34 + macd_norm * 0.33)
    
    # 新闻情绪因子 (20%) — 非STUB: 从RSS获取币种关键词新闻
    news_sentiment = news_data.get("sentiment", 0.0) if news_data else calc_news_sentiment(coin)
    news_score = (news_sentiment + 1) * 50  # -1~1 -> 0~100
    
    # 链上因子 (10%) — 非STUB: 从market_sentiment.json读取CEX流向
    onchain = onchain_data if onchain_data else calc_onchain_metrics(coin)
    exchange_flow = onchain.get("exchange_flow", 0.0)
    large_transfer = onchain.get("large_transfer", 0.0)
    onchain_score = 50 + (exchange_flow * 20 + large_transfer * 10)
    
    # 钱包因子 (10%) — 非STUB: 从CoinGecko获取持币集中度
    wallet = calc_wallet_metrics(coin)
    holder_conc = wallet.get("holder_concentration", 0.5)
    wallet_score = holder_conc * 100
    
    # 订单簿因子 (10%) — 微观结构信号: 买卖盘压力比
    orderbook_score = 50.0
    orderbook_pressure = 0.0
    orderbook_confidence = 0.0
    if orderbook_data:
        try:
            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])
            ob_signal = compute_orderbook_signal(bids, asks, depth=20)
            orderbook_pressure = ob_signal.pressure_score
            orderbook_confidence = ob_signal.confidence
            # pressure_score (-1~1) -> score (0~100)
            orderbook_score = 50 + orderbook_pressure * 50
        except Exception as e:
            logger.warning("订单簿因子计算失败: %s", e)
            orderbook_score = 50.0
    
    # 加权综合得分 — 只考虑启用的因子，权重重新归一化
    factors_cfg = CONFIG["factors"]
    
    # 收集启用的因子及其权重
    enabled_weights = {}
    for name, cfg in factors_cfg.items():
        if cfg.get("enabled", True):  # 默认启用
            enabled_weights[name] = cfg.get("weight", 0)
    
    if enabled_weights:
        total = sum(enabled_weights.values())
        # 归一化权重 (使总和为1.0)
        if total > 0:
            enabled_weights = {k: v/total for k, v in enabled_weights.items()}
    
    price_weight = enabled_weights.get("price_momentum", 0)
    news_weight = enabled_weights.get("news_sentiment", 0)
    onchain_weight = enabled_weights.get("onchain", 0)
    wallet_weight = enabled_weights.get("wallet", 0)
    orderbook_weight = enabled_weights.get("orderbook", 0)
    
    composite = (
        price_score * price_weight +
        news_score * news_weight +
        onchain_score * onchain_weight +
        wallet_score * wallet_weight +
        orderbook_score * orderbook_weight
    )
    
    return {
        "rsi": rsi,
        "adx": adx,
        "macd_hist": hist,
        "macd_signal": signal,
        "atr": atr,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "price_score": price_score,
        "news_sentiment": news_sentiment,
        "news_score": news_score,
        "exchange_flow": exchange_flow,
        "large_transfer": large_transfer,
        "onchain_score": onchain_score,
        "holder_concentration": holder_conc,
        "wallet_score": wallet_score,
        "orderbook_pressure": orderbook_pressure,
        "orderbook_confidence": orderbook_confidence,
        "orderbook_score": orderbook_score,
        "composite_score": composite,
        "weights": {
            "price": price_weight,
            "news": news_weight,
            "onchain": onchain_weight,
            "wallet": wallet_weight,
            "orderbook": orderbook_weight,
        }
    }

# ===== 趋势强度 =====

def calc_trend_strength(factors: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算趋势强度（0-100）
    
    Returns:
        (strength_0_100, level_label)
    """
    adx = factors["adx"]
    plus_di = factors["plus_di"]
    minus_di = factors["minus_di"]
    
    # ADX本身表示趋势强度
    adx_strength = min(adx, 100)
    
    # DI差值表示方向强度
    di_diff = abs(plus_di - minus_di)
    di_strength = min(di_diff * 2, 100)
    
    # 综合
    strength = int(adx_strength * 0.7 + di_strength * 0.3)
    
    if strength >= 70:
        label = "strong"
    elif strength >= 40:
        label = "medium"
    else:
        label = "weak"
    
    return strength, label

# ===== 杠杆计算 =====

def calc_leverage(trend_strength: int, confidence: int) -> Tuple[int, float]:
    """
    计算杠杆和仓位乘数
    
    Args:
        trend_strength: 趋势强度 0-100
        confidence: 因子置信度 0-100
        
    Returns:
        (leverage, position_multiplier)
    """
    lev_config = CONFIG["leverage"]
    
    # 确定趋势等级
    if trend_strength >= 70:
        level = lev_config["strong_trend"]
    elif trend_strength >= 40:
        level = lev_config["medium_trend"]
    else:
        level = lev_config["weak_trend"]
    
    base_leverage = level["leverage"]
    base_multiplier = level["multiplier"]
    
    # 置信度调整
    confidence_factor = confidence / 100  # 0-1
    
    # FOMC窗口期置信度降低50%
    from core.market_intel_base import get_fomc_confidence_multiplier
    confidence_factor = get_fomc_confidence_multiplier(confidence_factor * 100) / 100
    
    adjusted_multiplier = base_multiplier * (0.5 + confidence_factor * 0.5)
    
    return base_leverage, adjusted_multiplier

# ===== 仓位计算 =====

def calc_position_size(account_balance: float, entry_price: float, 
                       stop_loss_pct: float, leverage: float,
                       max_loss_pct: float = None) -> Tuple[float, float]:
    """
    计算仓位大小
    
    Args:
        account_balance: 账户余额（USD）
        entry_price: 入场价格
        stop_loss_pct: 止损百分比（小数，如0.02=2%）
        leverage: 杠杆倍数
        max_loss_pct: 最大单笔亏损比例（默认从配置读取）
    
    Returns:
        (position_size_usd, position_size_contracts)
    """
    if max_loss_pct is None:
        max_loss_pct = CONFIG["risk"]["max_loss_per_trade_pct"]
    
    pos_config = CONFIG["position"]
    base_pct = pos_config["base_position_pct"] / 100
    
    # 基础仓位
    base_position = account_balance * base_pct
    
    # 按固定1%账户风险计算仓位
    # 风险敞口 = 仓位 * 杠杆 * 止损%
    # 确保单笔亏损不超过 max_loss_pct * account_balance
    # 仓位(USD) = (max_loss_pct * account_balance) / (leverage * stop_loss_pct)
    
    risk_amount = account_balance * max_loss_pct  # 固定1%风险，如$100,000账户=$1,000
    stop_loss_as_pct = stop_loss_pct  # 止损%，如3%止损=0.03
    
    # 仓位 = 风险金额 / (杠杆 × 止损%)
    # 例：$1,000风险 / (3x杠杆 × 3%止损) = $1,000 / 0.09 = $11,111
    risk_per_dollar = leverage * stop_loss_as_pct
    if risk_per_dollar > 0:
        position = risk_amount / risk_per_dollar
    else:
        position = base_position * leverage
    
    # 限制最大仓位（不超过账户15%）
    max_pos_pct = pos_config["max_position_pct"] / 100
    max_position = account_balance * max_pos_pct
    position = min(position, max_position)
    
    # 同时限制不超过基础仓位的2倍
    position = min(position, base_position * leverage * 2)
    
    # 合约数量（假设BTC，1合约=1 USD保证金则需要/ entry_price）
    # 这里简化为 USD 计价，实际对接交易所时计算合约数
    contracts = position / entry_price if entry_price > 0 else 0
    
    return position, contracts


def calc_gradual_position_size(account_balance: float, entry_price: float,
                               stop_loss_pct: float, leverage: float,
                               current_step: int = 1,
                               max_loss_pct: float = None) -> Dict[str, Any]:
    """
    渐进式仓位计算 - 分步建仓
    
    策略：
    - Step 1: 初始仓位 base_position_pct
    - Step 2: 行情有利时加仓 step_increment_pct
    - Step 3: 再次加仓直到 max_position_pct
    - 每步都有独立的止损点
    
    Args:
        account_balance: 账户余额（USD）
        entry_price: 入场价格
        stop_loss_pct: 止损百分比（小数）
        leverage: 杠杆倍数
        current_step: 当前步数（1, 2, 3）
        max_loss_pct: 最大单笔亏损比例
    
    Returns:
        {
            "position_size": float,      # 仓位大小（USD）
            "contracts": float,          # 合约数量
            "position_pct": float,      # 仓位占比（%）
            "step": int,                 # 当前步数
            "is_final_step": bool,       # 是否最后一步
            "next_step_pct": float,      # 下一步仓位%（用于显示）
            "stop_loss_pct": float       # 本步止损%
        }
    """
    if max_loss_pct is None:
        max_loss_pct = CONFIG["risk"]["max_loss_per_trade_pct"]
    
    pos_config = CONFIG["position"]
    gradual_config = pos_config.get("gradual_steps", {})
    
    if not gradual_config.get("enabled", False):
        # 如果未启用渐进式，返回标准仓位
        position, contracts = calc_position_size(
            account_balance, entry_price, stop_loss_pct, leverage, max_loss_pct
        )
        return {
            "position_size": position,
            "contracts": contracts,
            "position_pct": position / account_balance * 100,
            "step": 1,
            "is_final_step": True,
            "next_step_pct": 0,
            "stop_loss_pct": stop_loss_pct
        }
    
    base_pct = pos_config["base_position_pct"] / 100
    max_pos_pct = pos_config["max_position_pct"] / 100
    max_step = gradual_config.get("max_step", 3)
    step_increment = gradual_config.get("step_increment_pct", 4.5) / 100
    
    # 计算当前步的仓位
    # Step 1: base_pct
    # Step 2: base_pct + step_increment
    # Step 3: min(base_pct + 2*step_increment, max_pos_pct)
    current_step = max(1, min(current_step, max_step))
    
    if current_step == 1:
        target_pct = base_pct
    elif current_step >= max_step:
        target_pct = max_pos_pct
    else:
        # 中间步：逐步增加
        target_pct = min(base_pct + (current_step - 1) * step_increment, max_pos_pct)
    
    # 计算基于风险的仓位
    risk_amount = account_balance * max_loss_pct
    risk_per_dollar = leverage * stop_loss_pct
    if risk_per_dollar > 0:
        risk_based_position = risk_amount / risk_per_dollar
    else:
        risk_based_position = account_balance * target_pct
    
    # 取风险仓位和目标仓位的较小值
    target_position = account_balance * target_pct
    position = min(risk_based_position, target_position)
    
    # 限制最大仓位
    max_position = account_balance * max_pos_pct
    position = min(position, max_position)
    
    # 计算下一步仓位
    if current_step < max_step:
        next_target_pct = min(base_pct + current_step * step_increment, max_pos_pct)
        next_step_pct = (next_target_pct - target_pct) * 100  # 转换为百分比
    else:
        next_step_pct = 0
    
    contracts = position / entry_price if entry_price > 0 else 0
    
    return {
        "position_size": position,
        "contracts": contracts,
        "position_pct": position / account_balance * 100,
        "step": current_step,
        "is_final_step": current_step >= max_step,
        "next_step_pct": next_step_pct,
        "stop_loss_pct": stop_loss_pct
    }

# ===== 止损百分比计算 =====

def calc_stop_loss_pct(entry_price: float, atr: float, account_balance: float,
                       position_value: float, account_risk_pct: float = 0.01) -> float:
    """
    计算止损百分比 - 统一为百分比单位
    核心原则：单笔亏损不超过账户的1%

    Args:
        entry_price: 入场价格
        atr: 平均真实波幅
        account_balance: 账户余额
        position_value: 仓位价值（USD）
        account_risk_pct: 账户风险比例，默认1%

    Returns:
        stop_loss_pct: 止损百分比（小数）
    """
    # 3xATR对应的止损百分比
    atr_stop_pct = (atr * 3) / entry_price if entry_price > 0 else 0.01

    # 账户1%风险对应的止损百分比
    # 仓位价值 × 止损% = 账户风险金额
    # 止损% = (账户余额 × 1%) / 仓位价值
    if position_value > 0:
        account_risk_stop_pct = (account_balance * account_risk_pct) / position_value
    else:
        account_risk_stop_pct = 0.01

    # 取较大值（更保守的保护）
    stop_pct = max(atr_stop_pct, account_risk_stop_pct)

    # 限制最大止损不超过入场价的10%
    return min(stop_pct, 0.10)

# ===== 止损检查 =====

def check_stops(position: Dict, current_price: float, 
                atr: float = None) -> Tuple[bool, str]:
    """
    检查是否触发止损
    
    Args:
        position: 持仓信息字典
        current_price: 当前价格
        atr: ATR值（可选）
    
    Returns:
        (should_exit, reason)
        reason: "none" | "sl" | "tp" | "structure" | "atr"
    """
    direction = position["direction"]
    entry_price = position["entry_price"]
    stop_loss = position["stop_loss"]
    take_profit = position["take_profit"]
    entry_time = position["entry_time"]
    
    risk_config = CONFIG["risk"]
    
    # 价格止损
    if direction == "long":
        if current_price <= stop_loss:
            return True, "sl"
        if current_price >= take_profit:
            return True, "tp"
    else:  # short
        if current_price >= stop_loss:
            return True, "sl"
        if current_price <= take_profit:
            return True, "tp"
    
    # ATR动态止损
    if atr is not None and atr > 0:
        atr_stop = entry_price * (1 - risk_config["atr_stop_multiplier"] * atr / entry_price)
        atr_stop_short = entry_price * (1 + risk_config["atr_stop_multiplier"] * atr / entry_price)
        if direction == "long" and current_price <= atr_stop:
            return True, "atr"
        elif direction == "short" and current_price >= atr_stop_short:
            return True, "atr"
    
    # 动态结构止损: 当持仓超过4h后，使用ATR动态跟踪止损替代时间止损
    # 结构止损原则：不在突破点反向，而是在趋势破坏点退出
    entry_dt = datetime.fromisoformat(entry_time)
    hold_hours = (datetime.now() - entry_dt).total_seconds() / 3600
    if hold_hours >= 4:  # 前4小时用固定SL
        # ATR动态止损：使用1.5倍ATR作为结构止损距离
        if atr is None or atr <= 0:
            # 如果没有ATR，使用价格波动率估算
            price_range = abs(take_profit - entry_price) if take_profit and entry_price else entry_price * 0.02
            atr_stop_distance = price_range * 0.5  # 50%价格范围作为动态止损
        else:
            atr_stop_distance = atr * 1.5
        
        if direction == "long":
            # 多头：结构止损 = 最高价 - 1.5*ATR（追踪高点，只跟踪不回头）
            highest = position.get("highest_price", entry_price)
            wm_key = f"{position.get('instId','')}_long"
            persisted = _get_watermark(wm_key, "highest", highest)
            if current_price > persisted or persisted == entry_price:
                position["highest_price"] = current_price
                _set_watermark(wm_key, "highest", current_price)
                highest = current_price
            else:
                highest = persisted
                position["highest_price"] = persisted
            struct_stop = highest - atr_stop_distance
            if struct_stop > stop_loss:
                if current_price <= struct_stop:
                    return True, "structure"
        else:
            lowest = position.get("lowest_price", entry_price)
            wm_key = f"{position.get('instId','')}_short"
            persisted = _get_watermark(wm_key, "lowest", lowest)
            if current_price < persisted or persisted == entry_price:
                position["lowest_price"] = current_price
                _set_watermark(wm_key, "lowest", current_price)
                lowest = current_price
            else:
                lowest = persisted
                position["lowest_price"] = persisted
            struct_stop = lowest + atr_stop_distance
            if struct_stop < stop_loss:
                if current_price >= struct_stop:
                    return True, "structure"
    
    return False, "none"


# P1 Fix: 结构止损水位线持久化（跨API调用）
_WATERMARK_STORE: Dict[str, Dict] = {}  # {instId_or_key: {"highest": float, "lowest": float}}


def _get_watermark(pos_id: str, key: str, default: float) -> float:
    """获取持久化的结构止损水位线"""
    return _WATERMARK_STORE.get(pos_id, {}).get(key, default)


def _set_watermark(pos_id: str, key: str, value: float) -> None:
    """设置持久化的结构止损水位线"""
    if pos_id not in _WATERMARK_STORE:
        _WATERMARK_STORE[pos_id] = {}
    _WATERMARK_STORE[pos_id][key] = value

# ===== 交易信号格式化 =====

def format_trade_signal(direction: str, entry_price: float,
                        factors: Dict[str, Any], account_balance: float) -> TradeSignal:
    """
    格式化交易信号
    
    Args:
        direction: "long" | "short"
        entry_price: 入场价格
        factors: 因子字典
        account_balance: 账户余额
    
    Returns:
        TradeSignal对象
    """
    risk_config = CONFIG["risk"]
    min_rr = risk_config["min_rr_ratio"]
    atr = factors.get("atr", entry_price * 0.02)
    
    # 计算趋势强度
    trend_strength, level = calc_trend_strength(factors)
    
    # 计算置信度
    confidence = factors["composite_score"]
    
    # 计算杠杆
    leverage, multiplier = calc_leverage(trend_strength, confidence)
    
    # 计算止损止盈
    if direction == "long":
        stop_loss = entry_price * (1 - risk_config["atr_stop_multiplier"] * atr / entry_price)
        # 赔率止损：止盈 = 止损 * RR
        take_profit = entry_price + (entry_price - stop_loss) * min_rr
    else:
        stop_loss = entry_price * (1 + risk_config["atr_stop_multiplier"] * atr / entry_price)
        take_profit = entry_price - (stop_loss - entry_price) * min_rr
    
    # 计算仓位
    stop_loss_pct = abs(entry_price - stop_loss) / entry_price
    position_usd, contracts = calc_position_size(
        account_balance, entry_price, stop_loss_pct, leverage
    )
    
    position_pct = position_usd / account_balance * 100
    
    # 实际RR
    rr = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
    
    return TradeSignal(
        direction=Direction(direction),
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=leverage,
        position_pct=position_pct,
        confidence=confidence,
        factors=factors,
        rr_ratio=rr,
        timestamp=datetime.now().isoformat()
    )

# ===== 交易记录 =====

def log_trade(trade: Trade, log_file: str = None) -> None:
    """
    记录交易到日志文件
    
    Args:
        trade: Trade对象
        log_file: 日志文件路径
    """
    if log_file is None:
        log_file = Path(__file__).parent / "trades.log"
    
    trade_data = asdict(trade)
    trade_data["direction"] = trade.direction.value
    
    with open(log_file, "a") as f:
        f.write(json.dumps(trade_data, ensure_ascii=False) + "\n")
    
    logger.info(f"Trade logged: {trade.id} {trade.direction.value} {trade.pnl:.2f}")

# ===== 自学习更新 =====

def update_factor_weights(trade_history: List[Trade], use_ic: bool = True) -> Dict[str, float]:
    """
    根据交易历史更新因子权重

    规则:
    - 胜率<40%的因子组合 → 自动降权50%
    - 连续5笔盈利 → 因子权重+10%
    
    Args:
        trade_history: 交易历史
        use_ic: 是否优先使用IC动态权重（默认True）
    
    Returns:
        Dict[str, float]: 更新的因子权重
    """
    # 如果启用IC且有IC历史，优先使用IC权重
    if use_ic:
        ic_weights = get_ic_adjusted_weights()
        # 检查是否有有效IC权重
        if ic_weights.get("price_momentum", 0) > 0:
            logger.info(f"使用IC动态权重: {ic_weights}")
            # ic_weights是扁平float字典，需要转换为嵌套结构
            nested_weights = {
                "price_momentum": {"enabled": True, "weight": ic_weights["price_momentum"]},
                "news_sentiment": {"enabled": True, "weight": ic_weights["news_sentiment"]},
                "onchain": {"enabled": True, "weight": ic_weights.get("onchain", 0.1)},
                "wallet": {"enabled": True, "weight": ic_weights.get("wallet", 0.1)}
            }
            update_config("factors", nested_weights)
            return nested_weights
    
    # 否则使用传统的胜率更新逻辑
    if len(trade_history) < 5:
        return copy.deepcopy(CONFIG["factors"])

    recent = trade_history[-5:]
    wins = sum(1 for t in recent if t.pnl > 0)

    # Create a deep copy to avoid mutating global state
    factors = copy.deepcopy(CONFIG["factors"])

    if wins / len(recent) < 0.4:
        # 降权所有因子50%
        for key in factors:
            if "weight" in factors[key]:
                factors[key]["weight"] *= 0.5
        # Normalize weights to sum to 1.0
        total = sum(f["weight"] for f in factors.values() if "weight" in f)
        if total > 0:
            for key in factors:
                if "weight" in factors[key]:
                    factors[key]["weight"] /= total
        logger.warning("Factor weights reduced by 50% due to low win rate")
    elif wins == len(recent):
        # 全胜，加权+10%，上限1.0
        for key in factors:
            if "weight" in factors[key]:
                factors[key]["weight"] = min(factors[key]["weight"] * 1.1, 1.0)
        # Normalize weights to sum to 1.0
        total = sum(f["weight"] for f in factors.values() if "weight" in f)
        if total > 0:
            for key in factors:
                if "weight" in factors[key]:
                    factors[key]["weight"] /= total
        logger.info("Factor weights increased by 10% due to winning streak (capped at 1.0)")

    # 将更新后的因子权重写回全局 CONFIG
    update_config("factors", factors)

    return factors

# ===== 交易频次控制 =====

def can_trade(trade_history: List[Dict], direction: str = None) -> Tuple[bool, str]:
    """
    检查是否可以交易
    
    Returns:
        (can_trade, reason)
    """
    trading = CONFIG["trading"]
    
    # 检查日交易上限
    today = datetime.now().date()
    today_trades = [
        t for t in trade_history 
        if datetime.fromisoformat(t["exit_time"]).date() == today
    ]
    
    if len(today_trades) >= trading["max_trades_per_day"]:
        return False, "daily_limit_reached"
    
    # 检查最小间隔
    if trade_history:
        last_trade_time = datetime.fromisoformat(trade_history[-1]["exit_time"])
        hours_since = (datetime.now() - last_trade_time).total_seconds() / 3600
        if hours_since < trading["min_trade_interval_hours"]:
            return False, f"min_interval_not_met ({hours_since:.1f}h)"
    
    # 检查同方向连续交易
    if direction:
        recent_same_dir = [
            t for t in today_trades 
            if t["direction"] == direction
        ]
        if len(recent_same_dir) >= trading["max_consecutive_same_direction"]:
            return False, "max_consecutive_same_direction"
    
    # 检查连续亏损暂停：验证最近2笔交易之间无盈利交易（确实是连续亏损）
    if len(trade_history) >= 2:
        recent_trades = trade_history[-2:]
        # 只有当最后两笔都是亏损且之间没有盈利时才触发
        if all(t["pnl"] < 0 for t in recent_trades):
            last_loss_time = datetime.fromisoformat(recent_trades[-1]["exit_time"])
            pause_hours = trading["consecutive_loss_pause_hours"]
            if (datetime.now() - last_loss_time).total_seconds() / 3600 < pause_hours:
                return False, "consecutive_loss_pause"
    
    return True, "ok"

# ===== 风险控制 =====

def check_risk_limits(account_balance: float, daily_pnl: float,
                     total_pnl: float, initial_balance: float) -> Tuple[bool, str]:
    """
    检查整体风险限制
    
    Returns:
        (within_limits, reason)
    """
    risk = CONFIG["risk"]
    
    # 日亏损限制
    daily_loss_pct = abs(daily_pnl) / account_balance
    if daily_pnl < 0 and daily_loss_pct >= risk["daily_loss_stop_pct"] / 100:
        return False, "daily_loss_limit"
    
    # 总回撤限制
    drawdown_pct = abs(total_pnl) / initial_balance
    if total_pnl < 0 and drawdown_pct >= risk["total_drawdown_stop_pct"] / 100:
        return False, "total_drawdown_limit"
    
    return True, "ok"

# ===== K线数据获取（币安免费公开API，无需Key）=====

def get_recent_price_data(symbol: str, days: int = 30, interval: str = "1h") -> Dict[str, List[float]]:
    """
    从币安公开API获取K线数据（无需API Key）
    interval: 1m, 5m, 15m, 1h, 4h, 1d
    返回: {"highs": [...], "lows": [...], "closes": [...]}
    """
    import requests

    sym_map = {
        "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
        "AVAX": "AVAXUSDT", "DOGE": "DOGEUSDT", "DOT": "DOTUSDT",
        "LINK": "LINKUSDT", "ADA": "ADAUSDT", "XRP": "XRPUSDT",
        "BNB": "BNBUSDT", "MATIC": "MATICUSDT", "ARB": "ARBUSDT",
    }
    binance_sym = sym_map.get(symbol, f"{symbol}USDT")

    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m",
                    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
    interval_binance = interval_map.get(interval, "1h")

    limit = min(days * 24 if interval == "1h" else days * 24 * 4, 1000)

    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": binance_sym, "interval": interval_binance, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"获取价格数据失败 [{symbol}]: HTTP {resp.status_code}")
            return None

        klines = resp.json()
        if not klines:
            logger.warning(f"获取价格数据失败 [{symbol}]: 空数据")
            return None

        highs = [float(k[2]) for k in klines]   # High
        lows = [float(k[3]) for k in klines]   # Low
        closes = [float(k[4]) for k in klines]  # Close

        return {"highs": highs, "lows": lows, "closes": closes}
    except DataFetchError as ed:
        # HTTP层以下的异常 → 直接抛出（_generate_fallback_price_data同上抛DataFetchError）
        logger.error(f"获取价格数据异常 [{symbol}]: {ed}")
        raise
    except Exception as e:
        logger.warning(f"获取价格数据失败 [{symbol}]: {e}")
        return None


def _generate_fallback_price_data(symbol: str, days: int) -> Dict[str, List[float]]:
    """当API不可用时，直接抛出异常，不返回随机数据"""
    raise DataFetchError("API不可用，无法获取实时行情")


# ===== 账户状态获取（OKX API）=====

def get_account_state(simulated: bool = None) -> Dict[str, Any]:
    """
    从环境变量读取OKX账户状态。
    需要设置: OKX_API_KEY, OKX_SECRET, OKX_PASSPHRASE
    如果未设置，返回模拟数据（仅演示用）。

    Args:
        simulated: 是否模拟盘。默认读环境变量 MIRACLE_SIMULATED_TRADING，
                  未设置时默认为 True（模拟盘）。真实交易时设为 False。
    """
    import base64
    import hashlib
    import hmac
    import os
    from datetime import datetime

    import requests

    api_key = os.getenv("OKX_API_KEY")
    secret = os.getenv("OKX_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")

    # 模拟盘开关：优先参数，其次环境变量，默认模拟盘
    if simulated is None:
        simulated = os.getenv("MIRACLE_SIMULATED_TRADING", "1") != "0"

    if not all([api_key, secret, passphrase]):
        logger.warning("OKX API密钥未设置，返回模拟账户状态")
        return _get_mock_account_state(simulated)

    def _sign(ts, method, path, body=""):
        msg = ts + method + path + body
        m = hmac.new(secret.encode(), msg.encode(), hashlib.sha256)
        return base64.b64encode(m.digest()).decode()

    def _req(method, path, body=""):
        ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')
        h = {
            "OK-ACCESS-KEY": api_key,
            "OK-ACCESS-SIGN": _sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json",
        }
        # 只有模拟盘才加此头，真实交易时去掉
        if simulated:
            h["x-simulated-trading"] = "1"
        r = requests.get(f"https://www.okx.com{path}", headers=h, timeout=10)
        return r.json()

    try:
        bal = _req("GET", "/api/v5/account/balance")
        total_equity = 0.0
        for ccy in bal.get("data", [{}])[0].get("details", []):
            if ccy.get("ccy") == "USDT":
                total_equity = float(ccy.get("eq", 0))
                break

        mode = "模拟盘" if simulated else "真实账户"
        logger.info(f"账户状态获取成功 [{mode}]: ${total_equity:.2f}")
        return {
            "total_equity": total_equity,
            "available_margin": total_equity,
            "position_margin": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "simulated": simulated,
        }
    except requests.exceptions.Timeout:
        logger.error("获取账户状态超时，使用模拟数据")
        _notify_fallback("OKX账户状态获取超时", "网络超时")
        return _get_mock_account_state(simulated)
    except requests.exceptions.HTTPError as e:
        logger.error(f"获取账户状态HTTP错误: {e}，使用模拟数据")
        _notify_fallback("OKX账户状态获取HTTP错误", str(e))
        return _get_mock_account_state(simulated)
    except Exception as e:
        logger.error(f"获取账户状态未知错误: {e}，使用模拟数据")
        _notify_fallback("OKX账户状态获取异常", str(e))
        return _get_mock_account_state(simulated)


def _notify_fallback(reason: str, detail: str):
    """网络异常时飞书通知（静默降级前通知用户）"""
    import os

    import requests as _req
    webhook = os.getenv("MIRACLE_FEISHU_WEBHOOK", os.getenv("FEISHU_WEBHOOK", ""))
    if not webhook:
        return
    try:
        payload = {
            "msg_type": "text",
            "content": {"text": f"[Miracle告警] {reason}\n{detail}\n系统已切换至模拟数据，请检查OKX连接"}
        }
        _req.post(webhook, json=payload, timeout=5)
    except Exception:
        pass


def _get_mock_account_state(simulated: bool = True) -> Dict[str, Any]:
    """模拟账户状态（无API Key时使用）"""
    return {
        "total_equity": 100000.0,
        "available_margin": 100000.0,
        "position_margin": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "simulated": simulated,
    }


# ===== 主入口（示例） =====

if __name__ == "__main__":
    # 示例数据
    import random
    
    prices = [100 + random.uniform(-5, 5) for _ in range(50)]
    closes = prices
    highs = [p + random.uniform(0, 2) for p in prices]
    lows = [p - random.uniform(0, 2) for p in prices]
    
    price_data = {"highs": highs, "lows": lows, "closes": closes}
    
    factors = calc_factors(price_data)
    print("Factors:", json.dumps(factors, indent=2))
    
    trend_strength, level = calc_trend_strength(factors)
    print(f"Trend: {trend_strength}/100 ({level})")
    
    leverage, multiplier = calc_leverage(trend_strength, factors["composite_score"])
    print(f"Leverage: {leverage}x (multiplier: {multiplier})")
    
    signal = format_trade_signal("long", closes[-1], factors, 10000)
    print(f"Signal: {signal}")
