"""
Miracle 1.0.1 - 核心交易引擎
高频趋势跟踪+事件驱动混合系统
"""

import copy
import json
import math
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

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
        from core.ic_weights import MiracleICTracker
        
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
        from core.ic_weights import MiracleICTracker
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
    with open(config_path, "r") as f:
        return json.load(f)

CONFIG = load_config()

# ===== 日志配置 =====

logger = logging.getLogger("miracle")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(Path(__file__).parent / "miracle.log")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

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

# ===== 因子计算 =====

def calc_rsi(prices: List[float], period: int = 14) -> float:
    """
    RSI (Relative Strength Index) 计算 — 正确实现Wilder平滑
    简化均值会导致RSI偏大/偏小，真实策略应使用Wilder递推。
    """
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    # Wilder平滑 — 递归形式
    alpha = 1.0 / period
    avg_gain = float(gains[0])
    avg_loss = float(losses[0])
    for i in range(1, len(gains)):
        avg_gain = avg_gain + alpha * (gains[i] - avg_gain)
        avg_loss = avg_loss + alpha * (losses[i] - avg_loss)

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float, float]:
    """
    计算ADX及+DI/-DI，返回(adx, plus_di, minus_di)
    
    正确实现Wilder平滑：
    1. ATR, +DI, -DI 全部用Wilder平滑（S_t = S_{t-1}*(n-1)/n + v_t/n）
    2. ADX = Wilder平滑(DX)
    
    参考: Wilder, J. Welles. "New Concepts in Technical Trading Systems" (1978)
    """
    # 需要至少 2*period 根K线数据才能正确初始化+递推
    min_required = 2 * period
    if len(closes) < min_required:
        return 25.0, 25.0, 25.0
    
    n = len(closes)
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []
    
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
        up = highs[i] - highs[i-1]
        dn = lows[i-1] - lows[i]
        if up > dn and up > 0:
            plus_dm_list.append(up)
            minus_dm_list.append(0)
        elif dn > up and dn > 0:
            plus_dm_list.append(0)
            minus_dm_list.append(dn)
        else:
            plus_dm_list.append(0)
            minus_dm_list.append(0)
    
    # ── Wilder平滑初始化 ──
    alpha = 1.0 / period
    # 第一个period的简单均值作为初始值
    atr = sum(tr_list[:period]) / period
    plus_di_smooth = sum(plus_dm_list[:period]) / period
    minus_di_smooth = sum(minus_dm_list[:period]) / period
    
    # ── Wilder递推（从第period个TR开始） ──
    for i in range(period, len(tr_list)):
        atr = atr * (1 - alpha) + tr_list[i] * alpha
        plus_di_smooth = plus_di_smooth * (1 - alpha) + plus_dm_list[i] * alpha
        minus_di_smooth = minus_di_smooth * (1 - alpha) + minus_dm_list[i] * alpha
    
    # 计算每个时间点的DX值并收集
    dx_values = []
    plus_di_series = []
    minus_di_series = []

    # 重新用Wilder平滑计算每个时间点的DX值
    atr = sum(tr_list[:period]) / period
    plus_di_smooth = sum(plus_dm_list[:period]) / period
    minus_di_smooth = sum(minus_dm_list[:period]) / period

    for i in range(period, len(tr_list)):
        atr = atr * (1 - alpha) + tr_list[i] * alpha
        plus_di_smooth = plus_di_smooth * (1 - alpha) + plus_dm_list[i] * alpha
        minus_di_smooth = minus_di_smooth * (1 - alpha) + minus_dm_list[i] * alpha

        di_sum = plus_di_smooth + minus_di_smooth
        if di_sum == 0 or atr == 0:
            dx_values.append(0.0)
            plus_di_series.append(0.0)
            minus_di_series.append(0.0)
        else:
            # +DI = 100 × (+DM_smoothed / ATR)
            # -DI = 100 × (-DM_smoothed / ATR)
            # DX = 100 × |+DI - -DI| / (+DI + -DI)
            # 代入后: DX = 100 × |+DM - -DM| / (+DM + -DM)
            # 注意：这里直接用smoothed DM值，不需要再÷atr
            plus_di_pct = 100 * plus_di_smooth / atr
            minus_di_pct = 100 * minus_di_smooth / atr
            dx_i = 100 * abs(plus_di_smooth - minus_di_smooth) / (plus_di_smooth + minus_di_smooth)
            dx_values.append(dx_i)
            plus_di_series.append(plus_di_pct)
            minus_di_series.append(minus_di_pct)

    if len(dx_values) == 0:
        return 0.0, 0.0, 0.0

    # ADX = Wilder平滑(DX) — 只需要period次递推即可收敛
    # 第一个ADX用DX的简单均值初始化
    if len(dx_values) >= period:
        adx = sum(dx_values[:period]) / period
    else:
        adx = sum(dx_values) / len(dx_values)

    # 后续ADX用Wilder递推（只进行period次，这是Wilder的标准做法）
    for i in range(period):
        if i < len(dx_values):
            adx = adx * (1 - alpha) + dx_values[i] * alpha

    # 返回最终的ADX和最后一组DI值
    plus_di_final = plus_di_series[-1] if plus_di_series else 0.0
    minus_di_final = minus_di_series[-1] if minus_di_series else 0.0

    return adx, plus_di_final, minus_di_final

def calc_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """计算MACD，返回(macd, signal, histogram)"""
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0
    
    def ema(data, n):
        k = 2 / (n + 1)
        result = [data[0]]
        for v in data[1:]:
            result.append(result[-1] * (1 - k) + v * k)
        return result
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(ema_fast))]
    signal_line = ema(macd_line, signal)
    
    macd = macd_line[-1]
    sig = signal_line[-1]
    hist = macd - sig
    return macd, sig, hist

def calc_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """
    计算ATR（平均真实波幅）— 正确实现Wilder平滑

    Wilder ATR = ATR_prev * (period-1)/period + TR_current / period
    不是简单均值（简单均值会使ATR偏小，导致止损设得太紧）

    正确实现：从period开始递推
    """
    n = len(closes)
    if n < period + 1:
        return (max(highs) - min(lows)) / min(lows) if min(lows) > 0 else 0.01

    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)

    # ── Wilder平滑 ──
    alpha = 1.0 / period
    # 第一个period用简单均值初始化
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0.0

    atr = sum(trs[:period]) / period  # 初始值

    # 从第period个元素开始递推（索引period到len(trs)-1，共len(trs)-period个）
    for i in range(period, len(trs)):
        atr = atr * (1 - alpha) + trs[i] * alpha

    return atr

def calc_onchain_metrics() -> Dict[str, float]:
    """
    计算链上因子（STUB - 需对接真实API）
    
    WARNING: 此函数为STUB实现，返回全零值。
    当前权重贡献已被静默设为0，避免污染综合得分。
    TODO: 对接链上数据API（OKX/Binance）
    """
    logger.warning("calc_onchain_metrics() called - STUB implementation returning zeros. Weight contribution disabled.")
    return {
        "exchange_flow": 0.0,   # 交易所净流量
        "large_transfer": 0.0   # 大额转账标记
    }

def normalize_macd_histogram(hist: float, price: float) -> float:
    """
    MACD直方图归一化 - 正确的归一化方法

    归一化到 -1~1 范围，表示多空动能强度

    Args:
        hist: MACD直方图值（MACD线 - Signal线）
        price: 当前价格

    Returns:
        normalized: 归一化后的值 (-1~1)
    """
    if price <= 0:
        return 0.0

    # 使用价格的一定比例作为归一化基准
    # 通常MACD直方图超过价格的0.5%被认为是强信号
    threshold = price * 0.005  # 0.5%的价格变动作为阈值

    if abs(hist) < threshold:
        return 0.0  # 低于阈值视为中性

    # 映射到 -1~1
    normalized = max(-1.0, min(1.0, hist / (threshold * 10)))
    return normalized

def calc_combined_score(price_score: float, news_score: float,
                        onchain_score: float, wallet_score: float,
                        weights: Dict[str, float] = None) -> float:
    """
    多因子融合 - 明确定义公式

    融合方法：加权几何平均 + 方向一致性修正

    Args:
        price_score: 价格因子得分 (-1~1)
        news_score: 新闻因子得分 (-1~1)
        onchain_score: 链上因子得分 (-1~1)
        wallet_score: 钱包因子得分 (-1~1)
        weights: 因子权重

    Returns:
        combined_score: 综合得分 (-1~1)
    """
    if weights is None:
        weights = {"price": 0.6, "news": 0.2, "onchain": 0.1, "wallet": 0.1}

    # 1. 加权算术平均（基础得分）
    weighted_sum = (
        price_score * weights["price"] +
        news_score * weights["news"] +
        onchain_score * weights["onchain"] +
        wallet_score * weights["wallet"]
    )

    # 2. 方向一致性修正
    # 计算方向一致的因子数量
    scores = [price_score, news_score, onchain_score, wallet_score]

    # 计算方向一致性因子（0.5~1.0）
    # 如果所有因子同向，一致性=1.0；如果完全矛盾，一致性=0.5
    sign_consistency = 0.5 + 0.5 * abs(sum(s/abs(s) if s != 0 else 0 for s in scores)) / len(scores)

    # 3. 最终得分 = 加权平均 × 一致性修正
    # 如果方向矛盾（如价格多头+新闻空头），降低置信度
    combined = weighted_sum * sign_consistency

    return max(-1.0, min(1.0, combined))

def calc_wallet_metrics() -> Dict[str, float]:
    """
    计算钱包分布因子（STUB - 需对接真实API）
    
    WARNING: 此函数为STUB实现，返回全零值。
    当前权重贡献已被静默设为0，避免污染综合得分。
    TODO: 对接区块链浏览器API
    """
    logger.warning("calc_wallet_metrics() called - STUB implementation returning zeros. Weight contribution disabled.")
    return {
        "holder_concentration": 0.0  # 持币地址集中度
    }

def calc_news_sentiment() -> float:
    """
    计算新闻情绪（STUB - 需对接新闻API）
    
    WARNING: 此函数为STUB实现，返回零（中性）。
    当前权重贡献已被静默设为0，避免污染综合得分。
    TODO: 对接新闻情绪分析API
    返回: 1.0=强烈利好, 0.0=中性, -1.0=强烈利空
    """
    logger.warning("calc_news_sentiment() called - STUB implementation returning 0.0 (neutral). Weight contribution disabled.")
    return 0.0

def calc_factors(price_data: Dict, onchain_data: Optional[Dict] = None, 
                 news_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    计算所有因子值
    
    Args:
        price_data: 包含 highs, lows, closes 的字典
        onchain_data: 链上数据（可选）
        news_data: 新闻数据（可选）
    
    Returns:
        因子字典
    """
    highs = price_data.get("highs", [])
    lows = price_data.get("lows", [])
    closes = price_data.get("closes", [])
    
    # 价格动量因子 (60%)
    rsi = calc_rsi(closes)
    adx, plus_di, minus_di = calc_adx(highs, lows, closes)
    macd, signal, hist = calc_macd(closes)
    atr = calc_atr(highs, lows, closes)
    
    # 归一化因子到0-100
    rsi_norm = rsi  # RSI本身0-100
    adx_norm = min(adx, 100)  # ADX 0-100
    # MACD归一化：使用正确的归一化方法，将直方图映射到-1~1
    macd_norm_float = normalize_macd_histogram(hist, closes[-1]) if closes[-1] > 0 else 0.0
    macd_norm = 50 + macd_norm_float * 50  # -1~1 -> 0~100
    
    price_score = (rsi_norm * 0.33 + adx_norm * 0.34 + macd_norm * 0.33)
    
    # 新闻情绪因子 (20%)
    news_sentiment = news_data.get("sentiment", 0.0) if news_data else calc_news_sentiment()
    news_score = (news_sentiment + 1) * 50  # -1~1 -> 0~100
    
    # 链上因子 (10%)
    onchain = onchain_data if onchain_data else calc_onchain_metrics()
    exchange_flow = onchain.get("exchange_flow", 0.0)
    large_transfer = onchain.get("large_transfer", 0.0)
    onchain_score = 50 + (exchange_flow * 20 + large_transfer * 10)
    
    # 钱包因子 (10%)
    wallet = calc_wallet_metrics()
    holder_conc = wallet.get("holder_concentration", 0.5)
    wallet_score = holder_conc * 100
    
    # 加权综合得分
    factors = CONFIG["factors"]
    price_weight = factors["price_momentum"]["weight"]
    news_weight = factors["news_sentiment"]["weight"]
    onchain_weight = factors["onchain"]["weight"]
    wallet_weight = factors["wallet"]["weight"]
    
    composite = (
        price_score * price_weight +
        news_score * news_weight +
        onchain_score * onchain_weight +
        wallet_score * wallet_weight
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
        "composite_score": composite,
        "weights": {
            "price": price_weight,
            "news": news_weight,
            "onchain": onchain_weight,
            "wallet": wallet_weight
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

def calc_leverage(trend_strength: int, confidence: float) -> Tuple[float, float]:
    """
    根据趋势强度和置信度计算杠杆
    
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
        reason: "none" | "sl" | "tp" | "time" | "atr" | "daily_loss"
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
        if direction == "long" and current_price <= atr_stop:
            return True, "atr"
        elif direction == "short" and current_price >= atr_stop:
            return True, "atr"
    
    # 时间止损
    max_hours = risk_config["max_hold_hours"]
    entry_dt = datetime.fromisoformat(entry_time)
    if datetime.now() - entry_dt > timedelta(hours=max_hours):
        return True, "time"
    
    return False, "none"

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
            return ic_weights
    
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
        logger.warning("Factor weights reduced by 50% due to low win rate")
    elif wins == len(recent):
        # 全胜，加权+10%，上限1.0
        for key in factors:
            if "weight" in factors[key]:
                factors[key]["weight"] = min(factors[key]["weight"] * 1.1, 1.0)
        logger.info("Factor weights increased by 10% due to winning streak (capped at 1.0)")

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
    
    # 检查连续亏损暂停
    recent_losses = trade_history[-2:] if len(trade_history) >= 2 else []
    if recent_losses and all(t["pnl"] < 0 for t in recent_losses):
        last_loss_time = datetime.fromisoformat(recent_losses[-1]["exit_time"])
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
        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": binance_sym, "interval": interval_binance, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"获取价格数据失败 [{symbol}]: HTTP {resp.status_code}，使用Fallback数据")
            return _generate_fallback_price_data(symbol, days)

        klines = resp.json()
        if not klines:
            logger.warning(f"获取价格数据失败 [{symbol}]: 空数据，使用Fallback数据")
            return _generate_fallback_price_data(symbol, days)

        highs = [float(k[2]) for k in klines]   # High
        lows = [float(k[3]) for k in klines]   # Low
        closes = [float(k[4]) for k in klines]  # Close

        return {"highs": highs, "lows": lows, "closes": closes}
    except DataFetchError as ed:
        # HTTP层以下的异常（如网络超时/DNS），走Fallback
        logger.warning(f"获取价格数据异常 [{symbol}]，使用Fallback数据: {ed}")
        return _generate_fallback_price_data(symbol, days)
    except Exception as e:
        logger.warning(f"获取价格数据失败 [{symbol}]: {e}，使用Fallback数据")
        return _generate_fallback_price_data(symbol, days)


def _generate_fallback_price_data(symbol: str, days: int) -> Dict[str, List[float]]:
    """
    当API不可用时，使用随机游走生成伪K线数据（仅用于测试/演示）。
    真实环境不应使用此数据。
    """
    import random
    n = days * 24
    base_price = {
        "BTC": 95000, "ETH": 3500, "SOL": 145, "AVAX": 38,
        "DOGE": 0.095, "DOT": 7.5, "LINK": 14.5, "ADA": 0.75,
        "XRP": 2.2, "BNB": 580, "MATIC": 0.85, "ARB": 1.1,
    }.get(symbol, 100.0)

    random.seed(symbol.encode().__hash__() if symbol else 0)
    import random as _r
    closes = [base_price]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + _r.gauss(0, 0.015)))  # 随机游走，μ=0，σ=1.5%/步
    highs = [c * (1 + abs(_r.uniform(0, 0.008))) for c in closes]
    lows = [c * (1 - abs(_r.uniform(0, 0.008))) for c in closes]
    random.seed()

    return {"highs": highs, "lows": lows, "closes": closes}


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
    import os
    import requests
    import hmac
    import base64
    import hashlib
    from datetime import datetime

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
