#!/usr/bin/env python3
"""
Miracle IC动态因子权重系统 v1.0
=================================

核心设计（参考Kronos voting_system.py IC权重实现）:
  - 使用Spearman秩相关系数计算IC，对异常值更鲁棒
  - 滚动IC追踪：每月更新一次因子权重
  - 权重更新公式: W_new = 0.7*W_old + 0.3*IC_last_month（指数衰减加权）
  - IC为负的因子权重置0，不参与投票
  - 单因子最大权重30%

与Kronos的差异:
  - Miracle使用更细粒度的子因子（RSI_Signal, RSI_Regime, ADX_Trend, etc.）
  - 额外支持News/Onchain/Wallet等非技术因子
  - 支持与Kronos IC权重共享（通过symlink或配置）

使用方式:
  tracker = MiracleICTracker()
  weights = tracker.get_all_weights()
  ic = tracker.get_ic('RSI')  # 获取某因子当前IC值
  tracker.record_ic('RSI', 0.08)  # 记录新IC
  tracker.compute_weights()  # 重新计算权重
"""

import os
import json
import copy
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ========== 常量定义 ==========

# 缓存文件路径
CACHE_FILE = os.path.expanduser('~/.hermes/miracle_ic_weights.json')

# 滚动窗口（天）
WINDOW_DAYS = 90

# 权重衰减参数
DECAY_ALPHA = 0.7  # 旧权重系数
NEW_WEIGHT_ALPHA = 0.3  # 新IC系数

# 单因子权重上限
MAX_WEIGHT_BTC = 0.15  # BTC因子最大权重（其IC是beta相关性，非方向预测）
MAX_WEIGHT_OTHER = 0.25  # 其他因子最大权重
MAX_WEIGHT_NEWS = 0.20  # 新闻因子最大权重（情绪驱动）

# 因子基准IC（无历史数据时使用）
BASE_ICS = {
    'RSI': 0.08,
    'ADX': 0.05,
    'MACD': 0.05,
    'Bollinger': 0.06,
    'Vol': 0.07,
    'BTC': 0.04,
    'Momentum': 0.06,
    'Trend': 0.05,
    'News': 0.03,
    'Onchain': 0.02,
    'Wallet': 0.02,
}


# ========== IC追踪器 ==========

class MiracleICTracker:
    """
    Miracle IC动态权重追踪器
    
    与Kronos ICTracker的对应关系:
    - Kronos因子: RSI, ADX, Bollinger, Vol, MACD, BTC, Gemma
    - Miracle因子: RSI, ADX, MACD, Bollinger, Vol, BTC, Momentum, Trend, News, Onchain, Wallet
    
    主要差异:
    - Miracle有更细粒度的子因子分解
    - 支持非技术因子（News, Onchain, Wallet）
    - 可选：与Kronos IC系统共享数据
    """

    # 类变量：跨实例共享
    _ic_history: Dict[str, List[Dict]] = {}
    _weights: Dict[str, float] = {}
    _last_update: Optional[str] = None
    _initialized: bool = False

    def __init__(self, sync_with_kronos: bool = True):
        """
        初始化IC追踪器
        
        Args:
            sync_with_kronos: 是否与Kronos IC系统同步（共享IC历史）
        """
        self.sync_with_kronos = sync_with_kronos
        self._load_cache()
        
    def _load_cache(self):
        """从磁盘加载IC历史和权重"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE) as f:
                    data = json.load(f)
                MiracleICTracker._ic_history = data.get('ic_history', {})
                MiracleICTracker._weights = data.get('weights', {})
                MiracleICTracker._last_update = data.get('last_update')
                MiracleICTracker._initialized = True
            except Exception as e:
                print(f"[ICTracker] 加载缓存失败: {e}")
        
        # 如果启用了Kronos同步，从Kronos加载IC历史
        if self.sync_with_kronos and not MiracleICTracker._ic_history:
            self._sync_from_kronos()

    def _sync_from_kronos(self):
        """从Kronos IC系统同步数据"""
        kronos_cache = os.path.expanduser('~/.hermes/kronos_ic_weights.json')
        if not os.path.exists(kronos_cache):
            return
            
        try:
            with open(kronos_cache) as f:
                kronos_data = json.load(f)
            
            # 映射Kronos因子名到Miracle因子名
            name_map = {
                'RSI': 'RSI',
                'ADX': 'ADX', 
                'Bollinger': 'Bollinger',
                'Vol': 'Vol',
                'MACD': 'MACD',
                'BTC': 'BTC',
                'Gemma': 'News',  # Gemma -> News（情绪分析）
            }
            
            kronos_history = kronos_data.get('ic_history', {})
            for kronos_name, history in kronos_history.items():
                miracle_name = name_map.get(kronos_name, kronos_name)
                if miracle_name not in MiracleICTracker._ic_history:
                    MiracleICTracker._ic_history[miracle_name] = history
                else:
                    # 合并历史（保留两者的并集）
                    existing_months = {h['month'] for h in MiracleICTracker._ic_history[miracle_name]}
                    for h in history:
                        if h['month'] not in existing_months:
                            MiracleICTracker._ic_history[miracle_name].append(h)
            
            # 同步权重（如果有）
            kronos_weights = kronos_data.get('weights', {})
            if kronos_weights and not MiracleICTracker._weights:
                for kronos_name, weight in kronos_weights.items():
                    miracle_name = name_map.get(kronos_name, kronos_name)
                    MiracleICTracker._weights[miracle_name] = weight
                    
            print(f"[ICTracker] 从Kronos同步IC数据完成")
        except Exception as e:
            print(f"[ICTracker] Kronos同步失败: {e}")

    def _save_cache(self):
        """持久化到磁盘"""
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        data = {
            'ic_history': MiracleICTracker._ic_history,
            'weights': MiracleICTracker._weights,
            'last_update': MiracleICTracker._last_update,
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def spearman_ic(signal_values: np.ndarray, future_returns: np.ndarray) -> float:
        """
        计算Spearman秩相关系数（IC值）
        
        Args:
            signal_values: 因子信号值序列（如RSI）
            future_returns: 未来收益率序列
            
        Returns:
            IC值: -1到1之间的相关系数
        """
        if len(signal_values) < 30 or len(future_returns) < 30:
            return 0.0
        if len(signal_values) != len(future_returns):
            min_len = min(len(signal_values), len(future_returns))
            signal_values = signal_values[-min_len:]
            future_returns = future_returns[-min_len:]

        # 去除NaN和Inf
        mask = ~(np.isnan(signal_values) | np.isnan(future_returns) | 
                  np.isinf(signal_values) | np.isinf(future_returns))
        sv = signal_values[mask]
        fr = future_returns[mask]
        if len(sv) < 30:
            return 0.0

        # Spearman: 用rank替代原始值
        def rank(x):
            order = np.argsort(np.argsort(x))
            return order / (len(x) - 1) * 2 - 1  # 归一化到[-1, 1]

        rank_sig = rank(sv)
        rank_ret = rank(fr)
        corr = np.corrcoef(rank_sig, rank_ret)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def record_ic(self, factor_name: str, ic_value: float):
        """
        记录某个因子本月的IC
        每月调用一次即可
        
        Args:
            factor_name: 因子名称
            ic_value: IC值
        """
        month_key = datetime.now().strftime('%Y-%m')

        if factor_name not in MiracleICTracker._ic_history:
            MiracleICTracker._ic_history[factor_name] = []

        # 更新或追加本月IC
        existing = [i for i in MiracleICTracker._ic_history[factor_name] if i['month'] == month_key]
        if existing:
            existing[0]['ic'] = ic_value
        else:
            MiracleICTracker._ic_history[factor_name].append({'month': month_key, 'ic': ic_value})

        # 只保留最近WINDOW_DAYS天的历史
        cutoff = (datetime.now() - timedelta(days=WINDOW_DAYS)).strftime('%Y-%m')
        MiracleICTracker._ic_history[factor_name] = [
            i for i in MiracleICTracker._ic_history[factor_name] if i['month'] >= cutoff
        ]

    def compute_weights(self) -> Dict[str, float]:
        """
        根据IC历史计算动态权重
        
        算法（与Kronos一致）:
        1. 对每个因子，计算指数衰减加权平均IC
        2. 新权重 = 0.7*旧权重 + 0.3*最新IC
        3. IC为负的因子权重置0
        4. 归一化使权重和=1
        5. 应用权重上限
        
        Returns:
            Dict[str, float]: 因子权重字典
        """
        weights = {}
        old_weights = dict(MiracleICTracker._weights)

        # 第一步：计算每个因子的指数衰减加权IC
        for factor, history in MiracleICTracker._ic_history.items():
            if not history:
                weights[factor] = 0.0
                continue

            # 取最近几个月的IC，用指数衰减计算加权平均
            recent_ics = [h['ic'] for h in history[-6:]]  # 最多6个月
            old_weight = old_weights.get(factor, 0.1)
            latest_ic = recent_ics[-1]

            # 指数衰减加权平均（更重视近期）
            decay_weights = [DECAY_ALPHA ** i for i in range(len(recent_ics))]
            decay_weights = [w / sum(decay_weights) for w in decay_weights]
            weighted_ic = sum(ic * w for ic, w in zip(recent_ics, decay_weights))

            # 新权重 = 0.7*旧权重 + 0.3*最新IC（平滑）
            new_ic_weight = DECAY_ALPHA * old_weight + NEW_WEIGHT_ALPHA * max(0, latest_ic)
            weights[factor] = new_ic_weight

        # 第二步：IC为负的因子权重置0
        for factor, history in MiracleICTracker._ic_history.items():
            if history and history[-1]['ic'] < 0:
                weights[factor] = 0.0

        # 第三步：归一化
        positive_weights = {k: v for k, v in weights.items() if v > 0}
        total = sum(positive_weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        # 第四步：应用权重上限
        weights = self._apply_weight_caps(weights, old_weights)

        MiracleICTracker._weights = weights
        MiracleICTracker._last_update = datetime.now().strftime('%Y-%m-%d %H:%M')
        self._save_cache()
        return weights

    def _apply_weight_caps(self, weights: Dict[str, float], 
                          old_weights: Dict[str, float]) -> Dict[str, float]:
        """
        应用权重上限，防止单一因子主导
        
        Args:
            weights: 当前计算的权重
            old_weights: 上一次的有效权重（用于计算释放量）
            
        Returns:
            应用上限后的权重
        """
        # 确定每个因子的上限
        def get_cap(factor: str) -> float:
            if factor == 'BTC':
                return MAX_WEIGHT_BTC
            elif factor in ('News', 'Sentiment'):
                return MAX_WEIGHT_NEWS
            else:
                return MAX_WEIGHT_OTHER

        # 记录每个因子cap前的值
        pre_cap = dict(weights)

        # 应用所有cap
        for k in weights:
            cap = get_cap(k)
            weights[k] = min(weights[k], cap)

        # 计算释放量
        excess = sum(pre_cap[k] - weights[k] for k in weights)

        # 找出可以吸收释放量的因子：cap前值 < cap值（说明它没被cap）
        absorbable = {k: get_cap(k) - pre_cap[k]
                      for k in weights if pre_cap[k] < get_cap(k)}
        absorbable_total = sum(absorbable.values())

        # 按可吸收量比例分配释放量
        if absorbable_total > 0 and excess > 0:
            for k in absorbable:
                cap = get_cap(k)
                boost = excess * (absorbable[k] / absorbable_total)
                weights[k] = min(weights[k] + boost, cap)

        return weights

    def get_weight(self, factor_name: str) -> float:
        """获取某因子的当前权重"""
        return MiracleICTracker._weights.get(factor_name, 0.0)

    def get_ic(self, factor_name: str) -> float:
        """获取某因子的当前IC值"""
        history = MiracleICTracker._ic_history.get(factor_name, [])
        if history:
            return history[-1]['ic']
        return BASE_ICS.get(factor_name, 0.0)

    def get_all_weights(self) -> Dict[str, float]:
        """获取所有因子的当前权重"""
        return dict(MiracleICTracker._weights)

    def get_all_ics(self) -> Dict[str, float]:
        """获取所有因子的当前IC值"""
        result = {}
        for factor, history in MiracleICTracker._ic_history.items():
            if history:
                result[factor] = history[-1]['ic']
            else:
                result[factor] = BASE_ICS.get(factor, 0.0)
        return result

    def get_ic_history(self, factor_name: str) -> List[Dict]:
        """获取某因子的IC历史"""
        return list(MiracleICTracker._ic_history.get(factor_name, []))


# ========== 因子IC计算工具 ==========

def compute_factor_ic_for_miracle(
    closes: List[float],
    highs: List[float] = None,
    lows: List[float] = None,
    volumes: List[float] = None
) -> Dict[str, float]:
    """
    为Miracle计算所有因子的IC值
    
    Args:
        closes: 价格列表
        highs: 最高价列表
        lows: 最低价列表
        volumes: 成交量列表
        
    Returns:
        Dict[str, float]: 因子IC字典
    """
    results = {}
    closes = np.array(closes)
    n = len(closes)
    
    if n < 30:
        return {k: 0.0 for k in BASE_ICS.keys()}
    
    # 计算未来收益率
    future_returns = np.diff(closes)  # n-1 个收益
    n = len(future_returns)  # = len(closes) - 1
    
    def safe_spearman(signal: np.ndarray) -> float:
        """安全的Spearman IC计算"""
        s = signal[-n:] if len(signal) > n else signal
        mask = ~(np.isnan(s) | np.isnan(future_returns) | np.isinf(s) | np.isinf(future_returns))
        if mask.sum() < 30:
            return 0.0
        s, t = s[mask], future_returns[mask]
        corr = np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(t)))[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    # RSI IC
    rsi_vals = _calc_rsi_series(closes)
    results['RSI'] = safe_spearman(rsi_vals)

    # ADX IC
    if highs is not None and lows is not None:
        adx_vals = _calc_adx_series(np.array(highs), np.array(lows), closes)
        results['ADX'] = safe_spearman(adx_vals)
    else:
        results['ADX'] = 0.0

    # MACD IC
    macd_hist = _calc_macd_series(closes)
    results['MACD'] = safe_spearman(macd_hist)

    # Bollinger IC（价格与布林带位置）
    bb_lower, bb_mid, bb_upper = _calc_bb_series(closes)
    bb_pos = (closes[-n:] - bb_lower[-n:]) / (bb_upper[-n:] - bb_lower[-n:] + 1e-10)
    results['Bollinger'] = safe_spearman(bb_pos)

    # Vol IC（成交量比率）
    if volumes is not None:
        volumes = np.array(volumes)
        vol_ma = np.convolve(volumes, np.ones(20)/20, mode='same')
        vol_ratio = volumes / (vol_ma + 1e-10)
        results['Vol'] = safe_spearman(vol_ratio[-n:])
    else:
        results['Vol'] = 0.0

    # Momentum IC
    momentum = np.diff(closes, prepend=closes[0])
    results['Momentum'] = safe_spearman(momentum)

    # Trend IC（用简单移动平均斜率）
    if n >= 20:
        ma = np.convolve(closes, np.ones(20)/20, mode='same')
        trend = np.diff(ma, prepend=ma[0])
        results['Trend'] = safe_spearman(trend)
    else:
        results['Trend'] = 0.0

    # News/Onchain/Wallet 默认IC（需要外部数据）
    results['News'] = BASE_ICS.get('News', 0.03)
    results['Onchain'] = BASE_ICS.get('Onchain', 0.02)
    results['Wallet'] = BASE_ICS.get('Wallet', 0.02)
    
    # BTC IC（如果有）
    results['BTC'] = BASE_ICS.get('BTC', 0.04)

    return results


# ========== 辅助函数 ==========

def _calc_rsi_series(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """计算RSI序列"""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gains = np.convolve(gains, np.ones(period)/period, mode='same')
    avg_losses = np.convolve(losses, np.ones(period)/period, mode='same')
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_adx_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """计算ADX序列（简化版）"""
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    plus_dm = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    atr = np.convolve(tr, np.ones(period)/period, mode='same')
    plus_di = 100 * np.convolve(plus_dm, np.ones(period)/period, mode='same') / (atr + 1e-10)
    minus_di = 100 * np.convolve(minus_dm, np.ones(period)/period, mode='same') / (atr + 1e-10)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = np.convolve(dx, np.ones(period)/period, mode='same')
    return adx


def _calc_macd_series(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """计算MACD直方图序列"""
    import pandas as pd
    s = pd.Series(prices)
    ema12 = s.ewm(span=fast).mean().values
    ema26 = s.ewm(span=slow).mean().values
    macd = ema12 - ema26
    signal_line = pd.Series(macd).ewm(span=signal).mean().values
    hist = macd - signal_line
    return hist


def _calc_bb_series(prices: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算布林带序列"""
    import pandas as pd
    s = pd.Series(prices)
    mid = s.rolling(period).mean().values
    std = s.rolling(period).std().values
    upper = mid + 2 * std
    lower = mid - 2 * std
    return lower, mid, upper


# ========== 自检 ==========

if __name__ == '__main__':
    print("=== Miracle IC动态因子权重系统自检 ===")
    print(f"缓存文件: {CACHE_FILE}")
    
    # 测试权重加载
    tracker = MiracleICTracker()
    weights = tracker.get_all_weights()
    ics = tracker.get_all_ics()
    
    print(f"\n当前因子权重:")
    for factor, w in sorted(weights.items(), key=lambda x: -x[1]):
        ic = ics.get(factor, 0)
        print(f"  {factor:12s} 权重={w:.2%}  IC={ic:+.4f}")
    
    if not weights:
        print("\n无历史权重，使用基准权重:")
        for factor, ic in BASE_ICS.items():
            print(f"  {factor:12s}  IC={ic:.4f}")
    
    print("\n=== 自检完成 ===")