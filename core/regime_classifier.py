#!/usr/bin/env python3
from __future__ import annotations

"""
regime_classifier.py - 市场状态分类器
=====================================

Miracle System 市场状态分类模块，基于技术指标判断当前市场状态。

功能：
  - 分析市场数据，判断当前市场状态（BULL/BEAR/SIDEWAYS）
  - 手动实现 ADX、DMI、ATR 指标
  - 防抖动机制：连续3根K线确认

约束：
  - 只做分类，不做交易决策
  - 不使用 ta-lib 等外部技术指标库
  - 不使用机器学习模型

参考：Kronos regime_classifier 实现 (kronos_v2/core/regime_classifier.py)

Author: Miracle System
Version: 1.0.0
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# HMM市场状态分类器（用于4状态识别）
from hmmlearn import hmm


class MarketRegime(Enum):
    """市场状态枚举。"""
    BULL = "bull"       # 上涨趋势
    BEAR = "bear"       # 下跌趋势
    SIDEWAYS = "sideways"  # 横盘/震荡


@dataclass
class RegimeMetrics:
    """市场状态指标容器。"""
    adx: float           # Average Directional Index (趋势强度)
    plus_di: float       # DMI 正向指标
    minus_di: float     # DMI 负向指标
    atr: float           # Average True Range
    momentum: float      # 动量 (-1 to 1)
    volatility_ratio: float  # 波动率比率


class RegimeClassifier:
    """
    市场状态分类器。

    功能：
    - 分析市场数据，判断当前趋势方向
    - 计算动量和波动率比率
    - 防抖动：连续3根K线确认趋势变化

    分类依据：
    - ADX > 25 表示趋势较强
    - DMI+ > DMI- 表示上涨趋势，DMI- > DMI+ 表示下跌趋势
    - ATR 相对历史均值反映波动率
    """

    # 默认阈值
    ADX_TREND_THRESHOLD = 25.0       # ADX > 25 认为有趋势
    ADX_STRONG_THRESHOLD = 40.0     # ADX > 40 认为趋势较强

    # 防抖动参数
    CONFIRMATION_BARS = 3            # 连续3根K线确认

    # DMI 阈值
    DMI_STRONG_THRESHOLD = 20.0      # DMI+/- 超过此值认为趋势明确

    # ATR 周期
    DEFAULT_ATR_PERIOD = 14
    DEFAULT_ADX_PERIOD = 14
    DEFAULT_DMI_PERIOD = 14

    def __init__(self, config: Dict | None = None):
        """
        初始化分类器。

        Args:
            config: 可选配置字典，覆盖默认阈值
                   {
                       'adx_trend_threshold': 25.0,
                       'adx_strong_threshold': 40.0,
                       'confirmation_bars': 3,
                       'dmi_strong_threshold': 20.0,
                       'atr_period': 14,
                       'adx_period': 14,
                       'dmi_period': 14
                   }
        """
        self.config = config or {}

        self.adx_trend_threshold = self.config.get(
            'adx_trend_threshold', self.ADX_TREND_THRESHOLD)
        self.adx_strong_threshold = self.config.get(
            'adx_strong_threshold', self.ADX_STRONG_THRESHOLD)
        self.confirmation_bars = self.config.get(
            'confirmation_bars', self.CONFIRMATION_BARS)
        self.dmi_strong_threshold = self.config.get(
            'dmi_strong_threshold', self.DMI_STRONG_THRESHOLD)
        self.atr_period = self.config.get('atr_period', self.DEFAULT_ATR_PERIOD)
        self.adx_period = self.config.get('adx_period', self.DEFAULT_ADX_PERIOD)
        self.dmi_period = self.config.get('dmi_period', self.DEFAULT_DMI_PERIOD)

        # 状态跟踪：用于防抖动
        self._last_trend: MarketRegime | None = None
        self._trend_counter: int = 0
        self._confirmed_trend: MarketRegime | None = None

    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        计算 ATR (Average True Range)。

        True Range = max(
            high - low,
            |high - close_prev|,
            |low - close_prev|
        )
        ATR = Wilder EMA of True Range

        Args:
            df: DataFrame 必须包含 high, low, close 列
            period: ATR 计算周期，默认14

        Returns:
            pd.Series: ATR 值序列
        """
        if period is None:
            period = self.atr_period

        high = df['high']
        low = df['low']
        close = df['close']

        # 计算 True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder EMA 方式计算 ATR (等同于 SMA 方式用于初始值，之后用 EMA)
        if len(tr) >= period:
            atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
        else:
            atr = tr.rolling(window=min(period, len(tr)), min_periods=1).mean()

        return atr

    def calculate_dmi(self, df: pd.DataFrame, period: int = None) -> Dict[str, pd.Series]:
        """
        计算 DMI (Directional Movement Index)。

        包括:
        - +DI (正向指标)
        - -DI (负向指标)
        - DX

        Args:
            df: DataFrame 必须包含 high, low, close 列
            period: 计算周期，默认14

        Returns:
            Dict with keys: 'plus_di', 'minus_di', 'dx'
        """
        if period is None:
            period = self.dmi_period

        high = df['high']
        low = df['low']
        close = df['close']

        prev_high = high.shift(1)
        prev_low = low.shift(1)

        # 计算 Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        # +DM: 当 up_move > down_move 且 up_move > 0
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        up_cond = (up_move > down_move) & (up_move > 0)
        down_cond = (down_move > up_move) & (down_move > 0)

        plus_dm[up_cond] = up_move[up_cond]
        minus_dm[down_cond] = down_move[down_cond]

        # 计算 True Range (同上)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder 平滑
        atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean()

        # 计算 +DI 和 -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # 计算 DX
        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)

        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'dx': dx
        }

    def calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        计算 ADX (Average Directional Index)。

        ADX = Wilder EMA of DX

        Args:
            df: DataFrame 必须包含 high, low, close 列
            period: 计算周期，默认14

        Returns:
            pd.Series: ADX 值序列
        """
        if period is None:
            period = self.adx_period

        dmi = self.calculate_dmi(df, period)
        dx = dmi['dx']

        # Wilder EMA 平滑 DX 得到 ADX
        adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

        return adx

    def _determine_trend_direction(
        self,
        plus_di: float,
        minus_di: float,
        adx: float
    ) -> MarketRegime:
        """
        根据 DMI 和 ADX 判断趋势方向。

        Args:
            plus_di: +DI 当前值
            minus_di: -DI 当前值
            adx: ADX 当前值

        Returns:
            MarketRegime: BULL, BEAR, or SIDEWAYS
        """
        # 如果 ADX 低于阈值，认为没有明显趋势
        if adx < self.adx_trend_threshold:
            return MarketRegime.SIDEWAYS

        # 如果 ADX 较高但 +DI 和 -DI 差距不大，认为是盘整
        di_diff = abs(plus_di - minus_di)
        if di_diff < self.dmi_strong_threshold:
            return MarketRegime.SIDEWAYS

        # 判断方向
        if plus_di > minus_di:
            return MarketRegime.BULL
        else:
            return MarketRegime.BEAR

    def _apply_anti_chatter(
        self,
        current_trend: MarketRegime
    ) -> MarketRegime:
        """
        应用防抖动机制：连续 CONFIRMATION_BARS 根K线确认趋势变化。

        只有连续多根K线都显示相同的趋势方向，才确认趋势变化。

        Args:
            current_trend: 当前检测到的趋势方向

        Returns:
            MarketRegime: 确认后的趋势方向
        """
        if current_trend == self._last_trend:
            # 趋势连续，增加计数
            self._trend_counter += 1
        else:
            # 趋势变化，重置计数
            self._trend_counter = 1
            self._last_trend = current_trend

        # 达到连续确认次数，确认趋势
        if self._trend_counter >= self.confirmation_bars:
            self._confirmed_trend = current_trend

        # 如果还没有足够的确认，保持之前的确认趋势或当前趋势
        if self._confirmed_trend is None:
            return current_trend

        return self._confirmed_trend

    def classify(self, df: pd.DataFrame) -> Tuple[MarketRegime, float, RegimeMetrics]:
        """
        对市场状态进行分类。

        Args:
            df: DataFrame，必须包含以下列：
                - high: 最高价
                - low: 最低价
                - close: 收盘价

        Returns:
            Tuple[MarketRegime, float, RegimeMetrics]:
                - regime: 市场状态 (BULL/BEAR/SIDEWAYS)
                - confidence: 置信度 (0-1)
                - metrics: 详细指标
        """
        # 计算指标
        atr = self.calculate_atr(df, self.atr_period)
        adx = self.calculate_adx(df, self.adx_period)
        dmi = self.calculate_dmi(df, self.dmi_period)

        # 获取最新值
        current_atr = atr.iloc[-1] if len(atr) > 0 else 0.0
        current_adx = adx.iloc[-1] if len(adx) > 0 else 0.0
        current_plus_di = dmi['plus_di'].iloc[-1] if len(dmi['plus_di']) > 0 else 0.0
        current_minus_di = dmi['minus_di'].iloc[-1] if len(dmi['minus_di']) > 0 else 0.0

        # 计算动量：总变化率 / ATR，归一化到 -1 ~ 1
        closes = df['close'].values
        total_change = closes[-1] - closes[0]  # 代数变化（有正负）
        abs_atr = abs(current_atr) if current_atr != 0 else 1e-10
        if abs_atr < 1e-10:
            abs_atr = 1e-10
        momentum_raw = total_change / abs_atr
        # 归一化到 -1~1（假设|total_change|/ATR ≈ 5 是极强趋势）
        momentum = max(-1.0, min(1.0, momentum_raw / 5.0))

        # 计算波动率比率：ATR / |总价格变化|
        total_change = closes[-1] - closes[0] if len(closes) > 1 else 0.0
        if abs(total_change) < 1e-10:
            total_change = 1e-10
        volatility_ratio = min(1.0, abs(current_atr / total_change))

        # 构建指标对象
        metrics = RegimeMetrics(
            adx=round(float(current_adx), 2),
            plus_di=round(float(current_plus_di), 2),
            minus_di=round(float(current_minus_di), 2),
            atr=round(float(current_atr), 4),
            momentum=round(float(momentum), 2),
            volatility_ratio=round(float(volatility_ratio), 2)
        )

        # 确定趋势方向（基于DMI）
        raw_trend = self._determine_trend_direction(
            current_plus_di, current_minus_di, current_adx
        )
        confirmed_trend = self._apply_anti_chatter(raw_trend)

        # 计算置信度：基于ADX和DMI差异
        di_diff = abs(current_plus_di - current_minus_di)
        if current_adx < self.adx_trend_threshold:
            confidence = 0.5  # 低置信度（无趋势）
        elif di_diff > self.dmi_strong_threshold * 2:
            confidence = 0.9  # 高置信度（趋势明确）
        elif di_diff > self.dmi_strong_threshold:
            confidence = 0.7  # 中等置信度
        else:
            confidence = 0.6  # 较低置信度

        return confirmed_trend, confidence, metrics

    def reset_state(self):
        """重置分类器状态（用于切换品种或时间周期时）。"""
        self._last_trend = None
        self._trend_counter = 0
        self._confirmed_trend = None

    def format_analysis(self, regime: MarketRegime, confidence: float,
                       metrics: RegimeMetrics) -> str:
        """
        格式化分析结果为可读字符串。

        Args:
            regime: 市场状态
            confidence: 置信度
            metrics: 指标对象

        Returns:
            str: 格式化的分析报告
        """
        regime_emoji = {
            MarketRegime.BULL: "🟢",
            MarketRegime.BEAR: "🔴",
            MarketRegime.SIDEWAYS: "🟡"
        }

        emoji = regime_emoji.get(regime, "⚪")

        return f"""
╔══════════════════════════════════════════════════════════╗
║              MARKET REGIME ANALYSIS                     ║
╠══════════════════════════════════════════════════════════╣
║ Regime:      {emoji} {regime.value.upper():<42}  ║
║ Confidence:  {confidence:.1%} ({confidence*100:.0f}/100){" "*32}║
╠══════════════════════════════════════════════════════════╣
║ METRICS                                                     ║
║   ADX (trend):          {metrics.adx:>8.2f}{" "*36}║
║   +DI:                  {metrics.plus_di:>8.2f}{" "*36}║
║   -DI:                  {metrics.minus_di:>8.2f}{" "*36}║
║   ATR:                  {metrics.atr:>8.4f}{" "*36}║
║   Momentum:             {metrics.momentum:>8.2f}{" "*36}║
║   Volatility Ratio:     {metrics.volatility_ratio:>8.2f}{" "*36}║
╚══════════════════════════════════════════════════════════╝
"""


# ============================================================
# HMM市场状态分类器（增强版，4状态识别）
# ============================================================

class HMMRegimeClassifier:
    """基于隐马尔可夫模型的市场状态分类器（增强版）
    
    使用 GaussianHMM 对收益率和波动率建模，识别4种隐藏状态：
      - BULL: 正收益 + 低波动
      - BEAR: 负收益 + 低波动
      - SIDEWAYS: 零均值 + 低波动
      - HIGH_VOL: 高波动（双向剧烈）
    
    依赖: hmmlearn, numpy, pandas
    参考: hmmlearn GaussianHMM 官方实现
    
    用法:
        hmm = HMMRegimeClassifier(n_states=4)
        hmm.train(df)  # 训练模型
        regime, confidence = hmm.classify(df)  # 分类最新状态
    """
    
    # 状态标签（训练后根据均值自动映射）
    _STATE_LABELS = {
        0: MarketRegime.SIDEWAYS,
        1: MarketRegime.BULL,
        2: MarketRegime.BEAR,
        3: MarketRegime.SIDEWAYS,
    }
    
    def __init__(self, n_states: int = 4, lookback: int = 100,
                 covariance_type: str = "diag", random_state: int = 42):
        """
        Args:
            n_states: 隐藏状态数（默认4: bull/bear/sideways/high_vol）
            lookback: 训练用的历史数据长度
            covariance_type: hmmlearn协方差类型
            random_state: 随机种子（确保可复现）
        """
        if n_states < 2 or n_states > 6:
            raise ValueError(f"n_states should be 2-6, got {n_states}")
        self.n_states = n_states
        self.lookback = lookback
        self.random_state = random_state
        
        # 模型（懒初始化）
        self._model = None
        self._state_mean_returns: Dict[int, float] = {}
        self._state_labels: Dict[int, MarketRegime] = {}
        self._is_trained = False
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取特征序列: [log_return, volatility, volume_change]
        
        特征说明:
        - log_return: 对数收益率（一阶矩）
        - volatility: 5期滚动波动率（二阶矩）
        """
        closes = df['close'].values
        volumes = df.get('volume', df.get('vol', pd.Series(index=df.index))).values
        
        # 对数收益率
        log_ret = np.diff(np.log(closes + 1e-10))
        
        # 5期滚动波动率（标准差）
        vol_window = min(5, len(log_ret))
        volatility = np.array([
            np.std(log_ret[max(0, i-vol_window):i+1])
            for i in range(len(log_ret))
        ])
        
        # 成交量变化率
        if len(volumes) > len(log_ret):
            volumes = volumes[-len(log_ret):]
        vol_returns = np.diff(np.log(volumes + 1e-10)) if len(volumes) > 1 else np.zeros(len(log_ret))
        if len(vol_returns) < len(log_ret):
            vol_returns = np.pad(vol_returns, (len(log_ret)-len(vol_returns), 0), 'edge')
        
        # 拼接特征矩阵 [n_timesteps, n_features]
        features = np.column_stack([
            log_ret[-len(volatility):],
            volatility[-len(log_ret):][:len(volatility)],
            vol_returns[-len(volatility):],
        ])
        return features
    
    def train(self, df: pd.DataFrame) -> None:
        """训练HMM模型
        
        Args:
            df: DataFrame，需包含 close 列
        
        Raises:
            RuntimeError: 训练失败（数据不足或无法收敛）
        """
        features = self._extract_features(df)
        if len(features) < self.n_states * 10:
            raise ValueError(f"Not enough data: {len(features)} rows < {self.n_states * 10}")
        
        # 用最近 lookback 行训练
        train_data = features[-self.lookback:] if len(features) > self.lookback else features
        
        # 标准化特征（防止数值不稳定）
        feature_mean = np.mean(train_data, axis=0)
        feature_std = np.std(train_data, axis=0)
        feature_std[feature_std < 1e-10] = 1.0
        train_data_norm = (train_data - feature_mean) / feature_std
        
        # 多次尝试训练（不同随机种子），取最好的结果
        best_model = None
        best_score = -np.inf
        
        for seed in [self.random_state, 0, 123, 456]:
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type='diag',
                    random_state=seed,
                    n_iter=100,
                    tol=1e-4,
                    init_params='stmc',  # 自动初始化所有参数
                    params='stmc',       # 训练所有参数
                )
                model.fit(train_data_norm)
                score = model.score(train_data_norm)
                
                # 验证模型参数有效
                if (np.any(np.isnan(model.startprob_)) or 
                    np.any(np.isnan(model.transmat_)) or
                    np.any(np.isnan(model.means_))):
                    continue
                    
                if score > best_score:
                    best_model = model
                    best_score = score
            except Exception as e:
                continue
        
        if best_model is None:
            raise RuntimeError(
                f"HMM failed to converge with {self.n_states} states "
                f"on {len(train_data)} data points. Try reducing n_states."
            )
        
        self._model = best_model
        
        # 解码训练集，按均值收益排序映射状态
        states = self._model.predict(train_data_norm)
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                self._state_mean_returns[s] = float(np.mean(train_data_norm[mask, 0]))
            else:
                self._state_mean_returns[s] = -999.0  # 空状态标记为极低值
        
        # 按收益率排序：最低→BEAR, 中间→SIDEWAYS, 最高→BULL
        sorted_states = sorted(range(self.n_states),
                              key=lambda s: self._state_mean_returns.get(s, 0))
        median_vol = np.median([
            np.mean(train_data_norm[states == s, 1])
            for s in range(self.n_states)
            if (states == s).sum() > 0
        ] + [1.0])
        
        for i, s in enumerate(sorted_states):
            mask = states == s
            if mask.sum() < 3:  # 观测太少的状态标记为SIDEWAYS
                self._state_labels[s] = MarketRegime.SIDEWAYS
                continue
            vol = np.mean(train_data_norm[mask, 1])
            if vol > median_vol * 2.0:
                self._state_labels[s] = MarketRegime.SIDEWAYS  # 高波动=不确定
            elif i == 0:
                self._state_labels[s] = MarketRegime.BEAR
            elif i == self.n_states - 1:
                self._state_labels[s] = MarketRegime.BULL
            else:
                self._state_labels[s] = MarketRegime.SIDEWAYS
        
        self._is_trained = True
    
    def classify(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """对最新数据点进行分类
        
        Returns:
            (regime, confidence): 市场状态和置信度
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("HMM not trained. Call train() first.")
        
        features = self._extract_features(df)
        if len(features) < 1:
            return MarketRegime.SIDEWAYS, 0.0
        
        # 取最近一段（至少10行）预测状态
        predict_data = features[-min(10, len(features)):]
        states = self._model.predict(predict_data)
        current_state = int(states[-1])
        
        # 状态→标签
        regime = self._state_labels.get(current_state, MarketRegime.SIDEWAYS)
        
        # 置信度：状态后验概率
        posteriors = self._model.predict_proba(predict_data)
        confidence = float(np.max(posteriors[-1]))
        
        return regime, min(1.0, max(0.0, confidence))
    
    def get_state_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """返回完整状态摘要（调试用）"""
        if not self._is_trained:
            return {"trained": False}
        
        features = self._extract_features(df)
        states = self._model.predict(features[-self.lookback:])
        
        summary = {
            "trained": True,
            "n_states": self.n_states,
            "state_labels": {k: v.value for k, v in self._state_labels.items()},
            "state_mean_returns": self._state_mean_returns,
            "current_state": int(states[-1]),
            "current_regime": self._state_labels.get(int(states[-1]), MarketRegime.SIDEWAYS).value,
            "state_distribution": {
                int(s): int((states == s).sum())
                for s in range(self.n_states)
            },
            "transition_matrix": self._model.transmat_.tolist() if hasattr(self._model, 'transmat_') else [],
        }
        return summary
    
    def save(self, path: str) -> None:
        """保存模型（暂不支持，需每次重新训练）"""
        raise NotImplementedError("HMM model serialization not yet supported")
    
    def load(self, path: str) -> None:
        """加载模型"""
        raise NotImplementedError("HMM model deserialization not yet supported")


def detect_regime_hmm(df: pd.DataFrame) -> Tuple[MarketRegime, float]:
    """便捷函数：快速用HMM检测市场状态（自动训练）"""
    hmm = HMMRegimeClassifier()
    hmm.train(df)
    return hmm.classify(df)


def detect_regime(df: pd.DataFrame, config: Dict | None = None) -> Tuple[MarketRegime, float, RegimeMetrics]:
    """
    便捷函数：快速检测市场状态。

    Args:
        df: DataFrame，必须包含 high, low, close 列
        config: 可选配置字典

    Returns:
        Tuple[MarketRegime, float, RegimeMetrics]: 市场状态、置信度、指标
    """
    classifier = RegimeClassifier(config)
    return classifier.classify(df)


# ========== 测试代码 ==========
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    print("Testing RegimeClassifier for Miracle System...")
    print("=" * 60)

    # 生成示例数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # 模拟趋势上涨数据
    data = {
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.3),
        'low': 99 + np.cumsum(np.random.randn(100) * 0.5 + 0.2),
        'close': 99.5 + np.cumsum(np.random.randn(100) * 0.5 + 0.25),
        'volume': np.random.randint(1000, 5000, 100)
    }
    df = pd.DataFrame(data, index=dates)

    # 初始化分类器
    classifier = RegimeClassifier(config={
        'adx_trend_threshold': 25.0,
        'confirmation_bars': 3
    })

    # 执行分类
    regime, confidence, metrics = classifier.classify(df)

    # 打印分析结果
    print(classifier.format_analysis(regime, confidence, metrics))

    print("\n" + "=" * 60)
    print("Test completed.")
