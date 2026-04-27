#!/usr/bin/env python3
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

from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


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

    def __init__(self, config: Optional[Dict] = None):
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
        self._last_trend: Optional[MarketRegime] = None
        self._trend_counter: int = 0
        self._confirmed_trend: Optional[MarketRegime] = None

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


def detect_regime(df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[MarketRegime, float, RegimeMetrics]:
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
    import pandas as pd
    import numpy as np

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
