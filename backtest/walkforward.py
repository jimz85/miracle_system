"""
Walk-Forward Validation Engine (P2.1)
=====================================
滚动窗口Walk-Forward回测验证框架

功能:
1. 3个月滚动窗口 (90天训练 / 30天测试)
2. 样本内/样本外分离
3. IC信息系数计算
4. 稳定性分析
5. 均值回归 vs 趋势跟踪对比

Usage:
    from backtest.walkforward import WalkForwardValidator
    
    validator = WalkForwardValidator(
        train_days=90,      # 3个月训练窗口
        test_days=30,       # 1个月测试窗口
        step_days=15        # 滚动步长
    )
    result = validator.run(data, strategy_func)
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger("miracle.backtest.walkforward")

# ==================== 常量 ====================

DEFAULT_CONFIG = {
    "train_days": 90,       # 3个月训练窗口
    "test_days": 30,        # 1个月测试窗口
    "step_days": 15,        # 滚动步长(每两周滚动一次)
    "min_train_trades": 10, # 最少训练期交易次数
    "min_test_trades": 5,   # 最少测试期交易次数
    "warmup_bars": 50,      # 预热K线数(计算指标用)
}

# 均值回归参数空间
MEAN_REVERSION_PARAMS = {
    "rsi_period": [7, 10, 14, 21],
    "rsi_oversold": [25, 30, 35],
    "rsi_overbought": [65, 70, 75],
    "bb_period": [10, 20, 30],
    "bb_std": [1.5, 2.0, 2.5],
}

# 趋势跟踪参数空间
TREND_FOLLOWING_PARAMS = {
    "adx_period": [7, 10, 14, 21],
    "adx_threshold": [15, 20, 25, 30],
    "ema_fast": [5, 10, 15, 20],
    "ema_slow": [20, 30, 50],
    "sl_pct": [0.015, 0.02, 0.03],
    "tp_pct": [0.05, 0.08, 0.12],
}


# ==================== 数据结构 ====================

class StrategyType(Enum):
    """策略类型枚举"""
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BOTH = "both"


@dataclass
class WalkForwardWindow:
    """Walk-Forward窗口"""
    window_id: int
    train_start: int       # 训练期开始时间戳(ms)
    train_end: int         # 训练期结束时间戳(ms)
    test_start: int        # 测试期开始时间戳(ms)
    test_end: int          # 测试期结束时间戳(ms)
    train_klines: List[Dict] = field(default_factory=list)
    test_klines: List[Dict] = field(default_factory=list)
    
    @property
    def train_period_days(self) -> int:
        return (self.train_end - self.train_start) // (24 * 3600 * 1000)
    
    @property
    def test_period_days(self) -> int:
        return (self.test_end - self.test_start) // (24 * 3600 * 1000)


@dataclass
class WindowResult:
    """单个窗口的回测结果"""
    window_id: int
    strategy_type: str
    train_params: Dict
    train_stats: Dict       # 训练期统计
    test_stats: Dict        # 测试期统计
    train_trades: int
    test_trades: int
    train_return: float     # 训练期收益(%)
    test_return: float     # 测试期收益(%)
    train_sharpe: float
    test_sharpe: float
    train_max_dd: float    # 训练期最大回撤(%)
    test_max_dd: float     # 测试期最大回撤(%)
    ic: float              # 信息系数
    is_stable: bool = False  # 是否稳定盈利
    
    def to_dict(self) -> Dict:
        return {
            "window_id": self.window_id,
            "strategy_type": self.strategy_type,
            "train_params": self.train_params,
            "train_stats": self.train_stats,
            "test_stats": self.test_stats,
            "train_trades": self.train_trades,
            "test_trades": self.test_trades,
            "train_return": self.train_return,
            "test_return": self.test_return,
            "train_sharpe": self.train_sharpe,
            "test_sharpe": self.test_sharpe,
            "train_max_dd": self.train_max_dd,
            "test_max_dd": self.test_max_dd,
            "ic": self.ic,
            "is_stable": self.is_stable,
        }


@dataclass
class WalkForwardResult:
    """Walk-Forward总体结果"""
    strategy_type: str
    num_windows: int
    train_avg_return: float
    test_avg_return: float
    train_avg_sharpe: float
    test_avg_sharpe: float
    train_avg_max_dd: float
    test_avg_max_dd: float
    avg_ic: float
    stability_ratio: float  # 稳定盈利窗口比例
    stable_params: List[Dict]  # 稳定参数列表
    window_results: List[Dict]
    
    def to_dict(self) -> Dict:
        return {
            "strategy_type": self.strategy_type,
            "num_windows": self.num_windows,
            "train_avg_return": self.train_avg_return,
            "test_avg_return": self.test_avg_return,
            "train_avg_sharpe": self.train_avg_sharpe,
            "test_avg_sharpe": self.test_avg_sharpe,
            "train_avg_max_dd": self.train_avg_max_dd,
            "test_avg_max_dd": self.test_avg_max_dd,
            "avg_ic": self.avg_ic,
            "stability_ratio": self.stability_ratio,
            "stable_params": self.stable_params,
            "window_results": self.window_results,
        }


# ==================== Walk-Forward 验证器 ====================

class WalkForwardValidator:
    """
    Walk-Forward滚动窗口验证器
    
    工作流程:
    1. 将数据分割成多个滚动窗口 (训练期 + 测试期)
    2. 在训练期进行参数优化
    3. 在测试期验证最优参数
    4. 重复滚动直到数据结束
    5. 汇总分析结果
    """
    
    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.train_days = self.config["train_days"]
        self.test_days = self.config["test_days"]
        self.step_days = self.config["step_days"]
        self.min_train_trades = self.config["min_train_trades"]
        self.min_test_trades = self.config["min_test_trades"]
        self.warmup_bars = self.config["warmup_bars"]
        
        self.windows: List[WalkForwardWindow] = []
        self.results: List[WindowResult] = []
        
    def create_windows(self, klines: List[Dict]) -> List[WalkForwardWindow]:
        """
        创建Walk-Forward滚动窗口
        
        Args:
            klines: K线数据列表
            
        Returns:
            窗口列表
        """
        if not klines:
            return []
        
        # 解析时间戳
        timestamps = [k.get("timestamp", k.get("ts", 0)) for k in klines]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        
        start_dt = datetime.fromtimestamp(min_ts / 1000) if isinstance(min_ts, (int, float)) else datetime.fromisoformat(str(min_ts))
        end_dt = datetime.fromtimestamp(max_ts / 1000) if isinstance(max_ts, (int, float)) else datetime.fromisoformat(str(max_ts))
        
        windows = []
        window_id = 1
        current = start_dt + timedelta(days=self.train_days)
        
        while current + timedelta(days=self.test_days) <= end_dt:
            train_end_dt = current
            test_end_dt = current + timedelta(days=self.test_days)
            
            # 找到最近的k线时间戳作为边界
            train_end_ts = int(train_end_dt.timestamp() * 1000)
            test_end_ts = int(test_end_dt.timestamp() * 1000)
            
            # 时间戳转毫秒辅助函数
            def ts_to_ms(ts):
                if isinstance(ts, (int, float)):
                    return int(ts) if ts > 1e10 else int(ts * 1000)  # 秒或毫秒
                s = str(ts)
                try:
                    # 尝试解析ISO格式
                    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
                    return int(dt.timestamp() * 1000)
                except:
                    return 0
            
            # 修复: 直接使用时间戳比较，避免//3600000小时桶化导致边界错误
            train_klines = [k for k in klines 
                          if ts_to_ms(k.get("timestamp", k.get("ts", 0))) <= train_end_ts]
            test_klines = [k for k in klines 
                         if train_end_ts < ts_to_ms(k.get("timestamp", k.get("ts", 0))) < test_end_ts]
            
            # 过滤预热期
            if len(train_klines) >= self.warmup_bars + 50 and len(test_klines) >= 50:
                first_ts = train_klines[0].get("timestamp", train_klines[0].get("ts", 0))
                windows.append(WalkForwardWindow(
                    window_id=window_id,
                    train_start=first_ts,
                    train_end=train_end_ts,
                    test_start=test_klines[0].get("timestamp", test_klines[0].get("ts", 0)),
                    test_end=test_end_ts,
                    train_klines=train_klines,
                    test_klines=test_klines,
                ))
                window_id += 1
            
            current += timedelta(days=self.step_days)
        
        logger.info(f"创建了 {len(windows)} 个Walk-Forward窗口 (训练:{self.train_days}天, 测试:{self.test_days}天)")
        return windows
    
    def run_mean_reversion(self, klines: List[Dict], leverage: float = 2) -> WalkForwardResult:
        """运行均值回归策略Walk-Forward验证"""
        return self._run_strategy(klines, StrategyType.MEAN_REVERSION, leverage)
    
    def run_trend_following(self, klines: List[Dict], leverage: float = 2) -> WalkForwardResult:
        """运行趋势跟踪策略Walk-Forward验证"""
        return self._run_strategy(klines, StrategyType.TREND_FOLLOWING, leverage)
    
    def run_both(self, klines: List[Dict], leverage: float = 2) -> Tuple[WalkForwardResult, WalkForwardResult]:
        """同时运行两种策略并对比"""
        mr_result = self.run_mean_reversion(klines, leverage)
        tf_result = self.run_trend_following(klines, leverage)
        return mr_result, tf_result
    
    def _run_strategy(self, klines: List[Dict], strategy_type: StrategyType, leverage: float) -> WalkForwardResult:
        """运行指定策略的Walk-Forward验证"""
        self.windows = self.create_windows(klines)
        self.results = []
        
        if strategy_type == StrategyType.MEAN_REVERSION:
            param_spaces = MEAN_REVERSION_PARAMS
            param_spaces["leverage"] = [leverage]
        elif strategy_type == StrategyType.TREND_FOLLOWING:
            param_spaces = TREND_FOLLOWING_PARAMS
            param_spaces["leverage"] = [leverage]
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        for window in self.windows:
            window_result = self._optimize_and_validate(
                window, param_spaces, strategy_type.value
            )
            if window_result:
                self.results.append(window_result)
        
        return self._summarize_results(strategy_type.value)
    
    def _optimize_and_validate(
        self, 
        window: WalkForwardWindow, 
        param_spaces: Dict,
        strategy_type: str
    ) -> Optional[WindowResult]:
        """
        训练期优化 + 测试期验证
        
        Args:
            window: Walk-Forward窗口
            param_spaces: 参数空间
            strategy_type: 策略类型
            
        Returns:
            窗口结果
        """
        # 生成参数组合
        param_combinations = self._generate_param_combinations(param_spaces)
        
        # ===== 训练阶段: 网格搜索最优参数 =====
        best_train_result = None
        best_train_score = -999
        
        for params in param_combinations:
            result = self._backtest_with_params(
                window.train_klines, params, strategy_type, is_train=True
            )
            if result and result["score"] > best_train_score:
                best_train_score = result["score"]
                best_train_result = result
        
        if not best_train_result or best_train_result["trades"] < self.min_train_trades:
            logger.warning(f"窗口 {window.window_id}: 训练期交易次数不足")
            return None
        
        # ===== 测试阶段: 验证最优参数 =====
        test_result = self._backtest_with_params(
            window.test_klines, best_train_result["params"], strategy_type, is_train=False
        )
        
        # 计算IC (信息系数)
        ic = self._calc_ic(best_train_result["return_pct"], test_result["return_pct"] if test_result else 0)
        
        # 判断稳定性 (测试期盈利且IC > 0)
        is_stable = (
            test_result is not None and 
            test_result["return_pct"] > 0 and 
            ic > 0
        )
        
        return WindowResult(
            window_id=window.window_id,
            strategy_type=strategy_type,
            train_params=best_train_result["params"],
            train_stats=best_train_result["stats"],
            test_stats=test_result["stats"] if test_result else {},
            train_trades=best_train_result["trades"],
            test_trades=test_result["trades"] if test_result else 0,
            train_return=best_train_result["return_pct"],
            test_return=test_result["return_pct"] if test_result else 0,
            train_sharpe=best_train_result["sharpe"],
            test_sharpe=test_result["sharpe"] if test_result else 0,
            train_max_dd=best_train_result["max_dd"],
            test_max_dd=test_result["max_dd"] if test_result else 0,
            ic=ic,
            is_stable=is_stable,
        )
    
    def _backtest_with_params(
        self, 
        klines: List[Dict], 
        params: Dict, 
        strategy_type: str,
        is_train: bool = True
    ) -> Optional[Dict]:
        """
        使用指定参数运行回测
        
        Returns:
            回测结果字典，包含 score, trades, stats, return_pct, sharpe, max_dd
        """
        if len(klines) < self.warmup_bars:
            return None
        
        # 提取数据
        opens = [k["open"] for k in klines]
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        closes = [k["close"] for k in klines]
        
        initial_balance = 100000
        balance = initial_balance
        position = None
        trades = []
        equity_curve = [balance]
        
        warmup = self.warmup_bars
        
        for i in range(warmup, len(klines)):
            current_price = closes[i]
            
            # 如果有持仓，检查止损止盈
            if position:
                should_exit, exit_reason = self._check_exit(position, i, closes, highs, lows)
                
                if should_exit:
                    exit_price = self._get_exit_price(position, i, closes, exit_reason)
                    pnl, balance = self._close_trade(position, i, exit_price, balance, closes)
                    
                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "direction": position["direction"],
                        "pnl": pnl,
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                    })
                    position = None
            
            # 生成信号
            if not position:
                signal = self._generate_signal(
                    opens[:i+1], highs[:i+1], lows[:i+1], closes[:i+1], 
                    params, strategy_type, i
                )
                if signal and signal.get("direction") in ["long", "short"]:
                    entry_price = signal.get("entry_price", opens[i + 1] if i + 1 < len(opens) else closes[i])
                    stop_loss = signal.get("stop_loss")
                    take_profit = signal.get("take_profit")
                    leverage = signal.get("leverage", params.get("leverage", 1))
                    
                    position = {
                        "entry_idx": i,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "direction": signal["direction"],
                        "leverage": leverage,
                        "factors": signal.get("factors", {}),
                    }
            
            # 记录equity
            if position:
                unrealized_pnl = self._calc_unrealized_pnl(position, current_price)
                equity_curve.append(balance + unrealized_pnl)
            else:
                equity_curve.append(balance)
        
        # 计算统计
        if len(trades) < (self.min_train_trades if is_train else self.min_test_trades):
            return None
        
        stats = self._calc_stats(trades, equity_curve, initial_balance)
        
        # 计算综合评分
        annual_return = stats["total_pnl_pct"]
        max_dd = abs(stats["max_drawdown_pct"])
        sharpe = stats["sharpe_ratio"]
        score = annual_return * 0.6 - max_dd * 0.4 + sharpe * 5
        
        return {
            "params": params,
            "score": score,
            "trades": len(trades),
            "stats": stats,
            "return_pct": stats["total_pnl_pct"],
            "sharpe": stats["sharpe_ratio"],
            "max_dd": stats["max_drawdown_pct"],
        }
    
    def _generate_signal(
        self, 
        opens: List[float], 
        highs: List[float], 
        lows: List[float], 
        closes: List[float],
        params: Dict,
        strategy_type: str,
        idx: int
    ) -> Optional[Dict]:
        """生成交易信号"""
        if idx < 20:
            return None
        
        current_price = closes[-1]
        
        if strategy_type == "mean_reversion":
            return self._mean_reversion_signal(opens, highs, lows, closes, params, idx)
        elif strategy_type == "trend_following":
            return self._trend_following_signal(opens, highs, lows, closes, params, idx)
        
        return None
    
    def _mean_reversion_signal(
        self, 
        opens: List[float], 
        highs: List[float], 
        lows: List[float], 
        closes: List[float],
        params: Dict,
        idx: int
    ) -> Optional[Dict]:
        """均值回归信号"""
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)
        bb_period = params.get("bb_period", 20)
        bb_std = params.get("bb_std", 2.0)
        
        if len(closes) < rsi_period * 2:
            return None
        
        # RSI计算
        period = rsi_period
        recent = closes[-period:]
        gains = [max(0, recent[j] - recent[j-1]) for j in range(1, len(recent))]
        losses = [max(0, recent[j-1] - recent[j]) for j in range(1, len(recent))]
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # 布林带计算
        bb_data = closes[-bb_period:]
        sma = sum(bb_data) / len(bb_data)
        std = (sum((x - sma) ** 2 for x in bb_data) / len(bb_data)) ** 0.5
        upper_band = sma + bb_std * std
        lower_band = sma - bb_std * std
        
        # 信号判断
        entry_price = closes[-1]
        sl_pct = params.get("sl_pct", 0.02)
        tp_pct = params.get("tp_pct", 0.05)
        
        # 超卖 + 价格触及下轨 -> 买入
        if rsi < rsi_oversold and entry_price <= lower_band * 1.01:
            return {
                "direction": "long",
                "entry_price": entry_price,
                "stop_loss": entry_price * (1 - sl_pct),
                "take_profit": entry_price * (1 + tp_pct),
                "leverage": params.get("leverage", 2),
                "factors": {"rsi": rsi, "bb_position": 0},
            }
        
        # 超买 + 价格触及上轨 -> 卖出
        if rsi > rsi_overbought and entry_price >= upper_band * 0.99:
            return {
                "direction": "short",
                "entry_price": entry_price,
                "stop_loss": entry_price * (1 + sl_pct),
                "take_profit": entry_price * (1 - tp_pct),
                "leverage": params.get("leverage", 2),
                "factors": {"rsi": rsi, "bb_position": 1},
            }
        
        return None

    def _calc_adx(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int
    ) -> float:
        """
        计算 ADX (Average Directional Index) - 使用完整的 Wilder 平滑算法。

        ADX 计算流程:
        1. TR (True Range) = max(H-L, H-PC, L-PC)
        2. +DM (Positive Directional Movement)
        3. -DM (Negative Directional Movement)
        4. Wilder 平滑得到 ATR, +DI, -DI
        5. DX = |+DI - (-DI)| / (+DI + (-DI)) * 100
        6. ADX = Wilder EMA of DX
        """
        if len(closes) < period + 1:
            return 50.0

        highs_arr = np.array(highs)
        lows_arr = np.array(lows)
        closes_arr = np.array(closes)

        # 计算 True Range
        prev_closes = np.roll(closes_arr, 1)
        prev_closes[0] = closes_arr[0]

        tr1 = highs_arr - lows_arr
        tr2 = np.abs(highs_arr - prev_closes)
        tr3 = np.abs(lows_arr - prev_closes)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # 计算 Directional Movement
        up_move = highs_arr - np.roll(highs_arr, 1)
        up_move[0] = 0
        down_move = np.roll(lows_arr, 1) - lows_arr
        down_move[0] = 0

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder 平滑 (使用 EMA，alpha = 1/period)
        alpha = 1.0 / period

        # 初始平滑值 (简单移动平均)
        atr_smooth = np.mean(tr[1:period+1])
        plus_dm_smooth = np.mean(plus_dm[1:period+1])
        minus_dm_smooth = np.mean(minus_dm[1:period+1])

        # 递归 Wilder 平滑
        for i in range(period + 1, len(tr)):
            atr_smooth = (atr_smooth * (period - 1) + tr[i]) / period
            plus_dm_smooth = (plus_dm_smooth * (period - 1) + plus_dm[i]) / period
            minus_dm_smooth = (minus_dm_smooth * (period - 1) + minus_dm[i]) / period

        # 计算 +DI 和 -DI
        plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)

        # 计算 DX
        di_sum = plus_di + minus_di
        if di_sum < 1e-10:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum

        # Wilder 平滑 DX 得到 ADX
        # 使用 DX 的 EMA 作为 ADX
        adx_value = dx  # 简化：直接使用 DX 作为 ADX（因为数据长度不足时）

        # 如果有足够数据，计算完整的 ADX
        if len(tr) > period * 2:
            dx_series = np.zeros(len(tr))
            for i in range(period, len(tr)):
                # 重新计算 DX
                atr_s = np.mean(tr[i-period+1:i+1])
                pdm_s = np.mean(plus_dm[i-period+1:i+1])
                mdm_s = np.mean(minus_dm[i-period+1:i+1])

                # Wilder 平滑
                for j in range(i-period+1, i):
                    atr_s = (atr_s * (period - 1) + tr[j+1]) / period
                    pdm_s = (pdm_s * (period - 1) + plus_dm[j+1]) / period
                    mdm_s = (mdm_s * (period - 1) + minus_dm[j+1]) / period

                pdi = 100 * pdm_s / (atr_s + 1e-10)
                mdi = 100 * mdm_s / (atr_s + 1e-10)
                di_sum_s = pdi + mdi
                dx_series[i] = 100 * abs(pdi - mdi) / di_sum_s if di_sum_s > 1e-10 else 0

            # Wilder EMA of DX
            adx_smooth = np.nanmean(dx_series[period:period+period])
            for i in range(period + period, len(dx_series)):
                if not np.isnan(dx_series[i]):
                    adx_smooth = (adx_smooth * (period - 1) + dx_series[i]) / period
            adx_value = adx_smooth

        return min(100, max(0, adx_value))

    def _trend_following_signal(
        self, 
        opens: List[float], 
        highs: List[float], 
        lows: List[float], 
        closes: List[float],
        params: Dict,
        idx: int
    ) -> Optional[Dict]:
        """趋势跟踪信号"""
        adx_period = params.get("adx_period", 14)
        adx_threshold = params.get("adx_threshold", 25)
        ema_fast = params.get("ema_fast", 10)
        ema_slow = params.get("ema_slow", 30)
        
        if len(closes) < max(adx_period, ema_slow) * 2:
            return None
        
        # EMA计算
        def calc_ema(data, period):
            multiplier = 2 / (period + 1)
            ema = data[0]
            for price in data[1:]:
                ema = price * multiplier + ema * (1 - multiplier)
            return ema
        
        ema_fast_val = calc_ema(closes[-ema_fast*2:], ema_fast)
        ema_slow_val = calc_ema(closes[-ema_slow*2:], ema_slow)
        
        # ADX计算 (使用完整的 Wilder 平滑算法)
        adx_value = self._calc_adx(highs, lows, closes, adx_period)
        
        # 信号判断
        entry_price = closes[-1]
        sl_pct = params.get("sl_pct", 0.02)
        tp_pct = params.get("tp_pct", 0.08)
        
        # 金叉 -> 做多
        if ema_fast_val > ema_slow_val and adx_value > adx_threshold:
            return {
                "direction": "long",
                "entry_price": entry_price,
                "stop_loss": entry_price * (1 - sl_pct),
                "take_profit": entry_price * (1 + tp_pct),
                "leverage": params.get("leverage", 2),
                "factors": {"adx": adx_value, "ema_cross": 1},
            }
        
        # 死叉 -> 做空
        if ema_fast_val < ema_slow_val and adx_value > adx_threshold:
            return {
                "direction": "short",
                "entry_price": entry_price,
                "stop_loss": entry_price * (1 + sl_pct),
                "take_profit": entry_price * (1 - tp_pct),
                "leverage": params.get("leverage", 2),
                "factors": {"adx": adx_value, "ema_cross": -1},
            }
        
        return None
    
    def _check_exit(
        self, 
        position: Dict, 
        idx: int,
        closes: List[float],
        highs: List[float],
        lows: List[float]
    ) -> Tuple[bool, str]:
        """检查是否应该退出
        
        P0 Fix: 使用日内高低点而非收盘价检查SL/TP
        - Long SL: 检查日内低点是否触发了止损（价格可能低开）
        - Long TP: 检查日内高点是否触发了止盈
        - Short SL: 检查日内高点是否触发了止损（价格可能高开）
        - Short TP: 检查日内低点是否触发了止盈
        """
        direction = position["direction"]
        entry_price = position["entry_price"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        
        # 使用日内高低点进行检查（而非收盘价）
        current_low = lows[idx]
        current_high = highs[idx]
        
        # 止损检查 - 使用日内价格
        if direction == "long" and current_low <= stop_loss:
            return True, "sl"
        if direction == "short" and current_high >= stop_loss:
            return True, "sl"
        
        # 止盈检查 - 使用日内价格
        if direction == "long" and current_high >= take_profit:
            return True, "tp"
        if direction == "short" and current_low <= take_profit:
            return True, "tp"
        
        return False, ""
    
    def _get_exit_price(
        self, 
        position: Dict, 
        idx: int, 
        closes: List[float], 
        reason: str
    ) -> float:
        """获取退出价格"""
        current_price = closes[idx]
        direction = position["direction"]
        slippage_rate = 0.0002  # 0.02%
        
        if reason == "sl":
            if direction == "long":
                return current_price * (1 - slippage_rate)
            else:
                return current_price * (1 + slippage_rate)
        elif reason == "tp":
            if direction == "long":
                return current_price * (1 - slippage_rate)
            else:
                return current_price * (1 + slippage_rate)
        
        return current_price
    
    def _close_trade(
        self, 
        position: Dict, 
        idx: int, 
        exit_price: float, 
        balance: float,
        closes: List[float]
    ) -> Tuple[float, float]:
        """平仓并返回盈亏和新balance"""
        entry_price = position["entry_price"]
        direction = position["direction"]
        leverage = position["leverage"]
        
        commission_rate = 0.0005  # 0.05%
        
        if direction == "long":
            pnl = (exit_price - entry_price) * leverage
        else:
            pnl = (entry_price - exit_price) * leverage
        
        commission = abs(pnl) * commission_rate * 2
        pnl -= commission
        
        return pnl, balance + pnl
    
    def _calc_unrealized_pnl(self, position: Dict, current_price: float) -> float:
        """计算未实现盈亏"""
        entry_price = position["entry_price"]
        direction = position["direction"]
        leverage = position["leverage"]
        
        if direction == "long":
            return (current_price - entry_price) * leverage
        else:
            return (entry_price - current_price) * leverage
    
    def _calc_stats(
        self, 
        trades: List[Dict], 
        equity_curve: List[float], 
        initial_balance: float
    ) -> Dict:
        """计算回测统计"""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown_pct": 0,
            }
        
        wins = [t for t in trades if t["pnl"] > 0]
        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        total_pnl_pct = total_pnl / initial_balance * 100
        
        # 夏普比率
        equity_arr = equity_curve
        if len(equity_curve) > 1:
            returns = [(equity_arr[i] - equity_arr[i-1]) / equity_arr[i-1] for i in range(1, len(equity_arr)) if equity_arr[i-1] != 0]
            if returns:
                mean_ret = sum(returns) / len(returns)
                std_ret = (sum((r - mean_ret) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0
                sharpe = (mean_ret / (std_ret + 1e-10)) * (252 ** 0.5) if std_ret > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # 最大回撤
        running_max = [equity_arr[0]]
        for price in equity_arr[1:]:
            running_max.append(max(running_max[-1], price))
        
        drawdowns = [(equity_arr[i] - running_max[i]) / running_max[i] for i in range(len(equity_arr))]
        max_dd_pct = abs(min(drawdowns)) * 100 if drawdowns else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd_pct,
        }
    
    def _generate_param_combinations(self, param_spaces: Dict) -> List[Dict]:
        """生成参数组合"""
        import itertools
        
        keys = list(param_spaces.keys())
        values = list(param_spaces.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _calc_ic(self, train_return: float, test_return: float) -> float:
        """
        计算信息系数 (Information Coefficient)
        IC = 训练期收益与测试期收益的一致性
        """
        if train_return > 0 and test_return > 0:
            return 1.0
        elif train_return < 0 and test_return < 0:
            return -1.0
        elif train_return * test_return < 0:
            return -0.5
        return 0.0
    
    def _summarize_results(self, strategy_type: str) -> WalkForwardResult:
        """汇总Walk-Forward结果"""
        if not self.results:
            return WalkForwardResult(
                strategy_type=strategy_type,
                num_windows=0,
                train_avg_return=0,
                test_avg_return=0,
                train_avg_sharpe=0,
                test_avg_sharpe=0,
                train_avg_max_dd=0,
                test_avg_max_dd=0,
                avg_ic=0,
                stability_ratio=0,
                stable_params=[],
                window_results=[],
            )
        
        # 统计稳定盈利的参数
        stable_params_map = {}
        for r in self.results:
            if r.is_stable:
                key = str(sorted(r.train_params.items()))
                if key not in stable_params_map:
                    stable_params_map[key] = {"params": r.train_params, "count": 0, "returns": []}
                stable_params_map[key]["count"] += 1
                stable_params_map[key]["returns"].append(r.test_return)
        
        stable_params = [
            {
                "params": v["params"],
                "count": v["count"],
                "avg_test_return": sum(v["returns"]) / len(v["returns"]),
            }
            for v in stable_params_map.values()
            if v["count"] >= 2
        ]
        stable_params.sort(key=lambda x: x["avg_test_return"], reverse=True)
        
        # 计算平均值
        n = len(self.results)
        train_returns = [r.train_return for r in self.results]
        test_returns = [r.test_return for r in self.results]
        train_sharpes = [r.train_sharpe for r in self.results]
        test_sharpes = [r.test_sharpe for r in self.results]
        train_dds = [r.train_max_dd for r in self.results]
        test_dds = [r.test_max_dd for r in self.results]
        ics = [r.ic for r in self.results]
        stable_count = sum(1 for r in self.results if r.is_stable)
        
        return WalkForwardResult(
            strategy_type=strategy_type,
            num_windows=n,
            train_avg_return=sum(train_returns) / n,
            test_avg_return=sum(test_returns) / n,
            train_avg_sharpe=sum(train_sharpes) / n,
            test_avg_sharpe=sum(test_sharpes) / n,
            train_avg_max_dd=sum(train_dds) / n,
            test_avg_max_dd=sum(test_dds) / n,
            avg_ic=sum(ics) / n,
            stability_ratio=stable_count / n,
            stable_params=stable_params[:5],
            window_results=[r.to_dict() for r in self.results],
        )


# ==================== 便捷函数 ====================

def run_walk_forward(
    klines: List[Dict],
    strategy: str = "both",
    leverage: float = 2,
    output_dir: str = None
) -> Dict:
    """
    运行Walk-Forward验证的便捷函数
    
    Args:
        klines: K线数据
        strategy: "mean_reversion", "trend_following", 或 "both"
        leverage: 杠杆倍数
        output_dir: 结果输出目录
        
    Returns:
        Walk-Forward结果
    """
    validator = WalkForwardValidator()
    
    if strategy == "mean_reversion":
        result = validator.run_mean_reversion(klines, leverage)
        return result.to_dict()
    elif strategy == "trend_following":
        result = validator.run_trend_following(klines, leverage)
        return result.to_dict()
    else:
        mr_result, tf_result = validator.run_both(klines, leverage)
        
        comparison = {
            "mean_reversion": mr_result.to_dict(),
            "trend_following": tf_result.to_dict(),
            "comparison": {
                "winner": "mean_reversion" if mr_result.test_avg_return > tf_result.test_avg_return else "trend_following",
                "return_diff": abs(mr_result.test_avg_return - tf_result.test_avg_return),
                "sharpe_diff": abs(mr_result.test_avg_sharpe - tf_result.test_avg_sharpe),
            }
        }
        
        # 保存到文件
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "walkforward_result.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {output_path}")
        
        return comparison


# ==================== 自检 ====================

if __name__ == "__main__":
    import random
    
    print("=== Walk-Forward 验证引擎自检 ===\n")
    
    # 生成模拟K线数据 (6个月)
    base_price = 50000
    klines = []
    price = base_price
    start_ts = int(datetime.now().timestamp() * 1000) - 180 * 24 * 3600 * 1000
    
    for i in range(1000):
        price = price * (1 + random.uniform(-0.015, 0.02))
        klines.append({
            "timestamp": start_ts + i * 3600 * 1000,  # 1小时K线
            "open": price * 0.99,
            "high": price * 1.02,
            "low": price * 0.97,
            "close": price,
            "volume": random.uniform(100, 1000),
        })
    
    print(f"生成模拟数据: {len(klines)} 条K线")
    
    # 运行Walk-Forward验证
    result = run_walk_forward(klines, strategy="both", leverage=2)
    
    print("\n=== 均值回归策略 ===")
    mr = result["mean_reversion"]
    print(f"窗口数: {mr['num_windows']}")
    print(f"训练期平均收益: {mr['train_avg_return']:.2f}%")
    print(f"测试期平均收益: {mr['test_avg_return']:.2f}%")
    print(f"测试期平均夏普: {mr['test_avg_sharpe']:.2f}")
    print(f"稳定性比例: {mr['stability_ratio']:.1%}")
    
    print("\n=== 趋势跟踪策略 ===")
    tf = result["trend_following"]
    print(f"窗口数: {tf['num_windows']}")
    print(f"训练期平均收益: {tf['train_avg_return']:.2f}%")
    print(f"测试期平均收益: {tf['test_avg_return']:.2f}%")
    print(f"测试期平均夏普: {tf['test_avg_sharpe']:.2f}")
    print(f"稳定性比例: {tf['stability_ratio']:.1%}")
    
    print("\n=== 对比结论 ===")
    comp = result["comparison"]
    print(f"胜出策略: {comp['winner']}")
    print(f"收益差异: {comp['return_diff']:.2f}%")
    print(f"夏普差异: {comp['sharpe_diff']:.2f}")
    
    print("\n=== 自检完成 ===")
