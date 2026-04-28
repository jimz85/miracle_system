from __future__ import annotations

"""
Miracle 1.0.1 - 回测引擎模块
================================
基于历史数据的策略回测引擎

功能:
1. 历史K线数据回测
2. 模拟交易执行
3. Walk-Forward验证
4. 多币种批量回测
5. 参数优化
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .stats import (
    BacktestStats,
    BacktestTrade,
    ICStats,
    calc_ic,
    calc_ic_simple,
    calc_rank_ic,
    calc_stats,
)

logger = logging.getLogger("miracle.backtest.engine")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================
# 回测引擎
# ============================================================

class BacktestEngine:
    """
    回测引擎

    使用方法:
    1. 创建引擎实例
    2. 加载历史数据
    3. 运行回测
    4. 获取报告
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.initial_balance = self.config.get("initial_balance", 100000)
        # 手续费: maker < taker (OKX: maker 0.02%, taker 0.05%)
        self.taker_commission_rate = self.config.get("taker_commission_rate", 0.0005)  # 0.05%
        self.maker_commission_rate = self.config.get("maker_commission_rate", 0.0002)  # 0.02%
        # 兼容旧 config
        self.commission_rate = self.config.get("commission_rate", self.taker_commission_rate)
        self.slippage_rate = self.config.get("slippage_rate", 0.0002)  # 0.02%
        # funding rate (每8小时, 从config读取，如 0.0001 = 0.01%)
        self.funding_rate = self.config.get("funding_rate", 0.0)

        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.balance = self.initial_balance

        # IC追踪数据
        self.ic_signals: List[Dict] = []  # 信号预测 vs 实际收益

    def load_klines(self, symbol: str, klines: List[Dict]):
        """
        加载K线数据

        Args:
            symbol: 币种
            klines: K线列表，每条包含 timestamp, open, high, low, close, volume
        """
        self.symbol = symbol
        self.klines = klines
        self.opens = [k["open"] for k in klines]
        self.prices = [k["close"] for k in klines]
        self.highs = [k["high"] for k in klines]
        self.lows = [k["low"] for k in klines]
        self.timestamps = [k.get("timestamp", k.get("time")) for k in klines]
        logger.info(f"加载 {symbol} K线数据: {len(klines)} 条")

    def run(self, signal_func, min_trades: int = 10) -> Tuple[bool, Dict]:
        """
        运行回测

        Args:
            signal_func: 信号生成函数，接收 (prices, highs, lows, index) 返回 signal dict 或 None
            min_trades: 最少交易次数，低于此数量认为回测无效

        Returns:
            (success, result_dict)
        """
        if not hasattr(self, 'klines') or not self.klines:
            return False, {"error": "未加载K线数据"}

        self.trades = []
        self.equity_curve = [self.balance]
        self.balance = self.initial_balance
        self.ic_signals = []  # 重置IC追踪

        n = len(self.klines)

        # 模拟交易
        position = None  # 当前持仓

        for i in range(50, n):  # 至少需要50根K线来计算指标
            current_price = self.prices[i]
            current_time = self.timestamps[i]

            # 如果有持仓，检查止损止盈
            if position:
                should_exit, exit_reason = self._check_exit(position, i)

                if should_exit:
                    exit_price = self._get_exit_price(position, i, exit_reason)
                    self._close_trade(position, i, exit_price, exit_reason)

                    # ========== IC追踪：记录入场信号对应的实际收益 ==========
                    entry_idx = position["entry_idx"]
                    if i > entry_idx and entry_idx >= 50:
                        entry_price_actual = position["entry_price"]
                        exit_price_actual = exit_price
                        direction = position["direction"]

                        # 实际收益(考虑方向)
                        if direction == "long":
                            actual_return = (exit_price_actual - entry_price_actual) / entry_price_actual
                        else:  # short
                            actual_return = (entry_price_actual - exit_price_actual) / entry_price_actual

                        # 预测信号强度 (long=1, short=-1)
                        signal_direction = 1 if direction == "long" else -1

                        # 收集因子数据
                        factors = position.get("factors", {})

                        self.ic_signals.append({
                            "signal_idx": entry_idx,
                            "exit_idx": i,
                            "signal_direction": signal_direction,
                            "actual_return": actual_return,
                            "factors": factors,
                            "confidence": position.get("confidence", 0.5),
                        })

            # 生成信号
            if not position:
                signal = signal_func(self.prices[:i], self.highs[:i], self.lows[:i], max(0, i - 1))
                if signal and signal.get("direction") in ["long", "short"]:
                    # 使用下根K线开盘价入场，避免未来函数
                    if i + 1 < n:
                        entry_price = signal.get("entry_price", self.opens[i + 1])
                    else:
                        continue
                    stop_loss = signal.get("stop_loss")
                    take_profit = signal.get("take_profit")
                    leverage = signal.get("leverage", 1)
                    position_size = self._calc_position_size(entry_price, stop_loss, leverage)

                    position = {
                        "entry_time": current_time,
                        "entry_idx": i,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "direction": signal["direction"],
                        "leverage": leverage,
                        "position_size": position_size,
                        "factors": signal.get("factors", {}),
                        "confidence": signal.get("confidence", 0.5)
                    }

            # 记录equity
            if position:
                unrealized_pnl = self._calc_pnl(position, current_price)
                self.equity_curve.append(self.balance + unrealized_pnl)
            else:
                self.equity_curve.append(self.balance)

        # 计算统计
        if len(self.trades) < min_trades:
            return False, {
                "error": f"交易次数不足 ({len(self.trades)} < {min_trades})",
                "trades": [asdict(t) for t in self.trades],
                "equity_curve": self.equity_curve
            }

        # 计算IC (Information Coefficient)
        self.ic_stats = calc_ic(self.ic_signals)

        stats = calc_stats(self.trades, self.equity_curve, self.initial_balance, self.ic_stats)
        result = {
            "stats": asdict(stats),
            "trades": [asdict(t) for t in self.trades],
            "equity_curve": self.equity_curve,
            "ic_stats": asdict(self.ic_stats) if self.ic_stats else None,
        }
        return True, result

    def _check_exit(self, position: Dict, current_idx: int) -> Tuple[bool, str]:
        """检查是否应该退出"""
        direction = position["direction"]
        position["entry_price"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        entry_time = position["entry_time"]
        current_price = self.prices[current_idx]
        current_time = self.timestamps[current_idx]

        # 止损检查
        if direction == "long" and current_price <= stop_loss:
            return True, "sl"
        if direction == "short" and current_price >= stop_loss:
            return True, "sl"

        # 止盈检查
        if direction == "long" and current_price >= take_profit:
            return True, "tp"
        if direction == "short" and current_price <= take_profit:
            return True, "tp"

        # 时间止损（24小时）
        if isinstance(entry_time, (int, float)):
            entry_dt = datetime.fromtimestamp(entry_time / 1000)
            exit_dt = datetime.fromtimestamp(current_time / 1000)
        else:
            entry_dt = datetime.fromisoformat(str(entry_time))
            exit_dt = datetime.fromisoformat(str(current_time))

        if (exit_dt - entry_dt).total_seconds() > 24 * 3600:
            return True, "time"

        return False, ""

    def _get_exit_price(self, position: Dict, current_idx: int, reason: str) -> float:
        """获取退出价格（考虑滑点）"""
        current_price = self.prices[current_idx]
        direction = position["direction"]

        if reason == "sl":
            # 止损：滑点不利方向
            if direction == "long":
                return current_price * (1 - self.slippage_rate)
            else:
                return current_price * (1 + self.slippage_rate)
        elif reason == "tp":
            # 止盈：滑点有利方向
            if direction == "long":
                return current_price * (1 - self.slippage_rate)
            else:
                return current_price * (1 + self.slippage_rate)
        else:
            return current_price

    def _close_trade(self, position: Dict, current_idx: int, exit_price: float, reason: str):
        """平仓"""
        entry_price = position["entry_price"]
        direction = position["direction"]
        leverage = position["leverage"]
        position_size = position["position_size"]

        # 手续费: 开仓(maker) + 平仓(taker)
        commission = (
            position_size * leverage * self.maker_commission_rate +  # 开仓 maker
            position_size * leverage * self.taker_commission_rate    # 平仓 taker
        )

        # Funding fee: 每日扣除 (按持仓时间计算)
        entry_time = position["entry_time"]
        current_time = self.timestamps[current_idx]
        if isinstance(entry_time, (int, float)):
            entry_dt = datetime.fromtimestamp(entry_time / 1000)
            exit_dt = datetime.fromtimestamp(current_time / 1000)
        else:
            entry_dt = datetime.fromisoformat(str(entry_time))
            exit_dt = datetime.fromisoformat(str(current_time))
        funding_hours = (exit_dt - entry_dt).total_seconds() / 3600.0
        # OKX funding每8小时结算，取整计算funding次数
        funding_periods = max(1, round(funding_hours / 8.0))
        funding_fee = position_size * leverage * self.funding_rate * funding_periods

        # 计算盈亏
        if direction == "long":
            pnl = (exit_price - entry_price) * position_size * leverage
        else:
            pnl = (entry_price - exit_price) * position_size * leverage

        pnl -= commission
        pnl -= funding_fee
        self.balance += pnl

        # 记录交易
        trade = BacktestTrade(
            entry_time=position["entry_time"],
            exit_time=self.timestamps[current_idx],
            symbol=self.symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            leverage=leverage,
            pnl=pnl,
            pnl_pct=pnl / (position_size * leverage) * 100,
            stop_triggered=reason,
            commission=commission,
            slippage=abs(exit_price - self.prices[current_idx])
        )
        self.trades.append(trade)

    def _calc_position_size(self, entry_price: float, stop_loss: float, leverage: float) -> float:
        """计算仓位大小"""
        risk_amount = self.balance * 0.01  # 1%账户风险
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return 0

        position_size = risk_amount / (stop_distance * leverage)
        # 限制最大仓位
        max_position = self.balance * 0.15  # 15%账户
        return min(position_size, max_position)

    def _calc_pnl(self, position: Dict, current_price: float) -> float:
        """计算未实现盈亏"""
        entry_price = position["entry_price"]
        direction = position["direction"]
        leverage = position["leverage"]
        position_size = position["position_size"]

        if direction == "long":
            return (current_price - entry_price) * position_size * leverage
        else:
            return (entry_price - current_price) * position_size * leverage


# ============================================================
# Walk-Forward 验证引擎
# ============================================================

class WalkForwardEngine:
    """
    Walk-Forward 验证引擎
    
    功能:
    1. 滚动窗口训练/测试分割
    2. 样本外参数验证
    3. 稳定性分析
    4. IC计算（信息系数）
    
    使用方法:
    1. 创建引擎实例
    2. 设置参数空间
    3. 运行Walk-Forward优化
    4. 获取稳定参数
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.initial_balance = self.config.get("initial_balance", 100000)
        self.taker_commission_rate = self.config.get("taker_commission_rate", 0.0005)
        self.maker_commission_rate = self.config.get("maker_commission_rate", 0.0002)
        self.commission_rate = self.config.get("commission_rate", self.taker_commission_rate)
        self.slippage_rate = self.config.get("slippage_rate", 0.0002)
        self.funding_rate = self.config.get("funding_rate", 0.0)
        
        # Walk-Forward配置
        self.train_days = self.config.get("wf_train_days", 90)   # 训练窗口
        self.test_days = self.config.get("wf_test_days", 30)    # 测试窗口
        self.step_days = self.config.get("wf_step_days", 15)    # 滚动步长
        self.min_trades = self.config.get("min_trades", 10)     # 最少交易次数
        
        # 参数空间
        self.param_spaces = self.config.get("param_spaces", {
            "rsi_period": [7, 10, 14, 21],
            "rsi_oversold": [25, 30, 35, 40],
            "rsi_overbought": [60, 65, 70, 75],
            "adx_threshold": [15, 20, 25, 30],
            "sl_pct": [0.015, 0.02, 0.025, 0.03],
            "tp_pct": [0.05, 0.08, 0.10, 0.12],
        })
        
        self.results: List[Dict] = []
        
    def _generate_param_combinations(self) -> List[Dict]:
        """生成所有参数组合"""
        import itertools
        keys = list(self.param_spaces.keys())
        values = list(self.param_spaces.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _score_params(self, stats: Dict, train_stats: Dict = None) -> float:
        """综合评分：收益 - 回撤惩罚"""
        annual_return = stats.get("total_pnl_pct", 0)
        max_dd = abs(stats.get("max_drawdown_pct", 0))
        sharpe = stats.get("sharpe_ratio", 0)
        
        # 综合评分：年化收益60%权重 + 回撤40%权重 + 夏普加分
        score = annual_return * 0.6 - max_dd * 0.4 + sharpe * 5
        return score
    
    def _backtest_with_params(self, klines: List[Dict], params: Dict,
                              train: bool = True) -> Optional[Dict]:
        """使用指定参数运行回测"""
        engine = BacktestEngine({
            "initial_balance": self.initial_balance,
            "taker_commission_rate": self.taker_commission_rate,
            "maker_commission_rate": self.maker_commission_rate,
            "slippage_rate": self.slippage_rate,
            "funding_rate": self.funding_rate,
        })
        
        # 简单信号生成器（基于RSI+ADX）
        def signal_func(prices, highs, lows, idx):
            if idx < params.get("rsi_period", 14) * 2:
                return None
            
            # 简单RSI计算
            period = params.get("rsi_period", 14)
            if len(prices) < period:
                return None
            
            recent = prices[-period:]
            gains = [max(0, recent[i] - recent[i-1]) for i in range(1, len(recent))]
            losses = [max(0, recent[i-1] - recent[i]) for i in range(1, len(recent))]
            avg_gain = sum(gains) / period if gains else 0
            avg_loss = sum(losses) / period if losses else 0
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            oversold = params.get("rsi_oversold", 30)
            overbought = params.get("rsi_overbought", 70)
            adx_thresh = params.get("adx_threshold", 20)
            
            # 简单ADX近似（使用波动率）
            if len(highs) > period and len(lows) > period:
                high_range = max(highs[-period:]) - min(lows[-period:])
                adx_approx = min(100, high_range / (prices[-1] + 1e-10) * 100)
            else:
                adx_approx = 50
            
            sl_pct = params.get("sl_pct", 0.02)
            tp_pct = params.get("tp_pct", 0.05)
            entry_price = prices[-1]
            
            # 买入信号
            if rsi < oversold and adx_approx > adx_thresh:
                return {
                    "direction": "long",
                    "entry_price": entry_price,
                    "stop_loss": entry_price * (1 - sl_pct),
                    "take_profit": entry_price * (1 + tp_pct),
                    "leverage": params.get("leverage", 2),
                    "factors": {"rsi": rsi, "adx": adx_approx}
                }
            # 卖出信号
            elif rsi > overbought:
                return {
                    "direction": "short",
                    "entry_price": entry_price,
                    "stop_loss": entry_price * (1 + sl_pct),
                    "take_profit": entry_price * (1 - tp_pct),
                    "leverage": params.get("leverage", 2),
                    "factors": {"rsi": rsi, "adx": adx_approx}
                }
            return None
        
        engine.load_klines("WF_TEST", klines)
        success, result = engine.run(signal_func, self.min_trades)
        
        if success:
            result["params"] = params
            result["score"] = self._score_params(result["stats"])
            return result
        return None
    
    def _create_windows(self, klines: List[Dict]) -> List[Dict]:
        """创建Walk-Forward窗口"""
        if not klines:
            return []
        
        # 解析时间戳
        timestamps = [k.get("timestamp", k.get("time", 0)) for k in klines]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        
        from datetime import datetime
        start_dt = datetime.fromtimestamp(min_ts / 1000) if isinstance(min_ts, (int, float)) else datetime.fromisoformat(str(min_ts))
        end_dt = datetime.fromtimestamp(max_ts / 1000) if isinstance(max_ts, (int, float)) else datetime.fromisoformat(str(max_ts))
        
        (end_dt - start_dt).days
        windows = []
        
        current = start_dt + timedelta(days=self.train_days)
        
        while current + timedelta(days=self.test_days) <= end_dt:
            train_end_ts = int(current.timestamp() * 1000)
            test_end_ts = int((current + timedelta(days=self.test_days)).timestamp() * 1000)
            
            train_klines = [k for k in klines if (k.get("timestamp", k.get("time", 0)) < train_end_ts)]
            test_klines = [k for k in klines 
                         if train_end_ts <= (k.get("timestamp", k.get("time", 0)) < test_end_ts)]
            
            if len(train_klines) >= 100 and len(test_klines) >= 50:
                windows.append({
                    "train_klines": train_klines,
                    "test_klines": test_klines,
                    "train_start": train_klines[0].get("timestamp", train_klines[0].get("time")),
                    "train_end": train_end_ts,
                    "test_start": test_klines[0].get("timestamp", test_klines[0].get("time")),
                    "test_end": test_end_ts,
                })
            
            current += timedelta(days=self.step_days)
        
        return windows
    
    def run(self, klines: List[Dict], leverage: float = 2) -> Dict:
        """
        运行Walk-Forward验证
        
        Args:
            klines: K线数据
            leverage: 杠杆倍数
            
        Returns:
            Walk-Forward结果字典
        """
        self.results = []
        self.param_spaces["leverage"] = [leverage]
        
        windows = self._create_windows(klines)
        if not windows:
            return {"success": False, "error": "数据不足，无法创建Walk-Forward窗口"}
        
        param_combinations = self._generate_param_combinations()
        
        for wi, window in enumerate(windows):
            train_klines = window["train_klines"]
            test_klines = window["test_klines"]
            
            # ===== 训练阶段：网格搜索 =====
            best_train_result = None
            best_train_score = -999
            
            for params in param_combinations:
                result = self._backtest_with_params(train_klines, params, train=True)
                if result and result["score"] > best_train_score:
                    best_train_score = result["score"]
                    best_train_result = result
            
            if not best_train_result:
                continue
            
            # ===== 测试阶段：验证最优参数 =====
            test_result = self._backtest_with_params(test_klines, best_train_result["params"], train=False)
            
            window_result = {
                "window": wi + 1,
                "train_period": f'{window["train_start"]} ~ {window["train_end"]}',
                "test_period": f'{window["test_start"]} ~ {window["test_end"]}',
                "train_trades": best_train_result["stats"]["total_trades"],
                "train_return": best_train_result["stats"]["total_pnl_pct"],
                "train_dd": best_train_result["stats"]["max_drawdown_pct"],
                "train_sharpe": best_train_result["stats"]["sharpe_ratio"],
                "train_params": best_train_result["params"],
                "test_trades": test_result["stats"]["total_trades"] if test_result else 0,
                "test_return": test_result["stats"]["total_pnl_pct"] if test_result else 0,
                "test_dd": test_result["stats"]["max_drawdown_pct"] if test_result else 0,
                "test_sharpe": test_result["stats"]["sharpe_ratio"] if test_result else 0,
                "ic": self._calc_ic(best_train_result["stats"], test_result["stats"]) if test_result else 0,
            }
            
            self.results.append(window_result)
        
        # 汇总结果
        return self._summarize_results()
    
    def _calc_ic(self, train_stats: Dict, test_stats: Dict) -> float:
        """
        计算信息系数（Information Coefficient）
        IC = 相关系数(训练收益, 测试收益)
        衡量参数过拟合程度：IC越高说明参数越稳定
        """
        train_return = train_stats.get("total_pnl_pct", 0)
        test_return = test_stats.get("total_pnl_pct", 0)
        
        # 简化为收益符号相关
        if train_return > 0 and test_return > 0:
            return 1.0
        elif train_return < 0 and test_return < 0:
            return -1.0
        elif train_return * test_return < 0:
            return -0.5
        return 0.0
    
    def _calc_rank_ic(self, train_results: List[Dict], test_results: List[Dict]) -> float:
        """
        计算排名信息系数（Rank IC）
        更稳健的IC计算方式
        """
        if len(train_results) < 2 or len(test_results) < 2:
            return 0.0
        
        train_returns = [r["stats"]["total_pnl_pct"] for r in train_results]
        test_returns = [r["stats"]["total_pnl_pct"] for r in test_results]
        
        if HAS_NUMPY:
            return float(np.corrcoef(train_returns, test_returns)[0, 1])
        return 0.0
    
    def _summarize_results(self) -> Dict:
        """汇总Walk-Forward结果"""
        if not self.results:
            return {"success": False, "error": "无有效结果"}
        
        # 统计各参数出现频次
        param_counts = {}
        param_test_returns = {}
        
        for r in self.results:
            p = r["train_params"]
            key = str(p)
            param_counts[key] = param_counts.get(key, 0) + 1
            if key not in param_test_returns:
                param_test_returns[key] = []
            param_test_returns[key].append(r["test_return"])
        
        # 找出稳定盈利的参数
        stable_params = []
        for key, returns in param_test_returns.items():
            if np.mean(returns) > 0 and param_counts[key] >= 2:
                stable_params.append({
                    "params": eval(key) if key.startswith("{") else {},
                    "count": param_counts[key],
                    "avg_test_return": np.mean(returns),
                    "std_test_return": np.std(returns) if len(returns) > 1 else 0,
                })
        
        stable_params.sort(key=lambda x: x["avg_test_return"], reverse=True)
        
        # 全局统计
        test_returns = [r["test_return"] for r in self.results]
        test_dds = [r["test_dd"] for r in self.results]
        ics = [r["ic"] for r in self.results]
        
        summary = {
            "success": True,
            "num_windows": len(self.results),
            "stable_params": stable_params[:5],
            "global_stats": {
                "avg_test_return": np.mean(test_returns),
                "std_test_return": np.std(test_returns),
                "avg_test_dd": np.mean(test_dds),
                "avg_ic": np.mean(ics),
            },
            "windows": self.results,
        }
        
        return summary


# ============================================================
# 多币种批量回测
# ============================================================

class MultiCoinBacktest:
    """
    多币种批量回测引擎
    
    功能:
    1. 批量回测多个币种
    2. 汇总统计
    3. 分币种详细报告
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.initial_balance = self.config.get("initial_balance", 100000)
        self.taker_commission_rate = self.config.get("taker_commission_rate", 0.0005)
        self.maker_commission_rate = self.config.get("maker_commission_rate", 0.0002)
        self.commission_rate = self.config.get("commission_rate", self.taker_commission_rate)
        self.slippage_rate = self.config.get("slippage_rate", 0.0002)
        self.funding_rate = self.config.get("funding_rate", 0.0)
        
        self.results: Dict[str, Dict] = {}
        
    def run(self, coin_data: Dict[str, List[Dict]], signal_func_map: Dict[str, Callable] = None,
            min_trades: int = 10) -> Dict:
        """
        运行多币种批量回测
        
        Args:
            coin_data: 币种数据字典 {"BTC": [...], "ETH": [...]}
            signal_func_map: 币种对应的信号函数
            min_trades: 最少交易次数
            
        Returns:
            批量回测结果
        """
        self.results = {}
        all_trades = []
        
        for symbol, klines in coin_data.items():
            signal_func = signal_func_map.get(symbol) if signal_func_map else None
            
            if signal_func is None:
                # 使用默认信号生成器
                signal_func = self._default_signal
            
            engine = BacktestEngine({
                "initial_balance": self.initial_balance,
                "commission_rate": self.commission_rate,
                "slippage_rate": self.slippage_rate
            })
            
            engine.load_klines(symbol, klines)
            success, result = engine.run(signal_func, min_trades)
            
            if success:
                self.results[symbol] = {
                    "success": True,
                    "stats": result["stats"],
                    "trades": result["trades"],
                    "equity_curve": result["equity_curve"],
                }
                all_trades.extend(result["trades"])
            else:
                self.results[symbol] = {
                    "success": False,
                    "error": result.get("error", "回测失败"),
                    "trades": result.get("trades", []),
                }
        
        # 汇总统计
        return self._summarize(all_trades)
    
    def _default_signal(self, prices, highs, lows, idx):
        """默认信号生成器"""
        return None
    
    def _summarize(self, all_trades: List[Dict]) -> Dict:
        """汇总统计"""
        successful_coins = [s for s, r in self.results.items() if r.get("success")]
        
        if not successful_coins:
            return {"success": False, "error": "所有币种回测均失败"}
        
        # 聚合统计
        total_trades = sum(len(self.results[s]["trades"]) for s in successful_coins)
        winning_trades = sum(
            sum(1 for t in self.results[s]["trades"] if t["pnl"] > 0)
            for s in successful_coins
        )
        total_pnl = sum(
            sum(t["pnl"] for t in self.results[s]["trades"])
            for s in successful_coins
        )
        
        # IC汇总（如果有Walk-Forward结果）
        avg_ic = 0
        if hasattr(self, 'wf_results') and self.wf_results:
            avg_ic = np.mean([r.get("ic", 0) for r in self.wf_results.get("windows", [])])
        
        return {
            "success": True,
            "num_coins": len(successful_coins),
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / (self.initial_balance * len(successful_coins)) * 100,
            "avg_ic": avg_ic,
            "coin_results": self.results,
        }


# ============================================================
# 参数优化器
# ============================================================

class ParameterOptimizer:
    """
    参数优化器
    
    功能:
    1. 网格搜索最优参数
    2. 遗传算法优化
    3. 敏感性分析
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.initial_balance = self.config.get("initial_balance", 100000)
        
        self.param_ranges = self.config.get("param_ranges", {})
        self.optimization_metric = self.config.get("optimization_metric", "sharpe_ratio")
        
    def grid_search(self, klines: List[Dict], signal_template: Callable,
                    min_trades: int = 10) -> Dict:
        """
        网格搜索最优参数
        
        Args:
            klines: K线数据
            signal_template: 信号模板函数，接收(params, prices, highs, lows, idx)
            min_trades: 最少交易次数
            
        Returns:
            最优参数和结果
        """
        import itertools
        
        # 生成参数组合
        keys = list(self.param_ranges.keys())
        values = list(self.param_ranges.values())
        combinations = list(itertools.product(*values))
        
        best_params = None
        best_score = -999
        all_results = []
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            def signal_func(prices, highs, lows, idx):
                return signal_template(params, prices, highs, lows, idx)
            
            engine = BacktestEngine({
                "initial_balance": self.initial_balance,
                "commission_rate": 0.0005,
                "slippage_rate": 0.0002
            })
            
            engine.load_klines("OPT", klines)
            success, result = engine.run(signal_func, min_trades)
            
            if success:
                stats = result["stats"]
                
                # 根据优化指标计算分数
                if self.optimization_metric == "sharpe_ratio":
                    score = stats["sharpe_ratio"]
                elif self.optimization_metric == "total_pnl":
                    score = stats["total_pnl"]
                elif self.optimization_metric == "win_rate":
                    score = stats["win_rate"]
                elif self.optimization_metric == "calmar":
                    score = stats["total_pnl_pct"] / (abs(stats["max_drawdown_pct"]) + 0.01)
                else:
                    score = stats["sharpe_ratio"]
                
                all_results.append({
                    "params": params,
                    "stats": stats,
                    "score": score,
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        return {
            "success": True,
            "best_params": best_params,
            "best_score": best_score,
            "all_results": sorted(all_results, key=lambda x: x["score"], reverse=True)[:50],
        }
    
    def sensitivity_analysis(self, klines: List[Dict], base_params: Dict,
                             signal_template: Callable, min_trades: int = 10) -> Dict:
        """
        敏感性分析：逐个参数变化，观察对结果的影响
        
        Returns:
            各参数敏感性报告
        """
        sensitivity = {}
        
        for param_name, base_value in base_params.items():
            if param_name in self.param_ranges:
                range_values = self.param_ranges[param_name]
                scores = []
                
                for value in range_values:
                    test_params = base_params.copy()
                    test_params[param_name] = value
                    
                    def signal_func(prices, highs, lows, idx):
                        return signal_template(test_params, prices, highs, lows, idx)
                    
                    engine = BacktestEngine({"initial_balance": self.initial_balance})
                    engine.load_klines("SEN", klines)
                    success, result = engine.run(signal_func, min_trades)
                    
                    if success:
                        if self.optimization_metric == "sharpe_ratio":
                            scores.append(result["stats"]["sharpe_ratio"])
                        else:
                            scores.append(result["stats"]["total_pnl_pct"])
                
                if scores:
                    sensitivity[param_name] = {
                        "values": range_values,
                        "scores": scores,
                        "max_impact": max(scores) - min(scores) if len(scores) > 1 else 0,
                        "best_value": range_values[scores.index(max(scores))],
                    }
        
        return {
            "success": True,
            "sensitivity": sensitivity,
            "most_sensitive_param": max(
                sensitivity.items(),
                key=lambda x: x[1]["max_impact"]
            )[0] if sensitivity else None,
        }


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    "BacktestEngine",
    "WalkForwardEngine",
    "MultiCoinBacktest",
    "ParameterOptimizer",
    "calc_ic_simple",
    "calc_rank_ic",
]
