from __future__ import annotations

"""
Backtest Package (P2.1)
======================
回测框架包

子模块:
- backtest.py: 主入口,提供统一API
- walkforward.py: Walk-Forward滚动窗口验证引擎

Usage:
    from backtest import run_backtest, BacktestRunner
    
    # 简单运行
    result = run_backtest("BTC", klines_data)
    
    # 完整运行(含Walk-Forward)
    runner = BacktestRunner()
    runner.load_data("BTC", klines_data)
    runner.run_walkforward(strategy="both")
    runner.save_results("output.json")
"""

from .backtest import (
    BacktestConfig,
    BacktestRunner,
    load_klines_from_csv,
    run_backtest,
)
from .walkforward import (
    MEAN_REVERSION_PARAMS,
    TREND_FOLLOWING_PARAMS,
    StrategyType,
    WalkForwardResult,
    WalkForwardValidator,
    WindowResult,
    run_walk_forward,
)

__all__ = [
    # Walk-Forward
    "WalkForwardValidator",
    "WalkForwardResult", 
    "WindowResult",
    "StrategyType",
    "run_walk_forward",
    "MEAN_REVERSION_PARAMS",
    "TREND_FOLLOWING_PARAMS",
    # Main
    "BacktestRunner",
    "BacktestConfig",
    "run_backtest",
    "load_klines_from_csv",
]
