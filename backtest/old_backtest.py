from __future__ import annotations

"""
Miracle 1.0.1 - 回测引擎 (重构版本)
====================================
基于历史数据的策略回测模块

功能:
1. 历史K线数据回测
2. 模拟交易执行
3. 性能指标计算（Sharpe, Sortino, MaxDD, WinRate）
4. 过拟合检测
5. 分币种/分时间段的详细报告

注意: 此模块已重构为三个子模块:
- engine.py: 回测引擎、WalkForward引擎、多币种回测、参数优化
- stats.py: 统计指标计算、IC信息系数
- reporter.py: 报告生成、便捷函数
"""

# 从子模块导入所有公开API，保持向后兼容
from .engine import (
    BacktestEngine,
    MultiCoinBacktest,
    ParameterOptimizer,
    WalkForwardEngine,
    calc_ic_simple,
    calc_rank_ic,
)
from .reporter import (
    _format_summary,
    run_simple_backtest,
)
from .stats import (
    BacktestStats,
    BacktestTrade,
    ICStats,
    calc_ic,
    calc_stats,
)

__all__ = [
    # engine
    "BacktestEngine",
    "WalkForwardEngine", 
    "MultiCoinBacktest",
    "ParameterOptimizer",
    "calc_ic_simple",
    "calc_rank_ic",
    # stats
    "BacktestTrade",
    "BacktestStats",
    "ICStats",
    "calc_stats",
    "calc_ic",
    # reporter
    "run_simple_backtest",
    "_format_summary",
]


if __name__ == "__main__":
    # 运行reporter中的演示代码
    from .reporter import _format_summary
    exec(open(__file__.replace('old_backtest.py', 'reporter.py')).read().split('if __name__ == "__main__":')[1])
