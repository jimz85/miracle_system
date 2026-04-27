"""
Miracle 1.0.1 - 回测报告模块
================================
回测结果报告生成

功能:
1. 格式化回测摘要
2. 便捷回测函数
3. 主入口测试
"""

import logging
from datetime import datetime
from typing import Dict, List, Callable

logger = logging.getLogger("miracle.backtest.reporter")

from dataclasses import asdict
from engine import BacktestEngine


# ============================================================
# 便捷函数
# ============================================================

def run_simple_backtest(
    symbol: str,
    klines: List[Dict],
    signal_generator: Callable,
    initial_balance: float = 100000,
    min_trades: int = 10
) -> Dict:
    """
    运行简单回测的便捷函数

    Args:
        symbol: 币种
        klines: K线数据
        signal_generator: 信号生成函数
        initial_balance: 初始资金
        min_trades: 最少交易次数

    Returns:
        回测结果字典
    """
    engine = BacktestEngine({
        "initial_balance": initial_balance,
        "commission_rate": 0.0005,
        "slippage_rate": 0.0002
    })

    engine.load_klines(symbol, klines)

    success, result = engine.run(signal_generator, min_trades)

    if success:
        return {
            "success": True,
            "symbol": symbol,
            "stats": result["stats"],
            "trades": result["trades"],
            "equity_curve": result["equity_curve"],
            "ic_stats": result.get("ic_stats"),
            "summary": _format_summary(symbol, result["stats"])
        }
    else:
        return {
            "success": False,
            "symbol": symbol,
            "error": result.get("error", "回测失败"),
            "trades": result.get("trades", []),
            "equity_curve": result.get("equity_curve", [])
        }


def _format_summary(symbol: str, stats: Dict) -> str:
    """格式化回测摘要"""
    ic_info = ""
    if stats.get('ic') is not None:
        ic_info = f"\nIC信息系数: {stats['ic']:.4f}"
        if stats.get('rank_ic') is not None:
            ic_info += f" | Rank IC: {stats['rank_ic']:.4f}"
        if stats.get('icir') is not None:
            ic_info += f" | ICIR: {stats['icir']:.4f}"
        ic_info += " (正值表示预测能力有效)"

    return f"""
{'='*50}
{symbol} 回测报告
{'='*50}
总交易次数: {stats['total_trades']}
胜率: {stats['win_rate']:.1%}
平均盈利: ${stats['avg_win']:.2f}
平均亏损: ${stats['avg_loss']:.2f}
盈亏比: {stats['avg_rr']:.2f}
总盈亏: ${stats['total_pnl']:.2f} ({stats['total_pnl_pct']:.1f}%)
夏普比率: {stats['sharpe_ratio']:.2f}
索提诺比率: {stats['sortino_ratio']:.2f}
最大回撤: ${stats['max_drawdown']:.2f} ({stats['max_drawdown_pct']:.1f}%)
平均持仓: {stats['avg_hold_hours']:.1f}小时
日均交易: {stats['avg_trades_per_day']:.2f}笔{ic_info}
{'='*50}
"""


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import random

    # 生成模拟K线数据
    base_price = 50000
    klines = []
    price = base_price
    for i in range(500):
        price = price * (1 + random.uniform(-0.02, 0.025))
        klines.append({
            "timestamp": int(datetime.now().timestamp() * 1000) - (500 - i) * 3600000,
            "open": price * 0.99,
            "high": price * 1.01,
            "low": price * 0.98,
            "close": price,
            "volume": random.uniform(1000, 10000)
        })

    # 简单信号生成器（用于演示）
    def demo_signal(prices, highs, lows, idx):
        if idx < 50:
            return None

        # 简单动量信号
        if idx % 20 == 0:  # 每20根K线发一次信号
            recent_return = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0

            if recent_return > 0.02:
                return {
                    "direction": "long",
                    "entry_price": prices[-1],
                    "stop_loss": prices[-1] * 0.97,
                    "take_profit": prices[-1] * 1.06,
                    "leverage": 2
                }
            elif recent_return < -0.02:
                return {
                    "direction": "short",
                    "entry_price": prices[-1],
                    "stop_loss": prices[-1] * 1.03,
                    "take_profit": prices[-1] * 0.94,
                    "leverage": 2
                }
        return None

    # 运行回测
    print("运行回测...")
    result = run_simple_backtest("BTC", klines, demo_signal, initial_balance=100000)

    if result["success"]:
        print(result["summary"])
        # 打印IC详情
        if result.get("ic_stats"):
            ic = result["ic_stats"]
            print("\n" + "="*50)
            print("IC (Information Coefficient) 分析")
            print("="*50)
            print(f"IC: {ic['ic']:.4f} (p-value: {ic['ic_pvalue']:.4f})")
            print(f"Rank IC: {ic['rank_ic']:.4f} (p-value: {ic['rank_ic_pvalue']:.4f})")
            print(f"IC均值: {ic['ic_mean']:.4f}")
            print(f"IC标准差: {ic['ic_std']:.4f}")
            print(f"ICIR: {ic['icir']:.4f}")
            if ic.get('factor_ic'):
                print("\n各因子IC:")
                for factor, factor_ic_val in sorted(ic['factor_ic'].items(), key=lambda x: -abs(x[1])):
                    print(f"  {factor}: {factor_ic_val:.4f}")
            print("="*50)
    else:
        print(f"回测失败: {result['error']}")
        print(f"交易次数: {len(result['trades'])}")
