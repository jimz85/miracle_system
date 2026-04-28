"""
Miracle 1.0.1 - 回测统计模块
================================
回测性能指标计算

功能:
1. 统计指标计算（Sharpe, Sortino, MaxDD, WinRate）
2. IC信息系数计算
3. 数据结构定义
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("miracle.backtest.stats")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BacktestTrade:
    """回测交易记录"""
    entry_time: str
    exit_time: str
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    exit_price: float
    position_size: float
    leverage: float
    pnl: float
    pnl_pct: float
    stop_triggered: str  # "sl" | "tp" | "time" | "manual"
    commission: float
    slippage: float

@dataclass
class ICStats:
    """IC(Information Coefficient)统计 - 衡量因子预测能力"""
    ic: float                      # IC - 皮尔逊相关系数(预测信号强度 vs 实际收益)
    ic_pvalue: float               # IC显著性水平
    rank_ic: float                 # Rank IC - 斯皮尔曼相关系数(更鲁棒)
    rank_ic_pvalue: float         # Rank IC显著性
    ic_periods: List[float]        # 各周期IC值(用于时序稳定性)
    ic_mean: float                 # IC均值
    ic_std: float                  # IC标准差(ICIR)
    icir: float                   # IC均值/IC标准差
    factor_ic: Dict[str, float]   # 各因子单独IC

@dataclass
class BacktestStats:
    """回测统计"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_rr: float
    total_pnl: float
    total_pnl_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_hold_hours: float
    avg_trades_per_day: float
    ic: Optional[float] = None         # IC信息系数
    rank_ic: Optional[float] = None    # Rank IC
    icir: Optional[float] = None       # ICIR


# ============================================================
# 统计计算函数
# ============================================================

def calc_stats(trades: List[BacktestTrade], 
               equity_curve: List[float],
               initial_balance: float,
               ic_stats: Optional[ICStats] = None) -> BacktestStats:
    """
    计算回测统计指标
    
    Args:
        trades: 交易列表
        equity_curve: 权益曲线
        initial_balance: 初始资金
        ic_stats: IC统计（可选）
        
    Returns:
        BacktestStats对象
    """
    if not trades:
        return BacktestStats(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, avg_rr=0,
            total_pnl=0, total_pnl_pct=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, max_drawdown_pct=0,
            avg_hold_hours=0, avg_trades_per_day=0
        )

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_trades = len(trades)
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_win = sum(t.pnl for t in wins) / winning_trades if winning_trades > 0 else 0
    avg_loss = abs(sum(t.pnl for t in losses) / losing_trades) if losing_trades > 0 else 0

    # 计算RR
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

    total_pnl = sum(t.pnl for t in trades)
    total_pnl_pct = total_pnl / initial_balance * 100

    # 计算夏普比率
    if HAS_NUMPY and len(equity_curve) > 1:
        equity_arr = np.array(equity_curve)
        daily_returns = np.diff(equity_arr) / equity_arr[:-1]
        daily_returns = daily_returns[daily_returns != 0]  # 过滤零值
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino比率
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0
    else:
        sharpe = sortino = 0

    # 最大回撤
    equity = np.array(equity_curve) if HAS_NUMPY else equity_curve
    running_max = np.maximum.accumulate(equity) if HAS_NUMPY else equity_curve
    drawdowns = (equity - running_max) / running_max
    max_drawdown_pct = abs(drawdowns.min()) * 100 if HAS_NUMPY else 0
    if HAS_NUMPY:
        dd_idx = drawdowns.argmin()
        max_drawdown = abs(equity[dd_idx] - running_max[dd_idx])
    else:
        max_drawdown = abs(min(equity) - max(equity))

    # 平均持仓时间
    hold_times = []
    for t in trades:
        if isinstance(t.exit_time, (int, float)) and isinstance(t.entry_time, (int, float)):
            entry_dt = datetime.fromtimestamp(t.entry_time / 1000)
            exit_dt = datetime.fromtimestamp(t.exit_time / 1000)
        else:
            entry_dt = datetime.fromisoformat(str(t.entry_time))
            exit_dt = datetime.fromisoformat(str(t.exit_time))
        hold_times.append((exit_dt - entry_dt).total_seconds() / 3600)
    avg_hold_hours = sum(hold_times) / len(hold_times) if hold_times else 0

    # 日均交易次数
    if trades:
        first_trade_time = trades[0].entry_time
        if isinstance(first_trade_time, (int, float)):
            first_dt = datetime.fromtimestamp(first_trade_time / 1000)
        else:
            first_dt = datetime.fromisoformat(str(first_trade_time))
        last_trade_time = trades[-1].entry_time
        if isinstance(last_trade_time, (int, float)):
            last_dt = datetime.fromtimestamp(last_trade_time / 1000)
        else:
            last_dt = datetime.fromisoformat(str(last_trade_time))

        days = max(1, (last_dt - first_dt).days)
        avg_trades_per_day = len(trades) / days
    else:
        avg_trades_per_day = 0

    return BacktestStats(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_rr=avg_rr,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        avg_hold_hours=avg_hold_hours,
        avg_trades_per_day=avg_trades_per_day,
        ic=ic_stats.ic if ic_stats else None,
        rank_ic=ic_stats.rank_ic if ic_stats else None,
        icir=ic_stats.icir if ic_stats else None,
    )


def calc_ic(ic_signals: List[Dict]) -> Optional[ICStats]:
    """
    计算IC(Information Coefficient)信息系数

    IC衡量信号预测能力:
    - IC (Pearson): 预测信号强度 vs 实际收益的相关性
    - Rank IC (Spearman): 更鲁棒，对异常值不敏感

    IC > 0.05 通常表示因子有预测能力
    ICIR (IC Mean/IC Std) > 0.5 表示因子稳定有效
    
    Args:
        ic_signals: 信号列表，每项包含 signal_direction, actual_return, factors, confidence
        
    Returns:
        ICStats对象或None
    """
    if not ic_signals or len(ic_signals) < 5:
        logger.warning(f"IC信号不足 ({len(ic_signals) if ic_signals else 0} < 5)，无法计算IC")
        return None

    # 提取预测方向和实际收益
    pred_directions = np.array([s["signal_direction"] for s in ic_signals])
    actual_returns = np.array([s["actual_return"] for s in ic_signals])
    confidences = np.array([s.get("confidence", 0.5) for s in ic_signals])

    # 过滤无效值
    valid_mask = ~(np.isnan(actual_returns) | np.isnan(pred_directions))
    if valid_mask.sum() < 5:
        return None

    pred_directions = pred_directions[valid_mask]
    actual_returns = actual_returns[valid_mask]
    confidences = confidences[valid_mask]

    # 1. 计算IC (Pearson相关系数)
    if HAS_SCIPY:
        ic_raw, ic_pvalue = scipy_stats.pearsonr(pred_directions, actual_returns)
        # 2. 计算Rank IC (Spearman相关系数)
        rank_ic_raw, rank_ic_pvalue = scipy_stats.spearmanr(pred_directions, actual_returns)
    elif HAS_NUMPY:
        # 无scipy时用numpy近似计算
        ic_raw = np.corrcoef(pred_directions, actual_returns)[0, 1]
        ic_pvalue = 0.0
        rank_ic_raw = np.corrcoef(np.argsort(np.argsort(pred_directions)),
                                   np.argsort(np.argsort(actual_returns)))[0, 1]
        rank_ic_pvalue = 0.0
    else:
        # 纯Python实现（简化版）
        ic_raw = _python_corr(pred_directions, actual_returns)
        rank_ic_raw = ic_raw
        ic_pvalue = rank_ic_pvalue = 0.0

    # 处理NaN
    ic = float(ic_raw) if not np.isnan(ic_raw) else 0.0
    rank_ic = float(rank_ic_raw) if not np.isnan(rank_ic_raw) else 0.0

    # 3. 计算IC时间序列统计 (按时间顺序的IC)
    ic_periods = []
    for j in range(0, len(ic_signals) - 1, max(1, len(ic_signals) // 10)):
        end_idx = min(j + max(1, len(ic_signals) // 10), len(ic_signals))
        if end_idx > j + 2:
            period_pred = pred_directions[j:end_idx]
            period_ret = actual_returns[j:end_idx]
            if len(period_pred) > 2 and np.std(period_pred) > 0 and np.std(period_ret) > 0:
                if HAS_NUMPY:
                    period_ic = np.corrcoef(period_pred, period_ret)[0, 1]
                else:
                    period_ic = _python_corr(period_pred, period_ret)
                if not np.isnan(period_ic):
                    ic_periods.append(float(period_ic))

    # 4. 计算IC均值和标准差
    if ic_periods:
        ic_mean = np.mean(ic_periods) if HAS_NUMPY else sum(ic_periods) / len(ic_periods)
        ic_std = np.std(ic_periods) if HAS_NUMPY else _python_std(ic_periods)
    else:
        ic_mean = ic
        ic_std = 0.0

    # ICIR = IC均值 / IC标准差
    icir = ic_mean / ic_std if ic_std > 0 else 0.0

    # 5. 计算各因子IC (如果提供了因子数据)
    factor_ic = {}
    if ic_signals and isinstance(ic_signals[0].get("factors"), dict):
        # 收集所有因子
        all_factors = set()
        for s in ic_signals:
            all_factors.update(s.get("factors", {}).keys())

        for factor in all_factors:
            factor_preds = []
            factor_returns = []
            for s in ic_signals:
                factor_val = s.get("factors", {}).get(factor)
                if factor_val is not None:
                    factor_preds.append(float(factor_val))
                    factor_returns.append(s["actual_return"])

            if len(factor_preds) >= 5:
                if HAS_NUMPY:
                    f_ic = np.corrcoef(factor_preds, factor_returns)[0, 1]
                else:
                    f_ic = _python_corr(factor_preds, factor_returns)
                if not np.isnan(f_ic):
                    factor_ic[factor] = float(f_ic)

    logger.info(f"IC计算完成: IC={ic:.4f}, Rank IC={rank_ic:.4f}, ICIR={icir:.4f}, 因子数={len(factor_ic)}")

    return ICStats(
        ic=ic,
        ic_pvalue=float(ic_pvalue) if ic_pvalue else 0.0,
        rank_ic=rank_ic,
        rank_ic_pvalue=float(rank_ic_pvalue) if rank_ic_pvalue else 0.0,
        ic_periods=ic_periods,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        factor_ic=factor_ic
    )


def _python_corr(x: List[float], y: List[float]) -> float:
    """纯Python皮尔逊相关系数（无numpy时备用）"""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = (sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
    return num / den if den != 0 else 0.0


def _python_std(data: List[float]) -> float:
    """纯Python标准差（无numpy时备用）"""
    n = len(data)
    if n < 2:
        return 0.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return variance ** 0.5


def calc_ic_simple(train_returns: List[float], test_returns: List[float]) -> float:
    """
    计算信息系数（Information Coefficient）
    
    Args:
        train_returns: 训练期收益列表
        test_returns: 测试期收益列表
        
    Returns:
        IC值 (-1 ~ 1)
    """
    if len(train_returns) != len(test_returns) or len(train_returns) < 2:
        return 0.0
    
    if HAS_NUMPY:
        return float(np.corrcoef(train_returns, test_returns)[0, 1])
    
    # 手动计算Pearson相关系数
    n = len(train_returns)
    mean_train = sum(train_returns) / n
    mean_test = sum(test_returns) / n
    
    numerator = sum((train_returns[i] - mean_train) * (test_returns[i] - mean_test) for i in range(n))
    denom_train = sum((x - mean_train) ** 2 for x in train_returns) ** 0.5
    denom_test = sum((x - mean_test) ** 2 for x in test_returns) ** 0.5
    
    if denom_train == 0 or denom_test == 0:
        return 0.0
    
    return numerator / (denom_train * denom_test)


def calc_rank_ic(train_returns: List[float], test_returns: List[float]) -> float:
    """
    计算排名信息系数（Rank IC）
    对极端值更稳健
    """
    if len(train_returns) != len(test_returns) or len(train_returns) < 2:
        return 0.0
    
    # 排名
    train_ranks = sorted(range(len(train_returns)), key=lambda i: train_returns[i])
    test_ranks = sorted(range(len(test_returns)), key=lambda i: test_returns[i])
    
    # 重新排列使排名一致
    train_sorted = [train_returns[i] for i in train_ranks]
    test_sorted = [test_returns[i] for i in test_ranks]
    
    return calc_ic_simple(train_sorted, test_sorted)
