from __future__ import annotations

"""
Agent-L: 学习迭代Agent for Miracle 1.0.1
高频趋势跟踪+事件驱动混合系统

职责:
1. 记录每笔交易结果
2. 更新因子权重
3. 淘汰表现差的因子
4. 发现新模式
5. 每周生成学习报告
6. 每月调整策略参数
"""

import json
import logging
import os
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 0. 真正的IC计算与Walk-Forward验证
# =============================================================================

def calc_information_coefficient(signals: List[float], returns: List[float]) -> Tuple[float, float]:
    """
    计算真正的Information Coefficient (IC)
    IC = Pearson相关系数(信号, 收益)

    Args:
        signals: 因子信号值列表
        returns: 对应的收益列表

    Returns:
        ic: IC值 (-1~1)
        p_value: 统计显著性
    """
    if len(signals) < 10:  # 至少需要10个样本
        return 0.0, 1.0

    # 转换为numpy数组
    signals_arr = np.array(signals)
    returns_arr = np.array(returns)

    # 过滤掉nan和inf
    valid_mask = np.isfinite(signals_arr) & np.isfinite(returns_arr)
    if np.sum(valid_mask) < 10:
        return 0.0, 1.0

    signals_arr = signals_arr[valid_mask]
    returns_arr = returns_arr[valid_mask]

    # 计算Pearson相关系数
    try:
        from scipy import stats as scipy_stats
        ic, p_value = scipy_stats.pearsonr(signals_arr, returns_arr)
        return float(ic), float(p_value)
    except Exception:
        # Fallback: 手动计算Pearson相关系数
        if np.std(signals_arr) == 0 or np.std(returns_arr) == 0:
            return 0.0, 1.0
        covariance = np.mean((signals_arr - np.mean(signals_arr)) * (returns_arr - np.mean(returns_arr)))
        ic = covariance / (np.std(signals_arr) * np.std(returns_arr))
        return float(ic), 1.0


def orthogonalize_factors(factors_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    因子正交化处理 - 去除共线性

    对价格类因子(RSI, MACD, ADX, Momentum)进行PCA处理，
    提取独立的主成分，减少共线性影响

    Args:
        factors_dict: 包含各因子值的字典

    Returns:
        添加了正交化因子（*_orthogonalized）的字典
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        # sklearn not available, return original dict
        return factors_dict

    # 提取价格因子
    price_factors = ["rsi", "macd", "adx", "momentum"]
    factor_matrix = []
    factor_names = []

    for name in price_factors:
        if name in factors_dict:
            value = factors_dict[name]
            # Handle case where value might be a list (historical data)
            if isinstance(value, list):
                if len(value) > 0:
                    factor_matrix.append(value)
                    factor_names.append(name)
            else:
                # Single value - can't do PCA with single values
                pass

    if len(factor_matrix) < 2:
        return factors_dict  # 因子不足，无法PCA

    # 转置：每列是一个因子，每行是一个时间点
    try:
        X = np.array(factor_matrix).T

        if X.shape[0] < 2 or X.shape[1] < 2:
            return factors_dict

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA降维，保留所有主成分
        n_components = min(len(factor_names), X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # 用主成分替换原始因子（添加正交化版本）
        for i, name in enumerate(factor_names):
            if i < X_pca.shape[1]:
                factors_dict[f"{name}_orthogonalized"] = float(X_pca[0, i]) if len(X_pca.shape) > 1 else float(X_pca[0])

        # 添加解释方差比例
        factors_dict["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist() if hasattr(pca.explained_variance_ratio_, 'tolist') else list(pca.explained_variance_ratio_)

    except Exception:
        pass  # PCA failed, return original dict

    return factors_dict


class WalkForwardValidator:
    """
    Walk-Forward分析 - 时序验证框架

    将数据分割为多个样本内/样本外窗口，
    验证策略在不同时间段的泛化能力
    """

    def __init__(self, train_ratio: float = 0.7, window_step: int = None):
        """
        Args:
            train_ratio: 样本内数据比例
            window_step: 滑动窗口步长（默认窗口大小的10%）
        """
        self.train_ratio = train_ratio
        self.window_step = window_step

    def split_data(self, data: List[Dict], n_windows: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        生成分割点列表

        Args:
            data: 历史数据列表
            n_windows: 分割窗口数

        Returns:
            splits: [(train_start, train_end, test_start, test_end), ...]
        """
        n = len(data)
        window_size = n // n_windows
        self.window_step or (window_size // 10)

        if window_size < 10:
            return []  # 数据太少

        splits = []
        for i in range(n_windows - 1):
            train_end = int((i + self.train_ratio) * window_size)
            test_start = train_end
            test_end = min(test_start + window_size - train_end, n)
            splits.append((i * window_size, train_end, test_start, test_end))

        return splits

    def validate(self, strategy_func, data: List[Dict], n_windows: int = 5) -> Dict[str, Any]:
        """
        验证策略的Walk-Forward表现

        Args:
            strategy_func: 策略函数，接受训练数据，返回策略参数和信号生成函数
            data: 历史数据列表，每个元素应包含 'return' 字段
            n_windows: 分割窗口数

        Returns:
            results: {
                "train_ic": [...],  # 样本内IC
                "test_ic": [...],   # 样本外IC
                "decay": [...],     # IC衰减（样本内-样本外）
                "avg_decay": float, # 平均IC衰减
                "avg_train_ic": float,
                "avg_test_ic": float
            }
        """
        splits = self.split_data(data, n_windows)
        if not splits:
            return {
                "train_ic": [],
                "test_ic": [],
                "decay": [],
                "avg_decay": 0.0,
                "avg_train_ic": 0.0,
                "avg_test_ic": 0.0
            }

        train_ics = []
        test_ics = []

        for train_start, train_end, test_start, test_end in splits:
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]

            if len(train_data) < 10 or len(test_data) < 5:
                continue

            try:
                # 样本内训练
                strategy_params = strategy_func(train_data)

                # 获取信号和收益
                train_signals = [self._get_signal(d, strategy_params) for d in train_data]
                train_returns = [d.get("return", 0.0) for d in train_data]
                train_ic, _ = calc_information_coefficient(train_signals, train_returns)
                train_ics.append(train_ic)

                # 样本外IC
                test_signals = [self._get_signal(d, strategy_params) for d in test_data]
                test_returns = [d.get("return", 0.0) for d in test_data]
                test_ic, _ = calc_information_coefficient(test_signals, test_returns)
                test_ics.append(test_ic)

            except Exception:
                continue

        if not train_ics or not test_ics:
            return {
                "train_ic": [],
                "test_ic": [],
                "decay": [],
                "avg_decay": 0.0,
                "avg_train_ic": 0.0,
                "avg_test_ic": 0.0
            }

        # 确保长度一致
        min_len = min(len(train_ics), len(test_ics))
        train_ics = train_ics[:min_len]
        test_ics = test_ics[:min_len]

        return {
            "train_ic": train_ics,
            "test_ic": test_ics,
            "decay": [t - e for t, e in zip(train_ics, test_ics)],
            "avg_decay": float(np.mean([t - e for t, e in zip(train_ics, test_ics)])) if train_ics and test_ics else 0.0,
            "avg_train_ic": float(np.mean(train_ics)),
            "avg_test_ic": float(np.mean(test_ics))
        }

    def _get_signal(self, data_point: Dict, strategy_params: Dict) -> float:
        """
        从数据点提取信号

        Args:
            data_point: 包含因子数据的字典
            strategy_params: 策略参数

        Returns:
            signal: 信号值 (-1~1)
        """
        # 默认实现：使用RSI作为信号
        rsi = data_point.get("rsi", 50)
        if rsi < 30:
            return 1.0  # 做多信号
        elif rsi > 70:
            return -1.0  # 做空信号
        else:
            return 0.0  # 中性


# =============================================================================
# 1. 交易记录存储
# =============================================================================

class TradeRecorder:
    """交易记录存储 - 使用SQLite"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,  -- 'long' or 'short'
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    pnl REAL,
                    rr REAL,  -- risk reward ratio
                    won INTEGER,  -- 1=win, 0=loss
                    factors TEXT,  -- JSON of factor signals at entry
                    pattern_key TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    market_regime TEXT,  -- 'BULL', 'BEAR', 'RANGE'
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entry_time ON trades(entry_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pattern ON trades(pattern_key)
            """)

    def record_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        记录完整交易
        trade_data: {
            symbol, direction, entry_time, entry_price,
            exit_time, exit_price, stop_loss, take_profit,
            factors (dict), pattern_key, market_regime
        }
        """
        entry_time = trade_data['entry_time']
        entry_price = trade_data['entry_price']
        exit_time = trade_data.get('exit_time')
        exit_price = trade_data.get('exit_price')
        stop_loss = trade_data.get('stop_loss')
        take_profit = trade_data.get('take_profit')

        pnl = None
        rr = None
        won = None

        if exit_price and exit_price > 0:
            if trade_data['direction'] == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price

            if stop_loss and stop_loss > 0:
                risk = abs(entry_price - stop_loss) / entry_price
                if risk > 0:
                    rr = abs(pnl / risk)
                    won = 1 if pnl > 0 else 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO trades (
                    symbol, direction, entry_time, entry_price,
                    exit_time, exit_price, pnl, rr, won,
                    factors, pattern_key, stop_loss, take_profit, market_regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['symbol'],
                trade_data['direction'],
                entry_time,
                entry_price,
                exit_time,
                exit_price,
                pnl,
                rr,
                won,
                json.dumps(trade_data.get('factors', {})),
                trade_data.get('pattern_key'),
                stop_loss,
                take_profit,
                trade_data.get('market_regime', 'RANGE')
            ))
            return cursor.lastrowid

    def get_trades(
        self,
        symbol: str | None = None,
        days: int = 30,
        direction: str | None = None
    ) -> List[Dict[str, Any]]:
        """查询交易记录"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        query = "SELECT * FROM trades WHERE entry_time >= ?"
        params = [cutoff]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if direction:
            query += " AND direction = ?"
            params.append(direction)

        query += " ORDER BY entry_time DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_stats(self, lookback_days: int = 30) -> Dict[str, Any]:
        """获取统计数据"""
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # 总交易数
            total = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE entry_time >= ? AND pnl IS NOT NULL",
                [cutoff]
            ).fetchone()[0]

            if total == 0:
                return {
                    "total_trades": 0, "win_rate": 0, "avg_rr": 0,
                    "best_trade": 0, "worst_trade": 0, "total_pnl": 0,
                    "sharpe_ratio": 0, "max_drawdown": 0
                }

            # 胜率
            wins = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE entry_time >= ? AND won = 1 AND pnl IS NOT NULL",
                [cutoff]
            ).fetchone()[0]
            win_rate = wins / total if total > 0 else 0

            # 平均RR
            avg_rr = conn.execute(
                "SELECT AVG(rr) FROM trades WHERE entry_time >= ? AND rr IS NOT NULL",
                [cutoff]
            ).fetchone()[0] or 0

            # 最佳/最差交易
            best = conn.execute(
                "SELECT MAX(pnl) FROM trades WHERE entry_time >= ?",
                [cutoff]
            ).fetchone()[0] or 0
            worst = conn.execute(
                "SELECT MIN(pnl) FROM trades WHERE entry_time >= ?",
                [cutoff]
            ).fetchone()[0] or 0

            # 总PnL
            total_pnl = conn.execute(
                "SELECT SUM(pnl) FROM trades WHERE entry_time >= ? AND pnl IS NOT NULL",
                [cutoff]
            ).fetchone()[0] or 0

            # 计算Sharpe Ratio (简化版)
            pnls = conn.execute(
                "SELECT pnl FROM trades WHERE entry_time >= ? AND pnl IS NOT NULL ORDER BY entry_time",
                [cutoff]
            ).fetchall()
            pnls = [p[0] for p in pnls]

            sharpe = 0
            max_dd = 0
            if len(pnls) > 1:
                import statistics
                mean_pnl = statistics.mean(pnls)
                stdev_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 0
                sharpe = (mean_pnl / stdev_pnl * (252 ** 0.5)) if stdev_pnl > 0 else 0

                # Max Drawdown
                cumulative = []
                running = 0
                for p in pnls:
                    running += p
                    cumulative.append(running)
                peak = cumulative[0]
                for c in cumulative:
                    if c > peak:
                        peak = c
                    dd = peak - c
                    if dd > max_dd:
                        max_dd = dd

            return {
                "total_trades": total,
                "win_rate": round(win_rate, 3),
                "avg_rr": round(avg_rr, 2),
                "best_trade": round(best, 4),
                "worst_trade": round(worst, 4),
                "total_pnl": round(total_pnl, 4),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_dd, 4)
            }


# =============================================================================
# 2. 因子表现分析
# =============================================================================

class FactorAnalyzer:
    """因子表现分析 - IC计算与权重更新"""

    def __init__(self, db_path: str, weights_path: str):
        self.db_path = db_path
        self.weights_path = weights_path
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self.factor_ic_history: Dict[str, List[float]] = defaultdict(list)
        self.factor_wins: Dict[str, List[int]] = defaultdict(list)
        self._load_weights()

    def _load_weights(self):
        """加载因子权重"""
        if os.path.exists(self.weights_path):
            with open(self.weights_path) as f:
                data = json.load(f)
                self.weights = data.get('weights', self._default_weights())
                self.ic_history = data.get('ic_history', {})
                self.factor_wins = defaultdict(list, self.ic_history)
        else:
            self.weights = self._default_weights()
            self.ic_history = {}

    def _default_weights(self) -> Dict[str, float]:
        """默认因子权重"""
        return {
            "rsi": 0.30,
            "macd": 0.20,
            "bollinger": 0.15,
            "volume": 0.10,
            "news_sentiment": 0.10,
            "onchain": 0.10,
            "market_regime": 0.05
        }

    def _save_weights(self):
        """保存因子权重"""
        with open(self.weights_path, 'w') as f:
            json.dump({
                'weights': self.weights,
                'ic_history': dict(self.factor_wins),
                'updated_at': datetime.now().isoformat()
            }, f, indent=2)

    def analyze_factor_performance(
        self,
        factor_name: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        分析单个因子的表现
        计算IC (Information Coefficient) = Pearson相关性(信号, 收益)
        """
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # 获取所有交易
            trades = conn.execute(
                "SELECT factors, pnl FROM trades WHERE entry_time >= ? AND pnl IS NOT NULL",
                [cutoff]
            ).fetchall()

            if len(trades) < 10:
                return {"ic": 0, "ic_p_value": 1.0, "signal_count": 0, "avg_pnl_with": 0, "avg_pnl_without": 0}

            # 提取信号和收益序列
            signals = []
            returns = []

            for t in trades:
                factors = json.loads(t['factors']) if t['factors'] else {}
                signal_value = factors.get(factor_name, 0.0)
                # 将信号值标准化到 -1~1 范围
                if isinstance(signal_value, (int, float)):
                    signal = max(-1.0, min(1.0, float(signal_value)))
                else:
                    signal = 0.0
                signals.append(signal)
                returns.append(t['pnl'])

            # 计算真正的IC (Pearson相关系数)
            ic, p_value = calc_information_coefficient(signals, returns)

            # 分离有信号和无信号的收益（用于辅助分析）
            with_signal = [returns[i] for i, s in enumerate(signals) if abs(s) > 0.1]
            without_signal = [returns[i] for i, s in enumerate(signals) if abs(s) <= 0.1]

            avg_with = sum(with_signal) / len(with_signal) if with_signal else 0
            avg_without = sum(without_signal) / len(without_signal) if without_signal else 0

            win_rate_with = sum(1 for p in with_signal if p > 0) / len(with_signal) if with_signal else 0

            return {
                "ic": round(ic, 4),
                "ic_p_value": round(p_value, 4),
                "signal_count": len(with_signal),
                "total_trades": len(trades),
                "avg_pnl_with": round(avg_with, 4),
                "avg_pnl_without": round(avg_without, 4),
                "win_rate_with": round(win_rate_with, 3),
                "trend": self._get_ic_trend(factor_name)
            }

    def _get_ic_trend(self, factor_name: str) -> str:
        """判断IC趋势"""
        history = self.factor_wins.get(factor_name, [])
        if len(history) < 5:
            return "insufficient_data"

        recent = history[-5:]
        if all(x > 0.05 for x in recent):
            return "improving"
        elif all(x < 0.02 for x in recent):
            return "declining"
        return "stable"

    def update_factor_weights(self) -> Dict[str, float]:
        """
        根据历史表现更新因子权重
        规则:
        - 因子IC持续<0.02超过5天 → 权重降低50%
        - 因子IC>0.05持续5天 → 权重恢复
        - 因子胜率<35% → 移入观察名单
        """
        watchlist = []
        changes = []

        for factor in self.weights:
            analysis = self.analyze_factor_performance(factor)

            ic = analysis['ic']
            wr = analysis.get('win_rate_with', 0)

            # 记录IC历史
            self.factor_wins[factor].append(ic)
            if len(self.factor_wins[factor]) > 30:
                self.factor_wins[factor] = self.factor_wins[factor][-30:]

            # 权重调整
            old_weight = self.weights[factor]

            if ic < 0.02 and len(self.factor_wins[factor]) >= 5:
                # 连续5天IC低于阈值，权重降低50%
                recent = self.factor_wins[factor][-5:]
                if all(x < 0.02 for x in recent):
                    self.weights[factor] *= 0.5
                    changes.append(f"{factor}: {old_weight:.3f} -> {self.weights[factor]:.3f} (IC偏低)")

            elif ic > 0.05 and len(self.factor_wins[factor]) >= 5:
                # 连续5天IC高于阈值，权重恢复(上限)
                recent = self.factor_wins[factor][-5:]
                if all(x > 0.05 for x in recent):
                    default = self._default_weights()[factor]
                    self.weights[factor] = min(self.weights[factor] * 1.2, default * 1.5)
                    changes.append(f"{factor}: {old_weight:.3f} -> {self.weights[factor]:.3f} (IC优秀)")

            # 胜率过低加入观察名单
            if wr > 0 and wr < 0.35:
                watchlist.append(factor)

        self._save_weights()

        return {
            "weights": self.weights,
            "changes": changes,
            "watchlist": watchlist
        }

    def get_weights(self) -> Dict[str, float]:
        """获取当前因子权重"""
        return self.weights.copy()


# =============================================================================
# 3. 模式学习
# =============================================================================

class PatternLearner:
    """模式学习 - 从交易中提取和学习模式"""

    def __init__(self, pattern_db_path: str):
        self.pattern_db_path = pattern_db_path
        os.makedirs(os.path.dirname(pattern_db_path), exist_ok=True)
        self.pattern_db: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """加载模式库"""
        if os.path.exists(self.pattern_db_path):
            with open(self.pattern_db_path) as f:
                self.pattern_db = json.load(f)
        else:
            self.pattern_db = {}

    def _save(self):
        """保存模式库（原子写）"""
        fd, tmp = tempfile.mkstemp(suffix='.json.tmp', dir=os.path.dirname(self.pattern_db_path))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.pattern_db, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.pattern_db_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def extract_pattern(self, trade_data: Dict[str, Any]) -> str:
        """
        从交易中提取模式
        pattern_key: "long_RSI30-40_ADX30+_BULL"
        """
        direction = trade_data.get('direction', 'long')
        factors = trade_data.get('factors', {})
        regime = trade_data.get('market_regime', 'RANGE')

        parts = [direction.upper()]

        # RSI
        rsi = factors.get('rsi')
        if rsi is not None:
            if rsi < 30:
                parts.append("RSI<30")
            elif rsi < 40:
                parts.append("RSI30-40")
            elif rsi > 70:
                parts.append("RSI>70")
            elif rsi > 60:
                parts.append("RSI60-70")

        # ADX
        adx = factors.get('adx')
        if adx is not None:
            if adx > 30:
                parts.append("ADX30+")
            elif adx > 20:
                parts.append("ADX20+")

        # MACD
        macd = factors.get('macd_signal')
        if macd is not None:
            parts.append("MACD_BULL" if macd > 0 else "MACD_BEAR")

        # Volume
        vol = factors.get('volume_ratio')
        if vol is not None and vol > 1.5:
            parts.append("HIGH_VOL")

        # Market regime
        if regime:
            parts.append(regime.upper())

        return "_".join(parts)

    def update_pattern(self, pattern_key: str, trade_result: Dict[str, Any]):
        """更新模式表现"""
        if not pattern_key:
            return

        if pattern_key not in self.pattern_db:
            self.pattern_db[pattern_key] = {
                "wins": 0,
                "losses": 0,
                "total_rr": 0,
                "rr_count": 0,
                "last_updated": datetime.now().isoformat()
            }

        p = self.pattern_db[pattern_key]
        p["wins"] += 1 if trade_result["won"] else 0
        p["losses"] += 1 if not trade_result["won"] else 0
        if trade_result.get("rr"):
            p["total_rr"] += trade_result["rr"]
            p["rr_count"] += 1
        p["last_updated"] = datetime.now().isoformat()

        self._save()

    def get_pattern_stats(self, pattern_key: str) -> Dict[str, Any]:
        """获取模式统计"""
        if pattern_key not in self.pattern_db:
            return {
                "total": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "avg_rr": 0
            }

        p = self.pattern_db[pattern_key]
        total = p["wins"] + p["losses"]
        win_rate = p["wins"] / total if total > 0 else 0
        avg_rr = p["total_rr"] / p["rr_count"] if p["rr_count"] > 0 else 0

        return {
            "total": total,
            "wins": p["wins"],
            "losses": p["losses"],
            "win_rate": round(win_rate, 3),
            "avg_rr": round(avg_rr, 2)
        }

    def get_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模式统计"""
        return {
            k: self.get_pattern_stats(k)
            for k in self.pattern_db
        }

    def add_to_blacklist(self, pattern_keys: List[str]):
        """添加模式到黑名单"""
        blacklist_path = self.pattern_db_path.replace('pattern_db.json', 'pattern_blacklist.json')
        blacklist = set()
        if os.path.exists(blacklist_path):
            with open(blacklist_path) as f:
                blacklist = set(json.load(f))
        blacklist.update(pattern_keys)
        fd, tmp = tempfile.mkstemp(suffix='.json.tmp', dir=os.path.dirname(blacklist_path))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(list(blacklist), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, blacklist_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        logger.info(f"Added patterns to blacklist: {pattern_keys}")


# =============================================================================
# 4. 白名单管理
# =============================================================================

class WhitelistManager:
    """基于历史表现管理交易白名单"""

    def __init__(self, whitelist_path: str, pattern_db_path: str):
        self.whitelist_path = whitelist_path
        self.pattern_db_path = pattern_db_path
        os.makedirs(os.path.dirname(whitelist_path), exist_ok=True)
        self.whitelist: List[str] = []
        self._load()

    def _load(self):
        """加载白名单"""
        if os.path.exists(self.whitelist_path):
            with open(self.whitelist_path) as f:
                data = json.load(f)
                self.whitelist = data.get('patterns', [])
                self.last_update = data.get('last_update', '')
        else:
            self.whitelist = []
            self.last_update = ''

    def _save(self):
        """保存白名单（原子写）"""
        fd, tmp = tempfile.mkstemp(suffix='.json.tmp', dir=os.path.dirname(self.whitelist_path))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump({
                    'patterns': self.whitelist,
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.whitelist_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def add_to_whitelist(
        self,
        pattern_key: str,
        min_wr: float = 0.45,
        min_trades: int = 5
    ):
        """将模式加入白名单"""
        if not pattern_key:
            return False

        # 从pattern_db读取统计
        pattern_db = {}
        if os.path.exists(self.pattern_db_path):
            with open(self.pattern_db_path) as f:
                pattern_db = json.load(f)

        p = pattern_db.get(pattern_key, {})
        total = p.get('wins', 0) + p.get('losses', 0)
        win_rate = p.get('wins', 0) / total if total > 0 else 0

        if total >= min_trades and win_rate >= min_wr:
            if pattern_key not in self.whitelist:
                self.whitelist.append(pattern_key)
                self._save()
            return True
        return False

    def remove_from_whitelist(self, pattern_key: str):
        """将模式移出白名单"""
        if pattern_key in self.whitelist:
            self.whitelist.remove(pattern_key)
            self._save()

    def get_whitelist(self) -> List[str]:
        """获取当前白名单"""
        return self.whitelist.copy()

    def is_whitelisted(self, pattern_key: str) -> bool:
        """检查模式是否在白名单"""
        return pattern_key in self.whitelist

    def auto_update_whitelist(self, pattern_learner: PatternLearner) -> Dict[str, Any]:
        """
        每月自动更新白名单
        淘汰：胜率<40% 或 平均RR<1.5
        新增：连续5笔盈利 + 胜率>50%
        """
        removed = []
        added = []
        kept = []

        # 评估现有模式
        new_whitelist = []
        for pattern_key in self.whitelist:
            stats = pattern_learner.get_pattern_stats(pattern_key)
            total = stats['total']
            wr = stats['win_rate']
            avg_rr = stats['avg_rr']

            if total >= 5:
                if wr >= 0.40 and avg_rr >= 1.5:
                    new_whitelist.append(pattern_key)
                    kept.append(pattern_key)
                else:
                    removed.append(pattern_key)

        # 扫描所有模式找新增候选
        all_patterns = pattern_learner.get_all_patterns()

        for pattern_key, stats in all_patterns.items():
            if pattern_key in new_whitelist:
                continue

            total = stats['total']
            wr = stats['win_rate']

            if total >= 5 and wr > 0.50:
                # 简单版：胜率>50%且总交易>=5
                if self.add_to_whitelist(pattern_key, min_wr=0.50, min_trades=5):
                    added.append(pattern_key)
                    new_whitelist.append(pattern_key)

        self.whitelist = new_whitelist
        self._save()

        return {
            "kept": kept,
            "removed": removed,
            "added": added,
            "total_whitelist": len(self.whitelist)
        }


# =============================================================================
# 5. 学习报告
# =============================================================================

class LearningReport:
    """学习报告生成"""

    def __init__(
        self,
        db_path: str,
        weights_path: str,
        pattern_db_path: str,
        whitelist_path: str
    ):
        self.trade_recorder = TradeRecorder(db_path)
        self.factor_analyzer = FactorAnalyzer(db_path, weights_path)
        self.pattern_learner = PatternLearner(pattern_db_path)
        self.whitelist_manager = WhitelistManager(whitelist_path, pattern_db_path)

    def generate_weekly_report(self) -> Dict[str, Any]:
        """生成每周学习报告"""
        # 计算周日期范围
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        period = f"{week_ago.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')}"

        # 获取周统计
        week_trades = self.trade_recorder.get_trades(days=7)
        completed = [t for t in week_trades if t.get('pnl') is not None]

        trades_count = len(completed)
        win_rate = 0
        total_pnl = 0

        if completed:
            wins = sum(1 for t in completed if t['won'] == 1)
            win_rate = wins / len(completed)
            total_pnl = sum(t['pnl'] for t in completed)
            total_pnl_str = f"+{total_pnl:.1%}" if total_pnl >= 0 else f"{total_pnl:.1%}"
        else:
            total_pnl_str = "0%"

        # 因子表现
        factor_performance = {}
        for factor in self.factor_analyzer.get_weights():
            analysis = self.factor_analyzer.analyze_factor_performance(factor, lookback_days=7)
            factor_performance[factor] = {
                "ic": analysis['ic'],
                "trend": analysis.get('trend', 'unknown')
            }

        # 模式表现
        patterns = {}
        for t in completed:
            pk = t.get('pattern_key')
            if pk and pk not in patterns:
                stats = self.pattern_learner.get_pattern_stats(pk)
                patterns[pk] = {
                    "wr": stats['win_rate'],
                    "trades": stats['total']
                }

        # 生成建议
        recommendations = []
        weights = self.factor_analyzer.get_weights()

        # 基于IC趋势调整权重
        for factor, perf in factor_performance.items():
            if perf['trend'] == 'improving':
                old = weights[factor]
                weights[factor] = min(old * 1.1, 0.5)
                recommendations.append(
                    f"{factor}因子权重从{old:.2f}调整到{weights[factor]:.2f} (IC改善)"
                )

        # 基于白名单生成建议
        whitelist = self.whitelist_manager.get_whitelist()
        for pk in whitelist:
            stats = self.pattern_learner.get_pattern_stats(pk)
            if stats['win_rate'] < 0.40:
                recommendations.append(
                    f"移除{pk}模式（胜率{stats['win_rate']:.0%}不足）"
                )

        return {
            "period": period,
            "trades_count": trades_count,
            "win_rate": round(win_rate, 3),
            "total_pnl": total_pnl_str,
            "factor_performance": factor_performance,
            "patterns": patterns,
            "recommendations": recommendations
        }

    def generate_monthly_report(self) -> Dict[str, Any]:
        """生成每月学习报告"""
        today = datetime.now()
        month_ago = today - timedelta(days=30)
        period = f"{month_ago.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')}"

        stats = self.trade_recorder.get_stats(lookback_days=30)

        # 因子月表现
        factor_performance = {}
        for factor in self.factor_analyzer.get_weights():
            analysis = self.factor_analyzer.analyze_factor_performance(factor, lookback_days=30)
            factor_performance[factor] = analysis

        # 白名单更新结果
        whitelist_update = self.whitelist_manager.auto_update_whitelist(self.pattern_learner)

        # 策略调整建议
        adjustments = []
        if stats['win_rate'] < 0.35:
            adjustments.append("周胜率低于35%，建议检查市场环境是否变化")
        if stats.get('avg_rr', 0) < 1.5:
            adjustments.append("平均RR低于1.5，建议收紧止损或提高止盈")
        if stats.get('max_drawdown', 0) > 0.15:
            adjustments.append("最大回撤超过15%，建议暂停策略等待人工审查")

        return {
            "period": period,
            "stats": stats,
            "factor_performance": factor_performance,
            "whitelist_update": whitelist_update,
            "adjustments": adjustments,
            "current_weights": self.factor_analyzer.get_weights()
        }


# =============================================================================
# 6. 策略调整
# =============================================================================

class StrategyOptimizer:
    """策略参数检查与调整"""

    def __init__(self, db_path: str, config_path: str):
        self.db_path = db_path
        self.config_path = config_path
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载策略配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                return json.load(f)
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "position_size": 1.0,
            "max_drawdown_pause": 0.15,
            "consecutive_loss_pause": 2,
            "last_adjustment": None,
            "adjustment_history": []
        }

    def _save_config(self):
        """保存配置（原子写）"""
        fd, tmp = tempfile.mkstemp(suffix='.json.tmp', dir=os.path.dirname(self.config_path))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.config_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def check_and_adjust(self) -> Dict[str, Any]:
        """
        检查并调整策略参数
        每周调用一次
        """
        import statistics

        today = datetime.now()
        week_ago = today - timedelta(days=7)
        two_weeks_ago = today - timedelta(days=14)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # 获取本周和上周交易
            week_trades = conn.execute(
                "SELECT pnl, won FROM trades WHERE entry_time >= ? AND pnl IS NOT NULL",
                [week_ago.isoformat()]
            ).fetchall()

            week_stats = self._calc_stats(week_trades)

            # 上周统计
            last_week_trades = conn.execute(
                "SELECT pnl, won FROM trades WHERE entry_time >= ? AND entry_time < ? AND pnl IS NOT NULL",
                [two_weeks_ago.isoformat(), week_ago.isoformat()]
            ).fetchall()
            last_week_stats = self._calc_stats(last_week_trades)

        actions = []
        should_pause = False

        # 规则1: 周胜率<35% → 检查市场环境
        if week_stats['win_rate'] < 0.35:
            actions.append({
                "type": "WEEK_WINRATE_LOW",
                "message": f"本周胜率{week_stats['win_rate']:.1%}低于35%阈值",
                "action": "检查市场环境是否变化"
            })

        # 规则2: 周平均RR<1.5 → 收紧止损或提高止盈
        if week_stats['avg_rr'] < 1.5 and week_stats['avg_rr'] > 0:
            actions.append({
                "type": "AVG_RR_LOW",
                "message": f"本周平均RR {week_stats['avg_rr']:.2f} 低于1.5",
                "action": "建议收紧止损或提高止盈目标"
            })

        # 规则3: 连续2周表现差 → 降低50%仓位
        if (week_stats['win_rate'] < 0.40 and last_week_stats['win_rate'] < 0.40) or \
           (week_stats['total_pnl'] < 0 and last_week_stats['total_pnl'] < 0):
            old_pos = self.config.get('position_size', 1.0)
            new_pos = old_pos * 0.5
            self.config['position_size'] = new_pos
            actions.append({
                "type": "POSITION_REDUCE",
                "message": "连续2周表现差",
                "action": f"仓位从{old_pos:.0%}降至{new_pos:.0%}"
            })

        # 规则4: 最大回撤>15% → 策略暂停
        if week_stats.get('max_drawdown', 0) > self.config['max_drawdown_pause']:
            should_pause = True
            actions.append({
                "type": "STRATEGY_PAUSE",
                "message": f"最大回撤{week_stats['max_drawdown']:.1%}超过15%阈值",
                "action": "策略暂停，等待人工审查"
            })

        # 记录调整
        adjustment = {
            "date": today.isoformat(),
            "week_stats": week_stats,
            "actions": actions,
            "position_size": self.config.get('position_size', 1.0),
            "paused": should_pause
        }

        self.config['last_adjustment'] = adjustment
        self.config['adjustment_history'].append(adjustment)
        self.config['adjustment_history'] = self.config['adjustment_history'][-10:]
        self._save_config()

        return {
            "this_week": week_stats,
            "last_week": last_week_stats,
            "actions": actions,
            "position_size": self.config.get('position_size', 1.0),
            "should_pause": should_pause
        }

    def _calc_stats(self, trades: List) -> Dict[str, Any]:
        """计算统计"""
        if not trades:
            return {"win_rate": 0, "avg_rr": 0, "total_pnl": 0, "max_drawdown": 0}

        pnls = [t['pnl'] for t in trades]
        wins = sum(1 for t in trades if t['won'] == 1)
        rr_list = [t['pnl'] / abs((t['entry_price'] - t['stop_loss']) / t['entry_price'])
                   for t in trades if t.get('stop_loss')]

        # Max drawdown
        running = 0
        peak = 0
        max_dd = 0
        for p in pnls:
            running += p
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        return {
            "win_rate": wins / len(trades) if trades else 0,
            "avg_rr": sum(rr_list) / len(rr_list) if rr_list else 0,
            "total_pnl": sum(pnls),
            "max_drawdown": max_dd
        }


# =============================================================================
# 主Agent: Agent-L
# =============================================================================

class AgentLearner:
    """
    Agent-L: 学习迭代主Agent
    整合所有子模块，统一接口
    """

    def __init__(self, base_dir: str = "~/Desktop/miracle-1.0.1"):
        self.base_dir = os.path.expanduser(base_dir)
        self.data_dir = os.path.join(self.base_dir, "data")
        self.agents_dir = os.path.join(self.base_dir, "agents")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.agents_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "miracle_trades.db")
        self.pattern_db_path = os.path.join(self.data_dir, "pattern_db.json")
        self.weights_path = os.path.join(self.data_dir, "factor_weights.json")
        self.whitelist_path = os.path.join(self.data_dir, "whitelist.json")
        self.config_path = os.path.join(self.data_dir, "strategy_config.json")

        self.trade_recorder = TradeRecorder(self.db_path)
        self.factor_analyzer = FactorAnalyzer(self.db_path, self.weights_path)
        self.pattern_learner = PatternLearner(self.pattern_db_path)
        self.whitelist_manager = WhitelistManager(self.whitelist_path, self.pattern_db_path)
        self.learning_report = LearningReport(
            self.db_path, self.weights_path,
            self.pattern_db_path, self.whitelist_path
        )
        self.strategy_optimizer = StrategyOptimizer(self.db_path, self.config_path)

    def on_trade_entry(self, trade_data: Dict[str, Any]) -> str:
        """交易入场时调用 - 记录并提取模式"""
        # 提取模式
        pattern_key = self.pattern_learner.extract_pattern(trade_data)
        trade_data['pattern_key'] = pattern_key

        # 检查是否白名单模式
        is_allowed = self.whitelist_manager.is_whitelisted(pattern_key) or \
                     pattern_key == ""

        # 记录交易
        trade_id = self.trade_recorder.record_trade(trade_data)

        return pattern_key, is_allowed, trade_id

    def on_trade_exit(self, trade_id: int, exit_data: Dict[str, Any]):
        """交易出场时调用 - 更新模式统计"""
        # 更新交易记录
        with sqlite3.connect(self.db_path) as conn:
            trade = conn.execute(
                "SELECT * FROM trades WHERE id = ?", [trade_id]
            ).fetchone()

        if trade:
            # 计算PnL
            entry_price = trade[3]
            exit_price = exit_data['exit_price']
            direction = trade[2]
            stop_loss = trade[11]

            if direction == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price

            risk = abs(entry_price - stop_loss) / entry_price if stop_loss else 0
            rr = abs(pnl / risk) if risk > 0 else 0
            won = 1 if pnl > 0 else 0

            # 更新数据库
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE trades SET
                        exit_time = ?, exit_price = ?,
                        pnl = ?, rr = ?, won = ?
                    WHERE id = ?
                """, (
                    exit_data['exit_time'],
                    exit_price,
                    pnl, rr, won, trade_id
                ))

            # 更新模式统计
            pattern_key = trade[10]
            self.pattern_learner.update_pattern(pattern_key, {
                "won": won == 1,
                "rr": rr
            })

    def run_weekly_learning(self) -> Dict[str, Any]:
        """运行每周学习迭代"""
        # 1. 更新因子权重
        weight_update = self.factor_analyzer.update_factor_weights()

        # 2. 自动更新白名单
        whitelist_update = self.whitelist_manager.auto_update_whitelist(self.pattern_learner)

        # 3. 检查策略参数
        strategy_check = self.strategy_optimizer.check_and_adjust()

        # 4. 生成报告
        report = self.learning_report.generate_weekly_report()

        return {
            "weight_update": weight_update,
            "whitelist_update": whitelist_update,
            "strategy_check": strategy_check,
            "weekly_report": report
        }

    def run_monthly_learning(self) -> Dict[str, Any]:
        """运行每月学习迭代"""
        return self.learning_report.generate_monthly_report()

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        stats = self.trade_recorder.get_stats(lookback_days=7)
        return {
            "stats_7d": stats,
            "factor_weights": self.factor_analyzer.get_weights(),
            "whitelist_count": len(self.whitelist_manager.get_whitelist()),
            "pattern_count": len(self.pattern_learner.pattern_db)
        }


# =============================================================================
# 便捷函数
# =============================================================================

def get_agent(base_dir: str = "~/Desktop/miracle-1.0.1") -> AgentLearner:
    """获取Agent实例"""
    return AgentLearner(base_dir)


if __name__ == "__main__":
    # 测试
    agent = get_agent()
    print("Agent-L initialized")
    print("Status:", agent.get_status())
