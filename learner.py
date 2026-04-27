"""
Miracle 1.0.2 - Adaptive Learning Core
=======================================
Self-learning system core components

Features:
1. Walk-forward validation
2. Decision Journal for tracking trading decisions
3. Pattern performance statistics
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sys
import gzip

sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from statistics import mean, stdev


# ===== 日志配置 =====

logger = logging.getLogger("miracle.adaptive_learner.learner")
logger.setLevel(logging.INFO)


# ===== Decision Journal =====

@dataclass
class DecisionJournalEntry:
    """
    决策日记条目 - 对标Kronos decision_journal.jsonl格式
    """
    ts: str                          # ISO格式时间戳
    equity: float                   # 当前权益
    position_count: int             # 持仓数量
    
    # 市场上下文
    local_context: Dict[str, Any] = field(default_factory=dict)
    # 市场上下文包含:
    #   - market_regime: str (bull/bear/range/volatile)
    #   - primary_direction: str (long/short/both)
    #   - overall_confidence: float (0-1)
    #   - emergency_level: str (none/warning/critical)
    #   - strategic_hint: str
    #   - data_quality: str (fresh/stale)
    
    # 持仓快照
    positions_snapshot: Dict[str, Any] = field(default_factory=dict)
    # 格式: {symbol: {direction, size, entry, price, pnl_pct, pnl_abs, sl_price, tp_price}}
    
    # 候选币种快照
    candidates_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    # 格式: [{coin, direction, score, rsi_1h, adx_1h, ...}]
    
    # LLM原始输出
    llm_raw_output: str = ""
    
    # 解析后的决策
    decision_parsed: Dict[str, str] = field(default_factory=dict)
    # 格式: {coin, decision, reason}
    # decision: open/close/hold/modify
    
    # 执行结果
    execution_result: str = ""
    execution_ok: Optional[bool] = None
    
    # Miracle特有：因子权重快照
    factor_weights_snapshot: Dict[str, float] = field(default_factory=dict)
    
    # Miracle特有：模式识别结果
    pattern_recognition: Dict[str, Any] = field(default_factory=dict)
    # 格式: {detected_patterns: [], pattern_key: str, confidence: float}
    
    # Miracle特有：学习反馈
    learning_feedback: Dict[str, Any] = field(default_factory=dict)
    # 格式: {overfitting_detected: bool, ic_decay: float, adjusted_weights: {}}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DecisionJournalEntry':
        """从字典创建"""
        return cls(**data)


class DecisionJournal:
    """
    决策日记 - 记录并对比交易决策
    
    功能:
    1. 记录每笔决策及其上下文
    2. 与Kronos decision_journal.jsonl对比分析
    3. 模式识别统计
    4. 因子权重学习历史
    """
    
    def __init__(self, journal_dir: str = None, kronos_journal_path: str = None):
        """
        Args:
            journal_dir: 日记存储目录
            kronos_journal_path: Kronos decision_journal.jsonl 路径
        """
        self.journal_dir = Path(journal_dir) if journal_dir else Path(__file__).parent / "data" / "decision_journal"
        self.kronos_journal_path = Path(kronos_journal_path) if kronos_journal_path else Path.home() / "kronos" / "decision_journal.jsonl"
        
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前日记文件
        self.current_journal_path = self.journal_dir / f"miracle_journal_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # 内存缓存（最近100条）
        self._cache: List[DecisionJournalEntry] = []
        self._cache_loaded = False
        
        # 统计计数器
        self._stats = {
            "total_decisions": 0,
            "execution_ok": 0,
            "execution_fail": 0,
            "pattern_counts": defaultdict(int),
            "regime_counts": defaultdict(int),
            "decision_counts": defaultdict(int)
        }
        
        logger.info(f"DecisionJournal initialized: {self.journal_dir}")
    
    def record_decision(self, entry: DecisionJournalEntry) -> None:
        """
        记录一条决策
        
        Args:
            entry: 决策日记条目
        """
        # 更新统计
        self._stats["total_decisions"] += 1
        
        if entry.execution_ok is True:
            self._stats["execution_ok"] += 1
        elif entry.execution_ok is False:
            self._stats["execution_fail"] += 1
        
        # 模式计数
        if entry.pattern_recognition:
            for pattern in entry.pattern_recognition.get("detected_patterns", []):
                self._stats["pattern_counts"][pattern] += 1
        
        # 市场状态计数
        regime = entry.local_context.get("market_regime", "unknown")
        self._stats["regime_counts"][regime] += 1
        
        # 决策计数
        decision = entry.decision_parsed.get("decision", "unknown")
        self._stats["decision_counts"][decision] += 1
        
        # 写入文件
        try:
            with open(self.current_journal_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write decision journal: {e}")
        
        # 更新缓存
        self._cache.append(entry)
        if len(self._cache) > 100:
            self._cache.popleft() if hasattr(self._cache, 'popleft') else self._cache.pop(0)
    
    def _ensure_cache_loaded(self):
        """确保缓存已加载"""
        if self._cache_loaded:
            return
        
        # 加载今天的日记
        if self.current_journal_path.exists():
            try:
                with open(self.current_journal_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._cache.append(DecisionJournalEntry.from_dict(data))
                self._cache_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load journal cache: {e}")
    
    def get_recent_decisions(self, n: int = 20) -> List[DecisionJournalEntry]:
        """
        获取最近N条决策
        
        Args:
            n: 返回数量
            
        Returns:
            最近的N条决策
        """
        self._ensure_cache_loaded()
        return self._cache[-n:] if len(self._cache) >= n else self._cache
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "journal_path": str(self.current_journal_path),
            "cache_size": len(self._cache)
        }
    
    def compare_with_kronos(self, 
                           time_window: Optional[str] = None,
                           limit: int = 100) -> Dict[str, Any]:
        """
        与Kronos decision_journal.jsonl对比分析
        
        Args:
            time_window: 时间窗口，如 "1h", "6h", "1d"
            limit: 最多读取Kronos条数
            
        Returns:
            对比分析报告
        """
        if not self.kronos_journal_path.exists():
            return {"error": f"Kronos journal not found: {self.kronos_journal_path}"}
        
        # 解析时间窗口
        time_cutoff = None
        if time_window:
            now = datetime.now()
            if time_window.endswith('h'):
                hours = int(time_window[:-1])
                time_cutoff = now - timedelta(hours=hours)
            elif time_window.endswith('d'):
                days = int(time_window[:-1])
                time_cutoff = now - timedelta(days=days)
        
        # 读取Kronos日记
        kronos_entries = []
        try:
            with open(self.kronos_journal_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    if line.strip():
                        data = json.loads(line)
                        # 时间过滤
                        if time_cutoff:
                            entry_ts = datetime.fromisoformat(data.get('ts', '2000-01-01'))
                            if entry_ts < time_cutoff:
                                continue
                        kronos_entries.append(data)
        except Exception as e:
            return {"error": f"Failed to read Kronos journal: {e}"}
        
        if not kronos_entries:
            return {"error": "No Kronos entries found in time window"}
        
        # 分析Kronos决策模式
        kronos_stats = self._analyze_kronos_entries(kronos_entries)
        
        # 获取Miracle最近决策
        miracle_recent = self.get_recent_decisions(n=min(limit, 100))
        miracle_stats = self._analyze_miracle_entries(miracle_recent)
        
        # 对比分析
        comparison = {
            "kronos": kronos_stats,
            "miracle": miracle_stats,
            "comparison": self._compute_comparison(kronos_stats, miracle_stats),
            "kronos_entries_analyzed": len(kronos_entries),
            "miracle_entries_analyzed": len(miracle_recent),
            "time_window": time_window,
            "kronos_latest_path": str(self.kronos_journal_path)
        }
        
        return comparison
    
    def _analyze_kronos_entries(self, entries: List[Dict]) -> Dict[str, Any]:
        """分析Kronos决策条目"""
        decisions = [e.get('decision_parsed', {}).get('decision', 'unknown') for e in entries]
        regimes = [e.get('local_context', {}).get('market_regime', 'unknown') for e in entries]
        equities = [e.get('equity', 0) for e in entries if e.get('equity')]
        
        # 执行成功率
        exec_ok = sum(1 for e in entries if e.get('execution_ok') is True)
        exec_fail = sum(1 for e in entries if e.get('execution_ok') is False)
        exec_total = exec_ok + exec_fail if exec_ok + exec_fail > 0 else 1
        
        # 决策分布
        decision_dist = defaultdict(int)
        for d in decisions:
            decision_dist[d] += 1
        
        # 市场状态分布
        regime_dist = defaultdict(int)
        for r in regimes:
            regime_dist[r] += 1
        
        return {
            "total_entries": len(entries),
            "decision_distribution": dict(decision_dist),
            "regime_distribution": dict(regime_dist),
            "execution_success_rate": exec_ok / exec_total if exec_total > 0 else 0,
            "avg_equity": mean(equities) if equities else 0,
            "equity_std": stdev(equities) if len(equities) > 1 else 0,
            "unique_positions": len(set.union(*[set(e.get('positions_snapshot', {}).keys()) for e in entries]))
        }
    
    def _analyze_miracle_entries(self, entries: List[DecisionJournalEntry]) -> Dict[str, Any]:
        """分析Miracle决策条目"""
        decisions = [e.decision_parsed.get('decision', 'unknown') for e in entries]
        regimes = [e.local_context.get('market_regime', 'unknown') for e in entries]
        equities = [e.equity for e in entries if e.equity]
        
        exec_ok = sum(1 for e in entries if e.execution_ok is True)
        exec_fail = sum(1 for e in entries if e.execution_ok is False)
        exec_total = exec_ok + exec_fail if exec_ok + exec_fail > 0 else 1
        
        # 模式分布
        pattern_dist = defaultdict(int)
        for e in entries:
            if e.pattern_recognition:
                for p in e.pattern_recognition.get('detected_patterns', []):
                    pattern_dist[p] += 1
        
        return {
            "total_entries": len(entries),
            "decision_distribution": dict(defaultdict(int, {d: decisions.count(d) for d in set(decisions)})),
            "regime_distribution": dict(defaultdict(int, {r: regimes.count(r) for r in set(regimes)})),
            "pattern_distribution": dict(pattern_dist),
            "execution_success_rate": exec_ok / exec_total if exec_total > 0 else 0,
            "avg_equity": mean(equities) if equities else 0,
            "equity_std": stdev(equities) if len(equities) > 1 else 0,
            "unique_positions": len(set.union(*[set(e.positions_snapshot.keys()) for e in entries]))
        }
    
    def _compute_comparison(self, kronos: Dict, miracle: Dict) -> Dict[str, Any]:
        """计算对比分析"""
        return {
            "execution_rate_diff": miracle.get("execution_success_rate", 0) - kronos.get("execution_success_rate", 0),
            "equity_diff_pct": ((miracle.get("avg_equity", 0) - kronos.get("avg_equity", 0)) / kronos.get("avg_equity", 1)) * 100 if kronos.get("avg_equity", 0) else 0,
            "decision_pattern_diff": {
                "kronos_top": sorted(kronos.get("decision_distribution", {}).items(), key=lambda x: -x[1])[:3],
                "miracle_top": sorted(miracle.get("decision_distribution", {}).items(), key=lambda x: -x[1])[:3]
            },
            "regime_focus_diff": {
                "kronos_primary": max(kronos.get("regime_distribution", {}).items(), key=lambda x: x[1])[0] if kronos.get("regime_distribution") else None,
                "miracle_primary": max(miracle.get("regime_distribution", {}).items(), key=lambda x: x[1])[0] if miracle.get("regime_distribution") else None
            }
        }
    
    def get_pattern_recognition_stats(self, pattern_key: str = None) -> Dict[str, Any]:
        """
        获取模式识别统计
        
        Args:
            pattern_key: 可选，特定模式键
            
        Returns:
            模式识别统计
        """
        self._ensure_cache_loaded()
        
        if pattern_key:
            # 特定模式统计
            matching = [e for e in self._cache if e.pattern_recognition.get("pattern_key") == pattern_key]
            if not matching:
                return {"pattern": pattern_key, "count": 0}
            
            wins = sum(1 for e in matching if e.execution_ok is True)
            return {
                "pattern": pattern_key,
                "count": len(matching),
                "success_rate": wins / len(matching) if matching else 0
            }
        else:
            # 全局模式统计
            pattern_stats = defaultdict(lambda: {"count": 0, "wins": 0})
            for e in self._cache:
                pk = e.pattern_recognition.get("pattern_key", "unknown")
                if pk != "unknown":
                    pattern_stats[pk]["count"] += 1
                    if e.execution_ok is True:
                        pattern_stats[pk]["wins"] += 1
            
            return {
                "total_patterns": len(pattern_stats),
                "patterns": [
                    {
                        "pattern": k,
                        "count": v["count"],
                        "success_rate": v["wins"] / v["count"] if v["count"] > 0 else 0
                    }
                    for k, v in pattern_stats.items()
                ]
            }
    
    def get_factor_weight_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取因子权重学习历史
        
        Returns:
            因子权重历史记录
        """
        history = defaultdict(list)
        
        for entry in self._cache:
            if entry.factor_weights_snapshot:
                for factor, weight in entry.factor_weights_snapshot.items():
                    history[factor].append({
                        "ts": entry.ts,
                        "weight": weight
                    })
        
        return dict(history)
    
    def export_journal(self, output_path: str = None, 
                      start_time: str = None,
                      end_time: str = None,
                      compress: bool = False) -> str:
        """
        导出日记到指定路径
        
        Args:
            output_path: 输出路径，默认导出到data/decision_journal/export
            start_time: ISO格式开始时间
            end_time: ISO格式结束时间
            compress: 是否gzip压缩
            
        Returns:
            导出文件路径
        """
        if output_path:
            out_path = Path(output_path)
        else:
            out_path = self.journal_dir / "export" / f"miracle_journal_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 时间过滤
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        
        entries = []
        for entry in self._cache:
            if start_dt or end_dt:
                entry_dt = datetime.fromisoformat(entry.ts)
                if start_dt and entry_dt < start_dt:
                    continue
                if end_dt and entry_dt > end_dt:
                    continue
            entries.append(entry.to_dict())
        
        if compress:
            out_path = Path(str(out_path) + '.gz')
            with gzip.open(out_path, 'wt', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        else:
            with open(out_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Journal exported to {out_path}")
        return str(out_path)


# ===== Walk-Forward Validator =====

class WalkForwardValidator:
    """
    Walk-Forward Validation - 防止过拟合的核心机制

    将数据分成多个时间窗口：
    - 样本内训练 (in-sample)
    - 样本外测试 (out-of-sample)

    检测IC衰减来判断是否过拟合
    """

    def __init__(self, train_window: int = 50, test_window: int = 20):
        """
        Args:
            train_window: 训练窗口大小
            test_window: 测试窗口大小
        """
        self.train_window = train_window
        self.test_window = test_window

    def validate(self, strategy_func, data: List[Dict],
                 n_windows: int = 5) -> Dict[str, Any]:
        """
        执行Walk-Forward验证

        Args:
            strategy_func: 策略函数，接受(训练数据, 测试数据)两个参数，
                          返回测试数据上的IC
            data: 时间序列数据
            n_windows: 验证窗口数量

        Returns:
            {
                "train_ic": [ic scores],
                "test_ic": [ic scores],
                "decay": [ic decay per window],
                "is_overfitting": bool
            }
        """
        if not HAS_NUMPY or len(data) < self.train_window + self.test_window:
            return {
                "train_ic": [0.0],
                "test_ic": [0.0],
                "decay": [0.0],
                "is_overfitting": False,
                "reason": "insufficient_data"
            }

        train_ics = []
        test_ics = []
        decays = []

        n_total = len(data)

        # Walk-Forward正确实现：
        # 训练窗口使用该测试窗口以前的数据
        # 测试时: 用训练好的参数生成测试数据信号 → 真实out-of-sample
        for i in range(n_windows):
            # 计算测试窗口位置（从后往前）
            test_end = n_total - i * self.test_window
            test_start = max(self.test_window, test_end - self.test_window)
            # 训练窗口：test_start之前的数据
            train_end = test_start
            train_start = max(0, train_end - self.train_window)

            if train_start >= train_end or test_start >= test_end:
                continue

            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]

            if len(train_data) < 10 or len(test_data) < 5:
                continue

            # 计算样本内IC (在训练数据上)
            train_result = strategy_func(train_data, train_data)
            train_ic = train_result.get("train_ic", 0.0) if isinstance(train_result, dict) else 0.0

            # 计算样本外IC: 用训练数据得到的参数应用到测试数据
            test_result = strategy_func(train_data, test_data)
            if isinstance(test_result, dict):
                test_ic = test_result.get("test_ic", 0.0)
            else:
                test_ic = 0.0

            train_ics.append(train_ic)
            test_ics.append(test_ic)

            # 计算IC衰减
            if train_ic != 0:
                decay = (train_ic - test_ic) / abs(train_ic)
            else:
                decay = 0.0
            decays.append(decay)

        avg_train_ic = float(np.mean(train_ics)) if train_ics else 0.0
        avg_test_ic = float(np.mean(test_ics)) if test_ics else 0.0
        avg_decay = float(np.mean(decays)) if decays else 0.0

        return {
            "train_ic": train_ics,
            "test_ic": test_ics,
            "decay": decays,
            "train_ic_avg": avg_train_ic,
            "test_ic_avg": avg_test_ic,
            "ic_decay": avg_decay,
            "is_overfitting": avg_decay > 0.3  # IC衰减超过30%认为过拟合
        }


# ===== Information Coefficient Calculator =====

def calc_information_coefficient(signals: List[float],
                                  returns: List[float]) -> Tuple[float, float]:
    """
    计算信息系数 (Information Coefficient)

    Args:
        signals: 信号列表
        returns: 收益列表

    Returns:
        (ic, p_value): IC值和P值
    """
    if not HAS_NUMPY:
        return 0.0, 1.0

    if len(signals) != len(returns) or len(signals) < 10:
        return 0.0, 1.0

    try:
        from scipy import stats
        ic, p_value = stats.spearmanr(signals, returns)
        return float(ic), float(p_value)
    except ImportError:
        # Fallback: simple correlation
        signals_arr = np.array(signals)
        returns_arr = np.array(returns)
        correlation = np.corrcoef(signals_arr, returns_arr)[0, 1]
        return float(correlation), 1.0


# ===== Module Exports =====

__all__ = [
    "DecisionJournal",
    "DecisionJournalEntry",
    "WalkForwardValidator",
    "calc_information_coefficient",
]
