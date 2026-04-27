"""
Fusion Memory Log System
=======================
文本记忆日志系统 - 用于辩论层决策记录与反馈学习

功能:
    - store_decision(): 存储辩论决策
    - update_with_outcome(): 用交易结果更新记忆
    - get_past_context(): 获取历史上下文
    - get_ic_feedback(): 获取IC反馈信号
    - ENTRY_END separator
    - 1000 entry rotation

Usage:
    from memory.fusion_memory import FusionMemoryLog, store_decision, update_with_outcome

    # 存储决策
    store_decision(
        ticker="BTC",
        bull_case="RSI超卖反弹",
        bear_case="趋势未反转",
        verdict="BUY",
        confidence=0.75,
        factors={"rsi": 28, "macd": "bullish"}
    )

    # 更新结果
    update_with_outcome(ticker="BTC", entry_time=datetime.now(), pnl_pct=2.5, outcome="WIN")

    # 获取上下文
    context = get_past_context(ticker="BTC", limit=5)

    # 获取IC反馈
    ic = get_ic_feedback(factor_name="rsi")
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)

# ==================== 常量 ====================

ENTRY_END = "=" * 60 + "ENTRY_END" + "=" * 60
MAX_ENTRIES = 1000
LOG_FILE = os.path.expanduser("~/.miracle_memory/fusion_log.json")
LOCK_FILE = os.path.expanduser("~/.miracle_memory/fusion_log.lock")

# ==================== 数据结构 ====================

@dataclass
class DecisionEntry:
    """决策条目"""
    id: int
    timestamp: str
    ticker: str
    bull_case: str
    bear_case: str
    verdict: str
    confidence: float
    factors: Dict[str, Any]
    market_context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None  # WIN/LOSS/PENDING
    pnl_pct: Optional[float] = None
    updated_at: Optional[str] = None


@dataclass
class ICFeedback:
    """IC反馈信号"""
    factor_name: str
    predicted_direction: int  # 1: LONG, -1: SHORT, 0: NEUTRAL
    actual_outcome: str        # WIN/LOSS
    ic_score: float            # 信息系数
    sample_count: int
    timestamp: str


# ==================== 内存存储 ====================

class _FusionMemoryStore:
    """
    内存中的双向存储:
        - _entries: deque[DecisionEntry] 按时间排序的决策记录
        - _ic_cache: Dict[factor_name, List[ICFeedback]] 因子IC缓存
        - _next_id: int 自增ID
    """
    __slots__ = ("_entries", "_ic_cache", "_next_id", "_lock", "_dirty")

    def __init__(self):
        self._entries: deque = deque(maxlen=MAX_ENTRIES)
        self._ic_cache: Dict[str, List] = {}
        self._next_id: int = 1
        self._lock = Lock()
        self._dirty = False

    def _load(self) -> None:
        """从磁盘加载"""
        if not os.path.exists(LOG_FILE):
            return
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = deque([DecisionEntry(**e) for e in data.get("entries", [])], maxlen=MAX_ENTRIES)
            self._next_id = data.get("next_id", 1)
            self._ic_cache = data.get("ic_cache", {})
            logger.info(f"[FusionMemory] 加载了 {len(self._entries)} 条决策记录")
        except Exception as e:
            logger.warning(f"[FusionMemory] 加载失败: {e}，使用空存储")

    def _save(self) -> None:
        """持久化到磁盘"""
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "entries": [asdict(e) for e in self._entries],
                    "next_id": self._next_id,
                    "ic_cache": self._ic_cache,
                    "saved_at": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            self._dirty = False
            logger.debug(f"[FusionMemory] 已保存 {len(self._entries)} 条记录")
        except Exception as e:
            logger.error(f"[FusionMemory] 保存失败: {e}")

    def add_entry(self, entry: DecisionEntry) -> int:
        """添加决策条目"""
        with self._lock:
            entry.id = self._next_id
            self._next_id += 1
            self._entries.append(entry)
            self._dirty = True
            self._save()
            return entry.id

    def update_entry_outcome(self, ticker: str, entry_time: datetime, outcome: str, pnl_pct: float) -> bool:
        """通过ticker+entry_time匹配并更新结果"""
        with self._lock:
            # 精确匹配: ticker相同且时间最接近
            best = None
            best_diff = float("inf")
            target_ts = entry_time.timestamp()

            for e in reversed(self._entries):
                if e.ticker != ticker:
                    continue
                if e.outcome is not None:
                    continue
                diff = abs(datetime.fromisoformat(e.timestamp).timestamp() - target_ts)
                if diff < best_diff:
                    best_diff = diff
                    best = e

            if best is None:
                logger.warning(f"[FusionMemory] 未找到匹配的pending决策: {ticker} @ {entry_time}")
                return False

            best.outcome = outcome
            best.pnl_pct = pnl_pct
            best.updated_at = datetime.now().isoformat()
            self._dirty = True
            self._save()
            return True

    def get_past_context(self, ticker: str, limit: int = 5, include_outcomes: bool = True) -> List[Dict[str, Any]]:
        """获取历史上下文"""
        with self._lock:
            results = []
            for e in reversed(self._entries):
                if e.ticker != ticker:
                    continue
                entry = {
                    "id": e.id,
                    "timestamp": e.timestamp,
                    "bull_case": e.bull_case,
                    "bear_case": e.bear_case,
                    "verdict": e.verdict,
                    "confidence": e.confidence,
                    "factors": e.factors,
                }
                if include_outcomes:
                    entry["outcome"] = e.outcome
                    entry["pnl_pct"] = e.pnl_pct
                results.append(entry)
                if len(results) >= limit:
                    break
            return results

    def get_pending_entries(self) -> List[DecisionEntry]:
        """获取所有待更新结果的决策"""
        with self._lock:
            return [e for e in self._entries if e.outcome is None]

    def add_ic_feedback(self, feedback: ICFeedback) -> None:
        """添加IC反馈"""
        with self._lock:
            if feedback.factor_name not in self._ic_cache:
                self._ic_cache[feedback.factor_name] = []
            self._ic_cache[feedback.factor_name].append(asdict(feedback))
            # 限制每个因子最多保留200条反馈
            if len(self._ic_cache[feedback.factor_name]) > 200:
                self._ic_cache[feedback.factor_name] = self._ic_cache[feedback.factor_name][-200:]
            self._dirty = True
            self._save()

    def get_ic_score(self, factor_name: str) -> Optional[float]:
        """计算IC分数"""
        with self._lock:
            feedbacks = self._ic_cache.get(factor_name, [])
            if len(feedbacks) < 10:
                return None

            correct = 0
            for fb in feedbacks:
                pred = fb["predicted_direction"]
                actual = fb["actual_outcome"]
                if (pred == 1 and actual == "WIN") or (pred == -1 and actual == "LOSS"):
                    correct += 1
            return correct / len(feedbacks)


# 全局单例
_store: Optional[_FusionMemoryStore] = None


def _get_store() -> _FusionMemoryStore:
    global _store
    if _store is None:
        _store = _FusionMemoryStore()
        _store._load()
    return _store


# ==================== 公开API ====================

def store_decision(
    ticker: str,
    bull_case: str,
    bear_case: str,
    verdict: str,
    confidence: float,
    factors: Dict[str, Any],
    market_context: Optional[Dict[str, Any]] = None
) -> int:
    """
    存储辩论决策到记忆日志

    Args:
        ticker: 交易标的
        bull_case: 多头论点
        bear_case: 空头论点
        verdict: 裁决 BUY/SELL/HOLD
        confidence: 置信度 0-1
        factors: 因子值 dict
        market_context: 市场上下文（可选）

    Returns:
        int: 决策ID
    """
    entry = DecisionEntry(
        id=0,
        timestamp=datetime.now().isoformat(),
        ticker=ticker,
        bull_case=bull_case,
        bear_case=bear_case,
        verdict=verdict,
        confidence=confidence,
        factors=factors,
        market_context=market_context or {},
        outcome=None,
        pnl_pct=None,
        updated_at=None
    )
    entry_id = _get_store().add_entry(entry)
    conf_str = f"{float(confidence):.2f}" if confidence else "N/A"
    logger.info(f"[FusionMemory] 存储决策 #{entry_id}: {ticker} {verdict} @{conf_str}")
    return entry_id


def update_with_outcome(
    ticker: str,
    entry_time: datetime,
    outcome: str,
    pnl_pct: float
) -> bool:
    """
    用交易结果更新记忆

    Args:
        ticker: 交易标的
        entry_time: 入场时间（用于匹配决策）
        outcome: 结果 WIN/LOSS
        pnl_pct: 盈亏百分比

    Returns:
        bool: 是否成功更新
    """
    ok = _get_store().update_entry_outcome(ticker, entry_time, outcome, pnl_pct)
    if ok:
        logger.info(f"[FusionMemory] 更新结果: {ticker} {outcome} {pnl_pct:+.2f}%")
    return ok


def get_past_context(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    获取历史上下文用于当前决策参考

    Args:
        ticker: 交易标的
        limit: 返回条数上限

    Returns:
        List[Dict]: 历史决策列表
    """
    return _get_store().get_past_context(ticker, limit)


def get_ic_feedback(factor_name: str) -> Optional[float]:
    """
    获取因子IC反馈分数

    Args:
        factor_name: 因子名称如 "rsi", "macd", "adx"

    Returns:
        float: IC分数 0-1，或None（样本不足）
    """
    return _get_store().get_ic_score(factor_name)


def sync_ic_feedback(factor_name: str, factor_value: float, predicted_direction: int, outcome: str) -> None:
    """
    同步因子IC反馈（当outcome确定时调用）

    Args:
        factor_name: 因子名称
        factor_value: 因子值
        predicted_direction: 1=LONG, -1=SHORT, 0=NEUTRAL
        outcome: WIN/LOSS
    """
    feedback = ICFeedback(
        factor_name=factor_name,
        predicted_direction=predicted_direction,
        actual_outcome=outcome,
        ic_score=0.0,  # 稍后计算
        sample_count=len(_get_store()._ic_cache.get(factor_name, [])) + 1,
        timestamp=datetime.now().isoformat()
    )
    _get_store().add_ic_feedback(feedback)
    new_ic = _get_store().get_ic_score(factor_name)
    if new_ic is not None:
        logger.info(f"[FusionMemory] {factor_name} IC更新: {new_ic:.3f}")


def format_log_entry(entry: Dict[str, Any]) -> str:
    """格式化单条记忆日志为可读字符串"""
    lines = [
        f"ID: {entry.get('id', 'N/A')}",
        f"时间: {entry.get('timestamp', 'N/A')}",
        f"标的: {entry.get('ticker', 'N/A')}",
        f"裁决: {entry.get('verdict', 'N/A')} (置信度: {entry.get('confidence', 0):.2f})",
        f"多头: {entry.get('bull_case', '')[:80]}",
        f"空头: {entry.get('bear_case', '')[:80]}",
    ]
    if entry.get("outcome"):
        lines.append(f"结果: {entry['outcome']} ({entry.get('pnl_pct', 0):+.2f}%)")
    lines.append(ENTRY_END)
    return "\n".join(lines)


def get_all_entries(ticker: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    获取所有（或按ticker过滤）记忆条目

    Args:
        ticker: 可选过滤器
        limit: 返回上限

    Returns:
        List[Dict]: 条目列表
    """
    store = _get_store()
    with store._lock:
        entries = list(store._entries)
    if ticker:
        entries = [e for e in entries if e.ticker == ticker]
    entries = entries[-limit:]
    return [asdict(e) for e in entries]


def rotate_log() -> Dict[str, int]:
    """
    执行日志轮转（保留最近1000条）

    Returns:
        Dict: 轮转统计
    """
    store = _get_store()
    with store._lock:
        before = len(store._entries)
        # deque maxlen已自动丢弃旧条目
        after = len(store._entries)
    store._save()
    return {"before": before, "after": after, "max_entries": MAX_ENTRIES}


def get_stats() -> Dict[str, Any]:
    """获取记忆系统统计"""
    store = _get_store()
    with store._lock:
        total = len(store._entries)
        with_outcome = sum(1 for e in store._entries if e.outcome is not None)
        pending = total - with_outcome
        tickers = set(e.ticker for e in store._entries)
        ic_factors = len(store._ic_cache)
    return {
        "total_entries": total,
        "with_outcome": with_outcome,
        "pending": pending,
        "unique_tickers": len(tickers),
        "ic_tracked_factors": ic_factors,
        "max_entries": MAX_ENTRIES
    }
