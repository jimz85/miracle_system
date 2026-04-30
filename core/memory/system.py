from __future__ import annotations

"""
Memory System - Unified Interface
================================
统一记忆系统接口 - 结合ChromaDB向量记忆和SQLite结构化记忆
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .structured_memory import (
    FactorPerformance,
    Lesson,
    MemoryType,
    StrategyParams,
    StructuredMemory,
    TradeRecord,
    get_structured_memory,
)
logger = logging.getLogger(__name__)


class MemorySystemInterface(Enum):
    """记忆系统接口类型"""
    VECTOR = "vector"
    STRUCTURED = "structured"
    BOTH = "both"


@dataclass
class MemoryQuery:
    """记忆查询请求"""
    query: str
    k: int = 5
    memory_types: List[str] | None = None  # vector memory types
    interface: MemorySystemInterface = MemorySystemInterface.BOTH
    
    # Structured memory filters
    table_name: str | None = None  # trade/factor/strategy/lesson
    symbol: str | None = None
    category: str | None = None
    limit: int = 10


@dataclass
class MemoryResult:
    """记忆查询结果"""
    id: str
    content: str
    source: MemorySystemInterface  # VECTOR or STRUCTURED
    memory_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity: float = 1.0  # 向量相似度，仅VECTOR有
    created_at: datetime | None = None


class MemorySystem:
    """
    统一记忆系统
    整合向量记忆(ChromaDB)和结构化记忆(SQLite)
    
    使用示例:
    ```python
    memory = MemorySystem()
    
    # 添加交易经验到两个系统
    memory.add_experience(
        content="BTC在RSI<30时反弹成功率很高",
        memory_type="lesson",
        structured_data={
            "category": "entry",
            "trigger_symbol": "BTC",
            "outcome": "WIN"
        }
    )
    
    # 检索记忆
    results = memory.search("RSI超卖")
    ```
    """
    
    def __init__(self, 
                 vector_memory: "VectorMemory | None" = None,
                 structured_memory: StructuredMemory | None = None):
        """
        初始化记忆系统
        
        Args:
            vector_memory: 向量记忆实例
            structured_memory: 结构化记忆实例
        """
        from .vector_memory import VectorMemory as _VectorMemory, get_vector_memory as _get_vector_memory
        self.vector = vector_memory or _get_vector_memory()
        self.structured = structured_memory or get_structured_memory()
    
    # ==================== Add Operations ====================
    
    def add_experience(self, content: str, 
                      memory_type: str = "general",
                      metadata: Dict[str, Any] | None = None,
                      structured_data: Dict[str, Any] | None = None,
                      embeddings: List[float] | None = None) -> str:
        """
        添加经验到记忆系统（同时写入向量和结构化存储）
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 (general/trade/lesson/pattern/market)
            metadata: 向量存储的元数据
            structured_data: 结构化存储的数据 (根据类型不同而不同)
            embeddings: 可选的预计算嵌入向量
            
        Returns:
            str: 记忆ID
        """
        import uuid
        
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["memory_type"] = memory_type
        
        # 1. 添加到向量存储
        try:
            self.vector.add(
                content=content,
                metadata=metadata,
                memory_type=memory_type,
                id=memory_id
            )
        except Exception as e:
            logger.error(f"Failed to add to vector memory: {e}")
        
        # 2. 根据类型添加到结构化存储
        try:
            if structured_data:
                self._add_structured(memory_type, memory_id, content, structured_data)
        except Exception as e:
            logger.error(f"Failed to add to structured memory: {e}")
        
        return memory_id
    
    def _add_structured(self, memory_type: str, memory_id: str, 
                       content: str, data: Dict[str, Any]):
        """根据类型添加到结构化存储"""
        
        if memory_type == "trade":
            # 交易记录
            trade = TradeRecord(
                id=None,  # 自增ID
                symbol=data.get("symbol", ""),
                direction=data.get("direction", ""),
                entry_price=data.get("entry_price", 0),
                exit_price=data.get("exit_price", 0),
                entry_time=data.get("entry_time", datetime.now()),
                exit_time=data.get("exit_time"),
                position_size=data.get("position_size", 0),
                pnl=data.get("pnl", 0),
                pnl_pct=data.get("pnl_pct", 0),
                strategy=data.get("strategy", ""),
                signals=data.get("signals", {}),
                market_context=data.get("market_context", {}),
                notes=content,
                status=data.get("status", "open")
            )
            return self.structured.add_trade(trade)
        
        elif memory_type == "lesson":
            # 教训
            lesson = Lesson(
                id=None,
                category=data.get("category", "general"),
                content=content,
                trigger_symbol=data.get("trigger_symbol", ""),
                trigger_direction=data.get("trigger_direction", ""),
                outcome=data.get("outcome", ""),
                actionable=data.get("actionable", True),
                tags=data.get("tags", [])
            )
            return self.structured.add_lesson(lesson)
        
        elif memory_type == "strategy":
            # 策略参数
            params = StrategyParams(
                strategy_name=data.get("strategy_name", ""),
                symbol=data.get("symbol", ""),
                params=data.get("params", {}),
                performance=data.get("performance", 0),
                notes=content
            )
            return self.structured.save_strategy_params(params)
        
        elif memory_type == "factor":
            # 因子表现
            factor = FactorPerformance(
                factor_name=data.get("factor_name", ""),
                symbol=data.get("symbol", ""),
                value=data.get("value", 0),
                signal_direction=data.get("signal_direction", ""),
                actual_outcome=data.get("actual_outcome"),
                market_regime=data.get("market_regime", "")
            )
            return self.structured.add_factor_performance(factor)
        
        logger.warning(f"Unknown memory type for structured storage: {memory_type}")
        return None
    
    def add_trade(self, trade: TradeRecord) -> int:
        """添加交易记录（同时添加到向量记忆）"""
        # 添加到结构化存储
        trade_id = self.structured.add_trade(trade)
        
        # 同步到向量记忆
        if trade.status == "closed":
            content = self._format_trade_for_memory(trade)
            self.add_experience(
                content=content,
                memory_type="trade",
                metadata={
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "pnl_pct": trade.pnl_pct,
                    "strategy": trade.strategy,
                    "trade_id": trade_id
                },
                structured_data={
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "strategy": trade.strategy,
                    "signals": trade.signals,
                    "market_context": trade.market_context,
                    "status": trade.status
                }
            )
        
        return trade_id
    
    def _format_trade_for_memory(self, trade: TradeRecord) -> str:
        """格式化交易记录为记忆文本"""
        outcome = "盈利" if trade.pnl > 0 else "亏损"
        return (
            f"{trade.direction}交易 {trade.symbol}，"
            f"入场{trade.entry_price}，出场{trade.exit_price}，"
            f"{outcome}{abs(trade.pnl_pct):.2f}%。"
            f"策略:{trade.strategy}。"
            f"持仓期间市场状态:{trade.market_context}。"
        )
    
    def add_lesson(self, category: str, content: str,
                  trigger_symbol: str = "",
                  trigger_direction: str = "",
                  outcome: str = "",
                  tags: List[str] | None = None) -> int:
        """添加教训"""
        lesson = Lesson(
            category=category,
            content=content,
            trigger_symbol=trigger_symbol,
            trigger_direction=trigger_direction,
            outcome=outcome,
            tags=tags or []
        )
        
        # 添加到结构化存储
        lesson_id = self.structured.add_lesson(lesson)
        
        # 同步到向量记忆
        self.add_experience(
            content=content,
            memory_type="lesson",
            metadata={
                "category": category,
                "trigger_symbol": trigger_symbol,
                "outcome": outcome,
                "lesson_id": lesson_id
            }
        )
        
        return lesson_id
    
    # ==================== Search/Retrieve Operations ====================
    
    def search(self, query: str, k: int = 5,
               memory_types: List[str] | None = None,
               symbol: str | None = None) -> List[MemoryResult]:
        """
        检索记忆（同时搜索向量和结构化存储）
        
        Args:
            query: 查询文本
            k: 返回数量
            memory_types: 记忆类型过滤
            symbol: 币种过滤
            
        Returns:
            List[MemoryResult]: 记忆结果列表
        """
        results = []
        
        # 1. 向量记忆检索
        try:
            # VectorMemory.search 只支持单个 memory_type，取第一个
            mem_type = memory_types[0] if memory_types else None
            vector_results = self.vector.search(
                query=query,
                k=k,
                memory_type=mem_type
            )
            
            for mem in vector_results:
                if symbol and mem.get("metadata", {}).get("symbol") != symbol:
                    continue
                    
                results.append(MemoryResult(
                    id=mem["id"],
                    content=mem["content"],
                    source=MemorySystemInterface.VECTOR,
                    memory_type=mem.get("metadata", {}).get("memory_type", "general"),
                    metadata=mem.get("metadata", {}),
                    similarity=mem.get("similarity", 1.0),
                    created_at=datetime.fromisoformat(
                        mem["metadata"].get("created_at", datetime.now().isoformat())
                    ) if mem.get("metadata", {}).get("created_at") else None
                ))
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        # 2. 结构化记忆检索
        try:
            # 检索教训
            lessons = self.structured.get_lessons(limit=k)
            for lesson in lessons:
                # 简单的文本匹配
                if query.lower() in lesson.content.lower():
                    results.append(MemoryResult(
                        id=str(lesson.id),
                        content=lesson.content,
                        source=MemorySystemInterface.STRUCTURED,
                        memory_type="lesson",
                        metadata={
                            "category": lesson.category,
                            "trigger_symbol": lesson.trigger_symbol,
                            "outcome": lesson.outcome,
                            "success_rate": lesson.success_rate
                        },
                        similarity=0.8,  # 假设匹配
                        created_at=lesson.created_at
                    ))
        except Exception as e:
            logger.error(f"Structured search failed: {e}")
        
        # 按相似度排序
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        return results[:k]
    
    def retrieve_relevant_experiences(self, query: str, k: int = 5) -> str:
        """
        检索相关经验并格式化为上下文字符串
        
        Args:
            query: 查询文本
            k: 返回数量
            
        Returns:
            str: 格式化的上下文
        """
        results = self.search(query, k=k)
        
        if not results:
            return ""
        
        context_parts = ["=== 相关历史经验 ==="]
        
        # 分组显示
        by_type = {}
        for r in results:
            if r.memory_type not in by_type:
                by_type[r.memory_type] = []
            by_type[r.memory_type].append(r)
        
        for mem_type, memories in by_type.items():
            context_parts.append(f"\n【{mem_type.upper()}】")
            for mem in memories[:3]:  # 每类型最多3条
                meta = mem.metadata
                meta.get("created_at", "")[:10] if meta.get("created_at") else ""
                context_parts.append(
                    f"• {mem.content[:200]}"
                    f"{' [OK]' if meta.get('outcome') == 'WIN' else ''}"
                )
        
        return "\n".join(context_parts)
    
    def get_trade_history(self, symbol: str | None = None,
                         limit: int = 50) -> List[TradeRecord]:
        """获取交易历史"""
        return self.structured.get_trades(symbol=symbol, status="closed", limit=limit)
    
    def get_open_positions(self) -> List[TradeRecord]:
        """获取未平仓交易"""
        return self.structured.get_open_positions()
    
    def get_lessons(self, category: str | None = None) -> List[Lesson]:
        """获取教训列表"""
        return self.structured.get_lessons(category=category)
    
    def get_active_strategies(self, symbol: str | None = None) -> List[StrategyParams]:
        """获取活跃策略"""
        return self.structured.get_active_strategies(symbol=symbol)
    
    def get_factor_performance(self, factor_name: str, symbol: str) -> float:
        """获取因子表现（IC）"""
        return self.structured.get_factor_ic(factor_name, symbol)
    
    # ==================== Update Operations ====================
    
    def close_trade(self, trade_id: int, exit_price: float,
                   pnl: float, pnl_pct: float, notes: str = "") -> bool:
        """平仓交易并更新相关记忆"""
        # 更新结构化记录
        self.structured.update_trade(
            trade_id,
            exit_price=exit_price,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            status="closed",
            notes=notes
        )
        
        # 获取交易信息用于更新教训
        trade = self.structured.get_trade(trade_id)
        if trade and trade.pnl_pct != 0:
            
            # 更新教训应用结果
            # 注意：这里简化处理，实际可能需要更复杂的匹配逻辑
            pass
        
        return True
    
    def update_lesson_outcome(self, lesson_id: int, success: bool):
        """更新教训应用结果"""
        self.structured.update_lesson_success(lesson_id, success)
    
    # ==================== Analytics ====================
    
    def analyze_trading_patterns(self, symbol: str) -> Dict[str, Any]:
        """分析交易模式"""
        trades = self.get_trade_history(symbol=symbol, limit=100)
        
        if not trades:
            return {"patterns": [], "insights": []}
        
        patterns = []
        insights = []
        
        # 分析赚钱交易
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        if winning_trades:
            avg_win_pct = sum(t.pnl_pct for t in winning_trades) / len(winning_trades)
            insights.append(f"平均盈利: {avg_win_pct:.2f}%")
            
            # 分析盈利交易的共同特征
            # 例如: RSI < 30 入场
            for trade in winning_trades:
                if trade.signals.get("rsi") and trade.signals["rsi"] < 35:
                    patterns.append(f"RSI={trade.signals['rsi']}时入场往往盈利")
                    break
        
        if losing_trades:
            avg_loss_pct = sum(t.pnl_pct for t in losing_trades) / len(losing_trades)
            insights.append(f"平均亏损: {avg_loss_pct:.2f}%")
        
        # 统计各策略表现
        strategy_stats = {}
        for trade in trades:
            strategy = trade.strategy or "unknown"
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"wins": 0, "losses": 0, "total_pnl": 0}
            
            if trade.pnl > 0:
                strategy_stats[strategy]["wins"] += 1
            else:
                strategy_stats[strategy]["losses"] += 1
            strategy_stats[strategy]["total_pnl"] += trade.pnl
        
        # 找出最佳策略
        best_strategy = max(strategy_stats.items(), 
                           key=lambda x: x[1]["total_pnl"], 
                           default=(None, None))
        if best_strategy[0]:
            insights.append(f"最佳策略: {best_strategy[0]} (盈利${best_strategy[1]['total_pnl']:.2f})")
        
        return {
            "total_trades": len(trades),
            "win_rate": len(winning_trades) / len(trades) if trades else 0,
            "patterns": list(set(patterns)),
            "insights": insights,
            "strategy_performance": strategy_stats
        }
    
    def generate_daily_summary(self) -> str:
        """生成每日记忆摘要"""
        from datetime import timedelta
        
        stats = self.structured.get_stats()
        
        # 获取今日交易
        today = datetime.now().date()
        today_trades = self.structured.get_trades(limit=100)
        today_trades = [t for t in today_trades 
                       if t.entry_time.date() == today]
        
        summary_parts = ["=== 每日记忆摘要 ==="]
        summary_parts.append(f"总交易数: {stats['total_trades']}")
        summary_parts.append(f"活跃策略数: {stats['active_strategies']}")
        summary_parts.append(f"教训数: {stats['total_lessons']}")
        
        if today_trades:
            wins = sum(1 for t in today_trades if t.pnl > 0)
            summary_parts.append(f"\n今日交易: {len(today_trades)}笔 ({wins}胜{len(today_trades)-wins}负)")
        
        # 获取最近的教训
        recent_lessons = self.structured.get_lessons(limit=3)
        if recent_lessons:
            summary_parts.append("\n最近教训:")
            for lesson in recent_lessons:
                summary_parts.append(f"• [{lesson.category}] {lesson.content[:50]}...")
        
        return "\n".join(summary_parts)
    
    # ==================== Utility ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计"""
        vector_stats = self.vector.get_stats()
        structured_stats = self.structured.get_stats()
        
        return {
            "vector": vector_stats,
            "structured": structured_stats,
            "total_memories": vector_stats.get("total", 0) + structured_stats.get("total_lessons", 0)
        }
    
    def reset(self, confirm: bool = False):
        """
        重置记忆系统（危险操作）
        
        Args:
            confirm: 必须设为True才会执行
        """
        if not confirm:
            logger.warning("Use reset(confirm=True) to actually reset the memory system")
            return
        
        self.vector.reset()
        self.structured.reset()
        logger.warning("Memory system has been reset")
    
    def export_memories(self, filepath: str) -> bool:
        """导出记忆到JSON文件"""
        import json
        
        try:
            data = {
                "export_time": datetime.now().isoformat(),
                "vector_stats": self.vector.get_stats(),
                "structured_stats": self.structured.get_stats(),
                # 导出教训
                "lessons": [
                    {
                        **lesson.__dict__,
                        "created_at": lesson.created_at.isoformat()
                    }
                    for lesson in self.structured.get_lessons(limit=1000)
                ],
                # 导出活跃策略
                "strategies": [
                    {
                        **{
                            k: v for k, v in s.__dict__.items() 
                            if k not in ["created_at", "updated_at"]
                        },
                        "created_at": s.created_at.isoformat(),
                        "updated_at": s.updated_at.isoformat()
                    }
                    for s in self.structured.get_active_strategies()
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Memories exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export memories: {e}")
            return False
    
    # ==================== Forgetting & Cleanup Mechanisms ====================
    
    def forget_low_value_memories(self, 
                                   min_success_rate: float = 0.3,
                                   min_applied: int = 5,
                                   dry_run: bool = True) -> Dict[str, Any]:
        """
        遗忘低价值记忆（遗忘机制）
        
        遗忘策略:
        - 教训: 成功率低于min_success_rate且应用次数>=min_applied的会被遗忘
        - 向量记忆: 清理过期的记忆条目
        
        Args:
            min_success_rate: 教训最小成功率阈值
            min_applied: 教训最小应用次数阈值  
            dry_run: 若为True，只统计不删除
            
        Returns:
            Dict: 遗忘统计
        """
        results = {
            "structured_lessons": {},
            "vector_expired": {},
            "total_forgotten": 0
        }
        
        # 遗忘低价值教训
        lesson_result = self.structured.cleanup_low_value_lessons(
            min_success_rate=min_success_rate,
            min_applied=min_applied,
            dry_run=dry_run
        )
        results["structured_lessons"] = lesson_result
        if not dry_run:
            results["total_forgotten"] += lesson_result.get("deleted", 0)
        
        # 清理过期向量记忆
        vector_result = self.vector.cleanup_expired(dry_run=dry_run)
        results["vector_expired"] = vector_result
        if not dry_run:
            results["total_forgotten"] += vector_result.get("deleted", 0)
        
        return results
    
    def run_memory_maintenance(self, 
                              trade_days: int = 90,
                              vector_age_days: int = 30,
                              lesson_success_rate: float = 0.3,
                              lesson_min_applied: int = 5,
                              strategy_min_trades: int = 10,
                              strategy_min_perf: float = -0.1,
                              dry_run: bool = True) -> Dict[str, Any]:
        """
        运行完整记忆维护（遗忘机制 + 过期清理）
        
        Args:
            trade_days: 保留最近N天的交易
            vector_age_days: 清理N天以上的向量记忆
            lesson_success_rate: 教训最小成功率阈值
            lesson_min_applied: 教训最小应用次数阈值
            strategy_min_trades: 策略最小交易次数
            strategy_min_perf: 策略最小收益率
            dry_run: 若为True，只统计不删除
            
        Returns:
            Dict: 各维护操作的统计
        """
        results = {}
        
        # 结构化记忆维护
        results["structured"] = self.structured.run_memory_maintenance(
            trade_days=trade_days,
            lesson_success_rate=lesson_success_rate,
            lesson_min_applied=lesson_min_applied,
            strategy_min_trades=strategy_min_trades,
            strategy_min_perf=strategy_min_perf,
            dry_run=dry_run
        )
        
        # 向量记忆按年龄清理
        results["vector_by_age"] = self.vector.cleanup_by_age(
            max_age_days=vector_age_days,
            dry_run=dry_run
        )
        
        # 清理明确的过期记忆
        results["vector_expired"] = self.vector.cleanup_expired(dry_run=dry_run)
        
        # 汇总
        total_deleted = 0
        for key, val in results.items():
            if isinstance(val, dict):
                total_deleted += val.get("deleted", 0)
        
        results["summary"] = {
            "dry_run": dry_run,
            "total_deleted": total_deleted,
            "message": f"Would forget {total_deleted} memories" if dry_run else f"Forgotten {total_deleted} memories"
        }
        
        return results
    
    def get_memory_health_report(self) -> Dict[str, Any]:
        """
        获取记忆健康报告
        
        Returns:
            Dict: 健康状况报告
        """
        vector_stats = self.vector.get_memory_stats_with_expiry()
        structured_stats = self.structured.get_stats()
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "vector_memory": vector_stats,
            "structured_memory": structured_stats,
            "issues": [],
            "recommendations": []
        }
        
        # 检查问题
        total_memories = vector_stats.get("total", 0)
        if total_memories > 10000:
            health["issues"].append("Vector memory exceeds 10000 entries - consider cleanup")
            health["recommendations"].append("Run cleanup to remove old memories")
        
        if vector_stats.get("expired", 0) > 0:
            health["recommendations"].append(f"Clean up {vector_stats['expired']} expired memories")
        
        if vector_stats.get("expiring_soon", 0) > 100:
            health["recommendations"].append(f"{vector_stats['expiring_soon']} memories expiring soon")
        
        # 总体状态
        if not health["issues"]:
            health["status"] = "healthy"
        elif len(health["issues"]) <= 2:
            health["status"] = "warning"
        else:
            health["status"] = "critical"
        
        return health


# 全局实例
_memory_system: MemorySystem | None = None

def get_memory_system() -> MemorySystem:
    """获取记忆系统全局实例"""
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem()
    return _memory_system
