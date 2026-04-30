from __future__ import annotations

"""
Memory System Module
====================
ChromaDB向量记忆 + SQLite结构化记忆

使用示例:
```python
from core.memory import get_memory_system, get_vector_memory, get_structured_memory

# 获取统一记忆系统
memory = get_memory_system()

# 添加经验
memory.add_experience(
    content="BTC在RSI<30时反弹成功率高",
    memory_type="lesson",
    metadata={"source": "trade_analysis"}
)

# 搜索记忆
results = memory.search("RSI超卖")

# 获取交易历史
trades = memory.get_trade_history(symbol="BTC")

# 获取教训
lessons = memory.get_lessons(category="entry")
```
"""

from .structured_memory import (
    FactorPerformance,
    Lesson,
    MemoryType,
    StrategyParams,
    StructuredMemory,
    TradeRecord,
    get_structured_memory,
)
from .system import (
    MemoryQuery,
    MemoryResult,
    MemorySystem,
    MemorySystemInterface,
    get_memory_system,
)
def __getattr__(name: str):
    """Lazy import to avoid loading sentence_transformers/torch at module import time."""
    if name in ("VectorMemory", "get_vector_memory", "MemoryEntry"):
        from . import vector_memory
        return getattr(vector_memory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Vector Memory (lazy)
    "VectorMemory",
    "get_vector_memory",
    "MemoryEntry",
    
    # Structured Memory
    "StructuredMemory",
    "get_structured_memory",
    "TradeRecord",
    "FactorPerformance",
    "StrategyParams",
    "Lesson",
    "MemoryType",
    
    # Unified System
    "MemorySystem",
    "MemorySystemInterface",
    "MemoryQuery",
    "MemoryResult",
    "get_memory_system"
]
