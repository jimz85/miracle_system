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

from .vector_memory import (
    VectorMemory,
    get_vector_memory,
    MemoryEntry
)

from .structured_memory import (
    StructuredMemory,
    get_structured_memory,
    TradeRecord,
    FactorPerformance,
    StrategyParams,
    Lesson,
    MemoryType
)

from .system import (
    MemorySystem,
    MemorySystemInterface,
    MemoryQuery,
    MemoryResult,
    get_memory_system
)

__all__ = [
    # Vector Memory
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
