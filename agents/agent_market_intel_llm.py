"""
Market Intel LLM 模块 - 向后兼容重导出
=========================================

本文件已重构为模块重导出，所有实现已移至 core/ 和 agents/ 下：

- core/market_intel_types.py     - 数据类、枚举、工具函数
- core/market_intel_llm.py       - LLM情感分析器
- core/market_intel_onchain.py   - 链上分析器
- core/market_intel_context.py   - 上下文构建器
- agents/market_intel_llm_agent.py - 主Agent类

现有导入不受影响，以下导入方式继续有效：
    from agents.agent_market_intel_llm import MarketIntelAgentLLM
"""

# 重导出所有公共接口
from agents.market_intel_llm_agent import (
    MarketIntelAgentLLM as MarketIntelAgentLLM,
    main_async as main_async,
    main as main,
)
from core.market_intel_context import ContextBuilder as ContextBuilder
from core.market_intel_llm import LLMSentimentAnalyzer as LLMSentimentAnalyzer
from core.market_intel_onchain import EnhancedOnChainAnalyzer as EnhancedOnChainAnalyzer
from core.market_intel_types import (
    CacheData as CacheData,
    IntelReport as IntelReport,
    LLMSentimentResult as LLMSentimentResult,
    MarketContext as MarketContext,
    OnChainPattern as OnChainPattern,
    SentimentLabel as SentimentLabel,
    SignalStrength as SignalStrength,
)

__all__ = [
    # 主类
    "MarketIntelAgentLLM",
    "main_async",
    "main",
    # 子模块
    "ContextBuilder",
    "LLMSentimentAnalyzer",
    "EnhancedOnChainAnalyzer",
    # 类型
    "SentimentLabel",
    "SignalStrength",
    "IntelReport",
    "LLMSentimentResult",
    "OnChainPattern",
    "MarketContext",
    "CacheData",
]
