"""
Market Intel LLM 模块 - 向后兼容重导出
=========================================

本文件已重构为模块重导出，所有实现已移至 core/ 下：

核心模块:
- core/market_intel_base.py         - 基础类、类型、枚举、工具函数、ContextBuilder
- core/market_intel_llm.py          - LLM情感分析器
- core/market_intel_sentiment.py   - 情绪数据分析
- core/market_intel_technicals.py   - 技术指标分析
- core/market_intel_onchain.py      - 增强链上分析
- core/market_intel_context.py      - 上下文构建器

Agent:
- agents/market_intel_llm_agent.py  - 主Agent类

现有导入不受影响，以下导入方式继续有效：
    from agents.agent_market_intel_llm import MarketIntelAgentLLM
    from agents.agent_market_intel_llm import LLMSentimentAnalyzer
    from agents.agent_market_intel_llm import ContextBuilder
"""

# 重导出所有公共接口
from agents.market_intel_llm_agent import (
    MarketIntelAgentLLM as MarketIntelAgentLLM,
    main_async as main_async,
    main as main,
)

# 基础模块
from core.market_intel_base import (
    # 枚举
    SentimentLabel as SentimentLabel,
    SignalStrength as SignalStrength,
    # 数据类
    IntelReport as IntelReport,
    LLMSentimentResult as LLMSentimentResult,
    OnChainPattern as OnChainPattern,
    MarketContext as MarketContext,
    CacheData as CacheData,
    # 工具函数
    get_timestamp as get_timestamp,
    load_cache as load_cache,
    save_cache as save_cache,
    api_request as api_request,
    # ContextBuilder
    ContextBuilder as ContextBuilder,
    # 配置
    API_CONFIG as API_CONFIG,
    DEFAULT_LLM_PROVIDER as DEFAULT_LLM_PROVIDER,
    CACHE_DIR as CACHE_DIR,
    SYMBOL_MAP as SYMBOL_MAP,
)

# LLM分析器
from core.market_intel_llm import LLMSentimentAnalyzer as LLMSentimentAnalyzer

# 情绪数据模块
from core.market_intel_sentiment import (
    NewsSentimentAnalyzer as NewsSentimentAnalyzer,
    KeywordSentimentAnalyzer as KeywordSentimentAnalyzer,
    SentimentAggregator as SentimentAggregator,
)

# 技术指标模块
from core.market_intel_technicals import (
    ExchangeFlowAnalyzer as ExchangeFlowAnalyzer,
    WhaleTracker as WhaleTracker,
    TechnicalPatternRecognizer as TechnicalPatternRecognizer,
)

# 增强链上分析
from core.market_intel_onchain import EnhancedOnChainAnalyzer as EnhancedOnChainAnalyzer

# 上下文构建器
from core.market_intel_context import ContextBuilder as ContextBuilder

__all__ = [
    # 主类
    "MarketIntelAgentLLM",
    "main_async",
    "main",
    # 基础类型
    "SentimentLabel",
    "SignalStrength",
    "IntelReport",
    "LLMSentimentResult",
    "OnChainPattern",
    "MarketContext",
    "CacheData",
    # 工具函数
    "get_timestamp",
    "load_cache",
    "save_cache",
    "api_request",
    # 构建器
    "ContextBuilder",
    # LLM分析器
    "LLMSentimentAnalyzer",
    "NewsSentimentAnalyzer",
    "KeywordSentimentAnalyzer",
    "SentimentAggregator",
    # 技术指标
    "ExchangeFlowAnalyzer",
    "WhaleTracker",
    "TechnicalPatternRecognizer",
    # 增强链上分析
    "EnhancedOnChainAnalyzer",
    # 配置
    "API_CONFIG",
    "DEFAULT_LLM_PROVIDER",
    "CACHE_DIR",
    "SYMBOL_MAP",
]
