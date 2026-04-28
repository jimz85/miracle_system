from __future__ import annotations

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
)
from agents.market_intel_llm_agent import (
    main as main,
)
from agents.market_intel_llm_agent import (
    main_async as main_async,
)
from core.market_intel_base import (
    # 配置
    API_CONFIG as API_CONFIG,
)
from core.market_intel_base import (
    CACHE_DIR as CACHE_DIR,
)
from core.market_intel_base import (
    DEFAULT_LLM_PROVIDER as DEFAULT_LLM_PROVIDER,
)
from core.market_intel_base import (
    SYMBOL_MAP as SYMBOL_MAP,
)
from core.market_intel_base import (
    CacheData as CacheData,
)
from core.market_intel_base import (
    # ContextBuilder
    ContextBuilder as ContextBuilder,
)
from core.market_intel_base import (
    # 数据类
    IntelReport as IntelReport,
)
from core.market_intel_base import (
    LLMSentimentResult as LLMSentimentResult,
)
from core.market_intel_base import (
    MarketContext as MarketContext,
)
from core.market_intel_base import (
    OnChainPattern as OnChainPattern,
)

# 基础模块
from core.market_intel_base import (
    # 枚举
    SentimentLabel as SentimentLabel,
)
from core.market_intel_base import (
    SignalStrength as SignalStrength,
)
from core.market_intel_base import (
    api_request as api_request,
)
from core.market_intel_base import (
    # 工具函数
    get_timestamp as get_timestamp,
)
from core.market_intel_base import (
    load_cache as load_cache,
)
from core.market_intel_base import (
    save_cache as save_cache,
)

# 上下文构建器
# LLM分析器
from core.market_intel_llm import LLMSentimentAnalyzer as LLMSentimentAnalyzer

# 增强链上分析
from core.market_intel_onchain import EnhancedOnChainAnalyzer as EnhancedOnChainAnalyzer
from core.market_intel_sentiment import (
    KeywordSentimentAnalyzer as KeywordSentimentAnalyzer,
)

# 情绪数据模块
from core.market_intel_sentiment import (
    NewsSentimentAnalyzer as NewsSentimentAnalyzer,
)
from core.market_intel_sentiment import (
    SentimentAggregator as SentimentAggregator,
)

# 技术指标模块
from core.market_intel_technicals import (
    ExchangeFlowAnalyzer as ExchangeFlowAnalyzer,
)
from core.market_intel_technicals import (
    TechnicalPatternRecognizer as TechnicalPatternRecognizer,
)
from core.market_intel_technicals import (
    WhaleTracker as WhaleTracker,
)

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
