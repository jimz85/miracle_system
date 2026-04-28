from __future__ import annotations

"""
Miracle 2.0 Core Module
========================
包含核心交易引擎和LLM驱动的Orchestrator协调器
"""

from core.config_manager import ConfigError, ConfigManager, get_config, reload_config
from core.feishu_notifier import (
    FeishuNotifier,
    get_notifier,
    is_feishu_configured,
    push_feishu,
    push_feishu_alert,
    push_feishu_report,
)

# LLM Provider (Miracle 2.0)
from core.llm_provider import (
    BaseLLMProvider,
    ClaudeProvider,
    DeepSeekProvider,
    GeminiProvider,
    GPTProvider,  # OpenAI/GPT Provider
    LLMConfig,
    LLMProviderManager,
    LLMProviderType,
    LLMResponse,
    Message,
    OllamaProvider,
    get_llm_manager,
    get_llm_provider,
)

# Orchestrator (Miracle 2.0) - TODO: implement
# from core.orchestrator import (
#     Orchestrator,
#     OrchestratorState,
#     TradingDecision,
#     TaskPlan,
#     Task,
#     TradeReflection,
#     DecisionType,
#     TaskStatus,
#     AgentM,
#     AgentS,
#     AgentR,
#     AgentE,
#     AgentL,
#     get_orchestrator,
# )
# Market Intel Modules
from core.market_intel_base import (
    API_CONFIG,
    CACHE_DIR,
    DEFAULT_LLM_PROVIDER,
    SYMBOL_MAP,
    CacheData,
    ContextBuilder,
    IntelReport,
    LLMSentimentResult,
    MarketContext,
    OnChainPattern,
    SentimentLabel,
    SignalStrength,
    api_request,
    get_timestamp,
    load_cache,
    save_cache,
)
from core.market_intel_context import ContextBuilder as MarketContextBuilder
from core.market_intel_llm import LLMSentimentAnalyzer
from core.market_intel_onchain import EnhancedOnChainAnalyzer
from core.market_intel_sentiment import (
    KeywordSentimentAnalyzer,
    NewsSentimentAnalyzer,
    SentimentAggregator,
)
from core.market_intel_technicals import (
    ExchangeFlowAnalyzer,
    TechnicalPatternRecognizer,
    WhaleTracker,
)

# Memory System (Miracle 2.0)
from core.memory import (
    FactorPerformance,
    Lesson,
    MemoryEntry,
    MemorySystem,
    MemoryType,
    StrategyParams,
    StructuredMemory,
    TradeRecord,
    VectorMemory,
    get_memory_system,
    get_structured_memory,
    get_vector_memory,
)
from core.regime_classifier import MarketRegime, RegimeClassifier, RegimeMetrics, detect_regime

# Coin Parameter Optimizer (per-coin parameter optimization, inspired by Kronos coin_strategy_map.json)
try:
    from coin_optimizer import (
        CoinParameterOptimizer,
        CoinParams,
        CoinSignalGenerator,
        OptimizationResult,
        StrategyType,
        get_coin_optimizer,
        get_coin_signal_generator,
    )
    HAS_COIN_OPTIMIZER = True
except ImportError:
    HAS_COIN_OPTIMIZER = False

__all__ = [
    # Config
    'ConfigManager', 'get_config', 'reload_config', 'ConfigError',
    # Regime
    'RegimeClassifier', 'MarketRegime', 'RegimeMetrics', 'detect_regime',
    # Feishu Notifier
    'FeishuNotifier', 'get_notifier',
    'push_feishu', 'push_feishu_alert', 'push_feishu_report',
    'is_feishu_configured',
    # LLM Provider
    'BaseLLMProvider',
    'ClaudeProvider',
    'GPTProvider',
    'GeminiProvider',
    'DeepSeekProvider',
    'OllamaProvider',
    'LLMProviderManager',
    'LLMProviderType',
    'LLMConfig',
    'LLMResponse',
    'Message',
    'get_llm_provider',
    'get_llm_manager',
    # Memory System
    'VectorMemory',
    'get_vector_memory',
    'StructuredMemory',
    'get_structured_memory',
    'MemorySystem',
    'get_memory_system',
    'TradeRecord',
    'FactorPerformance',
    'StrategyParams',
    'Lesson',
    'MemoryType',
    'MemoryEntry',
    # Market Intel
    'SentimentLabel',
    'SignalStrength',
    'IntelReport',
    'LLMSentimentResult',
    'OnChainPattern',
    'MarketContext',
    'CacheData',
    'get_timestamp',
    'load_cache',
    'save_cache',
    'api_request',
    'ContextBuilder',
    'MarketContextBuilder',
    'API_CONFIG',
    'DEFAULT_LLM_PROVIDER',
    'CACHE_DIR',
    'SYMBOL_MAP',
    'LLMSentimentAnalyzer',
    'NewsSentimentAnalyzer',
    'KeywordSentimentAnalyzer',
    'SentimentAggregator',
    'ExchangeFlowAnalyzer',
    'WhaleTracker',
    'TechnicalPatternRecognizer',
    'EnhancedOnChainAnalyzer',
    # Coin Parameter Optimizer
    'CoinParameterOptimizer',
    'CoinSignalGenerator',
    'CoinParams',
    'OptimizationResult',
    'StrategyType',
    'get_coin_optimizer',
    'get_coin_signal_generator'
]
