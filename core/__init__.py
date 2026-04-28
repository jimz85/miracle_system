"""
Miracle 2.0 Core Module
========================
包含核心交易引擎和LLM驱动的Orchestrator协调器
"""

from core.config_manager import ConfigManager, get_config, reload_config, ConfigError
from core.regime_classifier import RegimeClassifier, MarketRegime, RegimeMetrics, detect_regime
from core.feishu_notifier import (
    FeishuNotifier,
    get_notifier,
    push_feishu,
    push_feishu_alert,
    push_feishu_report,
    is_feishu_configured
)

# LLM Provider (Miracle 2.0)
from core.llm_provider import (
    BaseLLMProvider,
    ClaudeProvider,
    GPTProvider,  # OpenAI/GPT Provider
    GeminiProvider,
    DeepSeekProvider,
    OllamaProvider,
    LLMProviderManager,
    LLMProviderType,
    LLMConfig,
    LLMResponse,
    Message,
    get_llm_provider,
    get_llm_manager,
)

# Memory System (Miracle 2.0)
from core.memory import (
    VectorMemory,
    get_vector_memory,
    StructuredMemory,
    get_structured_memory,
    MemorySystem,
    get_memory_system,
    TradeRecord,
    FactorPerformance,
    StrategyParams,
    Lesson,
    MemoryType,
    MemoryEntry,
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
    SentimentLabel,
    SignalStrength,
    IntelReport,
    LLMSentimentResult,
    OnChainPattern,
    MarketContext,
    CacheData,
    get_timestamp,
    load_cache,
    save_cache,
    api_request,
    ContextBuilder,
    API_CONFIG,
    DEFAULT_LLM_PROVIDER,
    CACHE_DIR,
    SYMBOL_MAP,
)

from core.market_intel_llm import LLMSentimentAnalyzer
from core.market_intel_sentiment import (
    NewsSentimentAnalyzer,
    KeywordSentimentAnalyzer,
    SentimentAggregator,
)
from core.market_intel_technicals import (
    ExchangeFlowAnalyzer,
    WhaleTracker,
    TechnicalPatternRecognizer,
)
from core.market_intel_onchain import EnhancedOnChainAnalyzer
from core.market_intel_context import ContextBuilder as MarketContextBuilder

# Coin Parameter Optimizer (per-coin parameter optimization, inspired by Kronos coin_strategy_map.json)
try:
    from coin_optimizer import (
        CoinParameterOptimizer,
        CoinSignalGenerator,
        CoinParams,
        OptimizationResult,
        StrategyType,
        get_coin_optimizer,
        get_coin_signal_generator
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
