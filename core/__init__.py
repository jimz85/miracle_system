from __future__ import annotations

"""
Miracle 2.0 Core Module
========================
包含核心交易引擎和LLM驱动的Orchestrator协调器

P0 Fix: 所有慢模块改为懒加载（__getattr__），避免顶层导入导致30秒延迟。
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

# Lazy-load mapping: attribute name -> (module, names_tuple)
_LAZY_IMPORTS = {
    # LLM Provider
    'BaseLLMProvider': ('core.llm_provider', (
        'BaseLLMProvider', 'ClaudeProvider', 'DeepSeekProvider', 'GeminiProvider',
        'GPTProvider', 'LLMConfig', 'LLMProviderManager', 'LLMProviderType',
        'LLMResponse', 'Message', 'OllamaProvider', 'get_llm_manager', 'get_llm_provider',
    )),
    # Regime
    'RegimeClassifier': ('core.regime_classifier', (
        'MarketRegime', 'RegimeClassifier', 'RegimeMetrics', 'detect_regime',
    )),
    # Memory System
    'MemorySystem': ('core.memory', (
        'FactorPerformance', 'Lesson', 'MemoryEntry', 'MemorySystem', 'MemoryType',
        'StrategyParams', 'StructuredMemory', 'TradeRecord', 'VectorMemory',
        'get_memory_system', 'get_structured_memory', 'get_vector_memory',
    )),
    # Market Intel Base
    'ContextBuilder': ('core.market_intel_base', (
        'API_CONFIG', 'CACHE_DIR', 'DEFAULT_LLM_PROVIDER', 'SYMBOL_MAP',
        'CacheData', 'ContextBuilder', 'IntelReport', 'LLMSentimentResult',
        'MarketContext', 'OnChainPattern', 'SentimentLabel', 'SignalStrength',
        'api_request', 'get_timestamp', 'load_cache', 'save_cache',
    )),
    # Market Intel Sub-modules
    'MarketContextBuilder': ('core.market_intel_context', ('ContextBuilder',)),
    'LLMSentimentAnalyzer': ('core.market_intel_llm', ('LLMSentimentAnalyzer',)),
    'EnhancedOnChainAnalyzer': ('core.market_intel_onchain', ('EnhancedOnChainAnalyzer',)),
    'KeywordSentimentAnalyzer': ('core.market_intel_sentiment', (
        'KeywordSentimentAnalyzer', 'NewsSentimentAnalyzer', 'SentimentAggregator',
    )),
    'ExchangeFlowAnalyzer': ('core.market_intel_technicals', (
        'ExchangeFlowAnalyzer', 'TechnicalPatternRecognizer', 'WhaleTracker',
    )),
    # Coin Optimizer
    'CoinParameterOptimizer': ('coin_optimizer', (
        'CoinParameterOptimizer', 'CoinParams', 'CoinSignalGenerator',
        'OptimizationResult', 'StrategyType', 'get_coin_optimizer', 'get_coin_signal_generator',
    )),
}

# Cache for already-loaded lazy modules
_LOADED = {}

def __getattr__(name: str):
    """Lazy import - only loads slow modules when actually used."""
    if name in _LOADED:
        return _LOADED[name]

    for attr, (module, names) in _LAZY_IMPORTS.items():
        if name in names:
            import importlib
            mod = importlib.import_module(module)
            for n in names:
                _LOADED[n] = getattr(mod, n)
            return _LOADED[name]

    raise AttributeError(f"module 'core' has no attribute {name!r}")

# Coin optimizer flag (checked at import time, not lazy)
try:
    from coin_optimizer import (
        CoinParameterOptimizer, CoinParams, CoinSignalGenerator,
        OptimizationResult, StrategyType, get_coin_optimizer, get_coin_signal_generator,
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
    'BaseLLMProvider', 'ClaudeProvider', 'GPTProvider', 'GeminiProvider',
    'DeepSeekProvider', 'OllamaProvider', 'LLMProviderManager', 'LLMProviderType',
    'LLMConfig', 'LLMResponse', 'Message', 'get_llm_provider', 'get_llm_manager',
    # Memory System
    'VectorMemory', 'get_vector_memory', 'StructuredMemory', 'get_structured_memory',
    'MemorySystem', 'get_memory_system', 'TradeRecord', 'FactorPerformance',
    'StrategyParams', 'Lesson', 'MemoryType', 'MemoryEntry',
    # Market Intel
    'SentimentLabel', 'SignalStrength', 'IntelReport', 'LLMSentimentResult',
    'OnChainPattern', 'MarketContext', 'CacheData', 'get_timestamp',
    'load_cache', 'save_cache', 'api_request', 'ContextBuilder',
    'MarketContextBuilder', 'API_CONFIG', 'DEFAULT_LLM_PROVIDER', 'CACHE_DIR',
    'SYMBOL_MAP', 'LLMSentimentAnalyzer', 'NewsSentimentAnalyzer',
    'KeywordSentimentAnalyzer', 'SentimentAggregator', 'ExchangeFlowAnalyzer',
    'WhaleTracker', 'TechnicalPatternRecognizer', 'EnhancedOnChainAnalyzer',
    # Coin Parameter Optimizer
    'CoinParameterOptimizer', 'CoinSignalGenerator', 'CoinParams',
    'OptimizationResult', 'StrategyType', 'get_coin_optimizer', 'get_coin_signal_generator',
]
