"""
Miracle 1.0.1 Core Module
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

# Coin Parameter Optimizer (per-coin parameter optimization, inspired by Kronos coin_strategy_map.json)
from coin_optimizer import (
    CoinParameterOptimizer,
    CoinSignalGenerator,
    CoinParams,
    OptimizationResult,
    StrategyType,
    get_coin_optimizer,
    get_coin_signal_generator
)

__all__ = [
    # Config
    'ConfigManager', 'get_config', 'reload_config', 'ConfigError',
    # Regime
    'RegimeClassifier', 'MarketRegime', 'RegimeMetrics', 'detect_regime',
    # Feishu Notifier
    'FeishuNotifier', 'get_notifier',
    'push_feishu', 'push_feishu_alert', 'push_feishu_report',
    'is_feishu_configured',
    # Coin Parameter Optimizer
    'CoinParameterOptimizer',
    'CoinSignalGenerator', 
    'CoinParams',
    'OptimizationResult',
    'StrategyType',
    'get_coin_optimizer',
    'get_coin_signal_generator'
]
