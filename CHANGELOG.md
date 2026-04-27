# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-04-27

### Added
- **Risk Management Module** (`core/risk_management.py`)
  - `ATRCalculator`: Wilder's smoothing ATR calculation
  - `DynamicPositionSizer`: Position size = (Account × Risk%) / (ATR × Stop Multiplier)
  - `CrossCurrencyRiskMonitor`: Multi-coin exposure tracking and correlation analysis
  - `SlippageFeeSimulator`: Market impact and fee simulation for backtesting

- **LLM Cache** (`core/llm_cache.py`)
  - Redis-based LLM response caching
  - SHA256 hash keys with provider-specific buckets
  - TTL auto-expiration and hit rate statistics
  - `CachedLLMCaller` wrapper for transparent caching

- **Exchange Adapter** (`core/exchange_adapter.py`)
  - `ExchangeAdapter` base class with unified interface
  - `OKXAdapter`: OKX exchange implementation
  - `BinanceAdapter`: Binance exchange implementation
  - Factory function `create_exchange_adapter()`

- **Prometheus Metrics** (`core/metrics.py`)
  - `TradingMetrics`: equity, positions, trades, circuit breaker
  - `LLMMetrics`: requests, latency, errors, fallbacks, cache
  - `SystemMetrics`: uptime, API calls, signals
  - Flask metrics endpoint at `/metrics`

- **Structured Logging** (`core/logging_config.py`)
  - `JSONFormatter` for machine-readable logs
  - `get_json_logger()`, `get_trade_logger()`, `get_audit_logger()`
  - `log_trade_event()`, `log_signal_event()`, `log_risk_event()` helpers
  - Rotating file handler with size-based rotation

- **Strategy Version Control** (`core/strategy_version_control.py`)
  - `create_version()`, `rollback_to()`, `emergency_rollback()`
  - `compare_versions()`, `mark_as_stable()`
  - Automatic cleanup of old versions

### Changed
- Refactored `backtest.py` (53KB → 4 modules):
  - `engine.py`: BacktestEngine (900 lines)
  - `stats.py`: Statistics calculation (410 lines)
  - `reporter.py`: Report generation (179 lines)
  - `backtest.py`: Backward-compatible wrapper (65 lines)

- Refactored `adaptive_learner.py` (45KB → 4 modules):
  - `learner.py`: Core learning components (628 lines)
  - `evaluator.py`: Factor/pattern evaluation (294 lines)
  - `strategy_evolution.py`: Strategy evolution (287 lines)
  - `adaptive_learner.py`: Backward-compatible wrapper (482 lines)

- Added `Dockerfile` and `docker-compose.yml` for containerized deployment

### Fixed
- `.env.example` template for API key management
- LLM provider fallback mechanism (primary → backup on failure)
- Order idempotency key generation (prevents duplicate orders)
- Emergency stop API with 3-layer checking (env/file/API)
- Orchestrator degradation (LLM failure → rule engine fallback)
- Memory expiration and forgetting mechanism

## [2.0.0] - 2026-04-27

### Added
- **LLM-Enhanced Architecture** (v2.0)
  - `core/orchestrator.py`: LLM-powered decision orchestration
  - `core/llm_provider.py`: Multi-provider support (Claude, GPT, Gemini, DeepSeek)
  - `core/memory/`: ChromaDB vector + SQLite structured memory
  - `agents/agent_market_intel_llm.py`: LLM-enhanced market intelligence
  - `miracle_autonomous.py`: AutoResearch continuous learning loop

### Changed
- Moved from rule-based (v1.0) to LLM-driven decision making
- Added 11 factors vs 7 in v1.0
- Implemented continuous learning and strategy evolution

## [1.0.2] - 2026-04-24

### Added
- Initial release with basic multi-agent trading system
- OKX exchange integration
- Basic backtesting framework
