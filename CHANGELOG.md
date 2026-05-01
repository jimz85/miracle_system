# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## [2.0.2] - 2026-05-01

### Fixed
- **P0: SHORT ATR止损方向错误** — `check_stops()` 中SHORT的ATR止损改为入场价上方(`atr_stop_short`)
- **P0: ICWeightManager类名不匹配** — `MiracleICTracker` 不存在导致IC系统降级为硬编码
- **P0: 因子权重Disabled机制** — 配置加 `enabled` 字段，禁用后权重自动重归一化
- **P1: check_stops() fixed stop direction** — `format_trade_signal()` SHORT止损方向正确

### Changed
- **性能优化** — `agent_signal.py` 延迟导入 pandas/scipy，冷启动 27s→2.8s
- **配置矛盾** — `max_position_pct` 从15%降至13%（13%×3x=39%<40%）
- **配置局限** — `max_total_exposure` 40%保留，多仓叠加检查后续强化

### Added
- **每币种滑点配置** — 13个币种独立滑点（BTC 0.05% ~ BNT 0.8%）
- **FusionMemoryLog** — 别名类使文档API与实际代码兼容
- **熔断五级生存层** — MiracleCircuitBreaker接入AgentRisk决策流
- **Memory/Orchestrator** — 记忆系统接入信号生成+评分调整+Orchestrator验证
- **autoresearch硬编码修复** — bear_only 2025-08-01→动态滚动窗口

### Removed
- **文档清理** — 18份→7份核心文档，11份旧审计/设计文档移至`docs/archive/`
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
