# Contributing to Miracle System

Thank you for your interest in contributing to Miracle 2.0!

## Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/miracle_system.git
cd miracle_system

# 2. Setup environment
cp .env.example .env  # Configure your API keys
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v

# 4. Start in paper mode
python miracle.py --mode paper
```

## Development Workflow

### 1. Branch Naming

```bash
git checkout -b feature/your-feature-name
git checkout -b fix/bug-description
git checkout -b refactor/module-improvement
```

### 2. Making Changes

1. **Write code** following our style guidelines
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Follow the commit format** below

### 3. Commit Format

```
type(scope): description

Types:
  - feat: New feature
  - fix: Bug fix
  - refactor: Code refactoring
  - test: Adding/updating tests
  - docs: Documentation changes
  - chore: Build/tooling changes

Examples:
  feat(risk): add ATR-based dynamic position sizing
  fix(executor): prevent duplicate orders with idempotency key
  refactor(memory): split into vector and structured modules
```

### 4. Pull Request Process

1. Update CHANGELOG.md with your changes
2. Ensure all tests pass
3. Request review from maintainers
4. Merge after approval

## Code Guidelines

### Python Style

- Follow PEP 8
- Line length: 120 characters
- Use type hints where possible
- Run `black` and `isort` before committing:

```bash
black .
isort .
```

### Logging

**Standard logging (human-readable):**
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Position opened: BTC long 0.1 @ 50000")
```

**Structured logging (machine-readable):**
```python
from core.logging_config import get_json_logger, log_trade_event

logger = get_json_logger("my.module")
logger.info("Position opened", extra={"symbol": "BTC", "size": 0.1})

# For trade events, use helper:
trade_logger = get_trade_logger()
log_trade_event(trade_logger, "OPEN", "BTC", "long", 0.1, 50000)
```

**JSON log output:**
```json
{
  "timestamp": "2026-04-27T19:00:00.000Z",
  "level": "INFO",
  "logger": "miracle.trades",
  "message": "Position opened",
  "symbol": "BTC",
  "side": "long",
  "size": 0.1,
  "price": 50000
}
```

### Error Handling

- Never swallow exceptions silently
- Log errors with appropriate context
- Use custom exceptions for business logic errors

```python
# Good
try:
    result = api_call()
except APIError as e:
    logger.error(f"API failed: {e}", extra={"endpoint": url})
    raise TradingError(f"Failed to execute trade: {e}") from e

# Bad
try:
    result = api_call()
except:
    pass  # Never do this
```

### Testing

- Unit tests for all new functions/classes
- Integration tests for API interactions
- Use pytest fixtures for common setup

```python
# tests/test_risk_management.py
import pytest
from core.risk_management import DynamicPositionSizer

@pytest.fixture
def position_sizer():
    return DynamicPositionSizer(account_balance=10000)

def test_calculate_position(position_sizer):
    result = position_sizer.calculate_position(
        high=105, low=95, close=100,
        entry_price=100, direction="long"
    )
    assert result["position_size"] > 0
    assert result["stop_loss"] < 100  # Long position SL must be below entry
```

### Security

- Never commit API keys or secrets
- Use environment variables for sensitive data
- Validate and sanitize all inputs
- Log sensitive data (API keys, passwords) must be masked

## Project Structure

```
miracle_system/
├── agents/              # Trading agents (Signal, Risk, Executor)
├── core/                # Core modules
│   ├── memory/         # Vector + structured memory
│   ├── llm_provider.py # LLM interface
│   ├── orchestrator.py # Decision orchestration
│   ├── risk_management.py
│   ├── metrics.py      # Prometheus metrics
│   └── logging_config.py
├── backtest/           # Backtesting modules (refactored from backtest.py)
├── learner/            # Learning modules (refactored from adaptive_learner.py)
├── tests/              # Test suite
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Modules to Work On

### Priority Areas

1. **Core Trading Logic** (`agents/`)
   - `agent_signal.py`: Signal generation
   - `agent_risk.py`: Risk management
   - `agent_executor.py`: Order execution

2. **LLM Integration** (`core/llm_provider.py`, `core/orchestrator.py`)
   - Provider fallbacks
   - Response caching
   - Error handling

3. **Backtesting** (`backtest/`)
   - Walk-forward validation
   - Multi-coin testing
   - Performance optimization

4. **Learning System** (`learner/`, `miracle_autonomous.py`)
   - Hypothesis generation
   - Strategy evolution
   - Performance feedback

### Good First Issues

- Add unit tests for existing modules
- Improve error messages and logging
- Document unclear functions
- Optimize slow code paths

## Resources

- [GitHub Issues](https://github.com/jimz85/miracle_system/issues)
- [Wiki](https://github.com/jimz85/miracle_system/wiki)
- [Discord (if available)](#)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
