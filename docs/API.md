# Miracle 2.0 API Documentation

## Swagger UI

访问 `http://localhost:8000/docs` 查看交互式API文档

## Endpoints

### Dashboard API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard HTML页面 |
| `GET` | `/api/stats` | 获取交易统计 |
| `GET` | `/api/quotes` | 获取实时行情 |
| `GET` | `/api/signals` | 获取交易信号 |
| `GET` | `/api/positions` | 获取持仓 |
| `WS` | `/ws` | WebSocket实时推送 |

### Response Schemas

#### Stats Response
```json
{
  "total_trades": 150,
  "win_rate": 0.58,
  "total_pnl": 2450.75,
  "sharpe_ratio": 1.85,
  "max_drawdown": 8.5,
  "equity": 12450.75
}
```

#### Quote Response
```json
{
  "symbol": "BTC-USDT",
  "last_price": 67500.00,
  "bid": 67495.00,
  "ask": 67505.00,
  "high_24h": 69000.00,
  "low_24h": 66000.00,
  "volume_24h": 15234567.89,
  "change_24h": 1250.00,
  "change_pct_24h": 1.89,
  "timestamp": 1714248000000
}
```

#### Signal Response
```json
{
  "id": "SIG-1714248000001",
  "symbol": "BTC-USDT",
  "direction": "buy",
  "strength": 0.85,
  "indicators": {
    "RSI": 32.5,
    "MACD": 150.75,
    "Bollinger": 0.55
  },
  "reason": "BUY signal from RSI indicator",
  "timestamp": 1714248000000
}
```

#### Position Response
```json
{
  "symbol": "BTC-USDT",
  "side": "long",
  "size": 0.5,
  "entry_price": 67000.00,
  "current_price": 67500.00,
  "unrealized_pnl": 250.00,
  "unrealized_pnl_pct": 0.75,
  "timestamp": 1714248000000
}
```

---

## Prometheus Metrics API

访问 `http://localhost:9091/metrics` 获取指标

### Trading Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `miracle_equity_current` | Gauge | - | 当前权益 |
| `miracle_positions_open` | Gauge | - | 开仓数 |
| `miracle_position_pnl` | Gauge | symbol, side | 各持仓盈亏 |
| `miracle_trades_total` | Counter | symbol, side, result | 交易总数 |
| `miracle_circuit_breaker_tripped` | Gauge | - | 熔断器状态 |

### LLM Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `miracle_llm_requests_total` | Counter | provider, model, status | LLM请求数 |
| `miracle_llm_request_duration_seconds` | Histogram | provider, model | LLM延迟 |
| `miracle_llm_fallbacks_total` | Counter | from_provider, to_provider | 降级次数 |

### System Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `miracle_system_uptime_seconds` | Gauge | - | 运行时间 |
| `miracle_system_api_latency_seconds` | Histogram | service, endpoint | API延迟 |

---

## Emergency Stop API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/status` | 获取交易状态 |
| `POST` | `/emergency/stop` | 触发紧急停止 |
| `POST` | `/emergency/resume` | 恢复交易 |

### Stop Request
```bash
curl -X POST http://localhost:8080/emergency/stop \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Response
```json
{
  "success": true,
  "stopped_at": "2026-04-27T20:00:00Z",
  "reason": "Manual trigger"
}
```

---

## Configuration API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config` | 获取配置 |
| `PUT` | `/api/config` | 更新配置 |
| `GET` | `/api/version` | 获取版本信息 |

---

## WebSocket Protocol

连接: `ws://localhost:8000/ws`

### Message Types

#### stats
```json
{
  "type": "stats",
  "equity": 12450.75,
  "win_rate": 0.58,
  "sharpe_ratio": 1.85,
  "max_drawdown": 8.5,
  "total_trades": 150
}
```

#### quotes
```json
{
  "type": "quotes",
  "quotes": [...]
}
```

#### signals
```json
{
  "type": "signals",
  "signals": [...]
}
```

#### positions
```json
{
  "type": "positions",
  "positions": [...]
}
```
