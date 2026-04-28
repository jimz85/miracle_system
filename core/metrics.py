from __future__ import annotations

"""
Prometheus Metrics - 监控指标采集
暴露交易系统关键指标供Prometheus抓取
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 尝试导入prometheus_client
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning("prometheus_client not installed, metrics disabled")


# ========================
# 指标定义
# ========================

@dataclass
class MetricsConfig:
    """指标配置"""
    namespace: str = "miracle"
    subsystem: str = "trading"
    subsystem_llm: str = "llm"
    subsystem_system: str = "system"


class TradingMetrics:
    """
    交易指标收集器
    
    暴露的指标:
    - equity: 账户权益
    - positions: 当前持仓数
    - position_pnl: 各持仓盈亏
    - trades_total: 交易总数
    - trade_pnl: 单笔交易盈亏
    - circuit_breaker: 熔断器状态
    """
    
    def __init__(self, config: MetricsConfig | None = None):
        self.config = config or MetricsConfig()
        self._enabled = HAS_PROMETHEUS
        
        if not self._enabled:
            logger.warning("Prometheus not available, metrics disabled")
            return
        
        ns = self.config.namespace
        
        # 账户指标
        self.equity = Gauge(
            f"{ns}_equity_current",
            "Current account equity in USDT"
        )
        self.equity_start = Gauge(
            f"{ns}_equity_start",
            "Starting account equity"
        )
        self.equity_pct_change = Gauge(
            f"{ns}_equity_pct_change",
            "Equity percent change from start"
        )
        
        # 持仓指标
        self.positions_open = Gauge(
            f"{ns}_positions_open",
            "Number of open positions"
        )
        self.position_pnl = Gauge(
            f"{ns}_position_pnl",
            "Individual position PnL",
            ["symbol", "side"]
        )
        self.position_size = Gauge(
            f"{ns}_position_size",
            "Individual position size",
            ["symbol", "side"]
        )
        
        # 交易指标
        self.trades_total = Counter(
            f"{ns}_trades_total",
            "Total number of trades",
            ["symbol", "side", "result"]  # result: profit/loss
        )
        self.trade_pnl = Histogram(
            f"{ns}_trade_pnl",
            "Individual trade PnL distribution",
            ["symbol"],
            buckets=[-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000]
        )
        self.trade_duration = Histogram(
            f"{ns}_trade_duration_seconds",
            "Trade holding duration",
            ["symbol"],
            buckets=[60, 300, 900, 3600, 14400, 43200, 86400, 259200]  # 1m-3d
        )
        
        # 熔断器指标
        self.circuit_breaker_tripped = Gauge(
            f"{ns}_circuit_breaker_tripped",
            "Circuit breaker tripped (1=tripped, 0=ok)"
        )
        self.consecutive_losses = Gauge(
            f"{ns}_consecutive_losses",
            "Number of consecutive losing trades"
        )
        self.daily_loss = Gauge(
            f"{ns}_daily_loss",
            "Daily loss amount"
        )
        
        # 信号指标
        self.signals_generated = Counter(
            f"{ns}_signals_generated",
            "Number of signals generated",
            ["symbol", "direction", "confidence"]
        )
        
        self._last_update = time.time()
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def record_equity(self, current: float, start: float):
        """记录权益"""
        if not self._enabled:
            return
        self.equity.set(current)
        self.equity_start.set(start)
        if start > 0:
            pct = ((current - start) / start) * 100
            self.equity_pct_change.set(pct)
        self._last_update = time.time()
    
    def record_position_open(self, symbol: str, side: str, size: float, entry_price: float):
        """记录开仓"""
        if not self._enabled:
            return
        self.position_size.labels(symbol=symbol, side=side).set(size)
        self.positions_open.inc()
    
    def record_position_update(self, symbol: str, side: str, pnl: float, current_price: float):
        """更新持仓"""
        if not self._enabled:
            return
        self.position_pnl.labels(symbol=symbol, side=side).set(pnl)
    
    def record_position_close(self, symbol: str, side: str, pnl: float, duration: float):
        """记录平仓"""
        if not self._enabled:
            return
        result = "profit" if pnl > 0 else "loss"
        self.trades_total.labels(symbol=symbol, side=side, result=result).inc()
        self.trade_pnl.labels(symbol=symbol).observe(pnl)
        self.trade_duration.labels(symbol=symbol).observe(duration)
        self.positions_open.dec()
    
    def record_circuit_breaker(self, tripped: bool, consecutive_losses: int):
        """记录熔断器状态"""
        if not self._enabled:
            return
        self.circuit_breaker_tripped.set(1 if tripped else 0)
        self.consecutive_losses.set(consecutive_losses)
    
    def record_daily_loss(self, loss: float):
        """记录日亏损"""
        if not self._enabled:
            return
        self.daily_loss.set(loss)
    
    def record_signal(self, symbol: str, direction: str, confidence: float):
        """记录信号"""
        if not self._enabled:
            return
        conf_bucket = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        self.signals_generated.labels(symbol=symbol, direction=direction, confidence=conf_bucket).inc()
    
    def get_metrics(self) -> bytes | None:
        """获取所有指标"""
        if not self._enabled:
            return None
        return generate_latest()
    
    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST


class LLMMetrics:
    """
    LLM调用指标
    
    指标:
    - llm_requests_total: 请求总数
    - llm_request_duration: 请求延迟
    - llm_errors: 错误数
    - llm_fallbacks: 降级次数
    """
    
    def __init__(self, config: MetricsConfig | None = None):
        self.config = config or MetricsConfig()
        self._enabled = HAS_PROMETHEUS
        
        if not self._enabled:
            return
        
        ns = self.config.namespace
        llm_ns = f"{ns}_{self.config.subsystem_llm}"
        
        self.requests_total = Counter(
            f"{llm_ns}_requests_total",
            "Total LLM requests",
            ["provider", "model", "status"]
        )
        self.request_duration = Histogram(
            f"{llm_ns}_request_duration_seconds",
            "LLM request duration",
            ["provider", "model"],
            buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30]
        )
        self.errors = Counter(
            f"{llm_ns}_errors_total",
            "LLM errors",
            ["provider", "error_type"]
        )
        self.fallbacks = Counter(
            f"{llm_ns}_fallbacks_total",
            "LLM fallbacks to backup",
            ["from_provider", "to_provider"]
        )
        self.cache_hits = Counter(
            f"{llm_ns}_cache_hits_total",
            "LLM cache hits",
            ["provider"]
        )
        self.cache_misses = Counter(
            f"{llm_ns}_cache_misses_total",
            "LLM cache misses",
            ["provider"]
        )
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def record_request(self, provider: str, model: str, duration: float, status: str = "success"):
        """记录请求"""
        if not self._enabled:
            return
        self.requests_total.labels(provider=provider, model=model, status=status).inc()
        self.request_duration.labels(provider=provider, model=model).observe(duration)
    
    def record_error(self, provider: str, error_type: str):
        """记录错误"""
        if not self._enabled:
            return
        self.errors.labels(provider=provider, error_type=error_type).inc()
    
    def record_fallback(self, from_provider: str, to_provider: str):
        """记录降级"""
        if not self._enabled:
            return
        self.fallbacks.labels(from_provider=from_provider, to_provider=to_provider).inc()
    
    def record_cache_hit(self, provider: str):
        """记录缓存命中"""
        if not self._enabled:
            return
        self.cache_hits.labels(provider=provider).inc()
    
    def record_cache_miss(self, provider: str):
        """记录缓存未命中"""
        if not self._enabled:
            return
        self.cache_misses.labels(provider=provider).inc()


class SystemMetrics:
    """
    系统指标
    
    指标:
    - system_uptime: 运行时间
    - system_cpu: CPU使用率
    - system_memory: 内存使用
    - system_errors: 系统错误
    """
    
    def __init__(self, config: MetricsConfig | None = None):
        self.config = config or MetricsConfig()
        self._enabled = HAS_PROMETHEUS
        self._start_time = time.time()
        
        if not self._enabled:
            return
        
        ns = self.config.namespace
        sys_ns = f"{ns}_{self.config.subsystem_system}"
        
        self.uptime = Gauge(
            f"{sys_ns}_uptime_seconds",
            "System uptime in seconds"
        )
        self.processed_signals = Counter(
            f"{sys_ns}_processed_signals_total",
            "Total signals processed"
        )
        self.api_calls = Counter(
            f"{sys_ns}_api_calls_total",
            "Total API calls",
            ["service", "endpoint", "status"]
        )
        self.api_latency = Histogram(
            f"{sys_ns}_api_latency_seconds",
            "API call latency",
            ["service", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
        )
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def record_uptime(self):
        """更新运行时间"""
        if not self._enabled:
            return
        self.uptime.set(time.time() - self._start_time)
    
    def record_signal_processed(self):
        """记录已处理信号"""
        if not self._enabled:
            return
        self.processed_signals.inc()
    
    def record_api_call(self, service: str, endpoint: str, status: str, duration: float):
        """记录API调用"""
        if not self._enabled:
            return
        self.api_calls.labels(service=service, endpoint=endpoint, status=status).inc()
        self.api_latency.labels(service=service, endpoint=endpoint).observe(duration)


# ========================
# 全局指标实例
# ========================

_trading_metrics: TradingMetrics | None = None
_llm_metrics: LLMMetrics | None = None
_system_metrics: SystemMetrics | None = None


def get_trading_metrics() -> TradingMetrics:
    """获取交易指标实例"""
    global _trading_metrics
    if _trading_metrics is None:
        _trading_metrics = TradingMetrics()
    return _trading_metrics


def get_llm_metrics() -> LLMMetrics:
    """获取LLM指标实例"""
    global _llm_metrics
    if _llm_metrics is None:
        _llm_metrics = LLMMetrics()
    return _llm_metrics


def get_system_metrics() -> SystemMetrics:
    """获取系统指标实例"""
    global _system_metrics
    if _system_metrics is None:
        _system_metrics = SystemMetrics()
    return _system_metrics


def get_all_metrics() -> bytes | None:
    """获取所有指标"""
    trading = get_trading_metrics()
    if trading.enabled:
        return trading.get_metrics()
    return None


# ========================
# Flask/HTTP指标服务器
# ========================

def create_metrics_app():
    """创建Flask指标服务器"""
    try:
        from flask import Flask, Response
        app = Flask(__name__)
        
        trading = get_trading_metrics()
        get_llm_metrics()
        get_system_metrics()
        
        @app.route("/metrics")
        def metrics():
            if not trading.enabled:
                return Response("Metrics disabled", status=503)
            return Response(
                trading.get_metrics(),
                mimetype=trading.content_type
            )
        
        @app.route("/health")
        def health():
            """
            Enhanced health check with component-level status.
            
            Returns:
                {
                    "status": "ok" | "degraded" | "error",
                    "timestamp": ISO datetime,
                    "components": [
                        {"name": "exchange", "status": "ok" | "error", "critical": True, "message": "..."},
                        {"name": "memory", "status": "ok" | "error", "critical": False, "message": "..."},
                        ...
                    ]
                }
            """
            from datetime import datetime as dt
            
            components = []
            overall_status = "ok"
            
            # Check 1: Exchange connectivity (critical)
            try:
                from core.exchange_client import ExchangeClient
                from core.executor_config import ExecutorConfig
                config = ExecutorConfig()
                client = ExchangeClient("okx", config)
                balance = client.get_balance()
                components.append({
                    "name": "exchange_okx",
                    "status": "ok",
                    "critical": True,
                    "message": f"OKX reachable, balance={balance.get('available', 0):.2f} USDT"
                })
            except Exception as e:
                components.append({
                    "name": "exchange_okx",
                    "status": "error",
                    "critical": True,
                    "message": f"OKX unreachable: {e}"
                })
                overall_status = "degraded"
            
            # Check 2: Memory/DB (non-critical)
            try:
                from core.memory.vector_memory import get_vector_memory
                vm = get_vector_memory()
                stats = vm.get_stats()
                components.append({
                    "name": "memory",
                    "status": "ok",
                    "critical": False,
                    "message": f"VectorMemory: {stats.get('total', 0)} memories"
                })
            except Exception as e:
                components.append({
                    "name": "memory",
                    "status": "error",
                    "critical": False,
                    "message": f"Memory error: {e}"
                })
            
            # Check 3: Prometheus metrics collector (non-critical)
            if trading.enabled:
                components.append({
                    "name": "prometheus",
                    "status": "ok",
                    "critical": False,
                    "message": "Metrics collector enabled"
                })
            else:
                components.append({
                    "name": "prometheus",
                    "status": "error",
                    "critical": False,
                    "message": "Metrics collector disabled"
                })
            
            # Check 4: LLM metrics
            try:
                llm = get_llm_metrics()
                if llm.enabled:
                    components.append({
                        "name": "llm_metrics",
                        "status": "ok",
                        "critical": False,
                        "message": "LLM metrics enabled"
                    })
                else:
                    components.append({
                        "name": "llm_metrics",
                        "status": "error",
                        "critical": False,
                        "message": "LLM metrics disabled"
                    })
            except Exception as e:
                components.append({
                    "name": "llm_metrics",
                    "status": "error",
                    "critical": False,
                    "message": f"LLM metrics error: {e}"
                })
            
            # Determine overall status
            for comp in components:
                if comp.get("critical") and comp.get("status") == "error":
                    overall_status = "error"
                    break
                elif comp.get("status") == "error":
                    overall_status = "degraded"
            
            return {
                "status": overall_status,
                "timestamp": dt.now().isoformat(),
                "components": components
            }
        
        return app
    except ImportError:
        logger.error("Flask not installed, cannot create metrics app")
        return None


if __name__ == "__main__":
    print("=== Prometheus Metrics Test ===\n")
    
    # 测试交易指标
    trading = get_trading_metrics()
    print(f"Trading metrics enabled: {trading.enabled}")
    
    if trading.enabled:
        trading.record_equity(current=11500, start=10000)
        trading.record_position_open("BTC", "long", 0.1, 50000)
        trading.record_position_update("BTC", "long", 500, 51000)
        trading.record_position_close("BTC", "long", 500, 3600)
        trading.record_circuit_breaker(False, 0)
        trading.record_signal("ETH", "buy", 0.85)
        
        print(f"Metrics output:\n{trading.get_metrics().decode()[:500]}")
    else:
        print("Prometheus client not installed")
    
    # 测试LLM指标
    llm = get_llm_metrics()
    print(f"\nLLM metrics enabled: {llm.enabled}")
    
    # 测试系统指标
    system = get_system_metrics()
    print(f"System metrics enabled: {system.enabled}")
    
    print("\n=== Test complete ===")
