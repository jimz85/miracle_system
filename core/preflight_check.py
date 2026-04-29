from __future__ import annotations

"""
Preflight Check - 下单前环境校验
================================

在每次下单前执行以下校验:
1. 交易所状态 (连接+余额)
2. 最小下单量 (minSz)
3. 价格精度 (tickSz)
4. 持仓上限

用法:
    from core.preflight_check import preflight_check, PreflightResult

    result = preflight_check(
        client=exchange_client,
        symbol="BTC-USDT",
        side="long",
        size=0.01,
        price=72000.0,
        leverage=2
    )
    if not result.ok:
        raise PreflightError(result)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """单个检查项的结果"""
    name: str  # e.g. "exchange_status", "min_size"
    ok: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "✓" if self.ok else "✗"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class PreflightResult:
    """全部预检查结果"""
    ok: bool = True  # 全部通过才为True
    checks: List[CheckResult] = field(default_factory=list)
    symbol: str = ""
    exchange: str = ""
    blocked_reason: str = ""  # 如果 ok=False，说明原因
    
    def add(self, check: CheckResult):
        self.checks.append(check)
        if not check.ok:
            self.ok = False
            self.blocked_reason = f"{check.name}: {check.message}"
    
    def summary(self) -> str:
        lines = [f"PreflightResult [{self.exchange}] {self.symbol}:"]
        for c in self.checks:
            lines.append(f"  {c}")
        if not self.ok:
            lines.append(f"  → BLOCKED: {self.blocked_reason}")
        else:
            lines.append(f"  → ALL CHECKS PASSED")
        return "\n".join(lines)


class PreflightError(Exception):
    """预检查失败异常"""
    def __init__(self, result: PreflightResult):
        self.result = result
        super().__init__(result.blocked_reason)


# ============================================================
# Instrument Info Cache
# ============================================================
_INSTRUMENT_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_instrument_info(client, symbol: str) -> Optional[Dict[str, Any]]:
    """
    获取合约的 instrument 信息 (minSz, tickSz, maxSz, etc.)
    先从缓存读取，缓存失效则重新请求
    """
    import time
    cache_key = f"{client.exchange}:{symbol}"
    
    # Check cache
    if cache_key in _INSTRUMENT_CACHE:
        cached = _INSTRUMENT_CACHE[cache_key]
        if time.time() - cached["_cached_at"] < _CACHE_TTL_SECONDS:
            return cached
    
    # Fetch from exchange
    try:
        if client.exchange == "okx":
            inst_id = symbol.replace("-USDT", "-USDT-SWAP") if "-SWAP" not in symbol else symbol
            result = client._make_request(
                "GET",
                "/api/v5/public/instruments",
                params={"instId": inst_id, "instType": "SWAP"},
                signed=False
            )
            data = result.get("data", [{}])[0]
            info = {
                "inst_id": data.get("instId"),
                "min_sz": float(data.get("minSz", 0)),
                "tick_sz": float(data.get("tickSz", 0.1)),
                "max_sz": float(data.get("maxSz", 999999)),
                "lot_sz": float(data.get("lotSz", 1)),  # 数量精度
                "_cached_at": time.time()
            }
        elif client.exchange == "binance":
            binance_symbol = symbol.replace("-USDT", "USDT").replace("-SWAP", "")
            result = client._make_request(
                "GET",
                "/api/v3/exchangeInfo",
                params={},
                signed=False
            )
            for s in result.get("symbols", []):
                if s["symbol"] == binance_symbol:
                    for f in s.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            info = {
                                "min_sz": float(f["minQty"]),
                                "tick_sz": float(f["tickSize"]),
                                "max_sz": float(f["maxQty"]),
                                "lot_sz": float(f["stepSize"]),
                                "_cached_at": time.time()
                            }
                            break
                    else:
                        info = {"min_sz": 0.001, "tick_sz": 0.01, "max_sz": 999999, "lot_sz": 0.001, "_cached_at": time.time()}
                    break
            else:
                info = None
        else:
            info = None
        
        if info:
            _INSTRUMENT_CACHE[cache_key] = info
            return info
    except Exception as e:
        logger.warning(f"获取instrument信息失败 [{symbol}]: {e}")
    
    return None


def _check_exchange_status(client) -> CheckResult:
    """检查1: 交易所连通性和账户状态"""
    try:
        balance = client.get_balance()
        if balance.get("available", 0) <= 0:
            return CheckResult(
                name="exchange_status",
                ok=False,
                message="余额为0或无法获取有效余额",
                details=balance
            )
        return CheckResult(
            name="exchange_status",
            ok=True,
            message=f"连接正常, 可用余额={balance.get('available', 0):.2f} USDT",
            details=balance
        )
    except ConnectionError as e:
        return CheckResult(
            name="exchange_status",
            ok=False,
            message=f"交易所连接失败: {e}",
            details={}
        )
    except Exception as e:
        return CheckResult(
            name="exchange_status",
            ok=False,
            message=f"交易所状态异常: {e}",
            details={}
        )


def _check_min_size(client, symbol: str, side: str, size: float) -> CheckResult:
    """检查2: 最小下单量"""
    info = _get_instrument_info(client, symbol)
    if info is None:
        # 无法获取instrument信息，放行但警告
        return CheckResult(
            name="min_size",
            ok=True,
            message=f"无法获取instrument信息，跳过最小量检查",
            details={}
        )
    
    min_sz = info.get("min_sz", 0)
    if size < min_sz:
        return CheckResult(
            name="min_size",
            ok=False,
            message=f"下单量 {size} 小于最小要求 {min_sz}",
            details={"size": size, "min_sz": min_sz}
        )
    
    return CheckResult(
        name="min_size",
        ok=True,
        message=f"下单量 {size} >= 最小 {min_sz}",
        details={"size": size, "min_sz": min_sz}
    )


def _check_price_precision(client, symbol: str, price: float) -> CheckResult:
    """检查3: 价格精度 (tickSz)"""
    if price is None or price <= 0:
        return CheckResult(
            name="price_precision",
            ok=True,
            message="市价单，不检查价格精度",
            details={}
        )
    
    info = _get_instrument_info(client, symbol)
    if info is None:
        return CheckResult(
            name="price_precision",
            ok=True,
            message="无法获取instrument信息，跳过精度检查",
            details={}
        )
    
    tick_sz = info.get("tick_sz", 0.1)
    # 检查价格是否是对tickSz的整数倍（使用整数运算避免浮点精度问题）
    price_ticks = round(price / tick_sz)
    adjusted_price = price_ticks * tick_sz
    precision_ok = abs(price - adjusted_price) < 1e-9
    
    # 调整到合法精度
    if not precision_ok:
        adjusted_price = round(price / tick_sz) * tick_sz
        return CheckResult(
            name="price_precision",
            ok=False,
            message=f"价格 {price} 不符合精度 tickSz={tick_sz}，调整后为 {adjusted_price}",
            details={"price": price, "tick_sz": tick_sz, "adjusted": adjusted_price}
        )
    
    return CheckResult(
        name="price_precision",
        ok=True,
        message=f"价格精度正确 tickSz={tick_sz}",
        details={"price": price, "tick_sz": tick_sz}
    )


def _check_position_limit(client, symbol: str, side: str, size: float, leverage: int) -> CheckResult:
    """检查4: 持仓上限 (最大可开数量)"""
    try:
        positions = client.get_open_positions()
        current_pos = sum(
            float(p.get("size", 0)) for p in positions
            if p.get("symbol") == symbol and p.get("side", "").lower() in ("long", "short")
        )
    except Exception as e:
        logger.warning(f"获取持仓失败: {e}")
        current_pos = 0.0
    
    info = _get_instrument_info(client, symbol)
    max_sz = info.get("max_sz", 999999) if info else 999999
    
    # 估算保证金能开的最大量 (粗略估算，忽略保证金率差异)
    try:
        balance = client.get_balance()
        available = balance.get("available", 0)
        # 粗略估算：可用余额 / (价格 * 保证金率(约1/杠杆)) * 杠杆
        price = client.get_ticker(symbol) or 1000.0
        margin_ratio = 1.0 / leverage
        max_by_margin = (available / price) / margin_ratio
        max_allowed = min(max_sz, max_by_margin)
    except Exception:
        max_allowed = max_sz
    
    new_total = current_pos + size
    if new_total > max_allowed:
        return CheckResult(
            name="position_limit",
            ok=False,
            message=f"总持仓 {new_total} 超过上限 {max_allowed:.4f} (当前={current_pos}, 本次={size})",
            details={
                "current_pos": current_pos,
                "new_size": size,
                "new_total": new_total,
                "max_allowed": max_allowed
            }
        )
    
    return CheckResult(
        name="position_limit",
        ok=True,
        message=f"持仓量 {new_total} 在上限内 (max={max_allowed:.4f})",
        details={
            "current_pos": current_pos,
            "new_size": size,
            "new_total": new_total,
            "max_allowed": max_allowed
        }
    )


def preflight_check(
    client,
    symbol: str,
    side: str,
    size: float,
    price: Optional[float] = None,
    leverage: int = 1,
    skip_checks: Optional[List[str]] = None
) -> PreflightResult:
    """
    执行下单前的全部预检查
    
    Args:
        client: ExchangeClient 实例
        symbol: 交易对，如 "BTC-USDT"
        side: "long" / "short"
        size: 下单数量
        price: 下单价格（None表示市价单）
        leverage: 杠杆倍数
        skip_checks: 要跳过的检查项列表，如 ["position_limit"]
    
    Returns:
        PreflightResult: 包含全部检查结果
        如果 ok=False，抛出 PreflightError
    
    Raises:
        PreflightError: 当任何检查失败时
    """
    skip_checks = skip_checks or []
    result = PreflightResult(symbol=symbol, exchange=client.exchange)
    
    checks = [
        ("exchange_status", lambda: _check_exchange_status(client)),
        ("min_size", lambda: _check_min_size(client, symbol, side, size)),
        ("price_precision", lambda: _check_price_precision(client, symbol, price)),
        ("position_limit", lambda: _check_position_limit(client, symbol, side, size, leverage)),
    ]
    
    for name, check_fn in checks:
        if name in skip_checks:
            logger.debug(f"Preflight: skipping {name}")
            continue
        try:
            check_result = check_fn()
        except Exception as e:
            check_result = CheckResult(
                name=name,
                ok=False,
                message=f"检查执行异常: {e}",
                details={}
            )
        result.add(check_result)
        logger.debug(f"Preflight {name}: {check_result}")
    
    return result


def preflight_check_or_raise(client, symbol: str, side: str, size: float,
                             price: Optional[float] = None, leverage: int = 1) -> PreflightResult:
    """
    执行预检查，失败则抛出 PreflightError
    """
    result = preflight_check(client, symbol, side, size, price, leverage)
    if not result.ok:
        logger.warning(f"Preflight check failed: {result.summary()}")
        raise PreflightError(result)
    return result


if __name__ == "__main__":
    # 简单测试
    print("=== Preflight Check Test ===")
    from core.exchange_client import ExchangeClient
    from core.executor_config import ExecutorConfig
    
    config = ExecutorConfig()
    client = ExchangeClient("okx", config)
    
    try:
        result = preflight_check(
            client=client,
            symbol="BTC-USDT",
            side="long",
            size=0.001,
            price=72000.0,
            leverage=2
        )
        print(result.summary())
    except PreflightError as e:
        print(f"PreflightError: {e}")
    except Exception as e:
        print(f"其他错误: {e}")
