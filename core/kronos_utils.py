#!/usr/bin/env python3
from __future__ import annotations

"""
kronos_utils.py - Kronos共享工具函数
====================================
OKX API封装、Treasury检查、OCO验证、集中度检查、日志幂等

版本: 1.0.0
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

_logger = logging.getLogger(__name__)

# OKX API 配置
BASE_URL = 'https://www.okx.com'
OKX_FLAG = os.environ.get('OKX_FLAG', '1')

# ===== 基础工具 =====

def _sign(ts: str, method: str, path: str, body: str = '') -> str:
    """OKX API 签名"""
    secret = os.environ.get('OKX_SECRET', '')
    msg = ts + method + path + body
    mac = hmac.new(secret.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def okx_req(method: str, path: str, body: str = '', api_key: str = None,
            secret: str = None, passphrase: str = None, retries: int = 3) -> dict:
    """
    通用的 OKX API 请求，带重试逻辑
    返回 parsed JSON 或 {'error': ...}
    """
    ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.') + '%03dZ' % (int(time.time() * 1000) % 1000)
    key = api_key or os.environ.get('OKX_API_KEY', '')
    secret or os.environ.get('OKX_SECRET', '')
    phrase = passphrase or os.environ.get('OKX_PASSPHRASE', '')
    flag = os.environ.get('OKX_FLAG', '1')

    headers = {
        'OK-ACCESS-KEY': key,
        'OK-ACCESS-SIGN': _sign(ts, method, path, body),
        'OK-ACCESS-TIMESTAMP': ts,
        'OK-ACCESS-PASSPHRASE': phrase,
        'Content-Type': 'application/json',
        'x-simulated-trading': '1' if flag == '1' else '0',
    }
    last_result = None
    for attempt in range(retries):
        try:
            r = requests.request(method, BASE_URL + path, headers=headers, data=body, timeout=10)
            last_result = r
            if r.status_code == 429 or r.status_code >= 500:
                # Rate limited or server error - retry with backoff
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return r.json()
        except requests.exceptions.Timeout:
            last_result = None
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        except Exception as e:
            return {'error': str(e)}
    # Max retries exceeded
    if last_result is not None:
        return last_result.json()
    return {'code': '99999', 'msg': 'Max retries exceeded'}


# ═══════════════════════════════════════════════════════════
#  原子写入工具（防断电损坏）
# ═══════════════════════════════════════════════════════════
import os as _os
import tempfile


def atomic_write_json(path: Path, data, indent: int = 2) -> None:
    """
    原子级 JSON 文件写入（多进程安全版）。
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(suffix='.json.tmp', prefix='atomic_', dir=str(path.parent))
    try:
        with _os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp_path, str(path))
    except Exception:
        if _os.path.exists(tmp_path):
            _os.unlink(tmp_path)
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """原子级纯文本文件写入"""
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(content)
        f.flush()
        _os.fsync(f.fileno())
    _os.replace(str(tmp), str(path))


# ═══════════════════════════════════════════════════════════
#  Treasury 预检查系统
# ═══════════════════════════════════════════════════════════
TREASURY_LIMITS = {
    'hourly_loss_pct': 0.05,    # 5% per hour
    'daily_loss_pct': 0.10,     # 10% per day
    'per_trade_pct': 0.02,      # 2% per trade
    'reserve_pct': 0.20,        # Keep 20% reserve
    'max_single_trade_pct': 0.05,  # Max 5% per single trade
    'min_equity': 1000,         # Min equity to trade
}

def check_treasury_trade_allowed(equity: float, treasury_state: dict) -> Tuple[bool, str, dict]:
    """
    Treasury预检查：决定是否允许开仓
    
    Returns: (allowed, reason, details)
    """
    limits = TREASURY_LIMITS
    
    # 1. 最低权益检查
    if equity < limits['min_equity']:
        return False, f'权益${equity:.2f}低于最低门槛${limits["min_equity"]}', {}
    
    # 2. 获取快照
    hourly_snap = treasury_state.get('hourly_snapshot', equity)
    daily_snap = treasury_state.get('daily_snapshot', equity)
    session_snap = treasury_state.get('session_start', equity)
    
    # 3. 计算损失
    hourly_loss_pct = (hourly_snap - equity) / hourly_snap if hourly_snap > 0 else 0
    daily_loss_pct = (daily_snap - equity) / daily_snap if daily_snap > 0 else 0
    session_dd_pct = (session_snap - equity) / session_snap if session_snap > 0 else 0
    
    # 4. 熔断层检查
    if session_dd_pct >= 0.20:
        return False, f'回撤{session_dd_pct:.1%}≥20%触发熔断suspended', {
            'tier': 'suspended', 'session_dd_pct': session_dd_pct
        }
    if daily_loss_pct >= 0.10:
        return False, f'日亏{daily_loss_pct:.1%}≥10%触发熔断critical', {
            'tier': 'critical', 'daily_loss_pct': daily_loss_pct
        }
    if hourly_loss_pct >= 0.05:
        return False, f'小时亏{hourly_loss_pct:.1%}≥5%触发熔断caution', {
            'tier': 'caution', 'hourly_loss_pct': hourly_loss_pct
        }
    
    # 5. 储备金检查
    reserve_amount = equity * limits['reserve_pct']
    available = equity - reserve_amount
    if available < equity * limits['per_trade_pct']:
        return False, f'可用${available:.2f}低于最低交易额', {
            'available': available, 'reserve': reserve_amount
        }
    
    return True, 'OK', {
        'hourly_loss_pct': hourly_loss_pct,
        'daily_loss_pct': daily_loss_pct,
        'session_dd_pct': session_dd_pct,
        'available': available,
        'tier': 'normal'
    }


def check_treasury_tier(equity: float, treasury_state: dict) -> Tuple[str, bool, str, dict]:
    """
    5层熔断判定
    
    Returns: (tier, can_trade, reason, details)
    """
    hourly_snap = treasury_state.get('hourly_snapshot', equity)
    daily_snap = treasury_state.get('daily_snapshot', equity)
    session_snap = treasury_state.get('session_start', equity)
    
    hourly_loss_pct = (hourly_snap - equity) / hourly_snap if hourly_snap > 0 else 0
    daily_loss_pct = (daily_snap - equity) / daily_snap if daily_snap > 0 else 0
    session_dd_pct = (session_snap - equity) / session_snap if session_snap > 0 else 0
    
    if session_dd_pct >= 0.20:
        return 'suspended', False, f'回撤{session_dd_pct:.1%}触发熔断', {
            'hourly_loss_pct': hourly_loss_pct,
            'daily_loss_pct': daily_loss_pct,
            'session_dd_pct': session_dd_pct
        }
    elif daily_loss_pct >= 0.10:
        return 'critical', False, f'日亏{daily_loss_pct:.1%}触发熔断', {
            'hourly_loss_pct': hourly_loss_pct,
            'daily_loss_pct': daily_loss_pct,
            'session_dd_pct': session_dd_pct
        }
    elif hourly_loss_pct >= 0.05:
        return 'caution', True, f'小时亏{hourly_loss_pct:.1%}', {
            'hourly_loss_pct': hourly_loss_pct,
            'daily_loss_pct': daily_loss_pct,
            'session_dd_pct': session_dd_pct
        }
    else:
        return 'normal', True, '正常', {
            'hourly_loss_pct': hourly_loss_pct,
            'daily_loss_pct': daily_loss_pct,
            'session_dd_pct': session_dd_pct
        }


# ═══════════════════════════════════════════════════════════
#  集中度检查系统
# ═══════════════════════════════════════════════════════════
CONCENTRATION_LIMITS = {
    'max_per_coin_pct': 0.15,      # 单币最多15%
    'max_total_exposure_pct': 0.50, # 总暴露最多50%
    'max_correlated_group_pct': 0.30, # 相关币组最多30%
}

# 相关币组定义 (同一生态/概念)
CORRELATED_GROUPS = {
    'L1': ['ETH', 'SOL', 'AVAX', 'ADA', 'DOT', 'LINK'],  # L1公链
    'MEME': ['DOGE', 'SHIB'],  # Meme币
    'BTC_ECOSYSTEM': ['BTC', 'BNB'],  # BTC生态
}

def check_concentration(
    symbol: str, 
    new_trade_pct: float, 
    current_positions: List[dict],
    equity: float
) -> Tuple[bool, str, dict]:
    """
    集中度检查：检查开仓是否超过集中度限制
    
    Args:
        symbol: 交易币种
        new_trade_pct: 新交易占权益百分比 (0.0-1.0)
        current_positions: 当前持仓列表 [{instId, notional, ...}, ...]
        equity: 当前权益
    
    Returns: (allowed, reason, details)
    """
    limits = CONCENTRATION_LIMITS
    
    # 1. 计算当前总暴露
    total_exposure = sum(abs(pos.get('notional', 0)) for pos in current_positions)
    total_exposure_pct = total_exposure / equity if equity > 0 else 0
    
    # 2. 计算该币当前暴露
    symbol_key = symbol.replace('-USDT-SWAP', '')
    symbol_exposure = sum(
        abs(pos.get('notional', 0)) 
        for pos in current_positions 
        if symbol_key in pos.get('instId', '')
    )
    symbol_exposure_pct = symbol_exposure / equity if equity > 0 else 0
    
    # 3. 检查单币限制
    if symbol_exposure_pct + new_trade_pct > limits['max_per_coin_pct']:
        limit_pct = limits['max_per_coin_pct'] * 100
        return False, f'{symbol}总暴露{(symbol_exposure_pct+new_trade_pct)*100:.1f}%>{limit_pct:.1f}%单币限制', {
            'symbol_exposure_pct': symbol_exposure_pct,
            'new_trade_pct': new_trade_pct,
            'limit': limits['max_per_coin_pct']
        }
    
    # 4. 检查总暴露限制
    if total_exposure_pct + new_trade_pct > limits['max_total_exposure_pct']:
        limit_pct = limits['max_total_exposure_pct'] * 100
        return False, f'总暴露{(total_exposure_pct+new_trade_pct)*100:.1f}%>{limit_pct:.1f}%限制', {
            'total_exposure_pct': total_exposure_pct,
            'new_trade_pct': new_trade_pct,
            'limit': limits['max_total_exposure_pct']
        }
    
    # 5. 检查相关币组限制
    for group_name, coins in CORRELATED_GROUPS.items():
        if symbol_key in coins:
            group_exposure = sum(
                abs(pos.get('notional', 0))
                for pos in current_positions
                for coin in coins
                if coin in pos.get('instId', '')
            )
            group_exposure_pct = group_exposure / equity if equity > 0 else 0
            if group_exposure_pct + new_trade_pct > limits['max_correlated_group_pct']:
                limit_pct = limits['max_correlated_group_pct'] * 100
                return False, f'{group_name}组暴露{(group_exposure_pct+new_trade_pct)*100:.1f}%>{limit_pct:.1f}%限制', {
                    'group': group_name,
                    'group_exposure_pct': group_exposure_pct,
                    'new_trade_pct': new_trade_pct,
                    'limit': limits['max_correlated_group_pct']
                }
    
    return True, 'OK', {
        'total_exposure_pct': total_exposure_pct,
        'symbol_exposure_pct': symbol_exposure_pct,
        'new_trade_pct': new_trade_pct
    }


# ═══════════════════════════════════════════════════════════
#  OCO 订单验证
# ═══════════════════════════════════════════════════════════
def validate_oco_order(
    instId: str,
    side: str,
    sz: float,
    entry_price: float,
    sl_pct: float,
    tp_pct: float,
    equity: float,
    min_size: float = 10.0
) -> Tuple[bool, str, dict]:
    """
    OCO订单参数验证
    
    Returns: (valid, reason, details)
    """
    # 1. 基本参数检查
    if not instId or not side:
        return False, '缺少instId或side', {}
    
    if sz <= 0:
        return False, f'合约数量sz={sz}必须>0', {'sz': sz}
    
    if entry_price <= 0:
        return False, f'入场价{entry_price}必须>0', {'entry_price': entry_price}
    
    # 2. SL/TP百分比检查
    if sl_pct <= 0 or sl_pct > 0.50:
        return False, f'SL={sl_pct:.2%}必须在0-50%之间', {'sl_pct': sl_pct}
    
    if tp_pct <= 0 or tp_pct > 1.00:
        return False, f'TP={tp_pct:.2%}必须在0-100%之间', {'tp_pct': tp_pct}
    
    # SL距离检查 (防止SL太紧被扫)
    sl_distance_pct = sl_pct
    if sl_distance_pct < 0.01:  # 小于1%
        return False, f'SL={sl_pct:.2%}太小,容易被震荡扫出', {'sl_pct': sl_pct}
    
    # 3. 仓位价值检查
    position_value = sz * entry_price
    if position_value < min_size:
        return False, f'仓位价值${position_value:.2f}<最低${min_size}', {
            'position_value': position_value, 'min_size': min_size
        }
    
    # 4. 账户余额检查
    min_required = position_value * 1.1  # 10% buffer for margin
    if equity < min_required and OKX_FLAG != '1':  # 实盘检查
        return False, f'余额${equity:.2f}<所需${min_required:.2f}', {
            'equity': equity, 'min_required': min_required
        }
    
    # 5. SL/TP价格合理性检查
    if side == 'long':
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
        if sl_price >= entry_price:
            return False, f'多头SL价格{sl_price}>=入场价{entry_price}', {}
        if tp_price <= entry_price:
            return False, f'多头TP价格{tp_price}<=入场价{entry_price}', {}
    else:  # short
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)
        if sl_price <= entry_price:
            return False, f'空头SL价格{sl_price}<=入场价{entry_price}', {}
        if tp_price >= entry_price:
            return False, f'空头TP价格{tp_price}>=入场价{entry_price}', {}
    
    return True, 'OK', {
        'position_value': position_value,
        'sl_price': sl_price,
        'tp_price': tp_price,
        'risk_amount': position_value * sl_pct
    }


def check_existing_oco_orders(instId: str) -> Tuple[bool, str]:
    """
    检查是否已有该币的活跃OCO订单
    
    Returns: (has_active, order_info)
    """
    data = okx_req('GET', '/api/v5/trade/orders-algo-pending')
    if data.get('code') != '0':
        return False, ''
    
    orders = data.get('data', [])
    for order in orders:
        if order.get('instId') == instId and order.get('ordType') == 'oco':
            return True, f"已有活跃OCO: {order.get('ordId', '')}"
    
    return False, ''


# ═══════════════════════════════════════════════════════════
#  异步并发扫描
# ═══════════════════════════════════════════════════════════
def parallel_scan_coins(
    scan_func,
    coins: List[Tuple[str, str]],
    max_workers: int = 5,
    timeout: float = 30.0
) -> List[Any]:
    """
    并发扫描多个币种
    
    Args:
        scan_func: 单币扫描函数，签名为 (instId, symbol) -> result
        coins: [(instId, symbol), ...]
        max_workers: 最大并发数
        timeout: 单币超时时间
    
    Returns:
        有效结果列表
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_coin = {
            executor.submit(scan_func, instId, symbol): (instId, symbol)
            for instId, symbol in coins
        }
        
        for future in as_completed(future_to_coin, timeout=timeout):
            future_to_coin[future]
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    results.append(result)
            except Exception:
                # 单币失败不影响整体
                pass
    
    return results


# ═══════════════════════════════════════════════════════════
#  日志幂等系统
# ═══════════════════════════════════════════════════════════
import hashlib as hashlib_


def generate_trade_idempotency_key(
    symbol: str, 
    direction: str, 
    entry_price: float,
    size: float,
    timestamp: str = None
) -> str:
    """
    生成幂等键：相同参数组合产生相同key
    
    格式: SHA256(symbol_direction_entry_size_time)
    """
    ts = timestamp or datetime.now().isoformat()
    raw = f"{symbol}_{direction}_{entry_price:.6f}_{size:.6f}_{ts}"
    return hashlib_.sha256(raw.encode()).hexdigest()[:32]


def load_idempotent_log(log_file: Path) -> dict:
    """加载幂等日志记录"""
    if log_file.exists():
        try:
            with open(log_file) as f:
                return json.load(f)
        except Exception as e:
            _logger.debug(f"load_idempotent_log: 读取日志文件失败 {log_file}: {e}")
    return {'processed_keys': [], 'trades': []}


def check_and_record_idempotent(
    log_file: Path,
    key: str,
    trade_data: dict
) -> Tuple[bool, str]:
    """
    检查并记录幂等键
    
    Returns: (is_duplicate, message)
    """
    log = load_idempotent_log(log_file)
    
    if key in log.get('processed_keys', []):
        return True, f'重复交易key={key[:8]}...'
    
    # 记录新key
    log['processed_keys'].append(key)
    log['trades'].append({
        'key': key,
        'trade': trade_data,
        'recorded_at': datetime.now().isoformat()
    })
    
    # 保持最近1000条记录
    if len(log['processed_keys']) > 1000:
        log['processed_keys'] = log['processed_keys'][-500:]
        log['trades'] = log['trades'][-500:]
    
    atomic_write_json(log_file, log)
    
    return False, 'OK'


# ═══════════════════════════════════════════════════════════
#  PnL 计算工具
# ═══════════════════════════════════════════════════════════

def calculate_trade_pnl(trade: dict, exit_price: float) -> Tuple[float, float]:
    """
    根据入场价、出场价、方向计算交易盈亏
    trade: dict，含 direction, entry_price, contracts, leverage
    返回: (result_pct, pnl)
    """
    entry = trade.get('entry_price', 0)
    direction = trade.get('direction', 'LONG')
    contracts = trade.get('contracts', 0)
    lev = trade.get('leverage', 1)

    if direction == 'LONG':
        ret = (exit_price - entry) / entry
    else:
        ret = (entry - exit_price) / entry

    ret_with_lev = ret * lev
    result_pct = round(ret_with_lev * 100, 2)
    pnl = round(ret_with_lev * contracts, 4)
    return result_pct, pnl


def get_account_balance() -> dict:
    """获取OKX账户权益"""
    data = okx_req('GET', '/api/v5/account/balance')
    try:
        if data.get('code') == '0' and data.get('data'):
            return {'totalEq': float(data['data'][0].get('totalEq', 0))}
    except Exception as e:
        _logger.debug(f"get_account_balance: 解析余额失败: {e}")
    return {'totalEq': 0}
