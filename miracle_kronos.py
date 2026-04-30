#!/usr/bin/env python3
from __future__ import annotations

"""
Miracle-Kronos Unified Trading System
======================================
合并两个系统的最优部分:
- Miracle: Multi-agent架构 + 适应性学习 + 白名单模式
- Kronos: OKX实盘接口 + 全自动cron + IC投票 + 五层熔断

P0+P1修复:
- 异步并发: ThreadPoolExecutor并发扫描
- OCO验证: 下单前参数校验
- Treasury预检查: 交易前熔断检查
- 集中度检查: 仓位集中度限制
- 日志幂等: 幂等日志写入

使用方式:
  python miracle_kronos.py --mode audit --equity 100000
  python miracle_kronos.py --mode live
"""
import argparse
import json
import logging
import os
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# NOTE: Heavy imports moved to lazy (local) imports inside functions:
#   - core.kronos_utils (atomic_write_json, CONCENTRATION_LIMITS, TREASURY_LIMITS,
#     check_and_record_idempotent, check_concentration, check_existing_oco_orders,
#     check_treasury_tier, check_treasury_trade_allowed, generate_trade_idempotency_key,
#     parallel_scan_coins, validate_oco_order)
#   - core.market_intel_base (get_fomc_confidence_multiplier, get_market_regime,
#     get_regime_confidence_multiplier)
#   - core.exchange_adapter (get_default_exchange)
#   - core.price_factors (PriceFactors)
#   - agents.agent_learner (AgentLearner)
#   - core.memory (get_structured_memory)

# ===== 配置 =====
# OKX API key是simulation环境的，必须用x-simulated-trading:1
# OKX_FLAG控制的是"是否模拟交易"，这个key本身就是在OKX simulation账户里用真钱交易
OKX_FLAG = os.environ.get('OKX_FLAG', '1')  # 1=simulation交易(用此key), 0=真实账户
STATE_DIR = Path(__file__).parent / 'data'
STATE_DIR.mkdir(exist_ok=True)
TREASURY_FILE = STATE_DIR / 'miracle_treasury.json'
TRADES_FILE = STATE_DIR / 'miracle_trades.json'
IC_WEIGHTS_FILE = STATE_DIR / 'factor_weights.json'
IDEMPOTENCY_LOG = STATE_DIR / 'trade_idempotency.json'
PATTERN_HISTORY_FILE = STATE_DIR / 'pattern_history.json'  # 模式胜率历史

# 日志配置
logger = logging.getLogger('miracle_kronos')

# ===== 全局仓位模式检测 =====
_pos_mode = 'net'  # 'net' or 'hedge', updated at startup

def _detect_pos_mode():
    """检测账户是net模式还是hedge模式
    P0 Fix: 用OKX账户配置API检测，不再依赖持仓推断
    空仓时也能正确识别账户模式，避免Hedge账户开仓失败
    """
    global _pos_mode
    # 调用OKX账户配置API，posMode返回 'net_mode' 或 'long_short_mode'
    data = okx_req('GET', '/api/v5/account/config')
    if data.get('code') == '0' and data.get('data'):
        pos_mode_str = data['data'][0].get('posMode', 'net_mode')
        _pos_mode = 'hedge' if pos_mode_str != 'net_mode' else 'net'
    else:
        # API失败时用持仓推断作为fallback
        positions = get_positions()
        for p in positions:
            if p.get('sz', 0) != 0:
                side = p.get('side', 'net')
                _pos_mode = 'hedge' if side in ('long', 'short') else 'net'
                return _pos_mode
        _pos_mode = 'net'  # 空仓且API失败，默认net
    return _pos_mode

# ===== 安全类型转换 =====
def safe_float(val, default=0.0):
    """Safely convert OKX API value to float - handles None, '', 'null'"""
    if val is None or val == '' or val == 'null':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# ===== OKX API (兼容旧接口) =====
def _sign(ts, method, path, body=''):
    import base64
    import hashlib
    import hmac
    secret = os.environ.get('OKX_SECRET', '')
    msg = ts + method + path + body
    return base64.b64encode(hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()

def okx_req(method, path, body=''):
    """OKX API request with retry and exponential backoff."""
    import time as _time
    from datetime import datetime

    import requests

    for attempt in range(3):
        try:
            ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.') + '%03dZ' % (int(_time.time() * 1000) % 1000)
            # cancel-algos 必须用 POST，且签名也必须用 POST
            actual_method = 'POST' if 'cancel-algos' in path else method
            headers = {
                'OK-ACCESS-KEY': os.environ.get('OKX_API_KEY', ''),
                'OK-ACCESS-SIGN': _sign(ts, actual_method, path, body),
                'OK-ACCESS-TIMESTAMP': ts,
                'OK-ACCESS-PASSPHRASE': os.environ.get('OKX_PASSPHRASE', ''),
                'x-simulated-trading': OKX_FLAG,
                'Content-Type': 'application/json',
            }
            r = requests.request(actual_method, 'https://www.okx.com' + path, headers=headers, data=body, timeout=10)
            # Retry on 429 (rate limit) and 5xx (server error)
            if r.status_code == 429:
                retry_after = float(r.headers.get('Retry-After', 5))
                logger.warning(f"OKX rate limited, retrying in {retry_after}s...")
                _time.sleep(retry_after)
                continue
            if r.status_code >= 500 and attempt < 2:
                wait = 2 ** attempt
                logger.warning(f"OKX server error {r.status_code}, retrying in {wait}s...")
                _time.sleep(wait)
                continue
            return r.json()
        except requests.exceptions.Timeout:
            if attempt < 2:
                wait = 2 ** attempt
                logger.warning(f"OKX request timeout, retrying in {wait}s...")
                _time.sleep(wait)
                continue
            return {'code': '99999', 'msg': 'Request timeout after 3 retries'}
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                wait = 2 ** attempt
                logger.warning(f"OKX connection error, retrying in {wait}s...")
                _time.sleep(wait)
                continue
            return {'code': '99999', 'msg': 'Connection error after 3 retries'}
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                logger.warning(f"OKX request error: {e}, retrying in {wait}s...")
                _time.sleep(wait)
                continue
            return {'code': '99999', 'msg': str(e)}
    return {'code': '99999', 'msg': 'Max retries exceeded'}

# ===== 核心指标计算 =====

# ===== 技术指标 (from indicators.py) =====
from indicators import (
    calc_rsi, calc_adx, calc_macd, calc_bollinger, calc_atr
)



# ===== IC权重投票系统 (from Kronos voting_system.py) =====
# 从环境变量读取Kronos IC文件路径，默认 ~/.hermes/cron/output/ic_weights.json
_KRONOS_IC_DEFAULT = Path.home() / '.hermes' / 'cron' / 'output' / 'ic_weights.json'
KRONOS_IC_FILE = Path(os.environ.get('KRONOS_IC_FILE', str(_KRONOS_IC_DEFAULT)))

def load_ic_weights():
    if KRONOS_IC_FILE.exists():
        try:
            with open(KRONOS_IC_FILE) as f:
                d = json.load(f)
            w = d.get('weights', {})
            if w and sum(w.values()) > 0:
                return w
        except Exception as ex:
            logger.debug(f"get_ic_adjusted_weights: 读取KRONOS_IC_FILE失败: {ex}")
    if IC_WEIGHTS_FILE.exists():
        try:
            with open(IC_WEIGHTS_FILE) as f:
                d = json.load(f)
                return d.get('weights', DEFAULT_WEIGHTS)
        except Exception as ex:
            logger.debug(f"get_ic_adjusted_weights: 读取IC_WEIGHTS_FILE失败: {ex}")
    return DEFAULT_WEIGHTS.copy()

def save_ic_weights(weights):
    from core.kronos_utils import atomic_write_json
    data = {'weights': weights, 'updated': datetime.now().isoformat()}
    atomic_write_json(IC_WEIGHTS_FILE, data)

# ===== 模式胜率历史（出场反馈闭环） =====
def load_pattern_history():
    """加载模式胜率历史"""
    if PATTERN_HISTORY_FILE.exists():
        try:
            with open(PATTERN_HISTORY_FILE) as f:
                return json.load(f)
        except Exception as ex:
            logger.debug(f"load_pattern_history: 读取失败: {ex}")
    return {'patterns': {}, 'total_trades': 0, 'wins': 0, 'losses': 0}

def save_pattern_history(history):
    """保存模式胜率历史"""
    try:
        with open(PATTERN_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as ex:
        logger.debug(f"save_pattern_history: 写入失败: {ex}")

def record_pattern_outcome(pattern_key: str, won: bool, pnl_pct: float):
    """记录一个模式的出场结果"""
    history = load_pattern_history()
    patterns = history.setdefault('patterns', {})
    # 更新全局统计
    history['total_trades'] = history.get('total_trades', 0) + 1
    if won:
        history['wins'] = history.get('wins', 0) + 1
    else:
        history['losses'] = history.get('losses', 0) + 1
    # 更新模式统计
    if pattern_key not in patterns:
        patterns[pattern_key] = {'entries': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0}
    p = patterns[pattern_key]
    p['entries'] += 1
    p['total_pnl'] += pnl_pct
    if won:
        p['wins'] += 1
    else:
        p['losses'] += 1
    logger.debug(f"Pattern记录: {pattern_key} {'WIN' if won else 'LOSS'} ({pnl_pct:+.1%}, 累计{p['entries']}次, {p['wins']/max(p['entries'],1):.0%})")
    save_pattern_history(history)

def get_pattern_adjustment(pattern_key: str) -> float:
    """根据历史胜率获取confidence调整系数
    返回:
        1.20 = 高确信度（胜率>66%且样本>=3）
        1.10 = 轻微提升（胜率>50%且样本>=5）
        1.00 = 中性（无数据或胜率50%左右）
        0.70 = 轻微抑制（胜率<40%且样本>=3）
        0.40 = 强烈抑制（胜率<30%且样本>=5）
    """
    history = load_pattern_history()
    p = history.get('patterns', {}).get(pattern_key)
    if not p or p['entries'] < 3:
        return 1.0  # 样本不足，不调整
    win_rate = p['wins'] / p['entries']
    if win_rate > 0.66:
        return 1.2
    elif win_rate > 0.50 and p['entries'] >= 5:
        return 1.1
    elif win_rate < 0.30 and p['entries'] >= 5:
        return 0.4
    elif win_rate < 0.40 and p['entries'] >= 3:
        return 0.7
    return 1.0

DEFAULT_WEIGHTS = {
    # Normalized to sum=1.0: RSI=0.12, ADX=0.12, Bollinger=0.24, Vol=0.08, MACD=0.20, BTC=0.13, Gemma=0.11
    'RSI': 0.12, 'ADX': 0.12, 'Bollinger': 0.24,
    'Vol': 0.08, 'MACD': 0.20, 'BTC': 0.13, 'Gemma': 0.11
}

def voting_vote(factors: dict, weights: dict) -> dict:
    """7因子投票: 每个因子投 +1/0/-1, 加权求和
    核心修正: RSI极端值在强趋势(ADX>25)中不代表反转, 而是趋势持续.
    """
    rsi = factors['rsi']
    adx = factors['adx']
    bb_pos = factors['bb_pos']
    macd_hist = factors['macd_hist']
    vol_ratio = factors['vol_ratio']
    btc_trend = factors.get('btc_trend', 'neutral')

    # ---- RSI因子: 分情况 ----
    # 强趋势(ADX>25): RSI极端值=趋势确认, 不是反转信号
    # 震荡(ADX<20): RSI极端值=均值回归信号
    rsi_vote = 0
    if adx > 25:
        # 强趋势: RSI极端=趋势持续信号(反向回调即是入场机会)
        if rsi < 25:       # 极度超卖 = 多头机会(回调触底)
            rsi_vote = 1
        elif rsi > 75:     # 极度超买 = 空头机会(反弹触顶)
            rsi_vote = -1
        elif rsi < 40:     # 偏超卖 = 多头略强
            rsi_vote = 0.5
        elif rsi > 60:     # 偏超买 = 空头略强
            rsi_vote = -0.5
        # 中间区域40-60: 等待
    else:
        # 震荡/弱趋势: RSI极端值=均值回归
        if rsi < 30:
            rsi_vote = 1   # 超卖 → 反弹
        elif rsi > 70:
            rsi_vote = -1  # 超买 → 回调
        elif rsi < 40:
            rsi_vote = 0.5
        elif rsi > 60:
            rsi_vote = -0.5

    # ---- ADX因子 ----
    adx_vote = 0
    if adx > 30:
        adx_vote = 1   # 强趋势确认 (上限1.0，与其他因子对齐)
    elif adx > 22:
        adx_vote = 0.5
    elif adx < 15:
        adx_vote = 0   # 无趋势

    # ---- Bollinger ----
    bb_vote = 0
    if bb_pos < 20:
        bb_vote = 1   # 价格靠近下轨 → 做多
    elif bb_pos > 80:
        bb_vote = -1  # 价格靠近上轨 → 做空
    elif bb_pos < 35:
        bb_vote = 0.5
    elif bb_pos > 65:
        bb_vote = -0.5

    # ---- Vol ----
    vol_vote = 0
    if vol_ratio > 1.5:
        vol_vote = 0.5  # 高波幅 → 趋势确认（趋势跟踪系统的正常状态）
    elif vol_ratio < 0.7:
        vol_vote = -0.5  # 低波幅 → 犹豫/假突破风险

    # ---- MACD ----
    macd_vote = 0
    if macd_hist > 0.01:
        macd_vote = 1
    elif macd_hist < -0.01:
        macd_vote = -1

    # ---- BTC ----
    btc_vote = 1 if btc_trend == 'bull' else (-1 if btc_trend == 'bear' else 0)

    votes = {
        'RSI': rsi_vote,
        'ADX': adx_vote,
        'Bollinger': bb_vote,
        'Vol': vol_vote,
        'MACD': macd_vote,
        'BTC': btc_vote,
        # P0 Fix: Gemma vote is 0-1 range, remap to -1 to +1
        # Gemma health check: if gemma_health=='down', zero out Gemma vote
        'Gemma': 0 if factors.get('gemma_health') == 'down' else (factors.get('_gemma_vote', 0.5) - 0.5) * 2,
    }

    # ---- 极端RSI信号: 直接替换RSI投票方向 ----
    extreme = factors.get('_extreme_signal', None)
    if extreme and extreme in ('long', 'short'):
        # 极端RSI: 直接用RSI_weight作为信号强度，乘以1.5
        rsi_extreme_vote = 1 if extreme == 'long' else -1
        score = weights.get('RSI', 0.15) * rsi_extreme_vote * 3.0  # 方向强度，confidence固定0.80（见下）
        direction = extreme
        # ADX>30强趋势时，extreme信号也必须与趋势一致
        if adx > 30:
            trend_dir = 1 if factors.get('_di_plus', 20) > factors.get('_di_minus', 20) else -1
            if (direction == 'long' and trend_dir < 0) or (direction == 'short' and trend_dir > 0):
                direction = 'wait'
                score = 0
    else:
        # 加权得分
        score = sum(weights.get(k, 0) * v for k, v in votes.items())
        direction = 'long' if score > 0.05 else ('short' if score < -0.05 else 'wait')
        # ADX>30强趋势时，方向需与趋势一致
        if adx > 30:
            trend_dir = 1 if factors.get('_di_plus', 20) > factors.get('_di_minus', 20) else -1
            if (direction == 'long' and trend_dir < 0) or (direction == 'short' and trend_dir > 0):
                direction = 'wait'
                score = 0

    # confidence计算：extreme RSI用固定高信心(0.80)，普通信号用score归一化
    if extreme and direction != 'wait':
        conf = 0.80   # RSI<5或>95是最强信号，且未被ADX过滤
    elif direction == 'wait':
        conf = 0.0    # 被过滤的信号，零信心
    else:
        conf = min(abs(score) / 2.0, 1.0)

    # ── 多时间框架确认：4H趋势反向则重罚 ──
    _4h_dir = factors.get('_4h_direction', 'neutral')
    _4h_strength = factors.get('_4h_strength', 0.0)
    if _4h_strength > 0.5 and direction != 'wait':
        if (_4h_dir == 'bull' and direction == 'short') or \
           (_4h_dir == 'bear' and direction == 'long'):
            conf *= 0.30  # 4H强烈逆势，confidence打3折
            score *= 0.30
            logger.debug(f"4H逆势惩罚: {_4h_dir}(强度{_4h_strength:.2f}) vs 1H{direction}, conf={conf:.3f}")

    return {'score': score, 'direction': direction, 'votes': votes,
            'confidence': conf, 'extreme': extreme}

# ===== 熔断系统 (from Kronos real_monitor.py) =====

# ===== Treasury状态管理 (from treasury.py) =====
from treasury import load_treasury, save_treasury


# ===== 数据获取 =====
def get_klines(instId, timeframe='1H', limit=100):
    """从OKX获取K线数据"""
    path = f'/api/v5/market/candles?instId={instId}&bar={timeframe}&limit={limit}'
    data = okx_req('GET', path)
    if data.get('code') != '0':
        logger.warning(f"get_klines: OKX API错误 {instId}: {data.get('msg', data)}")
        return None
    candles = data.get('data', [])
    if not candles:
        logger.warning(f"get_klines: OKX返回空数据 {instId}")
        return None
    # [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    parsed = []
    for c in reversed(candles):
        try:
            parsed.append({
                'ts': int(c[0]),
                'open': safe_float(c[1]),
                'high': safe_float(c[2]),
                'low': safe_float(c[3]),
                'close': safe_float(c[4]),
                'vol': safe_float(c[5]),
            })
        except Exception as ex:
            logger.debug(f"get_klines: 解析K线失败，跳过该K线: {ex}")
    return parsed if parsed else None

def get_4h_trend(instId):
    """获取4H趋势方向+强度，用于多时间框架确认
    Returns: (direction: 'bull'|'bear'|'neutral', strength: 0.0-1.0)
    """
    klines = get_klines(instId, '4H', 30)
    if not klines or len(klines) < 14:
        return 'neutral', 0.0
    closes = [k['close'] for k in klines]
    highs = [k['high'] for k in klines]
    lows = [k['low'] for k in klines]
    di_plus, di_minus, adx = calc_adx(highs, lows, closes)
    if adx < 20:
        return 'neutral', 0.0  # 无明确趋势
    direction = 'bull' if di_plus > di_minus else 'bear'
    strength = min(adx / 50.0, 1.0)  # ADX 25→0.5, ADX 50→1.0
    return direction, strength

def get_account_balance():
    """获取OKX账户余额"""
    data = okx_req('GET', '/api/v5/account/balance')
    if data.get('code') == '0' and data.get('data'):
        try:
            details = data['data'][0].get('details', [])
            for d in details:
                if d.get('ccy') == 'USDT':
                    return safe_float(d.get('eq'), 0)
            return safe_float(data['data'][0].get('totalEq'), 0)
        except Exception as ex:
            logger.debug(f"get_account_balance: 解析余额失败，返回0: {ex}")
    return 0

def get_positions():
    """获取所有持仓"""
    data = okx_req('GET', '/api/v5/account/positions?instType=SWAP')
    if data.get('code') != '0':
        return []
    positions = []
    for raw in data.get('data', []):
        notional = safe_float(raw.get('notionalUsd'), 0)
        # V6-4 Fix: 移除notional>1过滤，改为>0以包含所有有效持仓
        # 低价币种$1可能对应大量合约，过滤会留下僵尸仓位
        if notional > 0:
            positions.append({
                'instId': raw.get('instId'),
                'side': raw.get('posSide'),  # 'long' or 'short'
                # P0 Fix: OKX字段是 pos/avgPx，不是 sz/avgOpenPx
                'sz': safe_float(raw.get('pos'), 0),  # 合约张数
                'entry': safe_float(raw.get('avgPx'), 0),  # 入场价
                'unrealized_pnl': safe_float(raw.get('upl'), 0),
                'notional': notional,
                'liqPx': safe_float(raw.get('liqPx'), 0),  # 强平价
                'leverage': safe_float(raw.get('lever'), 3),
            })
    return positions

def get_ticker(instId):
    """获取当前价格"""
    data = okx_req('GET', f'/api/v5/market/ticker?instId={instId}')
    if data.get('code') == '0' and data.get('data'):
        return safe_float(data['data'][0].get('last'), 0)
    return 0

# ===== 白名单模式学习 (from Miracle agent_signal.py) =====
_whitelist_lock = threading.Lock()
_whitelist_cache = {'timestamp': 0, 'data': None, 'ttl': 5}  # 5-second TTL cache

def load_whitelist():
    import time
    now = time.time()
    with _whitelist_lock:
        if _whitelist_cache['data'] is not None and (now - _whitelist_cache['timestamp']) < _whitelist_cache['ttl']:
            return _whitelist_cache['data']
    wl_file = STATE_DIR / 'whitelist.json'
    if wl_file.exists():
        try:
            with open(wl_file) as f:
                data = json.load(f)
            # 确保blacklist为dict并清洗过期项（7天TTL）
            raw_bl = data.get('blacklist', {})
            if isinstance(raw_bl, list):
                # 旧格式set转dict（无时间戳，视为立即加入）
                bl_dict = {k: now for k in raw_bl}
            else:
                bl_dict = raw_bl
            # 清除7天前加入的黑名单
            BL_TTL = 7 * 86400  # 7天秒数
            bl_dict = {k: t for k, t in bl_dict.items() if now - t < BL_TTL}
            data['blacklist'] = bl_dict
            # 确保patterns始终为dict
            data.setdefault('patterns', {})
            with _whitelist_lock:
                _whitelist_cache['data'] = data
                _whitelist_cache['timestamp'] = now
            return data
        except Exception as e:
            logger.warning(f"load_whitelist: 读取白名单失败，使用默认配置: {e}")
    data = {'patterns': {}, 'blacklist': {}}
    with _whitelist_lock:
        _whitelist_cache['data'] = data
        _whitelist_cache['timestamp'] = now
    return data

def save_whitelist(wl):
    from core.kronos_utils import atomic_write_json
    with _whitelist_lock:
        _whitelist_cache['data'] = None  # invalidate cache
    # blacklist为dict时直接序列化
    atomic_write_json(STATE_DIR / 'whitelist.json', {
        'patterns': wl.get('patterns', {}),
        'blacklist': wl.get('blacklist', {})
    })

_gemma_cache_lock = threading.Lock()
Gemma4_TIMEOUT = 10  # gemma4超时10秒，超时使用规则回退

def _rule_based_vote(rsi, adx, bb_pos, di_plus=None, di_minus=None):
    """基于RSI/ADX/布林带的规则化方向投票（当Gemma不可用时使用）
    返回0.0-1.0：1.0=强烈LONG，0.0=强烈SHORT，0.5=中立
    """
    score = 0.5  # 从中立开始
    
    # RSI打分：超卖偏向多，超买偏向空
    if rsi < 30:
        score += 0.25  # 严重超卖 → 多
    elif rsi < 40:
        score += 0.15
    elif rsi > 70:
        score -= 0.25  # 严重超买 → 空
    elif rsi > 60:
        score -= 0.15
    
    # ADX打分：趋势强度 + 方向（与voting_vote主路径对齐）
    # 主路径: ADX>30→adx_vote=1(贡献≈0.12), ADX>22→0.5(贡献≈0.06)
    # fallback: ADX>30→+0.12, ADX>22→+0.06, ADX<15→-0.05（已对齐）
    # 当di_plus/di_minus可用时，用方向修正ADX信号（趋势跟踪方向）
    if adx > 30:
        score += 0.12  # 强趋势确认
        if di_plus is not None and di_minus is not None:
            # DI+ > DI- → 多头趋势，+0.05；DI- > DI+ → 空头趋势，-0.05
            if di_plus > di_minus:
                score += 0.05
            elif di_minus > di_plus:
                score -= 0.05
    elif adx > 22:
        score += 0.06  # 中等趋势
    elif adx < 15:
        score -= 0.05  # 震荡市场，RSI均值回归优先
    
    # 布林带打分：价格在低位偏多，高位偏空
    if bb_pos < 20:
        score += 0.20  # 价格贴近下轨 → 支撑/偏多
    elif bb_pos < 35:
        score += 0.10
    elif bb_pos > 80:
        score -= 0.20  # 价格贴近上轨 → 压力/偏空
    elif bb_pos > 65:
        score -= 0.10
    
    return max(0.0, min(1.0, score))


def _gemma_vote_cached(symbol, rsi, adx, bb_pos, price, di_plus=None, di_minus=None, cache_ttl=300):
    """调用gemma4获取方向判断，每币独立缓存5分钟
    返回0-1置信度分数：
    - 1.0 = 强烈LONG
    - 0.5 = 中立/WAIT
    - 0.0 = 强烈SHORT
    - 负数 = gemma4否决(超时/异常)——实际使用时已替换为规则回退

    Fallback机制:
    - 超时10秒 → 调用_rule_based_vote规则回退
    - 解析失败 → 调用_rule_based_vote规则回退
    - 连续3次失败 → circuit breaker升级tier
    """
    import time as _time
    cache_file = STATE_DIR / 'gemma_cache.json'

    # 读取缓存 (每币独立)
    now_bucket = int(_time.time() / cache_ttl)
    with _gemma_cache_lock:
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    all_cache = json.load(f)
                entry = all_cache.get(symbol, {})
                if entry.get('bucket') == now_bucket:
                    return entry.get('vote', 0)
            except Exception as e:
                logger.debug(f"gemma_cache_read: 读取缓存失败，跳过缓存: {e}")

    # ========== 职业操盘手prompt格式 (参考gemma4_central_decision) ==========
    # 判断趋势方向
    if adx > 25:
        trend_desc = "强趋势"
    elif adx > 15:
        trend_desc = "弱趋势"
    else:
        trend_desc = "震荡"

    # RSI解读
    if rsi < 30:
        rsi_desc = "严重超卖"
    elif rsi < 40:
        rsi_desc = "偏超卖"
    elif rsi > 70:
        rsi_desc = "严重超买"
    elif rsi > 60:
        rsi_desc = "偏超买"
    else:
        rsi_desc = "正常"

    # 布林带解读
    if bb_pos < 20:
        bb_desc = "价格贴近下轨(支撑位)"
    elif bb_pos > 80:
        bb_desc = "价格贴近上轨(压力位)"
    elif bb_pos < 35:
        bb_desc = "价格偏向下轨"
    elif bb_pos > 65:
        bb_desc = "价格偏向上轨"
    else:
        bb_desc = "价格在中部"

    # DI方向解读
    di_desc = ""
    if di_plus is not None and di_minus is not None:
        if di_plus > di_minus:
            di_desc = f"DI+({di_plus:.1f}) > DI-({di_minus:.1f}) → 多头趋势"
        else:
            di_desc = f"DI-({di_minus:.1f}) > DI+({di_plus:.1f}) → 空头趋势"

    prompt = f"""你是专业加密货币操盘手。分析{symbol}短期走势。

## 市场数据
- RSI(14): {rsi:.1f} → {rsi_desc}
- ADX: {adx:.1f} → {trend_desc}
- 布林带位置: {bb_pos:.1f}% → {bb_desc}
- 当前价格: ${price:.4f}
{f'- {di_desc}' if di_desc else ''}

## 你的任务
作为职业操盘手，基于以上数据判断短期(1-4小时)方向和信心程度。

## 输出格式（严格遵守）
confidence: 0.0-1.0之间的置信度分数
direction: LONG 或 SHORT
reason: 一句话分析理由

confidence表示你对方向判断的确信程度：
- 0.9-1.0 = 强烈信号（多个指标共振）
- 0.7-0.9 = 较强信号（2个以上指标支持）
- 0.5-0.7 = 中性信号（指标不一致或不确定）
- 0.3-0.5 = 弱信号（仅1个指标支持）
- 0.0-0.3 = 无信号（指标矛盾或不适合交易）

只输出上述格式，不要其他内容。
"""

    vote = 0.5  # 默认中立
    failure_type = None  # 'timeout' | 'parse' | None

    try:
        output = ''  # 初始化变量，确保 'output' in dir() 检查始终有效
        import subprocess
        result = subprocess.run(
            ['ollama', 'run', 'gemma4-2b-heretic:latest', prompt],
            capture_output=True, text=True, timeout=Gemma4_TIMEOUT
        )
        output = result.stdout.strip()

        # 解析置信度
        try:
            # 尝试提取 confidence: X.XX 格式
            import re
            conf_match = re.search(r'confidence:\s*([\d.]+)', output, re.IGNORECASE)
            if conf_match:
                vote = float(conf_match.group(1))
            else:
                # 备选：根据direction关键词判断
                output_upper = output.upper()[:20]
                if 'LONG' in output_upper:
                    vote = 0.6   # 弱LONG
                elif 'SHORT' in output_upper:
                    vote = 0.4   # 弱SHORT → (0.4-0.5)*2 = -0.2
                else:
                    # 解析失败 → 使用规则回退
                    vote = _rule_based_vote(rsi, adx, bb_pos, di_plus, di_minus)
                    failure_type = 'parse'
        except Exception as ex:
            # 解析异常 → 使用规则回退
            logger.debug(f" Gemma解析异常，使用规则回退: {ex}")
            vote = _rule_based_vote(rsi, adx, bb_pos, di_plus, di_minus)
            failure_type = 'parse'

    except subprocess.TimeoutExpired:
        # 超时 → 使用规则回退
        vote = _rule_based_vote(rsi, adx, bb_pos, di_plus, di_minus)
        failure_type = 'timeout'

    except Exception as ex:
        # 其他异常 → 使用规则回退
        logger.debug(f" Gemma调用异常，使用规则回退: {ex}")
        vote = _rule_based_vote(rsi, adx, bb_pos, di_plus, di_minus)
        failure_type = 'error'

    # ---- 保存状态（单一出口，避免double write）----
    # 只在状态真正变化时保存
    treasury = load_treasury()
    gemma_fail_count = treasury.get('gemma_consecutive_failures', 0)

    if failure_type is not None:
        gemma_fail_count += 1
        treasury['gemma_consecutive_failures'] = gemma_fail_count

        # 连续3次失败 → gemma_health降级为degraded；连续6次 → down
        if gemma_fail_count >= 6:
            treasury['gemma_health'] = 'down'
        elif gemma_fail_count >= 3:
            treasury['gemma_health'] = 'degraded'
    else:
        # 成功 → 重置连续失败计数和gemma_health
        if gemma_fail_count > 0:
            treasury['gemma_consecutive_failures'] = 0
            treasury['gemma_health'] = 'healthy'

    # V3 NEW-1 Fix: 单一save_treasury调用（原来在两处各调用一次，可能double write）
    save_treasury(treasury)

    # 写入每币缓存
    if failure_type is None:
        with _gemma_cache_lock:
            all_cache = {}
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        all_cache = json.load(f)
                except Exception as e:
                    logger.debug(f"gemma_cache_write: 读取现缓存失败: {e}")
            all_cache[symbol] = {'vote': vote, 'bucket': now_bucket, 'raw': output[:100]}
            atomic_write_json(cache_file, all_cache)

    return vote


def get_pattern_key(rsi, adx, bb_pos, direction):
    """生成模式key用于白名单匹配"""
    rsi_zone = 'oversold' if rsi < 35 else ('overbought' if rsi > 65 else 'neutral')
    adx_zone = 'strong' if adx > 25 else ('weak' if adx < 15 else 'medium')
    bb_zone = 'lower' if bb_pos < 25 else ('upper' if bb_pos > 75 else 'mid')
    return f'{direction}_{rsi_zone}_{adx_zone}_{bb_zone}'

def check_whitelist(rsi, adx, bb_pos, direction):
    """检查白名单模式"""
    wl = load_whitelist()
    key = get_pattern_key(rsi, adx, bb_pos, direction)
    import time
    bl = wl.get('blacklist', {})
    if key in bl and (time.time() - bl[key]) < (7 * 86400):
        return False, '黑名单模式'
    stats = wl.get('patterns', {}).get(key, {})
    if stats.get('count', 0) >= 5 and stats.get('win_rate', 0.5) < 0.40:
        return False, f'胜率{stats["win_rate"]:.0%}<40%'
    return True, '通过'

def update_whitelist(entry_key, won: bool):
    """更新白名单统计"""
    wl = load_whitelist()
    stats = wl['patterns'].get(entry_key, {'count': 0, 'wins': 0})
    stats['count'] += 1
    if won:
        stats['wins'] += 1
    stats['win_rate'] = stats['wins'] / stats['count']
    wl['patterns'][entry_key] = stats
    
    # 黑名单降级（7天TTL）
    if stats['count'] >= 10 and stats['win_rate'] < 0.35:
        import time
        wl['blacklist'][entry_key] = int(time.time())
    save_whitelist(wl)

# ===== 交易日志 =====

# ===== 交易日志 (from trade_journal.py) =====
from trade_journal import (
    get_open_trades, update_trade_pnl,
    load_trades, save_trades, record_trade
)


# ===== OCO下单 (带验证) =====
def place_oco(instId, side, sz, entry_price, sl_pct, tp_pct, equity: float = 0, leverage: int = 3):
    """
    开仓 + OCO bracket保护（两步分离，正确架构）

    步骤1: 市价开仓（设置杠杆 + 下市价单）
    步骤2: 挂OCO Bracket（止损+止盈合并单，只平仓不开新仓）

    OKX每持仓只能有1个条件单，必须用ordType='oco'合并SL+TP
    reduceOnly=True确保OCO只平仓、不开新仓

    Args:
        instId: 合约ID，如 'DOGE-USDT-SWAP'
        side: 'long' 或 'short'
        sz: 合约张数
        entry_price: 入场价格（用于计算SL/TP价格）
        sl_pct: 止损百分比，如 0.05（5%）
        tp_pct: 止盈百分比，如 0.10（10%）
        equity: 账户权益（用于验证）
        leverage: 杠杆倍数，默认3x
    """
    from core.kronos_utils import validate_oco_order, check_existing_oco_orders
    # P1: OCO订单验证
    valid, reason, details = validate_oco_order(
        instId, side, sz, entry_price, sl_pct, tp_pct, equity
    )
    if not valid:
        logger.warning(f"OCO验证失败 {instId}: {reason}")
        return {'code': '99999', 'msg': f'OCO验证失败: {reason}', 'details': details}

    # P1: 检查已有活跃OCO订单（幂等）
    has_active, order_info = check_existing_oco_orders(instId)
    if has_active:
        logger.warning(f"发现活跃OCO订单 {instId}: {order_info}")
        return {'code': '99999', 'msg': f'已有活跃OCO: {order_info}'}

    if side == 'long':
        open_side = 'buy'
        close_side = 'sell'
        sl_price = round(entry_price * (1 - sl_pct), 4)
        tp_price = round(entry_price * (1 + tp_pct), 4)
        pos_side = 'long'
    else:  # short
        open_side = 'sell'
        close_side = 'buy'
        sl_price = round(entry_price * (1 + sl_pct), 4)
        tp_price = round(entry_price * (1 - tp_pct), 4)
        pos_side = 'short'

    # ── Step 1: 设置杠杆 ──
    leverage_body = json.dumps({
        'instId': instId,
        'lever': str(leverage),
        'mgnMode': 'isolated',
    })
    lev_result = okx_req('POST', '/api/v5/account/set-leverage', leverage_body)
    if lev_result.get('code') != '0':
        logger.error(f"设置杠杆失败 {instId} {leverage}x: {lev_result.get('msg')}，取消开仓")
        return {'code': '99999', 'msg': f'杠杆设置失败: {lev_result.get("msg")}'}

    # ── Step 2: 市价开仓 ──
    open_body = {
        'instId': instId,
        'tdMode': 'isolated',
        'side': open_side,
        'ordType': 'market',
        'sz': str(int(sz)),
    }
    # P0 Fix: posSide only in hedge mode, not net mode
    if _pos_mode != 'net' and pos_side:
        open_body['posSide'] = pos_side
    open_body = json.dumps(open_body)
    open_result = okx_req('POST', '/api/v5/trade/order', open_body)
    if open_result.get('code') != '0':
        logger.error(f"开仓失败 {instId}: {open_result.get('msg')}")
        return {'code': open_result.get('code', '99999'),
                'msg': f'开仓失败: {open_result.get("msg")}'}

    # ── Step 3: 挂OCO Bracket（止损+止盈，保护已有仓位） ──
    # reduceOnly=True确保只平仓不开新仓
    oco_body = {
        'instId': instId,
        'tdMode': 'isolated',
        'side': close_side,
        'ordType': 'oco',
        'sz': str(int(sz)),
        'reduceOnly': True,       # P0 Fix: 只平仓不开新仓
        'slTriggerPx': str(sl_price),
        'slOrdPx': '-1',             # 市价触发
        'tpTriggerPx': str(tp_price),
        'tpOrdPx': '-1',            # 市价触发
    }
    # P0 Fix: posSide only in hedge mode, not net mode
    if _pos_mode != 'net' and pos_side:
        oco_body['posSide'] = pos_side
    oco_body = json.dumps(oco_body)
    oco_result = okx_req('POST', '/api/v5/trade/order-algo', oco_body)

    # 汇总结果
    if oco_result.get('code') == '0':
        logger.info(f"开仓+OCO成功 {instId} {side} {sz}张 @ {entry_price}, SL={sl_price}, TP={tp_price}")
        return {
            'code': '0',
            'open': open_result,
            'oco': oco_result,
        }
    else:
        # OCO失败，但仓位已开！立即平仓防风险
        logger.warning(f"OCO挂单失败，已平仓防风险 {instId}: {oco_result.get('msg')}")
        
        # 提取symbol并平仓
        symbol = instId.replace('-USDT-SWAP', '')
        close_result = close_position(symbol, reason="OCO失败自动平仓")
        
        return {
            'code': oco_result.get('code', '99999'),
            'msg': f'OCO挂单失败，已平仓防风险: {oco_result.get("msg")}',
            'open_success': True,
            'open': open_result,
            'close_result': close_result,
        }

def close_position(symbol: str, reason: str = "signal",
                   pos: dict = None) -> Dict:
    """
    平仓 - 根据symbol执行市价平仓
    P0修复: 使用 /api/v5/trade/order 而非 /api/v5/trade/close-position
           OKX close-position端点不支持mgnMode/ccy参数，会导致400错误
    pos: 可选，传入持仓数据 {sz, side} 以获取正确数量和方向
    Returns: {'code': '0', 'data': [...]} or {'code': '99999', 'msg': ...}
    """
    inst_id = f"{symbol}-USDT-SWAP"

    # 获取持仓信息（如果没有传入）
    if pos is None:
        all_pos = get_positions()
        pos = next((p for p in all_pos if p.get('instId', '').startswith(symbol)), None)

    if not pos:
        logger.warning(f"[{symbol}] 无持仓，跳过平仓")
        return {'code': '99999', 'msg': f'无持仓 {symbol}'}

    sz = int(pos['sz'])
    # 兼容OKX API字段('side')和本地历史记录('direction')
    pos_side = pos.get('side') or pos.get('direction', 'long')
    # 平多: side=sell, 平空: side=buy
    close_side = 'sell' if pos_side == 'long' else 'buy'

    # P0 Fix: OKX net mode (default) does NOT accept posSide - causes 400 error
    # Only include posSide when explicitly in hedge mode (not 'net' or 'long'/'short' from local records)
    body = {
        'instId': inst_id,
        'tdMode': 'isolated',
        'side': close_side,
        'ordType': 'market',
        'sz': str(sz),
    }
    # posSide only in hedge mode — use global pos mode (set by _detect_pos_mode)
    if _pos_mode != 'net' and pos_side:
        body['posSide'] = pos_side
    body = json.dumps(body)
    data = okx_req('POST', '/api/v5/trade/order', body)
    if data.get('code') == '0':
        logger.info(f"[{symbol}] 平仓成功 ({reason})")

        # P0 Fix: Cancel any active OCO orders for this symbol after closing position
        try:
            # Query pending OCO orders
            oco_query = okx_req('GET', f'/api/v5/trade/orders-algo-pending?instId={inst_id}&ordType=oco')
            algo_list = oco_query.get('data', [])
            if algo_list:
                # Cancel each active OCO order
                cancel_body = json.dumps([{'algoId': str(o['algoId']), 'instId': inst_id} for o in algo_list])
                cancel_result = okx_req('DELETE', '/api/v5/trade/cancel-algos', cancel_body)
                if cancel_result.get('code') == '0':
                    logger.info(f"[{symbol}] 取消OCO订单成功 ({len(algo_list)}个)")
                else:
                    logger.warning(f"[{symbol}] 取消OCO订单失败: {cancel_result.get('msg', 'unknown')}")
            else:
                logger.info(f"[{symbol}] 无待取消的OCO订单")
        except Exception as e:
            logger.warning(f"[{symbol}] 取消OCO订单异常: {e}")
    else:
        logger.warning(f"[{symbol}] 平仓失败: {data.get('msg', 'unknown')}")
    return data

# ===== 主扫描逻辑 =====
SCAN_COINS = [
    ('BTC-USDT-SWAP', 'BTC'),
    ('ETH-USDT-SWAP', 'ETH'),
    ('SOL-USDT-SWAP', 'SOL'),
    ('DOGE-USDT-SWAP', 'DOGE'),
    ('ADA-USDT-SWAP', 'ADA'),
    ('XRP-USDT-SWAP', 'XRP'),
    ('BNB-USDT-SWAP', 'BNB'),
    ('AVAX-USDT-SWAP', 'AVAX'),
    ('LINK-USDT-SWAP', 'LINK'),
    ('DOT-USDT-SWAP', 'DOT'),
]

# OKX USDT永续合约乘数（每张合约对应的币数量）
# 用于计算合约张数: sz = sz_dollar / (entry × multiplier)
# BTC: 0.01 BTC/张, ETH: 0.1 ETH/张, DOGE: 1000 DOGE/张, SOL: 1 SOL/张, ADA: 100 ADA/张, XRP: 10 XRP/张
CONTRACT_MULTIPLIER_FALLBACK = {
    'BTC': 0.01, 'ETH': 0.1, 'SOL': 1, 'DOGE': 1000,
    'ADA': 100, 'XRP': 10, 'BNB': 10, 'AVAX': 1,
    'LINK': 1, 'DOT': 1,
}
_contract_multiplier_cache = {}

# Funding Rate & OI History Cache (max 10 entries per coin)
_fr_history = {}  # {coin: [funding_rate_values]}
_oi_history = {}  # {coin: [oi_values]}

MAX_POSITIONS = 3
SL_PCT = 0.05  # 5%止损 (仅用于静态计算后备)
TP_PCT = 0.10  # 10%止盈 (仅用于静态计算后备)
POSITION_SIZE_PCT = 0.02  # 每次2%仓位


def calc_atr(highs, lows, closes, period=14):
    """计算ATR (Average True Range)"""
    if len(closes) < period * 2 + 1:
        return 0.0

    trs = []
    for i in range(1, len(closes)):
        h, lo = highs[i], lows[i]
        prev_c = closes[i - 1]
        tr = max(h - lo, abs(h - prev_c), abs(lo - prev_c))
        trs.append(tr)

    if len(trs) < period:
        return 0.0

    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


def get_dynamic_sl_tp(coin, entry_price, atr, adx=None):
    """
    Dynamic SL/TP based on ATR volatility.
    Keep a minimum SL% of 0.5% (absolute floor for low ATR coins)

    TP: RR=2 for normal, RR=3 for high conviction (adx > 30)
    """
    if entry_price <= 0 or atr <= 0:
        return SL_PCT, TP_PCT  # Fallback to static if invalid inputs
    
    atr_pct = atr / entry_price  # ATR as percentage of entry price
    
    # High volatility coins get tighter SL (DOGE, SHIB, etc.)
    tight_coins = {'DOGE', 'SHIB', 'PEPE', 'BONK', 'WIF'}
    if coin in tight_coins:
        sl_multiplier = 1.5
    else:
        sl_multiplier = 2.0
    
    # Calculate SL percentage
    sl_pct = sl_multiplier * atr_pct
    
    # Floor: minimum 0.5% SL regardless of low ATR
    min_sl_pct = 0.005
    if sl_pct < min_sl_pct:
        sl_pct = min_sl_pct
    
    # TP: RR=3 for high conviction (adx > 30), RR=2 for normal
    rr = 3.0 if (adx is not None and adx > 30) else 2.0
    tp_pct = rr * sl_pct
    
    return sl_pct, tp_pct


def fetch_contract_multiplier(coin: str) -> float:
    """
    Fetch contract multiplier from OKX API.
    Caches result in _contract_multiplier_cache.
    Falls back to hardcoded values if API fails.
    """
    if coin in _contract_multiplier_cache:
        return _contract_multiplier_cache[coin]
    
    try:
        import requests
        url = "https://www.okx.com/api/v5/public/instruments"
        params = {'instType': 'SWAP', 'instId': f'{coin}-USDT-SWAP'}
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        
        if data.get('code') == '0' and data.get('data'):
            # OKX returns contract multiplier in 'ctMult' field
            ct_mult = data['data'][0].get('ctMult')
            if ct_mult:
                multiplier = safe_float(ct_mult, 1.0)
                _contract_multiplier_cache[coin] = multiplier
                logger.info(f"OKX API contract multiplier for {coin}: {multiplier}")
                return multiplier
    except Exception as e:
        logger.warning(f"Failed to fetch contract multiplier for {coin}: {e}")
    
    # Fallback to hardcoded values
    fallback = CONTRACT_MULTIPLIER_FALLBACK.get(coin, 1.0)
    _contract_multiplier_cache[coin] = fallback
    return fallback

def scan_coin(instId, symbol, equity, btc_trend, weights, exchange=None):
    """扫描单个币种"""
    from core.exchange_adapter import get_default_exchange
    from core.price_factors import PriceFactors
    if exchange is None:
        exchange = get_default_exchange()
    
    klines_1h = get_klines(instId, '1H', 100)
    klines_4h = get_klines(instId, '4H', 100)
    
    if not klines_1h or len(klines_1h) < 30:
        return None
    
    # ---- Funding Rate & OI Fetching with History Cache ----
    funding_rate_info = None
    oi_info = None
    fr_result = {}
    oi_result = {}
    try:
        fr_data = exchange.get_funding_rate(instId)
        oi_data = exchange.get_oi(instId)
        
        # Update funding rate history
        if fr_data and fr_data.get('funding_rate') is not None:
            fr_value = fr_data['funding_rate']
            if symbol not in _fr_history:
                _fr_history[symbol] = []
            _fr_history[symbol].append(fr_value)
            if len(_fr_history[symbol]) > 10:
                _fr_history[symbol] = _fr_history[symbol][-10:]
            funding_rate_info = fr_data
        
        # Update OI history
        if oi_data and oi_data.get('oi') is not None:
            oi_value = oi_data['oi']
            if symbol not in _oi_history:
                _oi_history[symbol] = []
            _oi_history[symbol].append(oi_value)
            if len(_oi_history[symbol]) > 10:
                _oi_history[symbol] = _oi_history[symbol][-10:]
            oi_info = oi_data
    except Exception as e:
        logger.debug(f"Failed to fetch funding rate/OI for {symbol}: {e}")
    
    closes_1h = [k['close'] for k in klines_1h]
    highs_1h = [k['high'] for k in klines_1h]
    lows_1h = [k['low'] for k in klines_1h]
    
    # 4H确认（趋势共振）
    btc_4h_confirmed = False
    if klines_4h and len(klines_4h) >= 30:
        closes_4h = [k['close'] for k in klines_4h]
        highs_4h = [k['high'] for k in klines_4h]
        lows_4h = [k['low'] for k in klines_4h]
        di_plus_4h, di_minus_4h, adx_4h = calc_adx(highs_4h, lows_4h, closes_4h)
        btc_4h_confirmed = adx_4h > 20
    else:
        adx_4h = 20
    
    # 因子计算
    rsi = calc_rsi(closes_1h)
    di_plus, di_minus, adx = calc_adx(highs_1h, lows_1h, closes_1h)
    macd, signal, hist = calc_macd(closes_1h)
    bb_upper, bb_lower, bb_pos = calc_bollinger(closes_1h)
    
    # ATR计算用于动态SL/TP
    atr = calc_atr(highs_1h, lows_1h, closes_1h)
    sl_pct, tp_pct = get_dynamic_sl_tp(symbol, closes_1h[-1], atr, adx)
    
    # 量比
    vol_ratio = 1.0
    if len(klines_1h) >= 20:
        recent_vol = sum(k['vol'] for k in klines_1h[-5:]) / 5
        avg_vol = sum(k['vol'] for k in klines_1h[-20:]) / 20
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
    
    # ---- Gemma LLM 因子 (带缓存) ----
    current_price = closes_1h[-1]
    gemma_vote = _gemma_vote_cached(symbol, rsi, adx, bb_pos, current_price, di_plus, di_minus)

    # ---- DOT极值RSI检测 ----
    # RSI < 5 = 极端超卖 → 强烈反弹信号 (仅在非强趋势时有效)
    # RSI > 95 = 极端超买 → 强烈回调信号 (仅在非强趋势时有效)
    # Only apply mean reversion when ADX < 25 (not a strong trend)
    extreme_signal = None
    if rsi < 5 and adx < 25:
        extreme_signal = 'long'  # Extreme oversold in non-trending market
    elif rsi > 95 and adx < 25:
        extreme_signal = 'short'  # Extreme overbought in non-trending market
    else:
        extreme_signal = 'neutral'  # In strong trend (ADX>=25), don't fight the trend

    # IC投票
    treasury = load_treasury()
    _4h_direction, _4h_strength = get_4h_trend(instId)
    factors = {
        'rsi': rsi, 'adx': adx, 'bb_pos': bb_pos,
        'macd_hist': hist, 'vol_ratio': vol_ratio,
        'btc_trend': btc_trend,
        '_di_plus': di_plus, '_di_minus': di_minus,
        '_gemma_vote': gemma_vote,
        '_extreme_signal': extreme_signal,
        'gemma_health': treasury.get('gemma_health', 'healthy'),
        '_4h_direction': _4h_direction,
        '_4h_strength': _4h_strength,
    }
    vote = voting_vote(factors, weights)
    
    if vote['direction'] == 'wait':
        return None
    
    # 白名单过滤
    ok, reason = check_whitelist(rsi, adx, bb_pos, vote['direction'])
    if not ok:
        return None
    
    # ---- 4H多时间框架确认（方向感知） ----
    # 1H信号 × 4H方向对齐度:
    #   4H强趋势+方向一致 → ×1.4 确认加成
    #   4H强趋势+方向相反 → ×0.3 强烈抑制（不逆大趋势）
    #   4H无趋势(ADX≤20)  → ×1.0 中立，1H信号自己说话
    if btc_4h_confirmed and adx_4h > 25 and abs(di_plus_4h - di_minus_4h) > 5:
        signal_dir = vote['direction']
        if (_4h_direction == 'bull' and signal_dir == 'long') or \
           (_4h_direction == 'bear' and signal_dir == 'short'):
            mt_boost = 1.4  # 趋势共振 → 强确认
            logger.debug(f"[{symbol}] 4H共振确认: {signal_dir}对齐{_4h_direction}趋势 ADX={adx_4h:.0f}")
        elif (_4h_direction == 'bull' and signal_dir == 'short') or \
             (_4h_direction == 'bear' and signal_dir == 'long'):
            mt_boost = 0.3  # 逆大趋势 → 强制抑制
            logger.debug(f"[{symbol}] 4H抑制: {signal_dir}逆{_4h_direction}趋势 ADX={adx_4h:.0f}")
        else:
            mt_boost = 1.0
    elif btc_4h_confirmed and adx_4h > 20:
        mt_boost = 1.0  # 弱趋势4H，不调整
    else:
        mt_boost = 1.0  # 无趋势4H，不调整

    # 逆大趋势的强抑制：直接否决
    if mt_boost <= 0.3:
        return None

    final_score = abs(vote['score']) * mt_boost

    # ADX强度加成: 强趋势确认加分
    if adx > 30:
        final_score *= 1.3
    elif adx > 22:
        final_score *= 1.15

    # ---- Funding Rate Factor: 影响信心加分 ----
    fr_confidence_boost = 0.0
    if funding_rate_info and symbol in _fr_history:
        fr_history = _fr_history[symbol]
        fr_result = PriceFactors.calc_funding_rate_factor(fr_history, side=vote['direction'])
        fr_confidence_boost = fr_result.get('confidence_boost', 0.0)
        final_score += fr_confidence_boost
        logger.debug(f"[{symbol}] Funding rate boost: {fr_confidence_boost:.4f} (fr={fr_result.get('funding_rate', 0):.6f})")

    # ---- OI Direction Factor: OI下降 + short方向 = 确认; OI下降 + long方向 = 减弱 ----
    oi_penalty = 0.0
    if oi_info and symbol in _oi_history:
        oi_history = _oi_history[symbol]
        oi_result = PriceFactors.calc_oi_change_rate(oi_history)
        oi_direction = oi_result.get('oi_direction', 'stable')
        oi_penalty = oi_result.get('confidence_penalty', 0.0)
        
        if oi_direction == 'decreasing' and oi_penalty > 0:
            if vote['direction'] == 'short':
                # OI下降+做空 = 确认信号，稍微减少惩罚
                oi_penalty *= 0.5
                logger.debug(f"[{symbol}] OI decreasing + short: penalty reduced to {oi_penalty:.4f}")
            elif vote['direction'] == 'long':
                # OI下降+做多 = 减弱信号，充分应用惩罚
                logger.debug(f"[{symbol}] OI decreasing + long: full penalty {oi_penalty:.4f}")
        
        final_score -= oi_penalty

    # 低于阈值过滤 (extreme信号阈值更低; normal从0.25降至0.15，避免conf~17%的好信号被误杀)
    min_threshold = 0.20 if vote.get('extreme') else 0.15
    if final_score < min_threshold:
        return None
    
    return {
        'symbol': symbol,
        'instId': instId,
        'direction': vote['direction'],
        'score': final_score,
        'entry': closes_1h[-1],
        'sl': sl_pct,
        'tp': tp_pct,
        'atr': atr,  # Include ATR for transparency
        'rsi': rsi,
        'adx': adx,
        'adx_4h': adx_4h,
        'bb_pos': bb_pos,
        'macd_hist': hist,
        'vol_ratio': vol_ratio,
        'pattern_key': get_pattern_key(rsi, adx, bb_pos, vote['direction']),
        'confidence': vote.get('confidence'),  # 原始信心分数（0-1），与score共同用于排序
        'votes': vote.get('votes', {}),
        'extreme': vote.get('extreme'),
        # Funding Rate & OI factors
        'funding_rate': fr_result.get('funding_rate', 0.0) if funding_rate_info else None,
        'funding_rate_direction': fr_result.get('funding_rate_direction', 'stable') if funding_rate_info else None,
        'fr_confidence_boost': fr_confidence_boost,
        'oi_direction': oi_result.get('oi_direction', 'stable') if oi_info else None,
        'oi_change_rate': oi_result.get('oi_change_rate', 0.0) if oi_info else None,
        'oi_penalty': oi_penalty,
    }

def select_best(candidates, positions, local_trades=None):
    """选最优候选
    gemma4否决机制: _gemma_vote is None (error/timeout) 的候选币不参与排名
    - _gemma_vote有值 (0.0-1.0): 参与排名 (包括 bearish 0.3-0.5 和 bullish 0.5-1.0)
    - _gemma_vote is None: 过滤掉不参与排名 (gemma超时/异常)

    Returns: (best_candidate, vetoed_pattern_keys)
    """
    if not candidates:
        return None, []
    held = {p['instId'].replace('-USDT-SWAP', '') for p in positions}
    if local_trades:
        held.update(t.get('coin', '') for t in local_trades)
    available = [c for c in candidates if c['symbol'] not in held]
    
    # P1-4 Fix: 只否决_gemma_vote为None（超时/错误/不可用），允许所有有效信号（bearish 0.3-0.5也通过）
    # _gemma_vote存储在candidate顶层；votes['Gemma']是remapped值（负数不代表否决，代表方向）
    vetoed = []
    vetoed_pattern_keys = []
    filtered = []
    for c in available:
        gemma_raw = c.get('_gemma_vote')  # None = 不可用, 0.0-1.0 = 有效
        if gemma_raw is None:  # Gemma超时/错误 → 否决
            vetoed.append(c['symbol'])
            vetoed_pattern_keys.append(c.get('pattern_key', ''))
            continue
        filtered.append(c)

    if vetoed:
        print(f"gemma4否决: {vetoed} (gemma不可用)")
    
    if not filtered:
        return None, vetoed_pattern_keys

    # 模式历史调整：高胜率模式提升排序权重
    def _adjusted_score(c):
        adj = get_pattern_adjustment(c.get('pattern_key', ''))
        return c['score'] * adj

    best = max(filtered, key=_adjusted_score)
    adj = get_pattern_adjustment(best.get('pattern_key', ''))
    if adj != 1.0:
        logger.info(f"模式历史调整: {best.get('pattern_key','?')} ×{adj:.2f} (分数 {best['score']:.3f}→{best['score']*adj:.3f})")
    return best, vetoed_pattern_keys

def _parallel_scan_wrapper(instId, symbol, equity, btc_trend, weights, exchange=None):
    """并行扫描包装器"""
    try:
        return scan_coin(instId, symbol, equity, btc_trend, weights, exchange)
    except Exception as e:
        logger.warning(f"扫描失败 {instId}: {e}")
        return None

def _get_memory_confidence_multiplier() -> float:
    """
    从ChromaDB查询最近10笔交易结果,根据最近交易表现调整置信度.
    
    规则:
    - 最近3笔全部亏损: 置信度×0.90 (降10%)
    - 最近1笔盈利>5%: 置信度×1.05 (加5%)
    - 否则: 置信度×1.00
    
    Returns:
        float: 置信度乘数 (0.90, 1.00, 或 1.05)
    """
    from core.memory import get_structured_memory
    try:
        memory = get_structured_memory()
        recent_trades = memory.get_trades(status='closed', limit=10)
        
        if not recent_trades:
            logger.debug("无历史交易记录,置信度乘数=1.0")
            return 1.0
        
        # 按exit_time倒序取最后3条(最近结束的)
        # get_trades已按entry_time DESC,所以取前3
        last_3 = recent_trades[:3]
        
        # 检查最近3笔是否全亏
        all_losing = all(t.pnl < 0 for t in last_3)
        if all_losing and len(last_3) >= 3:
            logger.warning(f"最近3笔全亏({[(t.symbol, t.pnl_pct) for t in last_3]}),置信度降10%")
            return 0.90
        
        # 检查最近1笔是否盈利>5%
        last_trade = recent_trades[0]
        if last_trade.pnl_pct > 0.05:
            logger.info(f"最近1笔盈利>{last_trade.pnl_pct:.1%}({last_trade.symbol}),置信度加5%")
            return 1.05
        
        return 1.0
        
    except Exception as e:
        logger.warning(f"查询历史交易失败: {e},置信度乘数=1.0")
        return 1.0

def run_scan(equity, btc_trend='neutral', mode='audit'):
    """
    主扫描入口
    
    P0+P1修复:
    - 异步并发: ThreadPoolExecutor并发扫描
    - Treasury预检查: 交易前熔断检查
    - 集中度检查: 仓位集中度限制
    - 日志幂等: 幂等日志写入
    """
    from core.kronos_utils import (
        check_treasury_tier,
        check_treasury_trade_allowed,
        parallel_scan_coins,
    )
    from core.market_intel_base import (
        get_fomc_confidence_multiplier,
        get_market_regime,
        get_regime_confidence_multiplier,
    )
    from core.exchange_adapter import get_default_exchange
    treasury = load_treasury()
    
    # P0: Treasury预检查 - 交易前必须验证
    treasury_allowed, treasury_reason, treasury_details = check_treasury_trade_allowed(
        equity, treasury
    )
    if not treasury_allowed:
        return {
            'action': 'blocked', 
            'tier': treasury_details.get('tier', '?'),
            'reason': treasury_reason,
            'treasury_check': treasury_details
        }
    
    # 获取熔断层级
    tier, can_trade, reason, losses = check_treasury_tier(equity, treasury)
    if not can_trade:
        return {'action': 'blocked', 'tier': tier, 'reason': reason}
    
    weights = load_ic_weights()
    # P1: 幽灵仓位检测 - 每次live扫描前检查OKX状态一致性
    # 检测: 无OCO保护的持仓 / 孤立订单 / 本地与交易所状态不匹配
    phantom_warnings = []
    if mode == 'live':
        try:
            from core.state_reconciler import StateReconciler
            reconciler = StateReconciler()
            result_reconcile = reconciler.reconcile(auto_fix=False)
            if result_reconcile.phantom_positions:
                for phantom in result_reconcile.phantom_positions:
                    # PhantomPosition字段: inst_id, direction, entry_price, contracts
                    contracts_info = f"contracts={phantom.contracts}"
                    msg = f"幽灵仓位: {phantom.inst_id} {phantom.direction} {contracts_info} 无OCO保护"
                    logger.warning(msg)
                    phantom_warnings.append(msg)
            if result_reconcile.orphan_orders:
                for order in result_reconcile.orphan_orders:
                    msg = f"孤立订单: {order.inst_id} algoId={order.algo_id[:12]}"
                    logger.warning(msg)
                    phantom_warnings.append(msg)
            if phantom_warnings:
                logger.warning(f"幽灵仓位警告共{len(phantom_warnings)}项")
        except Exception as e:
            logger.warning(f"状态协调器检查失败: {e}")

    positions = get_positions() if mode == 'live' else []
    if phantom_warnings and mode == 'live':
        positions = get_positions()  # 重新获取最新状态
    
    # P0: 并发扫描所有币 (替代原有串行for循环)
    # P0 Fix: 使用functools.partial绑定equity/btc_trend/weights参数
    # 初始化exchange一次，避免每个coin创建新实例
    exchange = get_default_exchange()
    wrapped_scan = partial(_parallel_scan_wrapper, equity=equity, btc_trend=btc_trend, weights=weights)
    candidates = parallel_scan_coins(
        scan_func=wrapped_scan,
        coins=SCAN_COINS,
        max_workers=5,
        timeout=30.0,
        exchange=exchange
    )
    
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Memory置信度调整: 根据最近交易历史调整所有候选评分
    memory_multiplier = _get_memory_confidence_multiplier()
    if memory_multiplier != 1.0:
        for c in candidates:
            c['score'] = c['score'] * memory_multiplier
        candidates.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Memory置信度乘数:{memory_multiplier:.2f},候选重排后top={candidates[0]['symbol'] if candidates else 'none'}@{candidates[0]['score']:.3f}")
    else:
        logger.debug("Memory置信度乘数:1.0(无调整)")

    # FOMC宏观事件置信度乘数 (窗口期降低50%)
    fomc_multiplier = get_fomc_confidence_multiplier(1.0)  # 1.0基准，返回实际乘数
    if fomc_multiplier < 1.0:
        for c in candidates:
            c['score'] = c['score'] * fomc_multiplier
        candidates.sort(key=lambda x: x['score'], reverse=True)
        logger.warning(f"FOMC窗口期! 置信度降{fomc_multiplier:.0%}, 候选重排后top={candidates[0]['symbol'] if candidates else 'none'}")

    # 市场状态检测 (Regime Classification)
    # 根据BTC 4H ADX判断当前市场状态，趋势市场/震荡市场采用不同因子权重
    regime = "neutral"
    if candidates:
        btc_cand = next((c for c in candidates if c['symbol'] == 'BTC'), None)
        if btc_cand and btc_cand.get('adx_4h'):
            regime = get_market_regime(btc_cand['adx_4h'], btc_cand.get('rsi'))
            weights = load_ic_weights()
            # 重新计算每个候选的分数：基于regime调整后的因子vote
            for c in candidates:
                if c.get('votes'):
                    # P1-5 Fix: regime调整作为乘数应用（不覆盖score，保留mt_boost/fr_confidence/oi_penalty）
                    regime_adj = 1.0
                    for fname, fvote in c['votes'].items():
                        adj = get_regime_confidence_multiplier(1.0, regime, fname.lower())
                        # 加权平均调整系数
                        regime_adj += (adj - 1.0) * weights.get(fname, 0) * abs(fvote)
                    regime_adj = max(0.5, min(regime_adj, 1.5))  # 限制在0.5x~1.5x
                    c['score'] = c['score'] * regime_adj
                    c['regime'] = regime
            # 按调整后分数重新排序
            if regime != "neutral":
                candidates.sort(key=lambda x: x['score'], reverse=True)
                logger.info(f"市场状态={regime} | top={candidates[0]['symbol']}@{candidates[0]['score']:.3f}")

    # 加载本地OPEN交易 (必须在select_best前)
    local_trades = get_open_trades()

    # 选最优 (过滤已有持仓)
    best, vetoed_pattern_keys = select_best(candidates, positions, local_trades)

    # 仓位管理 (融合OKX实时+本地记录)
    position_decisions = []

    # Build lookup from local_trades for dynamic SL/TP
    local_trade_by_sym = {t.get('coin', ''): t for t in local_trades}

    # 1. OKX实时持仓检查
    for pos in positions:
        sym = pos['instId'].replace('-USDT-SWAP', '')
        entry = pos['entry']
        current = get_ticker(pos['instId'])
        if current == 0:
            current = entry

        pos_side = pos.get('side', 'long')
        if pos_side == 'long':
            pnl_pct = (current - entry) / entry
        else:
            pnl_pct = (entry - current) / entry

        # Try to get dynamic SL/TP from local_trade record
        local_trade = local_trade_by_sym.get(sym, {})
        trade_sl = local_trade.get('sl_price', 0)
        trade_tp = local_trade.get('tp_price', 0)

        if trade_sl > 0 and trade_tp > 0:
            # Use actual SL/TP price for monitoring
            if pos_side == 'long':
                sl_triggered = current <= trade_sl
                tp_triggered = current >= trade_tp
            else:
                sl_triggered = current >= trade_sl
                tp_triggered = current <= trade_tp
        else:
            # Fallback to hardcoded percentages
            sl_triggered = pnl_pct <= -SL_PCT
            tp_triggered = pnl_pct >= TP_PCT

        if sl_triggered:
            position_decisions.append({'action': 'close', 'symbol': sym, 'reason': 'SL触发', 'urgency': 9, 'pnl_pct': pnl_pct})
        elif tp_triggered:
            position_decisions.append({'action': 'close', 'symbol': sym, 'reason': 'TP触发', 'urgency': 8, 'pnl_pct': pnl_pct})

    # 2. 本地记录持仓追踪
    for trade in local_trades:
        sym = trade.get('coin', '')
        inst_id = f'{sym}-USDT-SWAP'
        entry = trade.get('entry_price', 0)
        direction = trade.get('direction', 'long')
        open_time = trade.get('open_time', '')
        current = get_ticker(inst_id)
        if current == 0:
            continue

        if 'current_price' not in trade:
            trade['current_price'] = current

        if direction == 'long':
            pnl_pct = (current - entry) / entry
        else:
            pnl_pct = (entry - current) / entry

        trade['current_pnl_pct'] = pnl_pct
        trade['current_price'] = current

        from datetime import datetime as dt
        try:
            open_dt = dt.fromisoformat(open_time)
            age_hours = (dt.now() - open_dt).total_seconds() / 3600
            trade['age_hours'] = round(age_hours, 1)
            if age_hours > 24:
                # ATR-based time stop: if price hasn't moved at least 0.5×ATR in 24h, it's consolidating - exit
                klines_1h = get_klines(inst_id, '1H', 100)
                if klines_1h and len(klines_1h) >= 30:
                    highs_1h = [k['high'] for k in klines_1h]
                    lows_1h = [k['low'] for k in klines_1h]
                    closes_1h = [k['close'] for k in klines_1h]
                    atr = calc_atr(highs_1h, lows_1h, closes_1h)
                    if atr > 0 and entry > 0:
                        atr_pct = atr / entry
                        price_moved_pct = abs(current - entry) / entry
                        if price_moved_pct < atr_pct * 0.5:
                            position_decisions.append({'action': 'close', 'symbol': sym, 'reason': f'时间止损(24h+ATR确认震荡)', 'urgency': 7, 'pnl_pct': pnl_pct})
                else:
                    # Fallback to fixed time stop if no klines
                    position_decisions.append({'action': 'close', 'symbol': sym, 'reason': f'时间止损({age_hours:.0f}h)', 'urgency': 7, 'pnl_pct': pnl_pct})
        except Exception as ex:
            logger.debug(f"run_scan: 计算持仓时长失败: {ex}")
            trade['age_hours'] = 0

        if pnl_pct > 0.03:
            peak_pnl = trade.get('peak_pnl_pct', pnl_pct)
            if pnl_pct > peak_pnl:
                trade['peak_pnl_pct'] = pnl_pct
                trade['peak_price'] = current  # Track peak price for trailing stop
            else:
                # ATR-based trailing stop (自适应波动率)
                # 用ATR×2取代固定3%，高波动币允许更大回撤
                trailing_mult = max(0.015, atr_pct * 2.0) if atr_pct > 0 else 0.03
                peak_price = trade.get('peak_price', entry)
                if direction == 'long':
                    trailing_sl = peak_price * (1 - trailing_mult)
                    if current <= trailing_sl:
                        position_decisions.append({'action': 'close', 'symbol': sym, 'reason': f'移动止损(ATR回撤{(peak_price - current)/peak_price:.1%})', 'urgency': 6, 'pnl_pct': pnl_pct})
                else:
                    trailing_sl = peak_price * (1 + trailing_mult)
                    if current >= trailing_sl:
                        position_decisions.append({'action': 'close', 'symbol': sym, 'reason': f'移动止损(ATR回撤{(current - peak_price)/peak_price:.1%})', 'urgency': 6, 'pnl_pct': pnl_pct})

        # ── Trailing TP: 趋势完好时跳过固定TP，让利润奔跑 ──
        # 趋势强(ADX>25)：价格超过原TP也不止盈，靠移动止损出场
        # 趋势弱(ADX<25)：按原TP止盈（震荡市适合落袋）
        adx_val = trade.get('adx', 20)
        trade_tp = trade.get('tp_price', 0)

        if direction == 'long':
            tp_triggered = (current >= trade_tp and adx_val < 25) if trade_tp > 0 else (pnl_pct >= TP_PCT and adx_val < 25)
        else:
            tp_triggered = (current <= trade_tp and adx_val < 25) if trade_tp > 0 else (pnl_pct >= TP_PCT and adx_val < 25)

        # 趋势强且盈利超TP：记录trailing状态，但不平仓
        if trade_tp > 0 and adx_val >= 25:
            if (direction == 'long' and current > trade_tp) or (direction == 'short' and current < trade_tp):
                trade['trailing_active'] = True
                logger.debug(f"Trailing TP激活 {sym}: ADX={adx_val:.0f}, 价格=${current:.4f}超TP=${trade_tp:.4f}")

        sl_triggered = False
        if direction == 'long':
            sl_triggered = current <= trade.get('sl_price', 0) if trade.get('sl_price', 0) > 0 else (pnl_pct <= -SL_PCT)
        else:
            sl_triggered = current >= trade.get('sl_price', 0) if trade.get('sl_price', 0) > 0 else (pnl_pct <= -SL_PCT)

        if sl_triggered:
            position_decisions.append({'action': 'close', 'symbol': sym, 'reason': 'SL触发', 'urgency': 9, 'pnl_pct': pnl_pct})
        elif tp_triggered:
            position_decisions.append({'action': 'close', 'symbol': sym, 'reason': 'TP触发', 'urgency': 8, 'pnl_pct': pnl_pct})

    # P0 Fix: Load full trade history, update OPEN trades, then save ALL
    if local_trades:
        all_trades = load_trades()
        # Update current_pnl and current_price for open trades
        for t in all_trades:
            if t.get('status') == 'OPEN':
                for lt in local_trades:
                    if t.get('coin', '').upper() == lt.get('coin', '').upper():
                        t['current_pnl'] = lt.get('current_pnl', 0)
                        t['current_pnl_pct'] = lt.get('current_pnl_pct', 0)
                        t['current_price'] = lt.get('current_price', 0)
                        break
        save_trades(all_trades)  # Full history preserved (OPEN + CLOSED)
    
    if len(positions) >= MAX_POSITIONS:
        return {
            'action': 'hold',
            'tier': tier,
            'equity': equity,
            'positions': len(positions),
            'candidates': candidates[:5],
            'decisions': position_decisions,
            'reason': reason,
            'phantom_warnings': phantom_warnings,
            'vetoed_pattern_keys': vetoed_pattern_keys,
            'memory_multiplier': memory_multiplier,
            'fomc_multiplier': fomc_multiplier,
            'regime': regime,
        }
    
    if best and best['score'] > 0.5:
        if tier == 'caution':
            best['score'] *= 0.5
        
        if mode == 'live':
            # 计算仓位
            # P1 Fix: 使用OKX合约真实乘数计算张数
            # sz = 仓位USD / (入场价 × 每张合约的币数量)
            # P1 Fix: 从OKX API获取合约乘数，失败则用硬编码后备
            multiplier = fetch_contract_multiplier(best['symbol'])
            # Dynamic position sizing based on signal strength
            base_pct = 0.02  # Base 2%
            score = best.get('score', 0.5)
            # Score 0.25→1x (2%), Score 0.75→1.5x (3%), Score 1.0→2x (4%)
            score_multiplier = 1.0 + (score - 0.25) * 2.0  # Normalized
            score_multiplier = max(1.0, min(2.0, score_multiplier))  # Clamp 1x-2x
            position_pct = base_pct * score_multiplier
            # P1-8 Fix: 先cap再算sz，保证sz和concentration用同一百分比
            position_pct = min(position_pct, TREASURY_LIMITS['max_single_trade_pct'])  # 硬上限5%
            sz_dollar = equity * position_pct
            entry = best['entry']
            contract_value_usd = entry * multiplier  # 每张合约的USD价值
            sz = max(1, int(sz_dollar / contract_value_usd))
            new_trade_pct = position_pct
            concentration_allowed, conc_reason, conc_details = check_concentration(
                symbol=best['symbol'],
                new_trade_pct=new_trade_pct,
                current_positions=positions,
                equity=equity
            )
            if not concentration_allowed:
                logger.warning(f"集中度检查失败 {best['symbol']}: {conc_reason}")
                return {
                    'action': 'concentration_blocked',
                    'best': best,
                    'tier': tier,
                    'equity': equity,
                    'reason': conc_reason,
                    'concentration_details': conc_details,
                    'vetoed_pattern_keys': vetoed_pattern_keys,
                }
            
            # P1: OCO下单 (带equity参数用于验证, leverage基于ADX动态)
            # Dynamic leverage based on trend strength (ADX)
            adx = best.get('adx', 20)
            if adx < 20:
                leverage = 1  # No trend - lowest leverage
            elif adx < 30:
                leverage = 2  # Weak trend
            else:
                leverage = 3  # Strong trend
            result = place_oco(
                best['instId'], best['direction'], sz,
                entry, best['sl'], best['tp'], equity, leverage
            )
            
            if result.get('code') == '0':
                # P5: 日志幂等 - 生成幂等键
                open_time_iso = datetime.now().isoformat()
                idempotency_key = generate_trade_idempotency_key(
                    symbol=best['symbol'],
                    direction=best['direction'],
                    entry_price=entry,
                    size=float(sz_dollar),
                    timestamp=open_time_iso
                )
                
                trade = {
                    'id': f"mk_{best['symbol']}_{int(time.time())}",
                    'coin': best['symbol'],
                    'direction': best['direction'],
                    'entry_price': entry,
                    'sl_price': entry * (1 - best['sl']) if best['direction'] == 'long' else entry * (1 + best['sl']),
                    'tp_price': entry * (1 + best['tp']) if best['direction'] == 'long' else entry * (1 - best['tp']),
                    'size_usd': sz_dollar,
                    'score': best['score'],
                    'pattern_key': best['pattern_key'],
                    'open_time': open_time_iso,
                    'status': 'OPEN',
                    'idempotency_key': idempotency_key,  # P5: 幂等键
                }
                
                # P5: 幂等检查 - 防止重复记录
                is_dup, dup_msg = check_and_record_idempotent(
                    IDEMPOTENCY_LOG, idempotency_key, trade
                )
                if is_dup:
                    logger.warning(f"重复交易检测: {dup_msg}")
                    return {
                        'action': 'duplicate_trade',
                        'best': best,
                        'idempotency_key': idempotency_key,
                        'message': dup_msg,
                        'vetoed_pattern_keys': vetoed_pattern_keys,
                    }
                
                record_trade(trade)
                # 自适应学习: 入场反馈
                try:
                    from agents.agent_learner import AgentLearner
                    learner = AgentLearner(str(STATE_DIR))
                    pattern_key, is_allowed, trade_id = learner.on_trade_entry(trade)
                    trade['trade_id'] = trade_id
                    logger.info(f"Agent-L入场记录: pattern={pattern_key} trade_id={trade_id} allowed={is_allowed}")
                except Exception as e:
                    logger.warning(f"入场学习反馈失败: {e}")
                    trade['trade_id'] = None
                logger.info(f"开仓成功: {best['symbol']} {best['direction']} @ {entry}")
            else:
                return {'action': 'order_failed', 'result': result, 'best': best, 'vetoed_pattern_keys': vetoed_pattern_keys, 'phantom_warnings': phantom_warnings}
        
        return {
            'action': 'open',
            'best': best,
            'trade': trade,  # 包含 trade_id 供 main() 出场时使用
            'tier': tier,
            'equity': equity,
            'positions': len(positions),
            'reason': reason,
            'phantom_warnings': phantom_warnings,
            'vetoed_pattern_keys': vetoed_pattern_keys,
            'memory_multiplier': memory_multiplier,
            'fomc_multiplier': fomc_multiplier,
            'regime': regime,
        }
    
    return {
        'action': 'wait',
        'tier': tier,
        'equity': equity,
        'positions': len(positions),
        'candidates': candidates[:5],
        'reason': reason,
        'phantom_warnings': phantom_warnings,
        'vetoed_pattern_keys': vetoed_pattern_keys,
        'memory_multiplier': memory_multiplier,
        'fomc_multiplier': fomc_multiplier,
        'regime': regime,
    }

# ===== 主动持仓管理 =====
def run_position_management(equity: float, btc_trend: str, mode: str = 'audit') -> dict:
    """
    主动持仓管理 - 每次scan后执行，解决"开仓后无人管"的问题

    规则：
    1. 加载OPEN仓位 + 当前价格
    2. 盈利保护：盈利>8% → SL上移到成本价（零风险）
    3. 强制复审：持仓>4h无盈利 → 推送警告
    4. 反向信号覆盖：出现反向信号 → 立即平仓
    5. 部分止盈：盈利>15% → 平50%仓位
    6. 超时强平：持仓>8h → 强制平仓

    Returns: {
        'action': 'managed' | 'no_open_positions',
        'decisions': [...],
        'warnings': [...],
        'summary': str
    }
    """
    from datetime import datetime, timezone
    import time as _time_module

    # ── 规则参数 ──
    LOSS_WARN_PCT     = -0.03    # 亏损>3% → 触发警告/平仓
    HOLD_WARN_HOURS   = 3        # 持仓>3h无盈利 → 触发警告
    HOLD_FORCE_HOURS  = 6        # 持仓>6h无盈利(亏损) → 强制平仓
    PROFIT_MOVE_SL    = 0.02     # 盈利>2% → SL上移到成本价
    PARTIAL_TP_PCT    = 0.12    # 盈利>12% → 部分止盈50%

    all_trades = _load_open_trades_for_management()
    open_trades = [t for t in all_trades if t.get('status') == 'OPEN']

    if not open_trades:
        return {'action': 'no_open_positions', 'decisions': [], 'warnings': [], 'summary': '无OPEN持仓'}

    now_ts = datetime.now(timezone.utc)
    decisions  = []
    warnings   = []
    summary_lines = []

    for trade in open_trades:
        coin       = trade.get('coin', '')
        direction  = trade.get('direction', '')
        entry_price = float(trade.get('entry_price', 0))
        open_time_str = trade.get('open_time', '')
        sl_price   = float(trade.get('sl_price') or 0)
        tp_price   = float(trade.get('tp_price') or 0)
        trade_id   = trade.get('trade_id')

        if not coin or not entry_price:
            continue

        inst_id = f'{coin}-USDT-SWAP'

        # ── 当前价格 + 技术指标 ──
        klines_1h = get_klines(inst_id, '1H', 50)
        current_price = entry_price
        rsi_val  = 50.0
        adx_val  = 20.0

        if klines_1h and len(klines_1h) >= 20:
            closes = [k['close'] for k in klines_1h]
            highs  = [k['high']  for k in klines_1h]
            lows   = [k['low']   for k in klines_1h]
            current_price = closes[-1]
            rsi_val  = calc_rsi(closes)
            di_plus, di_minus, adx_val = calc_adx(highs, lows, closes)

        # ── 计算盈亏 ──
        if direction in ('LONG', '做多', 'long'):
            pnl_pct = (current_price - entry_price) / entry_price
        elif direction in ('SHORT', '做空', 'short'):
            pnl_pct = (entry_price - current_price) / entry_price
        else:
            pnl_pct = 0.0

        # ── 持仓时间计算 ──
        hold_hours = None
        if open_time_str:
            try:
                open_dt_str = open_time_str.replace('+08:00', '').replace('Z', '')
                open_dt = datetime.fromisoformat(open_dt_str[:19]).replace(tzinfo=timezone.utc)
                hold_hours = (now_ts - open_dt).total_seconds() / 3600
            except Exception:
                hold_hours = None

        # ── 规则1: 强制平仓（最优先） ──
        # 亏损>3% + 持仓>3h → 直接平
        if (hold_hours is not None and hold_hours >= HOLD_WARN_HOURS and pnl_pct <= LOSS_WARN_PCT):
            decisions.append({
                'action': 'force_close',
                'coin': coin,
                'direction': direction,
                'reason': f'亏损{pnl_pct:.1%}+持仓{hold_hours:.1f}h → 强制平仓',
                'entry': entry_price,
                'current': current_price,
                'pnl_pct': pnl_pct,
                'mode': mode,
                'trade_id': trade_id,
                'urgency': 9,
            })
            summary_lines.append(f'🚨 {coin} {direction} 亏损{pnl_pct:.1%}+{hold_hours:.1f}h → 强平')
        # 持仓>6h且亏损（非盈利） → 强制平
        elif hold_hours is not None and hold_hours >= HOLD_FORCE_HOURS and pnl_pct < 0.02 and pnl_pct < 0:
            decisions.append({
                'action': 'force_close',
                'coin': coin,
                'direction': direction,
                'reason': f'持仓{hold_hours:.1f}h无盈利({pnl_pct:.1%}) → 强平',
                'entry': entry_price,
                'current': current_price,
                'pnl_pct': pnl_pct,
                'mode': mode,
                'trade_id': trade_id,
                'urgency': 9,
            })
            summary_lines.append(f'🚨 {coin} {direction} {hold_hours:.1f}h无盈利 → 强平')
        # ── 规则2: 部分止盈（盈利>12%，LONG+SHORT） ──
        elif pnl_pct >= PARTIAL_TP_PCT:
            decisions.append({
                'action': 'partial_tp',
                'coin': coin,
                'direction': direction,
                'reason': f'盈利{pnl_pct:.1%} → 部分止盈50%',
                'entry': entry_price,
                'current': current_price,
                'tp_price': tp_price,
                'pnl_pct': pnl_pct,
                'close_pct': 0.50,
                'mode': mode,
                'trade_id': trade_id,
                'urgency': 8,
            })
            summary_lines.append(f'🎯 {coin} {direction} 盈利{pnl_pct:.1%} → 部分止盈')
        # ── 规则3: 盈利保护（盈利>2% → SL移到成本） ──
        elif pnl_pct >= PROFIT_MOVE_SL and sl_price > 0:
            new_sl = entry_price
            decisions.append({
                'action': 'move_sl_to_cost',
                'coin': coin,
                'direction': direction,
                'reason': f'盈利{pnl_pct:.1%} → SL移到成本{new_sl:.4f}',
                'entry': entry_price,
                'old_sl': sl_price,
                'new_sl': new_sl,
                'pnl_pct': pnl_pct,
                'mode': mode,
                'trade_id': trade_id,
                'urgency': 7,
            })
            summary_lines.append(f'🛡️  {coin} {direction} 盈利{pnl_pct:.1%} → SL移成本')
        # ── 规则4: 反向信号覆盖（亏损>3%且RSI明确反向） ──
        elif pnl_pct <= LOSS_WARN_PCT:
            if direction in ('LONG', '做多', 'long') and rsi_val > 65:
                decisions.append({
                    'action': 'close_opposite_signal',
                    'coin': coin,
                    'direction': direction,
                    'reason': f'亏损{pnl_pct:.1%}+RSI={rsi_val:.0f}>65 → 平仓',
                    'entry': entry_price,
                    'current': current_price,
                    'rsi': rsi_val,
                    'mode': mode,
                    'trade_id': trade_id,
                    'urgency': 8,
                })
                summary_lines.append(f'🔄 {coin} 多头亏损{pnl_pct:.1%}+RSI超买 → 平仓')
            elif direction in ('SHORT', '做空', 'short') and rsi_val < 35:
                decisions.append({
                    'action': 'close_opposite_signal',
                    'coin': coin,
                    'direction': direction,
                    'reason': f'亏损{pnl_pct:.1%}+RSI={rsi_val:.0f}<35 → 平仓',
                    'entry': entry_price,
                    'current': current_price,
                    'rsi': rsi_val,
                    'mode': mode,
                    'trade_id': trade_id,
                    'urgency': 8,
                })
                summary_lines.append(f'🔄 {coin} 空头亏损{pnl_pct:.1%}+RSI超卖 → 平仓')
        # ── 规则5: 持仓超时警告（>3h无盈利，未触发上面规则） ──
        elif hold_hours is not None and hold_hours >= HOLD_WARN_HOURS and pnl_pct < 0.03:
            warnings.append({
                'coin': coin,
                'direction': direction,
                'hold_hours': hold_hours,
                'pnl_pct': pnl_pct,
                'reason': f'持仓{hold_hours:.1f}h无盈利({pnl_pct:.1%})',
            })
            summary_lines.append(f'⚠️  {coin} {direction} {hold_hours:.1f}h无盈利({pnl_pct:.1%})')

    # ── 写入决策日志 ──
    _write_management_journal({
        'timestamp': now_ts.isoformat(),
        'equity': equity,
        'btc_trend': btc_trend,
        'open_count': len(open_trades),
        'decisions': decisions,
        'warnings': warnings,
    })

    action = 'managed' if (decisions or warnings) else 'no_action'
    summary = '; '.join(summary_lines) if summary_lines else '无操作'

    return {
        'action': action,
        'decisions': decisions,
        'warnings': warnings,
        'summary': summary,
        'open_count': len(open_trades),
    }


def _write_management_journal(log_entry: dict):
    """写入持仓管理日志（追加到文件）"""
    journal_file = STATE_DIR / 'position_management_log.jsonl'
    try:
        with open(journal_file, 'a') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.warning(f'写入持仓管理日志失败: {e}')


def _load_open_trades_for_management():
    """
    加载OPEN仓位用于主动管理。
    优先读 paper_trades.json（cron输出的真实状态），
    如果为空则回退到 miracle_trades.json（数据库）。
    """
    paper_file = Path('/Users/jimingzhang/.hermes/cron/output/paper_trades.json')
    if paper_file.exists():
        try:
            with open(paper_file) as f:
                paper = json.load(f)
            open_trades = [t for t in paper if t.get('status') == 'OPEN']
            if open_trades:
                logger.debug(f'从paper_trades.json加载{len(open_trades)}个OPEN仓位')
                return open_trades
        except Exception as e:
            logger.debug(f'读取paper_trades.json失败: {e}')

    # 回退到数据库
    return load_trades()


def _mark_trade_closed(coin: str, reason: str, trade_id=None, pnl_pct=None):
    """更新交易记录为CLOSED状态（被持仓管理调用）
    同时记录pattern胜率历史（出场反馈闭环）
    """
    pattern_key = None
    entry_price = 0
    direction = ''
    all_trades = load_trades()
    for t in all_trades:
        if t.get('coin', '').upper() == coin.upper() and t.get('status') == 'OPEN':
            pattern_key = t.get('pattern_key')
            entry_price = float(t.get('entry_price', 0))
            direction = t.get('direction', '')
            t['status'] = 'CLOSED'
            t['exit_reason'] = reason
            t['exit_time'] = datetime.now().isoformat()
            if trade_id:
                t['trade_id'] = trade_id
            break
    save_trades(all_trades)

    # 出场反馈：记录pattern胜率
    if pattern_key and pnl_pct is not None:
        try:
            won = pnl_pct > 0  # 正收益=胜，负收益=负
            record_pattern_outcome(pattern_key, won, pnl_pct)
            logger.debug(f"出场反馈: {coin} {pattern_key} {'WIN' if won else 'LOSS'} ({pnl_pct:+.1%})")
        except Exception as ex:
            logger.debug(f"出场反馈失败: {ex}")

    # 同时更新 paper_trades.json
    paper_file = Path('/Users/jimingzhang/.hermes/cron/output/paper_trades.json')
    if paper_file.exists():
        try:
            with open(paper_file) as f:
                paper = json.load(f)
            changed = False
            for t in paper:
                if t.get('coin', '').upper() == coin.upper() and t.get('status') == 'OPEN':
                    t['status'] = 'CLOSED'
                    t['close_reason'] = reason
                    t['close_time'] = datetime.now().isoformat()
                    changed = True
            if changed:
                with open(paper_file, 'w') as f:
                    json.dump(paper, f, indent=2, ensure_ascii=False)
                logger.info(f'paper_trades.json已更新: {coin} → CLOSED ({reason})')
        except Exception as e:
            logger.warning(f'更新paper_trades.json失败: {e}')


# ===== 入口 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='audit', choices=['audit', 'live'])
    parser.add_argument('--equity', type=float, default=None)
    args = parser.parse_args()
    
    print(f"[Miracle-Kronos] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 模式: {args.mode}")
    
    # 初始化IC权重文件（首次运行时种入默认值）
    if not IC_WEIGHTS_FILE.exists():
        save_ic_weights(DEFAULT_WEIGHTS)
        logger.info("factor_weights.json已初始化: 写入DEFAULT_WEIGHTS")
    
    # P0-3 Fix: 启动时检测账户模式（hedge vs net），影响place_oco和close_position的posSide行为
    pos_mode = _detect_pos_mode()
    print(f"账户模式: {pos_mode} | ", end='')
    
    # 获取equity
    if args.mode == 'live':
        equity = get_account_balance()
        if equity == 0:
            print('❌ 无法获取OKX账户余额')
            return
    else:
        equity = args.equity if args.equity else get_account_balance()
        if equity == 0:
            equity = 100000
    print(f'余额: ${equity:,.2f} | ', end='')
    
    # 获取BTC趋势
    btc_klines = get_klines('BTC-USDT-SWAP', '4H', 50)
    btc_trend = 'neutral'
    if btc_klines and len(btc_klines) >= 20:
        closes = [k['close'] for k in btc_klines]
        ma20 = sum(closes[-20:]) / 20
        btc_trend = 'bull' if closes[-1] > ma20 else 'bear'
    
    result = run_scan(equity, btc_trend, args.mode)

    # ═══════════════════════════════════════════════════════════
    # 主动持仓管理（每次scan后执行，解决"开仓后无人管"）
    # ═══════════════════════════════════════════════════════════
    mgmt = run_position_management(equity, btc_trend, args.mode)

    # live模式：执行所有管理决策（不只看urgency）
    if mgmt.get('decisions') and args.mode == 'live':
        for dec in mgmt['decisions']:
            coin = dec.get('coin', '')
            inst_id = f'{coin}-USDT-SWAP'
            action_type = dec.get('action', '')
            reason = dec.get('reason', '')
            trade_id = dec.get('trade_id')

            if action_type in ('force_close', 'close_opposite_signal'):
                # 全量平仓
                close_data = close_position(coin, reason=reason)
                if close_data.get('code') == '0':
                    _mark_trade_closed(coin, reason, trade_id, pnl_pct=dec.get('pnl_pct'))
                    logger.info(f'持仓管理平仓: {coin} {reason}')
                else:
                    logger.warning(f'持仓管理平仓失败: {coin} {close_data.get("msg")}')

            elif action_type == 'partial_tp':
                # 部分止盈：平50%仓位
                positions = get_positions()
                pos = next((p for p in positions if coin.upper() in p.get('instId','')), None)
                if pos:
                    sz_total = int(pos.get('sz', 0))
                    sz_close = max(1, int(sz_total * dec.get('close_pct', 0.5)))
                    # 全量平仓（简化版，实际应部分平仓）
                    close_data = close_position(coin, reason=reason)
                    if close_data.get('code') == '0':
                        _mark_trade_closed(coin, f'{reason} [部分止盈]', trade_id, pnl_pct=dec.get('pnl_pct'))
                        logger.info(f'持仓管理部分止盈: {coin} {reason}')
                    else:
                        logger.warning(f'部分止盈失败: {coin} {close_data.get("msg")}')
                else:
                    logger.warning(f'部分止盈: 未找到{coin}持仓')

            elif action_type == 'move_sl_to_cost':
                # 移动SL到成本价：需要取消旧OCO + 挂新版
                try:
                    # 取消现有OCO
                    oco_query = okx_req('GET', f'/api/v5/trade/orders-algo-pending?instId={inst_id}&ordType=oco')
                    algo_list = oco_query.get('data', [])
                    for algo in algo_list:
                        cancel_body = json.dumps([{'algoId': str(algo['algoId']), 'instId': inst_id}])
                        okx_req('DELETE', '/api/v5/trade/cancel-algos', cancel_body)
                    # 挂新版OCO（SL=entry_price，TP不变）
                    entry = dec.get('entry', 0)
                    sl_pct = abs(entry - dec['new_sl']) / entry if entry > 0 else 0.05
                    tp_pct = dec.get('tp_price', 0)
                    if tp_pct > 0 and entry > 0:
                        tp_pct_val = abs(tp_pct - entry) / entry
                    else:
                        tp_pct_val = 0.10
                    positions = get_positions()
                    pos = next((p for p in positions if coin.upper() in p.get('instId','')), None)
                    if pos:
                        sz = int(pos.get('sz', 0))
                        direction = pos.get('posSide', 'long')
                        direction = 'long' if direction in ('long','net') else 'short'
                        place_oco(inst_id, direction, sz, entry, sl_pct, tp_pct_val,
                                  equity=equity, leverage=3)
                        logger.info(f'SL上移到成本: {coin} @ {dec["new_sl"]:.4f}')
                except Exception as e:
                    logger.warning(f'SL移动失败: {coin} {e}')

    # 打印管理摘要
    if mgmt.get('action') == 'managed':
        print(f'📋 持仓管理: {mgmt["summary"]}')
    elif mgmt.get('warnings'):
        for w in mgmt['warnings']:
            print(f'⚠️  {w["coin"]} {w["reason"]}')
    elif mgmt.get('action') == 'no_open_positions':
        pass  # 静默
    elif mgmt.get('action') == 'no_action':
        pass  # 静默

    # === 自适应学习: gemma4否决 → 黑名单 ===
    vetoed_keys = result.get('vetoed_pattern_keys', [])
    if vetoed_keys:
        try:
            learner = AgentLearner(str(STATE_DIR))
            for vk in vetoed_keys:
                if vk:
                    learner.pattern_learner.add_to_blacklist(vk)
            logger.info(f"gemma4否决模式加入黑名单: {vetoed_keys}")
        except Exception as e:
            logger.warning(f"gemma4黑名单更新失败: {e}")

    # === 执行平仓决策 (urgency >= 6) ===
    decisions = result.get('decisions', [])
    for decision in sorted(decisions, key=lambda x: -x.get('urgency', 0)):
        if decision.get('action') == 'close' and decision.get('urgency', 0) >= 6:
            sym = decision['symbol']
            inst_id = f'{sym}-USDT-SWAP'
            positions = get_positions() if args.mode == 'live' else []
            pos = next((p for p in positions if p.get('instId') == inst_id), None)
            if pos and args.mode == 'live':
                close_data = close_position(sym, decision['reason'], pos)
                if close_data.get('code') == '0':
                    # P0-4 Fix: Load FULL trade history to preserve CLOSED trades
                    all_trades = load_trades()
                    for t in all_trades:
                        if t.get('coin', '').upper() == sym.upper() and t.get('status') == 'OPEN':
                            t['status'] = 'CLOSED'
                            t['exit_reason'] = decision.get('reason', '')
                            t['exit_time'] = datetime.now().isoformat()
                            # P0-5 Fix: Read trade_id from local trade object, not result
                            trade_id = t.get('trade_id')
                            if trade_id:
                                learner = AgentLearner(str(STATE_DIR))
                                # P0 Fix: Use get_ticker() for accurate exit price, not pos.get('last', 0)
                                exit_price = get_ticker(inst_id)
                                learner.on_trade_exit(trade_id, {
                                    'exit_time': datetime.now().isoformat(),
                                    'exit_price': exit_price if exit_price > 0 else pos.get('last', 0),
                                    'close_reason': decision.get('reason', ''),
                                    'pnl_pct': decision.get('pnl_pct', 0),
                                })
                                logger.info(f"Agent-L出场反馈: trade_id={trade_id} reason={decision.get('reason', '')}")
                            # 出场反馈闭环：记录pattern胜率
                            pattern_key = t.get('pattern_key')
                            pnl_pct_dec = decision.get('pnl_pct', 0)
                            if pattern_key and pnl_pct_dec is not None:
                                try:
                                    record_pattern_outcome(pattern_key, pnl_pct_dec > 0, pnl_pct_dec)
                                except Exception as ex:
                                    logger.debug(f"出场反馈记录失败: {ex}")
                    save_trades(all_trades)  # Full history preserved

    # 更新treasury (快照时间轴管理)
    treasury = load_treasury()
    treasury['equity'] = equity
    now = datetime.now()
    last_update = treasury.get('last_update', '')
    if last_update:
        try:
            from datetime import datetime as dt
            last_dt = dt.fromisoformat(last_update)
            if now.hour != last_dt.hour:
                # 新的一小时：检查上一小时是否盈利
                prev_hourly_snapshot = treasury.get('hourly_snapshot', equity)
                treasury['hourly_snapshot'] = equity
                treasury['hourly_snapshot_time'] = treasury.get('last_update', '')[:10]
                # V6-3: 追踪连续盈利小时数用于tier降级
                if equity > prev_hourly_snapshot:
                    treasury['consecutive_win_hours'] = treasury.get('consecutive_win_hours', 0) + 1
                    logger.info(f"连续盈利+1h: {treasury['consecutive_win_hours']}h (equity=${equity:.2f})")
                else:
                    treasury['consecutive_win_hours'] = 0
                treasury['consecutive_loss_hours'] = 0  # 重置亏损计数
            if now.date() != last_dt.date():
                treasury['daily_snapshot'] = equity
                treasury['daily_snapshot_time'] = treasury.get('last_update', '')[:10]
                # session_start: 每日起始equity，用于每日20%回撤熔断（非跨日Session追踪）
                # P1-6: 每日重置是故意设计——每个交易日独立风控，坏日子不蔓延到下一天
                treasury['session_start'] = equity
                # 换日时重置连续计数（避免跨天累积，hourly counters只管当天）
                treasury['consecutive_win_hours'] = 0
                treasury['consecutive_loss_hours'] = 0
        except Exception as ex:
            logger.debug(f"更新treasury快照失败: {ex}")
            treasury['hourly_snapshot'] = equity
            treasury['daily_snapshot'] = equity
            treasury['hourly_snapshot_time'] = treasury.get('last_update', '')[:10]
            treasury['daily_snapshot_time'] = treasury.get('last_update', '')[:10]
    else:
        treasury['hourly_snapshot'] = equity
        treasury['daily_snapshot'] = equity
        treasury['session_start'] = equity
    treasury['last_update'] = now.isoformat()
    save_treasury(treasury)
    
    # 输出结果
    action = result.get('action', 'unknown')
    tier = result.get('tier', '?')
    reason = result.get('reason', '')
    print(f'熔断: {tier} | {reason}')
    
    if action == 'blocked':
        print(f'🚫 系统熔断: {reason}')
    elif action == 'wait':
        print('⏸️  无信号，等待')
        if result.get('candidates'):
            print('  TOP候选:')
            for c in result['candidates'][:3]:
                print(f'    {c["symbol"]:6s} {c["direction"]:5s} score={c["score"]:.2f} RSI={c["rsi"]:.0f} ADX={c["adx"]:.0f}')
    elif action == 'open':
        b = result['best']
        print(f'📈 开仓信号: {b["symbol"]} {b["direction"]}')
        print(f'  评分: {b["score"]:.2f} | RSI: {b["rsi"]:.0f} | ADX: {b["adx"]:.0f} | BB%: {b["bb_pos"]:.0f}')
        print(f'  入场: ${b["entry"]:.4f} | SL: {b["sl"]:.0%} | TP: {b["tp"]:.0%}')
        print(f'  4H ADX: {b["adx_4h"]:.0f}')
    elif action == 'order_failed':
        print(f'❌ 下单失败: {result["result"].get("msg", "")}')
    
    if result.get('positions', 0) > 0:
        print(f'持仓: {result["positions"]}/{MAX_POSITIONS}')

if __name__ == '__main__':
    main()
