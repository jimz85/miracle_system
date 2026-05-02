#!/usr/bin/env python3
from __future__ import annotations

"""
Miracle Pilot驾驶舱 - 状态监控面板
====================================

对比Kronos kronos_pilot.py实现，包含:
1. 状态摘要 (Status Summary)
2. 持仓监控 (Position Monitoring)
3. 信号列表 (Signal List)
4. 风险仪表盘 (Risk Dashboard)

运行:
  python miracle_pilot.py              驾驶舱概览
  python miracle_pilot.py --status    纸质交易胜率统计
  python miracle_pilot.py --signals   实时信号列表
  python miracle_pilot.py --risk      风险仪表盘
  python miracle_pilot.py --positions 持仓监控
  python miracle_pilot.py --full      完整日报
  python miracle_pilot.py --log N     查看最近N行日志
"""


import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# 缓存目录
CACHE_DIR = Path.home() / '.hermes' / 'miracle' / 'output'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 状态文件
PAPER_TRADES_FILE = CACHE_DIR / 'paper_trades.json'
IC_WEIGHTS_FILE = CACHE_DIR / 'ic_weights.json'
PERFORMANCE_FILE = CACHE_DIR / 'performance.json'
SIGNALS_FILE = CACHE_DIR / 'signals.json'
RISK_STATE_FILE = CACHE_DIR / 'risk_state.json'

# ── 日志配置 ──────────────────────────────────────────────
_log_dir = SCRIPT_DIR / 'logs'
_log_dir.mkdir(exist_ok=True)
_pilot_logger = logging.getLogger('miracle_pilot')
_pilot_logger.setLevel(logging.DEBUG)
_pilot_logger.handlers.clear()

rf = RotatingFileHandler(
    _log_dir / 'miracle_pilot.log',
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)
rf.setLevel(logging.DEBUG)
rf.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
_pilot_logger.addHandler(rf)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
_pilot_logger.addHandler(sh)

def log_info(msg):
    _pilot_logger.info(msg)
    print(msg)

def log_warn(msg):
    _pilot_logger.warning(msg)
    print(f"⚠️  {msg}")

def log_error(msg):
    _pilot_logger.error(msg)
    print(f"❌ {msg}")

# ── 核心模块导入 ──────────────────────────────────────────
try:
    from agents.agent_risk import AccountState, AgentRisk, CircuitBreaker
    from agents.agent_signal import AgentSignal
    from core.data_fetcher import get_klines, get_ticker
    from miracle_core import (
        RiskMetrics,
        calc_factors,
        calc_trend_strength,
        format_trade_signal,
        get_ic_adjusted_weights,
        load_config,
    )
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    _pilot_logger.warning(f"核心模块导入失败: {e}，驾驶舱将以演示模式运行")


# ============================================================
# 1. 状态摘要 (Status Summary)
# ============================================================

def get_system_status() -> Dict[str, Any]:
    """
    获取系统整体状态摘要
    
    Returns:
        {
            'mode': str,           # 实盘/模拟盘
            'uptime': str,         # 运行时间
            'last_scan': str,     # 最后扫描时间
            'active_positions': int,
            'pending_signals': int,
            'system_health': str,  # healthy/warning/critical
            'account': {...},
            'circuit_breaker': {...}
        }
    """
    status = {
        'mode': 'SIMULATION',
        'uptime': 'unknown',
        'last_scan': 'never',
        'active_positions': 0,
        'pending_signals': 0,
        'system_health': 'unknown',
        'account': {},
        'circuit_breaker': {},
        'errors': []
    }
    
    # 检查配置
    if CORE_AVAILABLE:
        try:
            config = load_config()
            status['mode'] = config.get('mode', 'SIMULATION')
        except Exception as e:
            status['errors'].append(f"配置加载失败: {e}")
    
    # 检查日志文件获取最后扫描时间
    log_file = SCRIPT_DIR / 'miracle.log'
    if log_file.exists():
        try:
            lines = log_file.read_text(encoding='utf-8').strip().split('\n')
            if lines:
                # 尝试从最后一行提取时间
                last_line = lines[-1]
                status['last_scan'] = last_line[:19] if len(last_line) > 19 else last_line
        except Exception:
            logger.debug("pilot日志读取失败（非关键）")
            pass
    
    # 检查持仓
    trades = load_paper_trades()
    open_trades = [t for t in trades if t.get('status') == 'OPEN']
    status['active_positions'] = len(open_trades)
    
    # 检查信号
    signals = load_signals()
    status['pending_signals'] = len(signals)
    
    # 系统健康状态
    if status['active_positions'] >= 3:
        status['system_health'] = 'warning'
    elif status['errors']:
        status['system_health'] = 'critical'
    else:
        status['system_health'] = 'healthy'
    
    # 账户信息
    status['account'] = get_account_summary()
    
    # 熔断器状态
    status['circuit_breaker'] = get_circuit_breaker_status()
    
    return status


def get_account_summary() -> Dict[str, Any]:
    """获取账户摘要"""
    summary = {
        'equity': 0.0,
        'balance': 0.0,
        'unrealized_pnl': 0.0,
        'realized_pnl': 0.0,
        'total_pnl': 0.0,
        'margin_used': 0.0,
        'available_margin': 0.0,
        'win_rate': 0.0,
        'total_trades': 0
    }
    
    # 尝试从缓存读取
    equity_cache = SCRIPT_DIR / 'data' / 'last_equity.json'
    if equity_cache.exists():
        try:
            data = json.load(open(equity_cache))
            summary['equity'] = data.get('equity', 0.0)
        except Exception:
            logger.debug("pilot权益缓存读取失败")
            pass
    
    # 从交易记录计算
    trades = load_paper_trades()
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
    
    if closed_trades:
        wins = [t for t in closed_trades if (t.get('pnl', 0) or 0) > 0]
        summary['win_rate'] = len(wins) / len(closed_trades) * 100 if closed_trades else 0
        summary['total_trades'] = len(closed_trades)
        summary['realized_pnl'] = sum(t.get('pnl', 0) or 0 for t in closed_trades)
        summary['total_pnl'] = summary['realized_pnl']
    
    return summary


def get_circuit_breaker_status() -> Dict[str, Any]:
    """获取熔断器状态"""
    cb_status = {
        'daily_loss_triggered': False,
        'drawdown_triggered': False,
        'consecutive_loss_triggered': False,
        'loss_streak': 0,
        'cooldown_until': None,
        'can_trade': True,
        'reason': None
    }
    
    if not CORE_AVAILABLE:
        return cb_status
    
    try:
        config = load_config()
        cb = CircuitBreaker(config.get('risk', {}))
        
        # 检查交易记录获取连亏状态
        trades = load_paper_trades()
        closed = [t for t in trades if t.get('status') == 'CLOSED']
        
        # 计算最近亏损
        recent_losses = 0
        for t in reversed(closed[-10:]):
            if (t.get('pnl', 0) or 0) < 0:
                recent_losses += 1
            else:
                break
        
        cb_status['loss_streak'] = recent_losses
        
        # 检查是否可以交易
        account = get_account_summary()
        equity = account.get('equity', 0) or 1000  # 默认1000
        
        can_trade, reason, resume_time = cb.check_consecutive_losses(
            recent_losses, None
        )
        
        cb_status['can_trade'] = can_trade
        cb_status['reason'] = reason
        if resume_time:
            cb_status['cooldown_until'] = resume_time.isoformat()
        
        # 检查回撤
        if equity > 0 and account.get('total_pnl', 0) < 0:
            drawdown = abs(account['total_pnl']) / equity * 100
            cb_status['drawdown_triggered'] = drawdown > config.get('risk', {}).get('max_drawdown_pct', 20)
        
    except Exception as e:
        cb_status['reason'] = f"检查失败: {e}"
    
    return cb_status


def print_status_summary():
    """打印状态摘要"""
    status = get_system_status()
    
    print("\n" + "=" * 70)
    print("🖥️  Miracle Pilot 驾驶舱 - 系统状态")
    print("=" * 70)
    
    # 模式和时间
    mode_icon = "🟢" if status['mode'] == 'LIVE' else "🟡"
    health_icon = {"healthy": "✅", "warning": "⚠️", "critical": "🔴", "unknown": "❓"}.get(
        status['system_health'], "❓"
    )
    
    print(f"\n{mode_icon} 模式: {status['mode']} | {health_icon} 系统状态: {status['system_health'].upper()}")
    print(f"📊 最后扫描: {status['last_scan']}")
    
    # 账户信息
    acc = status['account']
    print("\n💰 账户信息:")
    print(f"   权益: ${acc.get('equity', 0):,.2f}")
    print(f"   总盈亏: ${acc.get('total_pnl', 0):+.2f}")
    print(f"   胜率: {acc.get('win_rate', 0):.1f}% ({acc.get('total_trades', 0)}笔)")
    
    # 持仓信息
    print("\n📈 持仓状态:")
    print(f"   活跃持仓: {status['active_positions']}/3")
    print(f"   待处理信号: {status['pending_signals']}")
    
    # 熔断器状态
    cb = status['circuit_breaker']
    cb_icon = "✅" if cb['can_trade'] else "🔴"
    print(f"\n🛡️ 熔断器 {cb_icon}:")
    if cb['can_trade']:
        print(f"   状态: 正常 (连亏: {cb['loss_streak']}笔)")
    else:
        print(f"   状态: 触发 ({cb['reason']})")
        if cb['cooldown_until']:
            print(f"   冷却至: {cb['cooldown_until']}")
    
    # 错误信息
    if status['errors']:
        print("\n❌ 错误:")
        for err in status['errors']:
            print(f"   - {err}")
    
    print("\n" + "=" * 70)


# ============================================================
# 2. 持仓监控 (Position Monitoring)
# ============================================================

def load_paper_trades() -> List[Dict]:
    """加载纸质交易记录"""
    if PAPER_TRADES_FILE.exists():
        try:
            with open(PAPER_TRADES_FILE) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return []


def get_positions_detail() -> List[Dict[str, Any]]:
    """获取持仓详细信息"""
    trades = load_paper_trades()
    open_trades = [t for t in trades if t.get('status') == 'OPEN']
    
    positions = []
    for t in open_trades:
        pos = {
            'symbol': t.get('symbol', t.get('coin', 'UNKNOWN')),
            'direction': t.get('direction', 'LONG'),
            'entry_price': t.get('entry_price', 0),
            'current_price': 0,
            'size': t.get('size', t.get('contracts', 0)),
            'unrealized_pnl': 0,
            'pnl_pct': 0,
            'hold_time': 'unknown',
            'stop_loss': t.get('stop_loss', 0),
            'take_profit': t.get('take_profit', 0),
            'leverage': t.get('leverage', 1),
            'opened_at': t.get('opened_at', t.get('open_time', '')),
        }
        
        # 尝试获取当前价格
        symbol = pos['symbol'].replace('-USDT', '')
        try:
            ticker = get_ticker(symbol)
            if ticker:
                pos['current_price'] = ticker.get('last', 0)
        except Exception:
            logger.warning(f"获取{pos.get('symbol', '?')} ticker失败，当前价格=0")
            pass
        
        # 计算未实现盈亏
        if pos['entry_price'] > 0 and pos['current_price'] > 0:
            direction = pos['direction'].lower()
            if direction == 'long':
                pnl = (pos['current_price'] - pos['entry_price']) / pos['entry_price']
            else:
                pnl = (pos['entry_price'] - pos['current_price']) / pos['entry_price']
            pos['pnl_pct'] = pnl * 100 * pos['leverage']
            pos['unrealized_pnl'] = pnl * pos['size'] * pos['leverage']
        
        # 计算持仓时间
        if pos['opened_at']:
            try:
                opened = datetime.fromisoformat(pos['opened_at'].replace('Z', '+00:00'))
                now = datetime.now()
                delta = now - opened
                hours = delta.total_seconds() / 3600
                if hours < 1:
                    pos['hold_time'] = f"{int(delta.total_seconds() / 60)}分钟"
                else:
                    pos['hold_time'] = f"{hours:.1f}小时"
            except Exception:
                logger.debug(f"持仓时间解析失败: {pos.get('opened_at', '?')}")
                pass
        
        positions.append(pos)
    
    return positions


def print_positions():
    """打印持仓监控"""
    positions = get_positions_detail()
    
    print("\n" + "=" * 70)
    print("📊 Miracle Pilot 驾驶舱 - 持仓监控")
    print("=" * 70)
    
    if not positions:
        print("\n🟡 当前无活跃持仓")
        print("\n" + "=" * 70)
        return
    
    print(f"\n📈 活跃持仓 ({len(positions)}/3):\n")
    print(f"{'币种':<8} {'方向':<6} {'入场价':<12} {'当前价':<12} {'杠杆':<6} {'未实现盈亏':<14} {'持仓时间':<10}")
    print("-" * 70)
    
    total_pnl = 0
    for p in positions:
        direction_icon = "🟢" if p['direction'].lower() == 'long' else "🔴"
        pnl_color = "🟢" if p['pnl_pct'] >= 0 else "🔴"
        pnl_str = f"${p['unrealized_pnl']:+.2f} ({pnl_color}{p['pnl_pct']:+.1f}%)"
        
        print(f"{p['symbol']:<8} {direction_icon}{p['direction']:<4} "
              f"${p['entry_price']:<11.4f} ${p['current_price']:<11.4f} "
              f"{p['leverage']}x    {pnl_str:<12} {p['hold_time']:<10}")
        
        total_pnl += p['unrealized_pnl']
    
    print("-" * 70)
    print(f"{'总计':<50} ${total_pnl:+.2f}")
    
    # 风险预警
    print("\n⚠️ 风险预警:")
    for p in positions:
        warnings = []
        
        # 止损检查
        if p['stop_loss'] > 0:
            if p['direction'].lower() == 'long' and p['current_price'] <= p['stop_loss']:
                warnings.append("接近止损!")
            elif p['direction'].lower() == 'short' and p['current_price'] >= p['stop_loss']:
                warnings.append("接近止损!")
        
        # 持仓时间过长
        if '小时' in p['hold_time']:
            hours = float(p['hold_time'].replace('小时', ''))
            if hours > 48:
                warnings.append("持仓超过48小时")
        
        if warnings:
            print(f"   {p['symbol']}: {' | '.join(warnings)}")
    
    print("\n" + "=" * 70)


# ============================================================
# 3. 信号列表 (Signal List)
# ============================================================

def load_signals() -> List[Dict]:
    """加载最新信号"""
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return data.get('signals', [])
        except (OSError, json.JSONDecodeError):
            pass
    return []


def generate_live_signals(symbols: List[str] = None) -> List[Dict[str, Any]]:
    """
    生成实时信号列表
    
    对比Kronos的generate_signals()实现
    """
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL", "AVAX", "DOGE", "DOT", "ADA", "XRP"]
    
    signals = []
    
    if not CORE_AVAILABLE:
        return signals
    
    try:
        config = load_config()
        signal_gen = AgentSignal(config)
        
        for symbol in symbols:
            try:
                # 获取数据
                klines = get_klines(symbol, "1h", 100)
                if not klines or len(klines) < 50:
                    continue
                
                prices = [k["close"] for k in klines]
                highs = [k["high"] for k in klines]
                lows = [k["low"] for k in klines]
                volumes = [k["volume"] for k in klines]
                
                price_data = {
                    "prices": prices,
                    "highs": highs,
                    "lows": lows,
                    "volumes": volumes
                }
                
                # 生成信号
                signal = signal_gen.process_intel(symbol, price_data, {}, price_data_4h=None)
                
                direction = signal.get("direction", "wait")
                if direction in ["long", "short"]:
                    conf = signal.get("confidence", 0)
                    trend = signal.get("trend_strength", 0)
                    
                    # 计算信号质量分数 (0-100%)
                    # 置信度范围0-1，趋势强度范围0-100
                    # 质量 = 置信度×60% + 趋势强度×40%（归一化到0-100）
                    quality = conf * 60 + trend * 0.4
                    
                    signals.append({
                        'symbol': symbol,
                        'direction': direction.upper(),
                        'direction_icon': "🟢" if direction == "long" else "🔴",
                        'confidence': conf,
                        'trend_strength': trend,
                        'quality_score': quality / 100,  # 存储为0-1方便显示
                        'entry_price': signal.get('entry_price', 0),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit': signal.get('take_profit', 0),
                        'factors': signal.get('factors', {}),
                        'rr_ratio': signal.get('rr_ratio', 0),
                        'generated_at': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                _pilot_logger.debug(f"信号生成失败 {symbol}: {e}")
                continue
                
    except Exception as e:
        _pilot_logger.warning(f"信号生成异常: {e}")
    
    # 按质量排序
    signals.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # 保存到缓存
    try:
        with open(SIGNALS_FILE, 'w') as f:
            json.dump({'signals': signals, 'updated': datetime.now().isoformat()}, f, indent=2)
    except Exception:
        logger.debug("pilot信号缓存写入失败")
        pass
    
    return signals


def print_signals():
    """打印信号列表"""
    print("\n" + "=" * 70)
    print("📡 Miracle Pilot 驾驶舱 - 信号列表")
    print("=" * 70)
    
    # 生成新信号
    print("\n🔄 正在生成实时信号...")
    signals = generate_live_signals()
    
    if not signals:
        print("\n🟡 当前无高置信度信号")
        print("\n" + "=" * 70)
        return
    
    print(f"\n📡 检测到 {len(signals)} 个信号:\n")
    print(f"{'#':<3} {'币种':<8} {'方向':<6} {'置信度':<10} {'趋势':<8} {'质量':<8} {'入场价':<12} {'止损':<12} {'止盈':<12}")
    print("-" * 70)
    
    for i, sig in enumerate(signals, 1):
        conf_bar = '█' * int(sig['confidence'] * 10) + '░' * (10 - int(sig['confidence'] * 10))
        trend_bar = '█' * int(sig['trend_strength'] / 10) + '░' * (10 - int(sig['trend_strength'] / 10))
        quality_bar = '█' * int(sig['quality_score'] * 10) + '░' * (10 - int(sig['quality_score'] * 10))
        
        print(f"{i:<3} {sig['symbol']:<8} {sig['direction_icon']}{sig['direction']:<4} "
              f"{sig['confidence']:.0%}{conf_bar} {sig['trend_strength']:.0f}{trend_bar} "
              f"{sig['quality_score']:.0%}{quality_bar}")
        print(f"     {'入场':<6} ${sig['entry_price']:<11.4f} SL:${sig['stop_loss']:<11.4f} TP:${sig['take_profit']:<11.4f}")
        
        # 显示主要因子
        if sig.get('factors'):
            fac = sig['factors']
            fac_str = []
            if fac.get('rsi'):
                fac_str.append(f"RSI={fac['rsi']:.0f}")
            if fac.get('adx'):
                fac_str.append(f"ADX={fac['adx']:.0f}")
            if fac.get('macd_direction'):
                fac_str.append(f"MACD={fac['macd_direction']}")
            if fac_str:
                print(f"     因子: {' | '.join(fac_str)}")
        
        print()
    
    # 信号统计
    long_signals = [s for s in signals if s['direction'] == 'LONG']
    short_signals = [s for s in signals if s['direction'] == 'SHORT']
    
    print("-" * 70)
    print(f"📊 信号统计: 🟢做多{len(long_signals)} | 🔴做空{len(short_signals)}")
    
    if long_signals and short_signals:
        print("⚠️  多空信号同时存在，市场可能处于震荡状态")
    
    print("\n" + "=" * 70)


# ============================================================
# 4. 风险仪表盘 (Risk Dashboard)
# ============================================================

def get_risk_metrics() -> Dict[str, Any]:
    """
    获取风险指标仪表盘
    对比Kronos的get_performance_stats()实现
    """
    metrics = {
        'var_95': 0.0,           # 95% VaR
        'cvar_95': 0.0,          # 95% CVaR
        'max_drawdown': 0.0,     # 最大回撤
        'sharpe_ratio': 0.0,     # Sharpe比率
        'sortino_ratio': 0.0,    # Sortino比率
        'win_rate': 0.0,         # 胜率
        'avg_win': 0.0,          # 平均盈利
        'avg_loss': 0.0,         # 平均亏损
        'wlr': 0.0,              # 盈亏比
        'total_trades': 0,       # 总交易数
        'open_positions': 0,      # 开放仓位
        'total_pnl': 0.0,        # 总盈亏
        'equity': 0.0,           # 当前权益
        'risk_level': 'LOW',     # 风险等级
    }
    
    # 加载交易记录
    trades = load_paper_trades()
    closed = [t for t in trades if t.get('status') == 'CLOSED']
    open_pos = [t for t in trades if t.get('status') == 'OPEN']
    
    metrics['total_trades'] = len(closed)
    metrics['open_positions'] = len(open_pos)
    
    if not closed:
        return metrics
    
    # 计算收益序列
    returns = []
    for t in closed:
        pnl_pct = t.get('result_pct', t.get('pnl_pct', 0))
        if pnl_pct:
            returns.append(pnl_pct / 100)  # 转换为小数
    
    if returns:
        # 计算风险指标
        metrics['var_95'] = abs(RiskMetrics.calculate_var(returns, 0.95)) * 100
        metrics['cvar_95'] = abs(RiskMetrics.calculate_cvar(returns, 0.95)) * 100
        metrics['max_drawdown'] = RiskMetrics.calculate_max_drawdown(returns) * 100
        metrics['sharpe_ratio'] = RiskMetrics.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = RiskMetrics.calculate_sortino_ratio(returns)
    
    # 计算交易统计
    wins = [t for t in closed if (t.get('pnl', 0) or 0) > 0]
    losses = [t for t in closed if (t.get('pnl', 0) or 0) <= 0]
    
    metrics['win_rate'] = len(wins) / len(closed) * 100 if closed else 0
    
    if wins:
        metrics['avg_win'] = sum(t.get('pnl', 0) or 0 for t in wins) / len(wins)
    if losses:
        metrics['avg_loss'] = abs(sum(t.get('pnl', 0) or 0 for t in losses) / len(losses))
    
    if metrics['avg_loss'] > 0:
        metrics['wlr'] = metrics['avg_win'] / metrics['avg_loss']
    
    metrics['total_pnl'] = sum(t.get('pnl', 0) or 0 for t in closed)
    
    # 获取权益
    acc = get_account_summary()
    metrics['equity'] = acc.get('equity', 0) or 1000
    
    # 风险等级评估
    risk_score = 0
    if metrics['max_drawdown'] > 15:
        risk_score += 3
    elif metrics['max_drawdown'] > 10:
        risk_score += 2
    elif metrics['max_drawdown'] > 5:
        risk_score += 1
    
    if metrics['win_rate'] < 40:
        risk_score += 2
    elif metrics['win_rate'] < 50:
        risk_score += 1
    
    if metrics['wlr'] < 1:
        risk_score += 2
    elif metrics['wlr'] < 1.5:
        risk_score += 1
    
    if metrics['open_positions'] >= 3:
        risk_score += 1
    
    if risk_score >= 6:
        metrics['risk_level'] = 'CRITICAL'
    elif risk_score >= 4:
        metrics['risk_level'] = 'HIGH'
    elif risk_score >= 2:
        metrics['risk_level'] = 'MEDIUM'
    else:
        metrics['risk_level'] = 'LOW'
    
    return metrics


def print_risk_dashboard():
    """打印风险仪表盘"""
    print("\n" + "=" * 70)
    print("📉 Miracle Pilot 驾驶舱 - 风险仪表盘")
    print("=" * 70)
    
    # 获取IC权重
    print("\n⚙️ IC自适应权重:")
    try:
        weights = get_ic_adjusted_weights()
        if weights:
            factor_labels = {
                'price_momentum': '价格动量',
                'news_sentiment': '新闻情绪',
                'onchain': '链上数据',
                'wallet': '钱包数据'
            }
            for factor, weight in weights.items():
                label = factor_labels.get(factor, factor)
                bar = '█' * int(weight * 30) + '░' * (30 - int(weight * 30))
                print(f"   {label:<12} {weight:.1%} {bar}")
        else:
            print("   权重数据不足，使用基准配置")
    except Exception as e:
        print(f"   ⚠️ 权重获取失败: {e}")
    
    # 风险指标
    metrics = get_risk_metrics()
    
    print("\n📊 风险指标:")
    print(f"   总交易: {metrics['total_trades']}笔 | 开放仓位: {metrics['open_positions']}/3")
    print(f"   总盈亏: ${metrics['total_pnl']:+.2f} | 权益: ${metrics['equity']:,.2f}")
    
    print("\n📉 风险度量:")
    risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🔴🔴"}.get(
        metrics['risk_level'], "❓"
    )
    print(f"   风险等级: {risk_icon} {metrics['risk_level']}")
    
    var_color = "🟢" if metrics['var_95'] < 5 else ("🟡" if metrics['var_95'] < 10 else "🔴")
    print(f"   VaR(95%):   {var_color} {metrics['var_95']:.2f}% (可能在1天内损失)")
    
    dd_color = "🟢" if metrics['max_drawdown'] < 10 else ("🟡" if metrics['max_drawdown'] < 20 else "🔴")
    print(f"   最大回撤:   {dd_color} {metrics['max_drawdown']:.2f}%")
    
    print(f"   Sharpe:     {metrics['sharpe_ratio']:.2f} (最优>1.0)")
    print(f"   Sortino:    {metrics['sortino_ratio']:.2f} (最优>1.5)")
    
    print("\n🎯 交易绩效:")
    wr_color = "🟢" if metrics['win_rate'] >= 50 else ("🟡" if metrics['win_rate'] >= 40 else "🔴")
    print(f"   胜率:       {wr_color} {metrics['win_rate']:.1f}%")
    print(f"   平均盈利:   ${metrics['avg_win']:+.2f}")
    print(f"   平均亏损:   ${metrics['avg_loss']:.2f}")
    
    wlr_color = "🟢" if metrics['wlr'] >= 2 else ("🟡" if metrics['wlr'] >= 1 else "🔴")
    print(f"   盈亏比:     {wlr_color} {metrics['wlr']:.2f}:1")
    
    # 熔断器状态
    cb = get_circuit_breaker_status()
    print("\n🛡️ 熔断器状态:")
    if cb['can_trade']:
        print(f"   ✅ 正常 (连亏: {cb['loss_streak']}笔)")
    else:
        print(f"   🔴 触发 - {cb['reason']}")
        if cb['cooldown_until']:
            print(f"   ⏰ 冷却至: {cb['cooldown_until']}")
    
    # 风控建议
    print("\n💡 风控建议:")
    suggestions = []
    
    if metrics['max_drawdown'] > 20:
        suggestions.append("⚠️ 最大回撤过高，考虑降低仓位")
    if metrics['win_rate'] < 40:
        suggestions.append("⚠️ 胜率偏低，检查策略有效性")
    if metrics['wlr'] < 1:
        suggestions.append("⚠️ 盈亏比<1，止损可能过紧")
    if metrics['open_positions'] >= 3:
        suggestions.append("⚠️ 仓位已满，等待信号出场")
    if cb['loss_streak'] >= 2:
        suggestions.append(f"⚠️ 连亏{cb['loss_streak']}笔，注意风险")
    
    if not suggestions:
        suggestions.append("✅ 系统运行正常，无特殊建议")
    
    for s in suggestions:
        print(f"   {s}")
    
    print("\n" + "=" * 70)


# ============================================================
# 5. 完整日报 (Full Report)
# ============================================================

def run_full_report():
    """运行完整报告，对比Kronos的run_full_report()"""
    print("\n" + "=" * 70)
    print("📋 Miracle Pilot 驾驶舱 - 完整日报")
    print(f"   生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 状态摘要
    print_status_summary()
    
    # 2. 持仓监控
    print_positions()
    
    # 3. 信号列表
    print_signals()
    
    # 4. 风险仪表盘
    print_risk_dashboard()
    
    # 底部信息
    print("\n" + "=" * 70)
    print("Miracle Pilot 驾驶舱 | 对比Kronos kronos_pilot.py实现")
    print("=" * 70)


# ============================================================
# 6. 查看日志
# ============================================================

def show_log(n: int = 20):
    """查看最近N行日志"""
    log_file = _log_dir / 'miracle_pilot.log'
    
    if not log_file.exists():
        print(f"📄 无日志文件: {log_file}")
        return
    
    try:
        lines = log_file.read_text(encoding='utf-8').strip().split('\n')
        print(f"\n📄 最近{min(n, len(lines))}行日志:")
        print("-" * 70)
        for line in lines[-n:]:
            print(line)
    except Exception as e:
        print(f"读取日志失败: {e}")


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Miracle Pilot 驾驶舱")
    parser.add_argument('--status', action='store_true', help='状态摘要')
    parser.add_argument('--positions', action='store_true', help='持仓监控')
    parser.add_argument('--signals', action='store_true', help='信号列表')
    parser.add_argument('--risk', action='store_true', help='风险仪表盘')
    parser.add_argument('--full', action='store_true', help='完整日报')
    parser.add_argument('--log', type=int, nargs='?', const=20, help='查看日志')
    
    args = parser.parse_args()
    
    # 默认显示状态摘要
    if args.status:
        print_status_summary()
    elif args.positions:
        print_positions()
    elif args.signals:
        print_signals()
    elif args.risk:
        print_risk_dashboard()
    elif args.full:
        run_full_report()
    elif args.log is not None:
        show_log(args.log)
    else:
        # 默认：完整驾驶舱概览
        run_full_report()
