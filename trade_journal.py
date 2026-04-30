"""
Trade Journal Module
====================
Extracted from miracle_kronos.py (Lines 889-933)
Manages local trade records, PnL tracking, and trade history.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.kronos_utils import atomic_write_json

logger = logging.getLogger('miracle_kronos')

STATE_DIR = Path(__file__).parent / 'data'
STATE_DIR.mkdir(exist_ok=True)
TRADES_FILE = STATE_DIR / 'miracle_trades.json'


def get_open_trades():
    """获取本地记录的OPEN交易"""
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE) as f:
                all_trades = json.load(f)
            return [t for t in all_trades if t.get('status') == 'OPEN']
        except Exception as ex:
            logger.debug(f"get_open_trades: 读取失败，返回空列表: {ex}")
    return []

def update_trade_pnl(trade, current_price):
    """更新交易浮动盈亏"""
    entry = trade.get('entry_price', 0)
    direction = trade.get('direction', 'long')
    if entry == 0 or current_price == 0:
        return trade
    if direction == 'long':
        pnl_pct = (current_price - entry) / entry
    else:
        pnl_pct = (entry - current_price) / entry
    trade['current_pnl_pct'] = pnl_pct
    trade['current_price'] = current_price
    return trade

def load_trades():
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE) as f:
                return json.load(f)
        except Exception as ex:
            logger.debug(f"load_trades: 读取失败，返回空列表: {ex}")
    return []

def save_trades(trades):
    atomic_write_json(TRADES_FILE, trades)

def record_trade(trade):
    trades = load_trades()
    trades.append(trade)
    if len(trades) > 1000:
        trades = trades[-500:]
    save_trades(trades)


