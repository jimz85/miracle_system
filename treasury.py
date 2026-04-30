"""
Treasury State Manager
=====================
Extracted from miracle_kronos.py (Lines 454-495)
Manages equity tracking, tier state, and consecutive win/loss hours.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger('miracle_kronos')

STATE_DIR = Path(__file__).parent / 'data'
STATE_DIR.mkdir(exist_ok=True)
TREASURY_FILE = STATE_DIR / 'miracle_treasury.json'


def load_treasury():
    if TREASURY_FILE.exists():
        try:
            with open(TREASURY_FILE) as f:
                return json.load(f)
        except Exception as ex:
            logger.debug(f"load_treasury: 读取失败，使用默认状态: {ex}")
    from datetime import date, datetime
    now = datetime.now().isoformat()
    return {
        'equity': 100000, 'hourly_snapshot': 100000, 'hourly_snapshot_time': now[:10],
        'daily_snapshot': 100000, 'daily_snapshot_time': now[:10],
        'tier': 'normal', 'consecutive_loss_hours': 0,
        'consecutive_win_hours': 0,
        'gemma_consecutive_failures': 0,  # Gemma tier upgrade tracking
        'last_update': now
    }

def save_treasury(state):
    # P0 Fix: 确保 daily_snapshot 在新的一天被重置
    from datetime import date as date_cls
    today = str(date_cls.today())
    ds_time = state.get('daily_snapshot_time', '')
    if ds_time and not ds_time.startswith(today):
        state['daily_snapshot'] = state.get('equity', state.get('daily_snapshot'))
        state['daily_snapshot_time'] = state.get('last_update', '')[:10]

    # V6-3 Fix: Treasury tier 降级机制——连续盈利时逐步降级
    current_tier = state.get('tier', 'normal')
    if current_tier != 'normal':
        # 追踪连续盈利小时数（由调用方在record_trade_outcome中更新）
        consecutive_win_hours = state.get('consecutive_win_hours', 0)
        if consecutive_win_hours >= 3:
            # 连续3小时盈利，降一级
            tier_order = ['normal', 'caution', 'critical', 'suspended']
            idx = tier_order.index(current_tier) if current_tier in tier_order else 0
            if idx > 0:
                old_tier = state['tier']
                state['tier'] = tier_order[idx - 1]
                state['consecutive_win_hours'] = 0  # 重置计数器
                logger.info(f"Tier降级: {old_tier}→{state['tier']} (连续{consecutive_win_hours}h盈利)")

    # Inline atomic write to avoid importing slow core.kronos_utils
    path = Path(TREASURY_FILE)
    fd, tmp = tempfile.mkstemp(suffix='.json.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

