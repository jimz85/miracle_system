#!/usr/bin/env python3
"""持仓主动管理 Cron — 每3分钟执行一次"""
import sys, os, json, threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['OKX_FLAG'] = '1'

result = {}
error_occurred = None

def worker():
    global result, error_occurred
    try:
        from miracle_kronos import run_position_management, get_account_balance, get_positions
        equity = get_account_balance()
        print(f'equity={equity}')
        if equity <= 0:
            equity = 67000
        result = run_position_management(equity=equity, btc_trend='neutral', mode='live')
    except Exception as e:
        import traceback
        error_occurred = str(e) + '\n' + traceback.format_exc()

t = threading.Thread(target=worker, daemon=True)
t.start()
t.join(timeout=50)

if error_occurred:
    print(f'pos_mgmt error: {error_occurred}')
    sys.exit(0)
elif t.is_alive():
    print('[SILENT]')
    sys.exit(0)

decisions = result.get('decisions', [])
warnings = result.get('warnings', [])

if not decisions and not warnings:
    print('[SILENT]')
    sys.exit(0)

if not decisions:
    for w in warnings:
        print(f'warn: {w.get("coin","?")} {w.get("reason","")}')
    sys.exit(0)

from miracle_kronos import close_position, _mark_trade_closed, get_positions

# P0 Fix: 预检实际OKX持仓，避免幽灵仓位
live_positions = get_positions()
live_inst_ids = {p.get('instId', '') for p in live_positions}
live_coins = set()
for inst_id in live_inst_ids:
    coin_part = inst_id.replace('-USDT-SWAP', '').replace('-USDT', '')
    live_coins.add(coin_part)

closed_coins = set()
executed_close = 0
failed_close = 0
skipped_ghost = 0

# dedup: 同一币种只处理第一个决策
processed = {}
for d in decisions:
    coin = d.get('coin', '')
    if coin and coin not in processed:
        processed[coin] = d

for coin, d in processed.items():
    action = d.get('action', '')
    reason = d.get('reason', '')
    pnl_pct = d.get('pnl_pct')
    is_close = 'close' in action or 'force' in action
    is_partial = 'partial' in action
    emoji = 'force:' if is_close else ('partial:' if is_partial else 'adj:')

    if is_close:
        if coin not in live_coins:
            ghost_reason = str(reason)[:25]
            print(f'ghost|{coin}|force_close跳过: OKX无持仓, reason={ghost_reason}')
            try:
                _mark_trade_closed(coin, reason=f'pos_mgmt_ghost:{ghost_reason}')
                skipped_ghost += 1
                print(f'ghost_cleared|{coin}|数据库已标记CLOSED')
            except Exception:
                pass
            continue

        print(f'{emoji}{coin}|action={action}|reason={reason}')
        if coin not in closed_coins:
            try:
                cr = close_position(coin, reason=f'pos_mgmt:{str(reason)[:30]}')
                if cr.get('code') == '0':
                    executed_close += 1
                    closed_coins.add(coin)
                    _mark_trade_closed(coin, reason='position_management', pnl_pct=pnl_pct)
                    print(f'closed|{coin}|成功')
                else:
                    failed_close += 1
                    print(f'close_fail|{coin}|{cr.get("msg","").strip()}')
            except Exception as e:
                failed_close += 1
                print(f'close_error|{coin}|{e}')
        else:
            print(f'already_closed|{coin}|trade marked closed')
    else:
        print(f'{emoji}{coin}|action={action}|reason={reason}')

parts = []
if skipped_ghost > 0:
    parts.append(f'ghost_skip={skipped_ghost}')
if executed_close > 0:
    parts.append(f'closed={executed_close}')
if failed_close > 0:
    parts.append(f'failed={failed_close}')
if parts:
    print(f'summary: {" | ".join(parts)}')
