#!/usr/bin/env python3
"""
持仓主动管理脚本 — 独立于主扫描循环

职责：
1. 每3分钟检查所有OPEN仓位
2. 用规则+Gemma4评估是否该平仓/调整
3. 执行close_position() / place_oco() 操作

与主扫描(scan)的关系：
- 本脚本只管理持仓，不开新仓
- 使用 fcntl.flock 文件锁防止和主扫描冲突
- 输出写 journal 文件，不写主状态文件

用法:
  python3 manage_positions.py              # 默认模式，执行并输出
  python3 manage_positions.py --dry-run    # 只检查不执行
"""

import os, sys, json, fcntl, time, logging
from pathlib import Path
from datetime import datetime, timezone

# 确保在项目目录下
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

# 文件锁 — 防止和主扫描同时运行
LOCK_FILE = SCRIPT_DIR / "manage_positions.lock"

def acquire_lock():
    """获取文件锁，失败则跳过本次运行"""
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (IOError, BlockingIOError):
        return None

def release_lock(fd):
    """释放文件锁"""
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        try:
            LOCK_FILE.unlink()
        except:
            pass

# ===== 导入核心函数 =====
# 从 miracle_kronos 导入所有需要的函数
from miracle_kronos import (
    run_position_management,
    get_account_balance,
    get_klines,
    get_positions,
    close_position,
    place_oco,
    calc_rsi,
    calc_adx,
    okx_req,
    load_trades,
    save_trades,
    _gemma_vote_cached,
    _pos_mode,
    _detect_pos_mode,
    calc_atr,
    _mark_trade_closed,
    get_ticker,
    logger,
)

# ===== 配置 =====
MANAGE_INTERVAL = 3  # 每3分钟
DRY_RUN = '--dry-run' in sys.argv

LOG_FILE = SCRIPT_DIR / "data" / "manage_journal.jsonl"


def append_journal(entry: dict):
    """追加管理日志"""
    LOG_FILE.parent.mkdir(exist_ok=True)
    entry['_ts'] = datetime.now().isoformat()
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.warning(f"管理日志写入失败: {e}")


def run_management_cycle(equity: float, btc_trend: str, dry_run: bool = False):
    """
    执行一个管理周期
    
    1. 调用 run_position_management() 获取决策
    2. 用 Gemma4 评估持仓
    3. 执行决策
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 持仓管理周期...")
    
    # Step 1: 获取当前持仓（同时从OKX和本地记录）
    positions = get_positions()
    all_trades = load_trades() if not dry_run else []
    open_trades = [t for t in all_trades if t.get('status') == 'OPEN'] if all_trades else []
    
    print(f"  OKX持仓: {len(positions)}  |  本地OPEN: {len(open_trades)}")
    
    # Step 2: 调用原有的 run_position_management()
    mgmt = run_position_management(equity, btc_trend, 'live' if not dry_run else 'audit')
    
    decisions = mgmt.get('decisions', [])
    warnings = mgmt.get('warnings', [])
    
    if mgmt.get('action') == 'no_open_positions':
        print(f"  无OPEN持仓，跳过")
        append_journal({'action': 'noop', 'reason': 'no_open_positions'})
        return
    
    # Step 3: Gemma4 持仓评估（只在有持仓时）
    if positions:
        for pos in positions:
            inst_id = pos.get('instId', '')
            sym = inst_id.replace('-USDT-SWAP', '')
            side = pos.get('side', 'long')
            entry = pos.get('entry', 0)
            sz = pos.get('sz', 0)
            upl = pos.get('unrealized_pnl', 0)
            
            # 获取技术指标
            current = get_ticker(inst_id)
            if current == 0:
                current = entry
            
            klines = get_klines(inst_id, '1H', 50)
            if klines and len(klines) >= 30:
                closes = [k['close'] for k in klines]
                highs = [k['high'] for k in klines]
                lows = [k['low'] for k in klines]
                
                rsi = calc_rsi(closes)
                _, _, adx = calc_adx(highs, lows, closes)
                bb_upper, bb_lower, bb_pos = None, None, 50
                try:
                    # calc_bollinger returns a tuple
                    from miracle_kronos import calc_bollinger
                    bb_result = calc_bollinger(closes)
                    if bb_result and len(bb_result) >= 3:
                        bb_upper, bb_lower, bb_pos = bb_result
                except:
                    pass
                
                # ⭐ Gemma4 持仓评估
                if rsi > 0 and adx > 0:
                    try:
                        gemma_vote = _gemma_vote_cached(sym, rsi, adx, bb_pos, current, 0, 0)
                        gemma_signal = (gemma_vote - 0.5) * 2  # → -1到+1
                        
                        # 计算盈亏方向
                        pnl_pct = (current - entry) / entry if side in ('long', 'net') else (entry - current) / entry
                        
                        print(f"  🤖 Gemma评估 {sym}: gemma_vote={gemma_vote:.2f} signal={gemma_signal:+.2f} pnl={pnl_pct:.2%}")
                        
                        # Gemma强烈看空持仓方向 + 亏损 → 平仓
                        if side in ('long', 'net') and gemma_signal < -0.5 and pnl_pct < 0:
                            decisions.append({
                                'action': 'force_close',
                                'coin': sym,
                                'direction': side,
                                'reason': f'Gemma看空(pnl={pnl_pct:.1%})',
                                'pnl_pct': pnl_pct,
                                'urgency': 9,
                                'source': 'gemma4',
                            })
                            print(f"  🚨 Gemma: {sym} 看空+亏损 → 平仓")
                        elif side == 'short' and gemma_signal > 0.5 and pnl_pct < 0:
                            decisions.append({
                                'action': 'force_close',
                                'coin': sym,
                                'direction': side,
                                'reason': f'Gemma看多(pnl={pnl_pct:.1%})',
                                'pnl_pct': pnl_pct,
                                'urgency': 9,
                                'source': 'gemma4',
                            })
                            print(f"  🚨 Gemma: {sym} 看多+亏损 → 平仓")
                    except Exception as e:
                        logger.debug(f"Gemma评估失败 {sym}: {e}")
    
    # Step 4: 执行决策
    if not decisions:
        print(f"  无需操作")
        append_journal({'action': 'noop', 'reason': 'no_decisions'})
        return
    
    if dry_run:
        print(f"\n  📋 DRY-RUN 决策 ({len(decisions)}项):")
        for d in decisions:
            print(f"    {d.get('action', '?'):15s} {d.get('coin','?'):6s} | {d.get('reason','')}")
        append_journal({'action': 'dry_run', 'decisions': decisions})
        return
    
    # 实际执行（和 main() 中一样的逻辑）
    executed = []
    for dec in sorted(decisions, key=lambda x: -x.get('urgency', 0)):
        coin = dec.get('coin', '')
        inst_id = f'{coin}-USDT-SWAP'
        action_type = dec.get('action', '')
        reason = dec.get('reason', '')
        pnl_pct = dec.get('pnl_pct', 0)
        
        if action_type in ('force_close', 'close_opposite_signal'):
            close_data = close_position(coin, reason=reason)
            if close_data.get('code') == '0':
                _mark_trade_closed(coin, reason, None, pnl_pct=pnl_pct)
                executed.append(f'平仓 {coin}: {reason}')
                print(f"  ✅ 平仓 {coin}: {reason}")
            else:
                print(f"  ❌ 平仓失败 {coin}: {close_data.get('msg')}")
        
        elif action_type == 'partial_tp':
            close_data = close_position(coin, reason=f'{reason} [部分止盈]')
            if close_data.get('code') == '0':
                _mark_trade_closed(coin, f'{reason} [部分止盈]', None, pnl_pct=pnl_pct)
                executed.append(f'部分止盈 {coin}: {reason}')
                print(f"  🎯 部分止盈 {coin}: {reason}")
            else:
                print(f"  ❌ 部分止盈失败 {coin}: {close_data.get('msg')}")
        
        elif action_type == 'move_sl_to_cost':
            try:
                oco_query = okx_req('GET', f'/api/v5/trade/orders-algo-pending?instId={inst_id}&ordType=oco')
                algo_list = oco_query.get('data', [])
                for algo in algo_list:
                    cancel_body = json.dumps([{'algoId': str(algo['algoId']), 'instId': inst_id}])
                    okx_req('DELETE', '/api/v5/trade/cancel-algos', cancel_body)
                entry = dec.get('entry', 0)
                sl_pct = abs(entry - dec['new_sl']) / entry if entry > 0 else 0.05
                pos = next((p for p in positions if coin.upper() in p.get('instId','')), None)
                if pos:
                    sz = int(pos.get('sz', 0))
                    direction = pos.get('posSide', 'long')
                    direction = 'long' if direction in ('long','net') else 'short'
                    place_oco(inst_id, direction, sz, entry, sl_pct, 0.10, equity=equity, leverage=3)
                    executed.append(f'SL上移 {coin}: {dec.get("new_sl",0):.4f}')
                    print(f"  🛡️  SL上移 {coin}: {dec.get('new_sl',0):.4f}")
            except Exception as e:
                print(f"  ❌ SL移动失败 {coin}: {e}")
    
    # 记录
    summary = '; '.join(executed) if executed else '无执行'
    append_journal({'action': 'executed', 'decisions': decisions, 'summary': summary})
    print(f"  📋 管理摘要: {summary}")


def main():
    # 获取文件锁（防止和主扫描冲突）
    lock_fd = acquire_lock()
    if lock_fd is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏭️  文件锁被占用（主扫描运行中），跳过本次")
        return
    
    try:
        if DRY_RUN:
            print("=" * 50)
            print("📋 持仓管理 DRY-RUN 模式（只检查不执行）")
            print("=" * 50)
        
        # 初始化
        _detect_pos_mode()
        
        # 获取余额
        equity = get_account_balance()
        if equity == 0:
            print("❌ 无法获取账户余额")
            return
        
        # 获取BTC趋势
        btc_klines = get_klines('BTC-USDT-SWAP', '4H', 50)
        btc_trend = 'neutral'
        if btc_klines and len(btc_klines) >= 20:
            closes = [k['close'] for k in btc_klines]
            ma20 = sum(closes[-20:]) / 20
            btc_trend = 'bull' if closes[-1] > ma20 else 'bear'
        
        print(f"余额: ${equity:,.2f} | BTC: {btc_trend} | 模式: {'DRY-RUN' if DRY_RUN else 'LIVE'}")
        
        # 执行管理周期
        run_management_cycle(equity, btc_trend, DRY_RUN)
        
    finally:
        release_lock(lock_fd)


if __name__ == '__main__':
    main()
