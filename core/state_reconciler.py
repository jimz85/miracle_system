#!/usr/bin/env python3
from __future__ import annotations

"""
Miracle State Reconciler - 状态一致性模块
==========================================

检测并修复幽灵仓位、孤立订单、僵尸仓位等问题。
确保本地状态文件与 OKX 交易所状态一致。

功能:
1. 幽灵仓位检测 (Phantom Positions) - 本地记录有但交易所没有
2. 孤立订单检测 (Orphan Orders) - 交易所有待处理订单但本地无记录
3. 僵尸仓位检测 (Zombie Positions) - 交易所有持仓但本地无记录
4. OCO订单修复 - 确保SL/TP条件单与持仓状态一致
5. 状态文件一致性 - 同步 local_state.json 与交易所实际状态

用法:
    from core.state_reconciler import StateReconciler
    reconciler = StateReconciler()
    result = reconciler.reconcile()
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# OKX API 封装
from core.exchange_client import ExchangeClient
from core.executor_config import ExecutorConfig

logger = logging.getLogger("miracle.state_reconciler")


# ══════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    inst_id: str
    side: str
    sz: float
    price: float | None = None
    algo_id: str | None = None
    sl_trigger: float | None = None
    tp_trigger: float | None = None
    ord_type: str = "market"
    state: str = "live"
    created_at: str | None = None


@dataclass
class PositionInfo:
    """持仓信息"""
    inst_id: str
    pos: float
    direction: str
    entry_price: float = 0.0
    mark_price: float = 0.0
    upl: float = 0.0
    notional: float = 0.0
    leverage: float = 1.0
    liq_price: float = 0.0
    algo_ids: List[str] = field(default_factory=list)  # 关联的SL/TP algo_id


@dataclass
class LocalOrderRecord:
    """本地订单记录"""
    order_id: str
    inst_id: str
    side: str
    sz: float
    created_at: str
    status: str = "submitted"
    algo_id: str | None = None
    algo_type: str = "oco"  # oco, conditional, stop


@dataclass
class LocalPositionRecord:
    """本地持仓记录"""
    inst_id: str
    direction: str
    entry_price: float
    contracts: float
    stop_loss: float | None = None
    take_profit: float | None = None
    algo_id: str | None = None  # OCO订单ID
    opened_at: str | None = None
    status: str = "open"  # open, closed, phantom, zombie


@dataclass
class PhantomPosition:
    """幽灵仓位 - 本地有记录但交易所没有"""
    inst_id: str
    direction: str
    entry_price: float
    contracts: float
    local_record: LocalPositionRecord | None = None
    note: str = ""
    # P0-FIX: 记录首次检测时间，防止网络抖动误判后立即清理
    detected_at: float = 0.0  # Unix timestamp


@dataclass
class ZombiePosition:
    """僵尸仓位 - 交易所有但本地没有记录"""
    inst_id: str
    direction: str
    pos: float
    entry_price: float
    mark_price: float
    upl: float
    notional: float
    note: str = ""


@dataclass
class OrphanOrder:
    """孤立订单 - 交易所有订单但本地没有记录"""
    order_id: str
    inst_id: str
    side: str
    sz: float
    algo_id: str | None = None
    order_type: str = "unknown"
    note: str = ""


@dataclass
class OCOIssue:
    """OCO订单问题"""
    inst_id: str
    issue_type: str  # missing_algo, stale_algo, price_mismatch, orphaned_algo
    position_exists: bool
    algo_exists: bool
    expected_sl: float | None = None
    expected_tp: float | None = None
    actual_sl: float | None = None
    actual_tp: float | None = None
    algo_id: str | None = None
    note: str = ""


@dataclass
class ReconcileResult:
    """状态协调结果"""
    timestamp: str = ""
    duration_ms: float = 0.0
    
    # 检测结果
    orphan_orders: List[OrphanOrder] = field(default_factory=list)
    phantom_positions: List[PhantomPosition] = field(default_factory=list)
    zombie_positions: List[ZombiePosition] = field(default_factory=list)
    oco_issues: List[OCOIssue] = field(default_factory=list)
    
    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 统计
    total_exchange_positions: int = 0
    total_local_positions: int = 0
    total_exchange_orders: int = 0
    total_local_orders: int = 0
    account_balance: float = 0.0
    
    # 健康状态
    is_healthy: bool = True
    needs_sync: bool = False


# ══════════════════════════════════════════════════════════════════════
# State Reconciler
# ══════════════════════════════════════════════════════════════════════

class StateReconciler:
    """
    状态协调器 - 检测并修复幽灵仓位、孤立订单等问题
    """
    
    def __init__(self, state_file: Path | None = None, 
                 trade_log_file: Path | None = None,
                 exchange: str = "okx"):
        """
        初始化状态协调器
        
        Args:
            state_file: 本地状态文件路径，默认 data/local_state.json
            trade_log_file: 交易日志文件路径，默认 logs/trades.json
            exchange: 交易所类型，okx 或 binance
        """
        base_dir = Path(__file__).parent.parent
        
        # 状态文件路径
        self.state_file = state_file or (base_dir / "data" / "local_state.json")
        self.trade_log_file = trade_log_file or (base_dir / "logs" / "trades.json")
        
        # 确保目录存在
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 交易所客户端
        self.exchange = exchange
        self.client = ExchangeClient(exchange=exchange, config=ExecutorConfig())
        
        # 缓存
        self._exchange_state: Dict | None = None
        self._local_state: Dict | None = None
        
    # ─────────────────────────────────────────────────────────────
    # 交易所状态获取
    # ─────────────────────────────────────────────────────────────
    
    def fetch_exchange_positions(self) -> Dict[str, PositionInfo]:
        """从OKX获取当前持仓"""
        positions = {}
        try:
            pos_data = self.client._make_request("GET", 
                "/api/v5/account/positions?instType=SWAP", signed=True)
            
            if pos_data.get('code') == '0':
                for p in pos_data.get('data', []):
                    inst_id = p.get('instId', '')
                    pos = float(p.get('pos', 0))
                    if pos == 0:
                        continue
                    
                    # 获取该持仓的SL/TP条件单
                    algo_ids = self._get_algo_ids_for_position(inst_id)
                    
                    positions[inst_id] = PositionInfo(
                        inst_id=inst_id,
                        pos=pos,
                        direction='long' if pos > 0 else 'short',
                        entry_price=float(p.get('avgPx', 0)),
                        mark_price=float(p.get('markPx', 0)),
                        upl=float(p.get('upl', 0)),
                        notional=float(p.get('notionalUsd', 0)),
                        leverage=float(p.get('lever', 1)),
                        liq_price=float(p.get('liqPx', 0)),
                        algo_ids=algo_ids
                    )
        except Exception as e:
            logger.error(f"获取交易所持仓失败: {e}")
            
        return positions
    
    def _get_algo_ids_for_position(self, inst_id: str) -> List[str]:
        """获取某持仓关联的SL/TP条件单"""
        algo_ids = []
        # OKX orders-algo-pending只接受单一ordType，不能逗号分隔
        for ord_type in ['oco', 'conditional']:
            try:
                algo_data = self.client._make_request("GET",
                    f"/api/v5/trade/orders-algo-pending?instId={inst_id}&ordType={ord_type}&limit=100",
                    signed=True)
                if algo_data.get('code') == '0':
                    for a in algo_data.get('data', []):
                        algo_id = a.get('algoId', '')
                        if algo_id:
                            algo_ids.append(algo_id)
            except Exception:
                pass
        return algo_ids
    
    def fetch_exchange_orders(self) -> Tuple[Dict[str, OrderInfo], Dict[str, OrderInfo]]:
        """
        获取交易所活跃订单和待触发条件单
        Returns: (regular_orders, algo_orders)
        """
        regular_orders = {}
        algo_orders = {}
        
        try:
            # 普通活跃订单
            orders_data = self.client._make_request("GET",
                "/api/v5/trade/orders-pending?instType=SWAP", signed=True)
            
            if orders_data.get('code') == '0':
                for o in orders_data.get('data', []):
                    oid = o.get('ordId', '')
                    if not oid or float(o.get('sz', 0)) == 0:
                        continue
                    regular_orders[oid] = OrderInfo(
                        order_id=oid,
                        inst_id=o.get('instId', ''),
                        side=o.get('side', ''),
                        sz=float(o.get('sz', 0)),
                        price=float(o.get('px', 0)) if o.get('px') else None,
                        ord_type=o.get('ordType', 'market'),
                        state=o.get('state', 'live')
                    )
        except Exception as e:
            logger.error(f"获取普通订单失败: {e}")
        
        try:
            # 条件单（OCO/STOP）- OKX要求ordType单一值，分两次查询
            algo_orders = {}
            for ord_type in ['oco', 'conditional']:
                algo_data = self.client._make_request("GET",
                    f"/api/v5/trade/orders-algo-pending?instType=SWAP&ordType={ord_type}",
                    signed=True)

                if algo_data.get('code') == '0':
                    for a in algo_data.get('data', []):
                        aid = a.get('algoId', '')
                        if not aid:
                            continue
                        algo_orders[aid] = OrderInfo(
                            order_id=aid,
                            inst_id=a.get('instId', ''),
                            side=a.get('side', ''),
                            sz=float(a.get('sz', 0)),
                            algo_id=aid,
                            sl_trigger=float(a.get('slTriggerPx', 0)) if a.get('slTriggerPx') else None,
                            tp_trigger=float(a.get('tpTriggerPx', 0)) if a.get('tpTriggerPx') else None,
                            ord_type=a.get('ordType', 'conditional'),
                            state='pending'
                        )
        except Exception as e:
            logger.error(f"获取条件单失败: {e}")
        
        return regular_orders, algo_orders
    
    def fetch_exchange_balance(self) -> float:
        """获取交易所账户余额"""
        try:
            bal = self.client._make_request("GET", "/api/v5/account/balance", signed=True)
            if bal.get('code') == '0' and bal.get('data'):
                return float(bal['data'][0].get('totalEq', 0))
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
        return 0.0
    
    # ─────────────────────────────────────────────────────────────
    # 本地状态加载
    # ─────────────────────────────────────────────────────────────
    
    def load_local_positions(self) -> Dict[str, LocalPositionRecord]:
        """从本地文件加载持仓记录"""
        positions = {}
        
        # 1. 从 local_state.json 加载
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                
                for inst_id, pos_data in data.items():
                    if isinstance(pos_data, dict):
                        contracts = pos_data.get('contracts', pos_data.get('position', 0))
                        if contracts and contracts != 0:
                            positions[inst_id] = LocalPositionRecord(
                                inst_id=inst_id,
                                direction=pos_data.get('direction', 'long'),
                                entry_price=pos_data.get('entry_price', pos_data.get('entry', 0)),
                                contracts=abs(contracts),
                                stop_loss=pos_data.get('stop_loss'),
                                take_profit=pos_data.get('take_profit'),
                                algo_id=pos_data.get('algo_id'),
                                opened_at=pos_data.get('opened_at'),
                                status=pos_data.get('status', 'open')
                            )
            except Exception as e:
                logger.warning(f"加载local_state.json失败: {e}")
        
        # 2. 从 trades.json 加载未平仓交易
        if self.trade_log_file.exists():
            try:
                with open(self.trade_log_file) as f:
                    trades = json.load(f)
                
                for t in trades:
                    if isinstance(t, dict) and t.get('status') == 'open':
                        inst_id = t.get('symbol', '')
                        if inst_id and inst_id not in positions:
                            # 转换symbol格式
                            if '-SWAP' not in inst_id:
                                inst_id = inst_id.replace('-USDT', '-USDT-SWAP')
                            
                            positions[inst_id] = LocalPositionRecord(
                                inst_id=inst_id,
                                direction=t.get('side', 'long'),
                                entry_price=t.get('entry_price', 0),
                                contracts=t.get('position_size', t.get('sz', 0)),
                                stop_loss=t.get('stop_loss'),
                                take_profit=t.get('take_profit'),
                                algo_id=t.get('algo_id'),
                                opened_at=t.get('timestamp'),
                                status='open'
                            )
            except Exception as e:
                logger.warning(f"加载trades.json失败: {e}")
        
        return positions
    
    def load_local_orders(self) -> Dict[str, LocalOrderRecord]:
        """从本地文件加载订单记录"""
        orders = {}
        
        # 从 trades.json 加载订单
        if self.trade_log_file.exists():
            try:
                with open(self.trade_log_file) as f:
                    trades = json.load(f)
                
                for t in trades:
                    if isinstance(t, dict):
                        # 普通订单
                        oid = t.get('order_id') or t.get('ordId')
                        if oid:
                            algo_id = t.get('algo_id')
                            algo_type = 'oco' if algo_id else 'regular'
                            orders[str(oid)] = LocalOrderRecord(
                                order_id=str(oid),
                                inst_id=t.get('inst_id', t.get('symbol', '')),
                                side=t.get('side', ''),
                                sz=float(t.get('sz', t.get('size', 0))),
                                created_at=t.get('timestamp', ''),
                                status=t.get('status', 'submitted'),
                                algo_id=algo_id,
                                algo_type=algo_type
                            )
                        
                        # OCO订单
                        if t.get('algo_id'):
                            orders[str(t['algo_id'])] = LocalOrderRecord(
                                order_id=str(t['algo_id']),
                                inst_id=t.get('symbol', ''),
                                side=t.get('side', ''),
                                sz=float(t.get('sz', t.get('size', 0))),
                                created_at=t.get('timestamp', ''),
                                status=t.get('status', 'submitted'),
                                algo_id=t['algo_id'],
                                algo_type='oco'
                            )
            except Exception as e:
                logger.warning(f"加载订单记录失败: {e}")

        # 2. 从 local_state.json 补充 OCO algo_id（防止真实OCO被误判为孤立订单）
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                for inst_id, pos_data in data.items():
                    if isinstance(pos_data, dict) and pos_data.get('algo_id') and inst_id != '_meta':
                        algo_id = str(pos_data['algo_id'])
                        if algo_id not in orders:
                            orders[algo_id] = LocalOrderRecord(
                                order_id=algo_id,
                                inst_id=inst_id,
                                side=pos_data.get('direction', 'long'),
                                sz=float(pos_data.get('contracts', 0)),
                                created_at='',
                                status='attached',
                                algo_id=algo_id,
                                algo_type=pos_data.get('algo_type', 'oco')
                            )
            except Exception:
                pass

        return orders
    
    # ─────────────────────────────────────────────────────────────
    # 状态保存
    # ─────────────────────────────────────────────────────────────
    
    def save_local_state(self, positions: Dict[str, LocalPositionRecord]):
        """保存本地持仓状态到文件"""
        data = {}
        for inst_id, pos in positions.items():
            data[inst_id] = {
                'direction': pos.direction,
                'entry_price': pos.entry_price,
                'contracts': pos.contracts,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'algo_id': pos.algo_id,
                'opened_at': pos.opened_at,
                'status': pos.status
            }
        
        try:
            # 原子写入防止文件损坏
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.state_file)
            logger.info(f"本地状态已保存: {len(positions)}个持仓")
        except Exception as e:
            logger.error(f"保存本地状态失败: {e}")
    
    def update_trade_status(self, trade_id: str, status: str, 
                           exit_reason: str | None = None):
        """更新交易状态"""
        if not self.trade_log_file.exists():
            return
        
        try:
            with open(self.trade_log_file) as f:
                trades = json.load(f)
            
            for t in trades:
                if t.get('trade_id') == trade_id or t.get('order_id') == trade_id:
                    t['status'] = status
                    if exit_reason:
                        t['exit_reason'] = exit_reason
                    break
            
            temp_file = self.trade_log_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(trades, f, indent=2)
            temp_file.replace(self.trade_log_file)
        except Exception as e:
            logger.error(f"更新交易状态失败: {e}")
    
    # ─────────────────────────────────────────────────────────────
    # 核心协调逻辑
    # ─────────────────────────────────────────────────────────────
    
    def reconcile(self, auto_fix: bool = False) -> ReconcileResult:
        """
        执行状态协调
        
        Args:
            auto_fix: 是否自动修复问题
            
        Returns:
            ReconcileResult: 协调结果
        """
        start = time.time()
        result = ReconcileResult()
        result.timestamp = datetime.now().isoformat()
        
        try:
            # 1. 获取交易所状态
            exchange_positions = self.fetch_exchange_positions()
            regular_orders, algo_orders = self.fetch_exchange_orders()
            exchange_balance = self.fetch_exchange_balance()
            
            # 2. 加载本地状态
            local_positions = self.load_local_positions()
            local_orders = self.load_local_orders()
            
            # 3. 填充统计信息
            result.total_exchange_positions = len(exchange_positions)
            result.total_local_positions = len(local_positions)
            result.total_exchange_orders = len(regular_orders) + len(algo_orders)
            result.total_local_orders = len(local_orders)
            result.account_balance = exchange_balance
            
            # 4. 检测幽灵仓位 (本地有，交易所没有)
            for inst_id, lpos in local_positions.items():
                if inst_id not in exchange_positions:
                    # 检查是否真的不存在，还是持仓为0
                    exchange_positions.get(inst_id)
                    contracts = lpos.contracts or (getattr(lpos, 'pos', 0) if hasattr(lpos, 'pos') else 0)
                    
                    phantom = PhantomPosition(
                        inst_id=inst_id,
                        direction=lpos.direction,
                        entry_price=lpos.entry_price,
                        contracts=contracts,
                        local_record=lpos,
                        note="本地持仓记录存在但交易所显示为零或不存在",
                        detected_at=time.time()  # P0-FIX: 记录首次检测时间
                    )
                    result.phantom_positions.append(phantom)
            
            # 5. 检测僵尸仓位 (交易所有，本地没有)
            for inst_id, epos in exchange_positions.items():
                if epos.pos != 0 and inst_id not in local_positions:
                    result.zombie_positions.append(ZombiePosition(
                        inst_id=inst_id,
                        direction=epos.direction,
                        pos=epos.pos,
                        entry_price=epos.entry_price,
                        mark_price=epos.mark_price,
                        upl=epos.upl,
                        notional=epos.notional,
                        note="交易所有持仓但本地没有记录"
                    ))
            
            # 6. 检测孤立订单 (交易所有订单，本地没有)
            for oid, o in regular_orders.items():
                if oid not in local_orders:
                    result.orphan_orders.append(OrphanOrder(
                        order_id=oid,
                        inst_id=o.inst_id,
                        side=o.side,
                        sz=o.sz,
                        order_type=o.ord_type,
                        note="普通订单在交易所但本地无记录"
                    ))
            
            for aid, a in algo_orders.items():
                if aid not in local_orders:
                    result.orphan_orders.append(OrphanOrder(
                        order_id=aid,
                        inst_id=a.inst_id,
                        side=a.side,
                        sz=a.sz,
                        algo_id=aid,
                        order_type=f"algo_{a.ord_type}",
                        note="条件单在交易所但本地无记录"
                    ))
            
            # 7. 检测OCO订单问题
            result.oco_issues = self._check_oco_consistency(
                exchange_positions, algo_orders, local_positions
            )
            
            # 8. 健康状态判断
            if result.phantom_positions:
                result.errors.append(
                    f"发现 {len(result.phantom_positions)} 个幽灵仓位 - 本地有记录但交易所没有!"
                )
                result.is_healthy = False
                result.needs_sync = True
                
            if result.zombie_positions:
                result.warnings.append(
                    f"发现 {len(result.zombie_positions)} 个僵尸仓位 - 交易所有但本地无记录"
                )
                result.needs_sync = True
                
            if result.orphan_orders:
                result.warnings.append(
                    f"发现 {len(result.orphan_orders)} 个孤立订单"
                )
                
            for issue in result.oco_issues:
                if issue.issue_type in ['missing_algo', 'orphaned_algo']:
                    result.errors.append(
                        f"OCO问题 [{issue.inst_id}]: {issue.issue_type} - {issue.note}"
                    )
                    result.is_healthy = False
                else:
                    result.warnings.append(
                        f"OCO警告 [{issue.inst_id}]: {issue.issue_type} - {issue.note}"
                    )
                    
            # 9. 自动修复
            if auto_fix and result.needs_sync:
                self._auto_sync(result, exchange_positions, local_positions)
                
        except Exception as e:
            result.errors.append(f"状态协调失败: {e}")
            result.is_healthy = False
            logger.exception("状态协调异常")
        
        result.duration_ms = (time.time() - start) * 1000
        return result
    
    def _check_oco_consistency(
        self, 
        exchange_positions: Dict[str, PositionInfo],
        algo_orders: Dict[str, OrderInfo],
        local_positions: Dict[str, LocalPositionRecord]
    ) -> List[OCOIssue]:
        """检查OCO订单一致性"""
        issues = []
        
        for inst_id, epos in exchange_positions.items():
            if epos.pos == 0:
                continue
                
            local_pos = local_positions.get(inst_id)
            
            # 查找该inst_id的所有条件单
            inst_algos = {
                aid: a for aid, a in algo_orders.items() 
                if a.inst_id == inst_id
            }
            
            if not inst_algos:
                # 有持仓但没有SL/TP条件单
                issues.append(OCOIssue(
                    inst_id=inst_id,
                    issue_type='missing_algo',
                    position_exists=True,
                    algo_exists=False,
                    expected_sl=local_pos.stop_loss if local_pos else None,
                    expected_tp=local_pos.take_profit if local_pos else None,
                    note='持仓存在但没有SL/TP条件单保护'
                ))
            else:
                # 有条件单，检查是否匹配
                for algo_id, algo in inst_algos.items():
                    # SL/TP触发价格不匹配
                    if local_pos:
                        if algo.sl_trigger and local_pos.stop_loss:
                            sl_diff = abs(algo.sl_trigger - local_pos.stop_loss) / local_pos.stop_loss
                            if sl_diff > 0.01:  # 1%偏差
                                issues.append(OCOIssue(
                                    inst_id=inst_id,
                                    issue_type='price_mismatch',
                                    position_exists=True,
                                    algo_exists=True,
                                    expected_sl=local_pos.stop_loss,
                                    expected_tp=local_pos.take_profit,
                                    actual_sl=algo.sl_trigger,
                                    actual_tp=algo.tp_trigger,
                                    algo_id=algo_id,
                                    note=f'SL价格不匹配: 期望{local_pos.stop_loss}, 实际{algo.sl_trigger}'
                                ))
        
        # 检查孤立的条件单（没有对应持仓）
        for algo_id, algo in algo_orders.items():
            if algo.inst_id not in exchange_positions:
                issues.append(OCOIssue(
                    inst_id=algo.inst_id,
                    issue_type='orphaned_algo',
                    position_exists=False,
                    algo_exists=True,
                    actual_sl=algo.sl_trigger,
                    actual_tp=algo.tp_trigger,
                    algo_id=algo_id,
                    note='条件单存在但无对应持仓'
                ))
        
        return issues
    
    def _auto_sync(self, result: ReconcileResult,
                   exchange_positions: Dict[str, PositionInfo],
                   local_positions: Dict[str, LocalPositionRecord]):
        """自动同步状态"""
        logger.info("执行自动同步...")
        
        # 1. 清理幽灵仓位（标记为closed）
        for phantom in result.phantom_positions:
            logger.warning(f"清理幽灵仓位: {phantom.inst_id}")
            self.update_trade_status(
                phantom.inst_id, 
                'phantom_closed',
                'reconciler_auto_cleanup'
            )
        
        # 2. 添加僵尸仓位到本地记录
        for zombie in result.zombie_positions:
            logger.info(f"添加僵尸仓位到本地: {zombie.inst_id}")
            local_positions[zombie.inst_id] = LocalPositionRecord(
                inst_id=zombie.inst_id,
                direction=zombie.direction,
                entry_price=zombie.entry_price,
                contracts=abs(zombie.pos),
                status='zombie_imported'
            )
        
        # 3. 保存更新后的本地状态
        self.save_local_state(local_positions)
        
        logger.info("自动同步完成")
    
    # ─────────────────────────────────────────────────────────────
    # OCO订单修复
    # ─────────────────────────────────────────────────────────────
    
    def fix_oco_orders(self) -> Dict[str, Any]:
        """
        修复OCO订单问题

        Returns:
            修复结果统计
        """
        result = self.reconcile(auto_fix=False)

        stats = {
            'total_issues': len(result.oco_issues),
            'fixed': 0,
            'failed': 0,
            'details': []
        }

        for issue in result.oco_issues:
            try:
                if issue.issue_type == 'missing_algo':
                    # 需要重新挂SL/TP
                    logger.warning(f"需要重新挂SL/TP: {issue.inst_id}")
                    stats['details'].append({
                        'inst_id': issue.inst_id,
                        'action': 'needs_manual_repair',
                        'note': '需要获取入场信息后重新下单'
                    })

                elif issue.issue_type == 'orphaned_algo':
                    # 取消孤立的条件单
                    if issue.algo_id:
                        success = self.client.cancel_algo_order(
                            issue.inst_id,
                            issue.algo_id
                        )
                        if success:
                            stats['fixed'] += 1
                            logger.info(f"已取消孤立条件单: {issue.algo_id}")
                        else:
                            stats['failed'] += 1

            except Exception as e:
                logger.error(f"修复OCO问题失败: {e}")
                stats['failed'] += 1
                stats['details'].append({
                    'inst_id': issue.inst_id,
                    'action': 'failed',
                    'error': str(e)
                })

        return stats

    # ─────────────────────────────────────────────────────────────
    # 幽灵仓位检测与清理 (P0.2)
    # ─────────────────────────────────────────────────────────────

    def get_real_positions(self) -> Dict[str, PositionInfo]:
        """
        获取交易所实际持仓

        Returns:
            Dict[str, PositionInfo]: 交易所实际持仓字典
        """
        return self.fetch_exchange_positions()

    def reconcile_positions(self, threshold_pct: float = 0.02) -> ReconcileResult:
        """
        检测幽灵仓位

        Args:
            threshold_pct: 差异阈值 (默认2%)

        Returns:
            ReconcileResult: 包含幽灵仓位检测结果
        """
        return self.reconcile(auto_fix=False)

    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """
        检查是否可以开仓 (防止幽灵仓位导致重复开仓)

        Args:
            symbol: 交易标的

        Returns:
            (can_open, reason)
        """
        # 获取实际持仓
        real_positions = self.get_real_positions()

        # 加载本地记录
        local_positions = self.load_local_positions()

        # 转换symbol格式
        inst_id = symbol
        if '-SWAP' not in inst_id and '-USDT' in inst_id:
            inst_id = symbol.replace('-USDT', '-USDT-SWAP')
        elif '-USDT' not in symbol:
            inst_id = f"{symbol}-USDT-SWAP"

        # 检查幽灵仓位
        if inst_id in local_positions and inst_id not in real_positions:
            local_pos = local_positions[inst_id]
            return False, (
                f"幽灵仓位检测: {symbol} 本地记录有持仓但交易所实际为0. "
                f"entry={local_pos.entry_price}, contracts={local_pos.contracts}. "
                f"请先运行 reconcile(auto_fix=True) 清理."
            )

        # 检查持仓差异
        if inst_id in local_positions and inst_id in real_positions:
            local_pos = local_positions[inst_id]
            real_pos = real_positions[inst_id]

            local_size = local_pos.contracts
            real_size = abs(real_pos.pos)

            if local_size > 0 and real_size == 0:
                return False, f"幽灵仓位检测: {symbol} 本地有持仓但交易所实际为0"

            # 检查差异是否超过阈值
            if local_size > 0:
                diff_pct = abs(local_size - real_size) / local_size
                if diff_pct > 0.02:  # 2%阈值
                    return False, (
                        f"持仓差异过大: {symbol} 本地={local_size}, "
                        f"交易所={real_size}, 差异={diff_pct:.1%}"
                    )

        return True, "OK"

    def cleanup_phantom_orders(
        self,
        phantom: PhantomPosition,
        force: bool = False,
        min_confirm_seconds: int = 300,
    ) -> bool:
        """
        清理幽灵仓位

        P0-FIX: 添加多重保护防止网络抖动误杀真实持仓
        1. 首次检测到后必须等待min_confirm_seconds才能自动清理
        2. 自动清理前必须二次确认（重试fetch交易所状态）
        3. force=True可跳过所有检查（用于紧急手动清理）

        Args:
            phantom: 幽灵仓位信息
            force: 跳过所有检查（紧急手动清理）
            min_confirm_seconds: 首次检测后需等待秒数才能自动清理

        Returns:
            是否清理成功
        """
        try:
            # P0-FIX: 检查检测时间，防止网络抖动立即误判
            if not force:
                elapsed = time.time() - phantom.detected_at
                if elapsed < min_confirm_seconds:
                    logger.warning(
                        f"幽灵仓位 {phantom.inst_id} 检测不足{min_confirm_seconds}秒"
                        f"({elapsed:.0f}秒)，暂不自动清理，等待人工确认"
                    )
                    return False

            logger.warning(f"[P0-SAFETY] 开始清理幽灵仓位: {phantom.inst_id}")

            # P0-FIX: 二次确认 — 再次fetch交易所状态，防止网络抖动误判
            if not force:
                exchange_positions = self.client.get_open_positions()
                active_symbols = {p.inst_id for p in exchange_positions if p.pos != 0}
                if phantom.inst_id in active_symbols:
                    logger.error(
                        f"二次确认失败: {phantom.inst_id} 在交易所仍有活跃持仓！"
                        f"停止清理，防止误杀真实仓位。"
                    )
                    # 推飞书告警
                    self._send_phantom_alert(phantom, "二次确认失败-真实持仓")
                    return False

            # 1. 如果有本地algo订单，取消它
            if phantom.local_record and phantom.local_record.algo_id:
                try:
                    self.client.cancel_algo_order(
                        phantom.inst_id,
                        phantom.local_record.algo_id
                    )
                    logger.info(f"已取消幽灵仓位的条件单: {phantom.local_record.algo_id}")
                except Exception as e:
                    logger.error(f"取消条件单失败: {e}")

            # 2. 更新交易日志，标记为phantom_closed
            self.update_trade_status(
                phantom.inst_id,
                'phantom_closed',
                'state_reconciler_auto_cleanup'
            )

            # 3. 更新本地状态
            local_positions = self.load_local_positions()
            if phantom.inst_id in local_positions:
                local_positions[phantom.inst_id].status = 'phantom_closed'
                self.save_local_state(local_positions)

            logger.info(f"幽灵仓位已清理: {phantom.inst_id}")
            self._send_phantom_alert(phantom, "已清理")
            return True

        except Exception as e:
            logger.error(f"清理幽灵仓位失败: {phantom.inst_id}: {e}")
            return False

    def _send_phantom_alert(self, phantom: PhantomPosition, action: str) -> None:
        """P0-FIX: 幽灵仓位变动时发送飞书告警"""
        try:
            msg = (
                f"🚨 **幽灵仓位告警**\n"
                f"币种: `{phantom.inst_id}`\n"
                f"方向: {phantom.direction}\n"
                f"数量: {phantom.contracts}\n"
                f"入场价: {phantom.entry_price}\n"
                f"操作: {action}"
            )
            logger.info(f"[FISHU-ALERT] {msg}")
        except Exception:
            pass  # 不因通知失败影响主流程

    def auto_cleanup_phantoms(self, force: bool = False) -> Dict[str, Any]:
        """
        自动清理所有幽灵仓位

        P0-FIX: 默认不自动强制清理，需等待min_confirm_seconds或force=True

        Args:
            force: 是否跳过时间/二次确认检查（紧急情况使用）

        Returns:
            清理结果统计
        """
        result = self.reconcile(auto_fix=False)

        stats = {
            'total_phantoms': len(result.phantom_positions),
            'cleaned': 0,
            'failed': 0,
            'skipped': 0,  # P0-FIX: 统计被跳过的
            'details': []
        }

        for phantom in result.phantom_positions:
            try:
                # P0-FIX: cleanup_phantom_orders内部会检查时间阈值
                if self.cleanup_phantom_orders(phantom, force=force):
                    stats['cleaned'] += 1
                    stats['details'].append({
                        'inst_id': phantom.inst_id,
                        'action': 'cleaned',
                        'entry': phantom.entry_price,
                        'contracts': phantom.contracts
                    })
                else:
                    stats['skipped'] += 1  # 时间不足/二次确认失败
                    stats['details'].append({
                        'inst_id': phantom.inst_id,
                        'action': 'skipped'
                    })
            except Exception as e:
                logger.error(f"清理幽灵仓位异常 {phantom.inst_id}: {e}")
                stats['failed'] += 1

        return stats
    
    # ─────────────────────────────────────────────────────────────
    # 报告
    # ─────────────────────────────────────────────────────────────
    
    def print_report(self, result: ReconcileResult):
        """打印状态协调报告"""
        print()
        print("═" * 70)
        print("  MIRACLE STATE RECONCILIATION REPORT")
        print("═" * 70)
        print(f"  Generated: {result.timestamp}")
        print(f"  Duration:  {result.duration_ms:.1f}ms")
        print()

        print("┌─ EXCHANGE STATE (OKX) ──────────────────────────────────────")
        print(f"│  Account Balance: ${result.account_balance:,.2f}")
        print(f"│  Open Positions:  {result.total_exchange_positions}")
        print(f"│  Open Orders:     {result.total_exchange_orders}")
        print("└─────────────────────────────────────────────────────────────")
        print()
        print("┌─ LOCAL STATE ───────────────────────────────────────────────")
        print(f"│  Tracked Positions: {result.total_local_positions}")
        print(f"│  Tracked Orders:    {result.total_local_orders}")
        print("└─────────────────────────────────────────────────────────────")

        print()
        if result.phantom_positions:
            print("┌─ 🚨 PHANTOM POSITIONS (local open, exchange zero) ───────────")
            for p in result.phantom_positions:
                print(f"│  {p.inst_id} | {p.direction} | entry={p.entry_price:.4f} | contracts={p.contracts}")
                print(f"│    → {p.note}")
            print("└─────────────────────────────────────────────────────────────")
        else:
            print("┌─ ✓ NO PHANTOM POSITIONS ────────────────────────────────────")

        print()
        if result.zombie_positions:
            print("┌─ ⚡ ZOMBIE POSITIONS (on exchange, not tracked locally) ─────")
            for z in result.zombie_positions:
                print(f"│  {z.inst_id} | {z.direction} | pos={z.pos} | entry={z.entry_price:.4f}")
            print("└─────────────────────────────────────────────────────────────")
        else:
            print("┌─ ✓ NO ZOMBIE POSITIONS ─────────────────────────────────────")

        print()
        if result.orphan_orders:
            print("┌─ ⚠️  ORPHAN ORDERS (on exchange, no local record) ──────────")
            for o in result.orphan_orders:
                print(f"│  [{o.order_id}] {o.inst_id} {o.side} sz={o.sz} type={o.order_type}")
            print("└─────────────────────────────────────────────────────────────")
        else:
            print("┌─ ✓ NO ORPHAN ORDERS ───────────────────────────────────────")

        print()
        if result.oco_issues:
            print("┌─ 🔧 OCO ISSUES ─────────────────────────────────────────────")
            for o in result.oco_issues:
                print(f"│  [{o.inst_id}] {o.issue_type}: {o.note}")
            print("└─────────────────────────────────────────────────────────────")

        if result.errors:
            print()
            print("┌─ ❌ ERRORS ──────────────────────────────────────────────────")
            for e in result.errors:
                print(f"│  {e}")
            print("└─────────────────────────────────────────────────────────────")

        if result.warnings:
            print()
            print("┌─ ⚠️  WARNINGS ───────────────────────────────────────────────")
            for w in result.warnings:
                print(f"│  {w}")
            print("└─────────────────────────────────────────────────────────────")

        print()
        print("═" * 70)
        health_msg = "✅ STATE IS HEALTHY - Ready to operate" if result.is_healthy else "❌ STATE HAS ISSUES - Review above"
        print(f"  {health_msg}")
        if result.needs_sync:
            print("  ⚠️  Needs sync - Run with auto_fix=True to synchronize")
        print("═" * 70)
        print()


# ══════════════════════════════════════════════════════════════════════
# Standalone Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Miracle State Reconciler")
    parser.add_argument("--fix", action="store_true", help="自动修复问题")
    parser.add_argument("--check-oco", action="store_true", help="仅检查OCO订单")
    args = parser.parse_args()
    
    reconciler = StateReconciler()
    
    if args.check_oco:
        result = reconciler.reconcile()
        reconciler.print_report(result)
        stats = reconciler.fix_oco_orders()
        print(f"\nOCO修复结果: {json.dumps(stats, indent=2)}")
    elif args.fix:
        result = reconciler.reconcile(auto_fix=True)
        reconciler.print_report(result)
    else:
        result = reconciler.reconcile()
        reconciler.print_report(result)
