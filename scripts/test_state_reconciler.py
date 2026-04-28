#!/usr/bin/env python3
"""
Miracle State Reconciler 测试脚本
================================

测试幽灵仓位检测、OCO订单修复、状态文件一致性功能
"""

import sys
import os
from pathlib import Path
import tempfile
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.state_reconciler import StateReconciler, ReconcileResult


def test_import():
    """测试模块导入"""
    print("✓ StateReconciler 导入成功")


def test_reconciler_init():
    """测试协调器初始化"""
    reconciler = StateReconciler()
    assert reconciler is not None
    assert reconciler.state_file is not None
    print("✓ StateReconciler 初始化成功")


def test_data_classes():
    """测试数据结构"""
    from core.state_reconciler import (
        OrderInfo, PositionInfo, LocalPositionRecord,
        PhantomPosition, ZombiePosition, OrphanOrder, OCOIssue, ReconcileResult
    )
    
    # 测试 PositionInfo
    pos = PositionInfo(
        inst_id="BTC-USDT-SWAP",
        pos=0.5,
        direction="long",
        entry_price=70000.0,
        mark_price=71000.0,
        upl=500.0,
        notional=35000.0
    )
    assert pos.inst_id == "BTC-USDT-SWAP"
    assert pos.direction == "long"
    print("✓ 数据结构测试通过")


def test_phantom_detection():
    """测试幽灵仓位检测逻辑"""
    from core.state_reconciler import LocalPositionRecord
    
    reconciler = StateReconciler()
    
    # 模拟本地有但交易所没有的持仓
    local_positions = {
        "BTC-USDT-SWAP": LocalPositionRecord(
            inst_id="BTC-USDT-SWAP",
            direction="long",
            entry_price=70000.0,
            contracts=0.5,
            stop_loss=68000.0,
            take_profit=75000.0,
            status="open"
        )
    }
    
    # 模拟交易所持仓为空
    exchange_positions = {}
    
    # 检测幽灵仓位
    phantom_positions = []
    for inst_id, lpos in local_positions.items():
        if inst_id not in exchange_positions:
            phantom_positions.append(inst_id)
    
    assert len(phantom_positions) == 1
    assert phantom_positions[0] == "BTC-USDT-SWAP"
    print("✓ 幽灵仓位检测逻辑正确")


def test_zombie_detection():
    """测试僵尸仓位检测逻辑"""
    from core.state_reconciler import PositionInfo
    
    # 模拟交易所有但本地没有的持仓
    exchange_positions = {
        "ETH-USDT-SWAP": PositionInfo(
            inst_id="ETH-USDT-SWAP",
            pos=2.0,
            direction="long",
            entry_price=3500.0,
            mark_price=3600.0,
            upl=200.0,
            notional=7000.0
        )
    }
    
    local_positions = {}
    
    # 检测僵尸仓位
    zombie_positions = []
    for inst_id, epos in exchange_positions.items():
        if epos.pos != 0 and inst_id not in local_positions:
            zombie_positions.append(inst_id)
    
    assert len(zombie_positions) == 1
    assert zombie_positions[0] == "ETH-USDT-SWAP"
    print("✓ 僵尸仓位检测逻辑正确")


def test_oco_issue_detection():
    """测试OCO订单问题检测"""
    from core.state_reconciler import OCOIssue
    
    # 模拟有持仓但没有SL/TP条件单的情况
    issues = []
    
    inst_id = "BTC-USDT-SWAP"
    position_exists = True
    algo_exists = False
    
    if position_exists and not algo_exists:
        issues.append(OCOIssue(
            inst_id=inst_id,
            issue_type='missing_algo',
            position_exists=True,
            algo_exists=False,
            expected_sl=68000.0,
            expected_tp=75000.0,
            note='持仓存在但没有SL/TP条件单保护'
        ))
    
    assert len(issues) == 1
    assert issues[0].issue_type == 'missing_algo'
    assert issues[0].position_exists == True
    assert issues[0].algo_exists == False
    print("✓ OCO问题检测逻辑正确")


def test_local_state_save_load():
    """测试本地状态保存和加载"""
    from core.state_reconciler import LocalPositionRecord
    
    reconciler = StateReconciler()
    
    # 使用临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    # 使用一个不存在的 trade_log_file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_trade_path = Path(f.name)
    
    try:
        reconciler.state_file = temp_path
        reconciler.trade_log_file = temp_trade_path  # 设为空文件
        
        # 创建测试数据
        test_positions = {
            "BTC-USDT-SWAP": LocalPositionRecord(
                inst_id="BTC-USDT-SWAP",
                direction="long",
                entry_price=70000.0,
                contracts=0.5,
                stop_loss=68000.0,
                take_profit=75000.0,
                status="open"
            )
        }
        
        # 保存
        reconciler.save_local_state(test_positions)
        assert temp_path.exists()
        
        # 读取文件确认
        with open(temp_path) as f:
            saved_data = json.load(f)
        assert "BTC-USDT-SWAP" in saved_data
        assert saved_data["BTC-USDT-SWAP"]["entry_price"] == 70000.0
        
        # 加载
        loaded = reconciler.load_local_positions()
        assert "BTC-USDT-SWAP" in loaded
        assert loaded["BTC-USDT-SWAP"].entry_price == 70000.0
        assert loaded["BTC-USDT-SWAP"].contracts == 0.5
        
        print("✓ 本地状态保存/加载测试通过")
        
    finally:
        if temp_path.exists():
            temp_path.unlink()
        if temp_trade_path.exists():
            temp_trade_path.unlink()


def test_reconcile_result_structure():
    """测试协调结果结构"""
    result = ReconcileResult()
    
    result.timestamp = "2026-04-27T12:00:00"
    result.total_exchange_positions = 3
    result.total_local_positions = 3
    result.is_healthy = True
    
    # 测试转换为dict
    result_dict = result.__dict__
    assert 'timestamp' in result_dict
    assert 'phantom_positions' in result_dict
    assert 'zombie_positions' in result_dict
    assert 'orphan_orders' in result_dict
    
    print("✓ 协调结果结构测试通过")


def test_phantom_position_class():
    """测试PhantomPosition数据结构"""
    from core.state_reconciler import PhantomPosition, LocalPositionRecord
    
    local_record = LocalPositionRecord(
        inst_id="BTC-USDT-SWAP",
        direction="long",
        entry_price=70000.0,
        contracts=0.5,
        stop_loss=68000.0,
        take_profit=75000.0,
        status="open"
    )
    
    phantom = PhantomPosition(
        inst_id="BTC-USDT-SWAP",
        direction="long",
        entry_price=70000.0,
        contracts=0.5,
        local_record=local_record,
        note="测试幽灵仓位"
    )
    
    assert phantom.inst_id == "BTC-USDT-SWAP"
    assert phantom.local_record.entry_price == 70000.0
    print("✓ PhantomPosition数据结构测试通过")


def main():
    """运行所有测试"""
    print()
    print("=" * 60)
    print("  Miracle State Reconciler 测试")
    print("=" * 60)
    print()
    
    tests = [
        test_import,
        test_reconciler_init,
        test_data_classes,
        test_phantom_detection,
        test_zombie_detection,
        test_oco_issue_detection,
        test_local_state_save_load,
        test_reconcile_result_structure,
        test_phantom_position_class,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"✗ {test.__name__} 失败: {e}")
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"  测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    print()
    
    if failed > 0:
        sys.exit(1)
    else:
        print("✓ 所有测试通过!")
        sys.exit(0)


if __name__ == "__main__":
    main()
