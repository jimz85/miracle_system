# Miracle OKX 集成优化说明

**日期:** 2026-04-27  
**优化内容:** 幽灵仓位检测、OCO订单修复、状态文件一致性

---

## 1. 新增模块: `core/state_reconciler.py`

### 功能概述
状态协调器，检测并修复幽灵仓位、孤立订单、僵尸仓位等问题，确保本地状态文件与 OKX 交易所状态一致。

### 核心功能

#### 1.1 幽灵仓位检测 (Phantom Positions)
- **定义:** 本地有持仓记录，但交易所显示持仓为零
- **原因:** 
  - 部分成交导致本地记录与实际不符
  - 网络异常导致订单状态丢失
  - 程序崩溃导致状态未同步
- **检测逻辑:**
  ```python
  for inst_id, lpos in local_positions.items():
      if inst_id not in exchange_positions:
          # 幽灵仓位!
  ```

#### 1.2 僵尸仓位检测 (Zombie Positions)
- **定义:** 交易所有持仓，但本地没有任何记录
- **原因:**
  - 手动在交易所开仓
  - 其他程序/脚本开的仓
  - 状态文件被清除或损坏
- **检测逻辑:**
  ```python
  for inst_id, epos in exchange_positions.items():
      if epos.pos != 0 and inst_id not in local_positions:
          # 僵尸仓位!
  ```

#### 1.3 孤立订单检测 (Orphan Orders)
- **定义:** 交易所有待处理订单，但本地没有对应记录
- **问题:** 可能导致重复下单或资金锁定

#### 1.4 OCO订单问题检测
- **missing_algo:** 有持仓但没有SL/TP条件单
- **orphaned_algo:** 条件单存在但无对应持仓
- **price_mismatch:** 条件单价格与本地记录不匹配

### 数据结构
```python
@dataclass class PositionInfo:
    inst_id: str
    pos: float
    direction: str
    entry_price: float
    algo_ids: List[str]  # 关联的SL/TP algo_id

@dataclass class LocalPositionRecord:
    inst_id: str
    direction: str
    entry_price: float
    contracts: float
    algo_id: Optional[str]  # OCO订单ID
    status: str  # open, closed, phantom, zombie
```

---

## 2. 修复的问题

### 2.1 OCO订单algo_id未保存
**问题:** `execute_signal` 中 OCO 订单的 `algo_id` 没有被保存到交易记录

**修复:** 在 `trade_record` 中添加 `algo_id` 字段
```python
# 提取 algo_id (OCO订单会有)
algo_id = order_result.get("algo_id") if order_result else None

trade_record = {
    ...
    "algo_id": algo_id,  # OCO订单ID
    ...
}
```

### 2.2 平仓时未取消OCO条件单
**问题:** 平仓时 OCO 条件单仍然存在，可能导致二次触发

**修复:** 在 `_close_trade` 中添加取消 OCO 条件单的逻辑
```python
# 取消OCO条件单（如果存在）
if algo_id and self.active_client.exchange == "okx":
    inst_id = symbol.replace("-USDT", "-USDT-SWAP") if "-SWAP" not in symbol else symbol
    self.active_client.cancel_algo_order(inst_id, algo_id)
```

---

## 3. 使用方法

### 3.1 状态检查
```python
from core.state_reconciler import StateReconciler

reconciler = StateReconciler()
result = reconciler.reconcile()

if not result.is_healthy:
    reconciler.print_report(result)
```

### 3.2 自动同步
```python
# 自动修复问题（清理幽灵仓位，导入僵尸仓位）
result = reconciler.reconcile(auto_fix=True)
```

### 3.3 OCO订单修复
```python
stats = reconciler.fix_oco_orders()
print(f"修复结果: {stats}")
```

---

## 4. 文件结构
```
miracle_system/
├── core/
│   ├── __init__.py
│   ├── state_reconciler.py  ← 新增
│   └── ...
├── agents/
│   └── agent_executor.py    ← 修复
└── test_state_reconciler.py ← 新增测试
```

---

## 5. 与Kronos的对比

| 功能 | Kronos | Miracle |
|------|--------|---------|
| 幽灵仓位检测 | ✅ `reconcile_state.py` | ✅ `state_reconciler.py` |
| 僵尸仓位检测 | ✅ | ✅ |
| 孤立订单检测 | ✅ | ✅ |
| OCO订单修复 | 部分 | ✅ 完整实现 |
| 自动同步 | ✅ | ✅ |
| 状态文件一致性 | ✅ | ✅ |

---

## 6. 测试结果
```
✓ StateReconciler 导入成功
✓ StateReconciler 初始化成功
✓ 数据结构测试通过
✓ 幽灵仓位检测逻辑正确
✓ 僵尸仓位检测逻辑正确
✓ OCO问题检测逻辑正确
✓ 本地状态保存/加载测试通过
✓ 协调结果结构测试通过
✓ PhantomPosition数据结构测试通过

============================================================
  测试结果: 9 通过, 0 失败
============================================================
```

---

## 7. 待完善功能

1. **手动同步界面** - 提供命令行工具手动触发同步
2. **飞书告警集成** - 发现幽灵仓位/OCO问题时发送告警
3. **定时检查** - 集成到主循环，每N分钟自动检查
4. **历史追踪** - 记录每次同步的结果和操作
