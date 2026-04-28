# Miracle System — 入口文件说明

本文档说明 Miracle System 的 5 个主要入口文件（入口点）的用途、命令行参数和使用场景。

---

## 1. `miracle.py` — 主程序（5 Agent协作）

**类型**: 交互式 / 守护进程交易扫描器  
**架构**: 5 Agent 协作系统

### 功能
多币种（BTC/ETH/SOL/AVAX/DOGE/DOT）高频趋势跟踪 + 事件驱动混合交易系统。

### Agent 架构
| Agent | 职责 |
|-------|------|
| Agent-M | 市场情报（新闻/链上/钱包） |
| Agent-S | 信号生成（多因子融合） |
| Agent-R | 风险管理（仓位/杠杆/熔断） |
| Agent-E | 执行引擎（OKX/Binance） |
| Agent-L | 学习迭代（自适应优化） |

### 策略参数
- 最小 Risk-Reward: 2.0
- 最大杠杆: 3x
- 最大仓位: 15%
- 日交易上限: 5笔
- 熔断阈值: 日亏5% / 回撤20%

### 命令行用法
```bash
python miracle.py                          # 单次扫描全部币种
python miracle.py --symbol BTC             # 指定币种
python miracle.py --daemon --interval 30   # 守护进程模式，每30分钟扫描
python miracle.py --backtest               # 回测模式
python miracle.py --info                   # 显示系统信息
python miracle.py --test                   # 测试模式（不执行交易）
```

---

## 2. `miracle_autonomous.py` — 自主研究循环

**类型**: 参数自动搜索 / Walk-Forward 优化  
**架构**: Ralph 风格的策略淘汰系统

### 功能
整合 **数据收集 → 假设生成 → 回测验证 → 反思改进** 的完整闭环，自动寻找最优策略参数。

### 工作流
1. 从 Memory Log 获取历史决策和结果
2. 生成新的策略假设
3. Walk-Forward 验证（expanding / rolling / rolling_recent）
4. 自适应淘汰低效策略，保留高效策略
5. 迭代 `n` 次后输出最佳配置

### 命令行用法
```bash
python miracle_autonomous.py                              # 默认50次实验
python miracle_autonomous.py --experiments 100            # 最大100次实验
python miracle_autonomous.py --coins BTC,ETH,SOL          # 指定币种列表
python miracle_autonomous.py --timeframe 4h               # K线时间周期
python miracle_autonomous.py --max-time 240                # 最大运行240分钟
python miracle_autonomous.py --wf-mode expanding          # Walk-Forward模式
python miracle_autonomous.py --resume                     # 从上次状态恢复
```

### 输出
- `results/best_config.json` — 最佳策略配置
- `results/autonomous_loop_state.json` — 恢复点（支持 `--resume`）

---

## 3. `miracle_core.py` — 核心交易引擎

**类型**: 因子计算 + 策略信号引擎  
**架构**: 纯函数式因子库（无状态）

### 功能
提供 Miracle 1.0.1 的核心因子计算和信号生成能力，包括：
- 趋势强度计算（`calc_trend_strength`）
- 杠杆计算（`calc_leverage`）
- 交易信号格式化（`format_trade_signal`）

### 主要导出函数
```python
from miracle_core import (
    calc_factors,        # 计算多因子信号（RSI/ADX/MACD/Bollinger/Momentum）
    calc_trend_strength, # 计算趋势强度 (0-100)
    calc_leverage,       # 计算建议杠杆
    format_trade_signal, # 格式化交易信号
    get_ic_adjusted_weights,  # 获取IC动态权重
)
```

### 命令行用法（自检/示例）
```bash
python miracle_core.py  # 运行示例数据演示因子计算和信号生成
```

---

## 4. `miracle_kronos.py` — Miracle-Kronos 统一交易系统

**类型**: 实盘/模拟盘交易引擎  
**架构**: Miracle + Kronos 双系统融合

### 功能
合并 Kronos 和 Miracle 两个系统的最优部分，实现统一交易决策：
- OKX USDT 永续合约交易
- BTC 4H 趋势过滤
- 多币种持仓管理（MAX_POSITIONS）
- 熔断机制（Tier 1-4）
- gemma4 自适应学习否决 + 黑名单
- Treasury 快照管理（日/小时）

### 模式
| 模式 | 说明 |
|------|------|
| `audit`（默认） | 只读审计模式，不下单 |
| `live` | 实盘交易模式 |

### 命令行用法
```bash
python miracle_kronos.py                          # 审计模式（默认）
python miracle_kronos.py --mode live             # 实盘交易模式
python miracle_kronos.py --mode audit --equity 50000  # 指定初始equity
```

### 核心流程
1. 获取 OKX 账户 equity
2. 获取 BTC 4H 趋势（bull/bear/neutral）
3. 运行 `run_scan()` 生成交易决策
4. gemma4 否决 → 加入黑名单
5. 高 urgency 平仓决策 → 执行 close
6. 更新 Treasury 快照

---

## 5. `miracle_pilot.py` — 驾驶舱状态监控面板

**类型**: 监控/报表面板  
**架构**: 命令行 UI（无 Web 服务）

### 功能
提供 Miracle 交易系统的实时状态监控，包含四大模块：
1. **状态摘要** (`--status`) — 纸质交易胜率统计
2. **持仓监控** (`--positions`) — 当前持仓明细
3. **信号列表** (`--signals`) — 实时信号列表
4. **风险仪表盘** (`--risk`) — 风险指标展示
5. **完整日报** (`--full`) — 综合日报
6. **日志查看** (`--log N`) — 查看最近 N 行日志

### 命令行用法
```bash
python miracle_pilot.py              # 默认：完整驾驶舱概览
python miracle_pilot.py --status     # 纸质交易胜率统计
python miracle_pilot.py --positions   # 持仓监控
python miracle_pilot.py --signals     # 实时信号列表
python miracle_pilot.py --risk        # 风险仪表盘
python miracle_pilot.py --full        # 完整日报
python miracle_pilot.py --log 50      # 查看最近50行日志
```

---

## 入口文件对比

| 文件 | 类型 | 持久化运行 | 交易执行 | 搜索/优化 | 监控面板 |
|------|------|-----------|---------|----------|---------|
| `miracle.py` | 交互式/守护进程 | ✅ (`--daemon`) | ✅ | ❌ | ❌ |
| `miracle_autonomous.py` | 参数搜索 | ❌ | ❌ | ✅ | ❌ |
| `miracle_core.py` | 因子计算库 | ❌ | ❌ | ❌ | ❌ |
| `miracle_kronos.py` | 实盘引擎 | ✅ | ✅ | ❌ | ❌ |
| `miracle_pilot.py` | 监控面板 | ❌ | ❌ | ❌ | ✅ |

---

## 入口文件依赖关系

```
miracle.py
  └── miracle_core.py (calc_factors, get_ic_adjusted_weights)
        └── core/ic_weights.py (IC动态权重)

miracle_kronos.py
  ├── core/ic_weights.py (IC动态权重)
  ├── agents/agent_learner.py (自适应学习)
  └── agents/agent_signal.py (信号生成)

miracle_autonomous.py
  ├── miracle_core.py (因子计算)
  └── Walk-Forward 验证框架

miracle_pilot.py
  ├── miracle_kronos.py (状态读取)
  └── state/ (Treasury/Trade日志)
```
