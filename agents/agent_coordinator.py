#!/usr/bin/env python3
from __future__ import annotations

"""
Miracle 1.0.1 - 协调器Agent (Agent-Coordinator)
================================================
负责任务调度和Agent间通信

协调流程：
  1. 每30分钟触发一次扫描
  2. 并行调度 Agent-M, Agent-S
  3. Agent-S完成后调度 Agent-R
  4. Agent-R完成后调度 Agent-E
  5. 交易结束后调度 Agent-L 记录学习

用法：
  python agent_coordinator.py              # 扫描一次
  python agent_coordinator.py --daemon     # 持续运行
  python agent_coordinator.py --status     # 查看状态
"""

import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from miracle_core import calc_factors, calc_trend_strength, format_trade_signal, log_trade

# ==================== 日志 ====================
# 只在root logger尚无handler时配置（防止多次调用basicConfig覆盖）
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('logs/coordinator.log'),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
CONFIG_PATH = Path(__file__).parent.parent / "miracle_config.json"
STATE_PATH = Path(__file__).parent.parent / "data" / "coordinator_state.json"

# ==================== Agent接口抽象 ====================

class MarketIntelAgent(ABC):
    """市场情报Agent接口"""
    @abstractmethod
    def generate_intel_report(self, symbol: str) -> Dict[str, Any]:
        """生成市场情报报告"""
        pass

class SignalGeneratorAgent(ABC):
    """信号生成Agent接口"""
    @abstractmethod
    def generate_signal(self, symbol: str, price_data: Dict, intel_report: Dict) -> Dict[str, Any]:
        """生成交易信号"""
        pass

class RiskManagerAgent(ABC):
    """风险管理Agent接口"""
    @abstractmethod
    def process_signal(self, signal: Dict, account_state: Dict) -> Dict[str, Any]:
        """处理交易信号，返回风控审批结果"""
        pass

class ExecutorAgent(ABC):
    """执行Agent接口"""
    @abstractmethod
    def execute_signal(self, approved_signal: Dict) -> Dict[str, Any]:
        """执行已批准的信号"""
        pass

class LearnerAgent(ABC):
    """学习Agent接口"""
    @abstractmethod
    def on_trade_entry(self, trade_data: Dict) -> tuple:
        """记录交易入场，返回(pattern_key, is_allowed, trade_id)"""
        pass

# ==================== Agent导入 ====================
try:
    from agents.agent_executor import Executor, TradeLogger
    from agents.agent_learner import AgentLearner, FactorAnalyzer, TradeRecorder
    from agents.agent_market_intel import NewsIntel, OnChainIntel, WalletIntel
    from agents.agent_risk import AgentRisk as RiskManager
    from agents.agent_risk import CircuitBreaker
    from agents.agent_signal import PriceFactors, SignalGenerator
    AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agent导入失败: {e}，协调器将仅做演示")
    AGENTS_AVAILABLE = False


class CoordinatorState:
    """协调器状态管理"""
    
    def __init__(self, state_path):
        self.state_path = state_path
        self.state = self._load_state()
        
    def _load_state(self):
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "last_scan_time": None,
            "last_trade_time": None,
            "consecutive_losses": 0,
            "today_trades": 0,
            "today_pnl": 0.0,
            "daily_loss_stops": 0,
            "total_trades": 0,
            "is_paused": False,
            "pause_until": None
        }
        
    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)
            
    def can_trade(self):
        """检查是否可以交易"""
        # 检查暂停状态
        if self.state.get("is_paused"):
            pause_until = self.state.get("pause_until")
            if pause_until and datetime.now() < datetime.fromisoformat(pause_until):
                return False, "系统暂停中"
            else:
                self.state["is_paused"] = False
                self.save()
                
        # 检查日交易次数
        if self.state.get("today_trades", 0) >= 5:
            return False, "已达今日交易上限(5笔)"
            
        # 检查交易间隔
        last_trade = self.state.get("last_trade_time")
        if last_trade:
            last_time = datetime.fromisoformat(last_trade)
            if datetime.now() - last_time < timedelta(hours=2):
                return False, "交易间隔不足2小时"
                
        return True, "可以交易"
        
    def record_trade(self, trade_result):
        """记录交易结果"""
        self.state["last_trade_time"] = datetime.now().isoformat()
        self.state["today_trades"] += 1
        self.state["total_trades"] += 1
        
        if trade_result.get("pnl", 0) < 0:
            self.state["consecutive_losses"] += 1
            if self.state["consecutive_losses"] >= 2:
                self.state["is_paused"] = True
                self.state["pause_until"] = (datetime.now() + timedelta(hours=24)).isoformat()
                logger.warning(f"连续亏损{self.state['consecutive_losses']}笔，系统暂停24小时")
        else:
            self.state["consecutive_losses"] = 0
            
        self.save()


class MiracleCoordinator:
    """
    Miracle 1.0.1 协调器

    调度流程：
      scan() -> Agent-M + Agent-S (并行) -> Agent-R -> Agent-E -> Agent-L

    支持依赖注入，可通过构造函数传入自定义Agent实现。
    """

    def __init__(self, symbols=None,
                 market_intel: MarketIntelAgent | None = None,
                 signal_gen: SignalGeneratorAgent | None = None,
                 risk_mgr: RiskManagerAgent | None = None,
                 executor: ExecutorAgent | None = None,
                 learner: LearnerAgent | None = None):
        """初始化协调器"""
        self.symbols = symbols or ["BTC", "ETH", "SOL", "AVAX", "DOGE", "DOT"]

        # 加载配置
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)

        # 初始化状态
        self.state = CoordinatorState(STATE_PATH)

        # 依赖注入：如果提供了Agent实例则使用，否则创建默认实例
        if market_intel is not None:
            self.market_intel = market_intel
        elif AGENTS_AVAILABLE:
            self.news_intel = NewsIntel()
            self.onchain_intel = OnChainIntel()
            self.wallet_intel = WalletIntel()
            self.price_factors = PriceFactors()
            # 组合市场情报Agent
            self.market_intel = self._create_default_market_intel()
        else:
            self.market_intel = None

        if signal_gen is not None:
            self.signal_gen = signal_gen
        elif AGENTS_AVAILABLE:
            self.signal_gen = SignalGenerator(self.config)
        else:
            self.signal_gen = None

        if risk_mgr is not None:
            self.risk_mgr = risk_mgr
        elif AGENTS_AVAILABLE:
            self.risk_mgr = RiskManager(self.config)
        else:
            self.risk_mgr = None

        if executor is not None:
            self.executor = executor
        elif AGENTS_AVAILABLE:
            self.executor = Executor(self.config)
        else:
            self.executor = None

        if learner is not None:
            self.agent_learner = learner
        elif AGENTS_AVAILABLE:
            self.agent_learner = AgentLearner(base_dir=str(Path(__file__).parent.parent))
        else:
            self.agent_learner = None

        if not all([self.market_intel, self.signal_gen, self.risk_mgr, self.executor]):
            logger.warning("Agent模块未完整加载，协调器运行在演示模式")

        logger.info(f"Miracle 1.0.1 协调器初始化完成，监控币种: {self.symbols}")

    def _create_default_market_intel(self):
        """创建默认市场情报Agent（组合NewsIntel, OnChainIntel, WalletIntel）"""
        class CombinedMarketIntel(MarketIntelAgent):
            def __init__(self, news_intel, onchain_intel, wallet_intel, price_factors):
                self.news_intel = news_intel
                self.onchain_intel = onchain_intel
                self.wallet_intel = wallet_intel
                self.price_factors = price_factors

            def generate_intel_report(self, symbol: str) -> Dict[str, Any]:
                report = self.news_intel.generate_intel_report(symbol)
                report["onchain"] = self.onchain_intel.generate_intel_report(symbol)
                report["wallet"] = self.wallet_intel.generate_intel_report(symbol)
                return report

        return CombinedMarketIntel(
            self.news_intel, self.onchain_intel, self.wallet_intel, self.price_factors
        )
        
    def scan(self, symbol):
        """
        单币种扫描流程
        
        Returns:
            dict: 扫描结果，包含是否执行了交易
        """
        logger.info(f"=== 开始扫描 {symbol} ===")
        result = {"symbol": symbol, "trades_executed": 0, "signals": []}
        
        # === 阶段1: Agent-M (市场情报) ===
        intel_report = None
        if AGENTS_AVAILABLE:
            try:
                intel_report = self._run_market_intel(symbol)
                logger.info(f"  市场情报: 新闻={intel_report.get('news_sentiment',{}).get('score',0):.2f}, "
                          f"链上={intel_report.get('onchain',{}).get('exchange_flow_signal',0):.2f}, "
                          f"钱包={intel_report.get('wallet',{}).get('concentration_signal',0):.2f}")
            except Exception as e:
                logger.error(f"  Agent-M执行失败: {e}")
                intel_report = self._get_default_intel_report(symbol)
        else:
            intel_report = self._get_default_intel_report(symbol)
            
        # === 阶段2: Agent-S (信号生成) ===
        signal = None
        if AGENTS_AVAILABLE:
            try:
                signal = self._run_signal_generator(symbol, intel_report)
                logger.info(f"  信号生成: {signal.get('direction', 'wait')} | "
                          f"置信度={signal.get('confidence', 0):.2f} | "
                          f"趋势强度={signal.get('trend_strength', 0):.0f} | "
                          f"RR={signal.get('rr_ratio', 0):.2f}")
            except Exception as e:
                logger.error(f"  Agent-S执行失败: {e}")
                signal = None
        else:
            signal = self._demo_signal(symbol)
            
        if signal and signal.get("direction") != "wait":
            result["signals"].append(signal)
            
            # === 阶段3: Agent-R (风险管理) ===
            approved_signal = None
            if AGENTS_AVAILABLE:
                try:
                    approved_signal = self._run_risk_manager(signal)
                    logger.info(f"  风控审批: {'通过' if approved_signal.get('approved') else '拒绝'} | "
                              f"杠杆={approved_signal.get('leverage', 1)}x | "
                              f"仓位={approved_signal.get('position_size_pct', 0):.1%}")
                except Exception as e:
                    logger.error(f"  Agent-R执行失败: {e}")
                    approved_signal = None
            else:
                approved_signal = self._demo_risk_approval(signal)
                
            if approved_signal and approved_signal.get("approved"):
                # === 阶段4: Agent-E (执行) ===
                can_trade, reason = self.state.can_trade()
                if can_trade:
                    try:
                        trade_result = self._run_executor(approved_signal)
                        if trade_result:
                            self.state.record_trade(trade_result)
                            result["trades_executed"] += 1
                    except Exception as e:
                        logger.error(f"  Agent-E执行失败: {e}")
                else:
                    logger.info(f"  跳过执行: {reason}")
            else:
                logger.info(f"  信号被风控拒绝: {approved_signal.get('rejection_reason', '未知原因') if approved_signal else '无信号'}")
        else:
            logger.info("  无交易信号，继续观望")
            
        logger.info(f"=== {symbol} 扫描完成 ===")
        return result
        
    def scan_all(self):
        """扫描所有币种"""
        results = []
        for symbol in self.symbols:
            try:
                result = self.scan(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"扫描{symbol}时出错: {e}")
        return results
        
    def _run_market_intel(self, symbol):
        """执行市场情报Agent"""
        if hasattr(self.market_intel, 'generate_intel_report'):
            return self.market_intel.generate_intel_report(symbol)
        # 兼容旧的分开调用方式
        report = self.news_intel.generate_intel_report(symbol)
        report["onchain"] = self.onchain_intel.generate_intel_report(symbol)
        report["wallet"] = self.wallet_intel.generate_intel_report(symbol)
        return report
        
    def _run_signal_generator(self, symbol, intel_report):
        """执行信号生成Agent"""
        # 获取价格数据（简化版，实际应从数据源获取）
        from miracle_core import get_recent_price_data
        price_data = get_recent_price_data(symbol, days=30)
        
        signal = self.signal_gen.generate_signal(symbol, price_data, intel_report)
        return signal
        
    def _run_risk_manager(self, signal):
        """执行风险管理Agent"""
        from miracle_core import get_account_state
        account_state = get_account_state()
        approved_signal = self.risk_mgr.process_signal(signal, account_state)
        return approved_signal
        
    def _run_executor(self, approved_signal):
        """执行执行引擎Agent + 记录到Agent-L学习系统"""
        trade_result = self.executor.execute_signal(approved_signal)

        # === 阶段5: Agent-L 记录入场（自学习系统）===
        if trade_result and AGENTS_AVAILABLE and hasattr(self, 'agent_learner'):
            try:
                trade_data = {
                    "symbol": approved_signal.get("symbol", "").replace("-USDT-SWAP", "").replace("-USDT", ""),
                    "direction": approved_signal.get("side", "long"),
                    "entry_time": datetime.now().isoformat(),
                    "entry_price": approved_signal.get("entry_price", 0),
                    "stop_loss": approved_signal.get("stop_loss", 0),
                    "take_profit": approved_signal.get("take_profit", 0),
                    "factors": approved_signal.get("factors", {}),
                    "market_regime": approved_signal.get("market_regime", "RANGE"),
                }
                pattern_key, is_allowed, trade_id = self.agent_learner.on_trade_entry(trade_data)
                # 将trade_id存入trade_result，出场时用于自学习反馈
                trade_result["trade_id"] = trade_id
                trade_result["pattern_key"] = pattern_key
                logger.info(f"  Agent-L记录: 模式={pattern_key} trade_id={trade_id}")
            except Exception as e:
                logger.error(f"  Agent-L记录失败: {e}")

            # 注册出场回调：平仓时自动触发自学习反馈
            self._register_exit_callback()

        return trade_result

    def _get_pattern_key_for_trade(self, trade_id) -> str:
        """从数据库读取交易的pattern_key"""
        if not hasattr(self.agent_learner, 'trade_recorder'):
            return ""
        try:
            import sqlite3
            db_path = self.agent_learner.db_path
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT pattern_key FROM trades WHERE id = ?", [trade_id]
                ).fetchone()
                return row[0] if row else ""
        except Exception:
            return ""

    def _register_exit_callback(self):
        """注册出场回调：平仓时自动更新因子/模式统计"""
        if not hasattr(self.executor, 'register_callback'):
            return

        def on_exit_callback(trade_record):
            """出场时更新Agent-L和SignalGenerator的统计"""
            try:
                trade_id = trade_record.get("trade_id")
                if not trade_id:
                    trade_id = trade_record.get("id")

                exit_data = {
                    "exit_time": trade_record.get("exit_time", datetime.now().isoformat()),
                    "exit_price": trade_record.get("exit_price") or trade_record.get("_exit_price", 0),
                    "pnl": trade_record.get("pnl", 0),
                    "exit_reason": trade_record.get("exit_reason", "unknown"),
                }

                # 1. 更新Agent-L的统计
                self.agent_learner.on_trade_exit(trade_id, exit_data)
                logger.info(f"  Agent-L出场更新: trade_id={trade_id} exit={exit_data['exit_reason']} pnl={exit_data['pnl']:.2f}")

                # 2. 同步更新SignalGenerator的内存pattern_db
                if hasattr(self, 'signal_gen') and hasattr(self.signal_gen, 'generator'):
                    # 从数据库直接读取pattern_key（trade_logger可能不保留它）
                    pattern_key = self._get_pattern_key_for_trade(trade_id)
                    if pattern_key:
                        pnl = exit_data.get("pnl", 0)
                        rr = abs(trade_record.get("rr", 0))
                        self.signal_gen.generator.update_pattern_db({
                            "pattern_key": pattern_key,
                            "won": pnl > 0,
                            "actual_rr": rr
                        })
                        logger.info(f"  SignalGenerator模式更新: {pattern_key} won={pnl>0}")
            except Exception as e:
                logger.error(f"  Agent-L出场回调失败: {e}")

        self.executor.register_callback("on_exit", on_exit_callback)
        
    def _get_default_intel_report(self, symbol):
        """获取默认情报报告（演示模式）"""
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "news_sentiment": {"score": 0.0, "labels": ["中性:100%"]},
            "onchain": {"exchange_flow_signal": 0.0, "large_transfer_count": 0},
            "wallet": {"concentration_signal": 0.0, "top10_pct": 45.0},
            "combined_score": 0.0,
            "recommendation": "观望",
            "confidence": 0.0
        }
        
    def _demo_signal(self, symbol):
        """演示模式信号（无Agent时）"""
        import random
        directions = ["long", "short", "wait"]
        direction = random.choice(directions)
        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": 70000 if direction != "wait" else None,
            "stop_loss": 68600 if direction == "long" else (71400 if direction == "short" else None),
            "take_profit": 74200 if direction == "long" else (68600 if direction == "short" else None),
            "rr_ratio": 3.0,
            "confidence": 0.65,
            "trend_strength": 55,
            "leverage_recommended": 2,
            "position_size_pct": 0.02
        }
        
    def _demo_risk_approval(self, signal):
        """演示模式风控审批"""
        return {
            "approved": signal.get("direction") != "wait",
            "modified_signal": signal,
            "leverage": 2,
            "position_size_pct": 0.02,
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "risk_reward": signal.get("rr_ratio", 2.0)
        }


def run_daemon(interval_minutes=30):
    """持续运行协调器"""
    coordinator = MiracleCoordinator()
    logger.info(f"协调器守护进程启动，每 {interval_minutes} 分钟扫描一次")
    
    while True:
        try:
            coordinator.scan_all()
        except Exception as e:
            logger.error(f"扫描出错: {e}")
            
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Miracle 1.0.1 协调器')
    parser.add_argument('--daemon', action='store_true', help='持续运行模式')
    parser.add_argument('--status', action='store_true', help='查看状态')
    parser.add_argument('--symbol', type=str, help='指定扫描币种')
    args = parser.parse_args()
    
    if args.status:
        state_path = STATE_PATH
        if state_path.exists():
            with open(state_path) as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print("无状态记录")
            
    elif args.daemon:
        run_daemon()
        
    else:
        coordinator = MiracleCoordinator()
        if args.symbol:
            coordinator.scan(args.symbol)
        else:
            results = coordinator.scan_all()
            print(json.dumps(results, indent=2, default=str))
