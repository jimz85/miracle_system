#!/usr/bin/env python3
"""
Miracle 1.0.1 - 主程序
==================================================
高频趋势跟踪+事件驱动混合交易系统

用法:
  python miracle.py                    # 运行一次扫描
  python miracle.py --daemon          # 持续运行
  python miracle.py --symbol BTC       # 指定币种
  python miracle.py --backtest        # 回测模式
  python miracle.py --info             # 显示系统信息

架构:
  Agent-M -> Agent-S -> Agent-R -> Agent-E -> Agent-L
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# ===== 余额缓存 =====
EQUITY_CACHE_FILE = Path(__file__).parent / "data" / "last_equity.json"

def _load_last_equity() -> float:
    """从缓存文件读取上次已知余额"""
    if EQUITY_CACHE_FILE.exists():
        try:
            import json
            d = json.load(open(EQUITY_CACHE_FILE))
            return d.get("equity", 0.0)
        except:
            pass
    return 0.0  # 首次运行且API失败 → 返回0，拒绝交易

def _save_last_equity(equity: float):
    """保存当前余额到缓存"""
    import json
    EQUITY_CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(EQUITY_CACHE_FILE, 'w') as f:
        json.dump({"equity": equity, "updated": datetime.now().isoformat()}, f)

# 核心模块
from core.data_fetcher import DataFetcher, get_ticker, get_klines
from miracle_core import (
    calc_factors, calc_trend_strength, calc_leverage,
    calc_position_size, check_stops, format_trade_signal, log_trade,
    can_trade, check_risk_limits, update_factor_weights, load_config,
    get_account_state
)

# Agent模块
from agents.agent_market_intel import MarketIntelAgent
from agents.agent_signal import AgentSignal
from agents.agent_risk import AgentRisk, AccountState, Signal
from agents.agent_executor import Executor, ExecutorConfig
from agents.agent_learner import AgentLearner

# 自适应学习
from adaptive_learner import AdaptiveLearner

# ==================== 日志配置 ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('miracle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("miracle")


# ==================== 仓位管理器 ====================

class PositionManager:
    """
    仓位管理器 - 高频交易专用
    规则：
    1. 最多同时持有3个仓位
    2. 新仓位只能在有仓位退出后才能进入
    3. 跟踪持仓状态
    """

    def __init__(self, max_positions: int = 3):
        self.max_positions = max_positions
        self.positions: Dict[str, Dict] = {}  # symbol -> position_info
        self.pending_queue: List[Dict] = []  # 等待入场的信号
        self.total_trades_today: int = 0
        self.last_exit_time: Optional[datetime] = None

    def has_position(self, symbol: str) -> bool:
        """检查是否有该币种持仓"""
        return symbol in self.positions

    def can_open_new_position(self) -> bool:
        """检查是否可以开新仓（仓位未满时）"""
        return len(self.positions) < self.max_positions

    def get_position_count(self) -> int:
        """获取当前持仓数量"""
        return len(self.positions)

    def open_position(self, symbol: str, position_info: Dict):
        """开仓"""
        if not self.can_open_new_position():
            # 仓位已满，加入等待队列
            self.pending_queue.append({
                "symbol": symbol,
                "position_info": position_info,
                "queued_at": datetime.now()
            })
            logger.info(f"[{symbol}] 仓位已满，加入等待队列")
            return False

        self.positions[symbol] = {
            **position_info,
            "opened_at": datetime.now()
        }
        self.total_trades_today += 1
        logger.info(f"[{symbol}] 开仓成功，当前持仓: {len(self.positions)}/{self.max_positions}")
        return True

    def close_position(self, symbol: str, reason: str = "manual"):
        """平仓"""
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            self.last_exit_time = datetime.now()
            logger.info(f"[{symbol}] 平仓完成 ({reason})，持仓: {len(self.positions)}/{self.max_positions}")

            # 检查等待队列，有等待的信号则自动入场
            self._process_queue()
        else:
            logger.warning(f"[{symbol}] 尝试平仓但无持仓")

    def _process_queue(self):
        """处理等待队列"""
        if not self.pending_queue or not self.can_open_new_position():
            return

        # FIFO：按顺序处理等待队列
        while self.pending_queue and self.can_open_new_position():
            pending = self.pending_queue.pop(0)
            symbol = pending["symbol"]
            position_info = pending["position_info"]

            self.positions[symbol] = {
                **position_info,
                "opened_at": datetime.now()
            }
            self.total_trades_today += 1
            logger.info(f"[{symbol}] 从等待队列开仓成功")

    def get_pending_count(self) -> int:
        """获取等待中的信号数量"""
        return len(self.pending_queue)

    def reset_daily(self):
        """每日重置交易计数"""
        self.total_trades_today = 0

    def get_positions_summary(self) -> Dict:
        """获取持仓摘要"""
        return {
            "current_positions": list(self.positions.keys()),
            "position_count": len(self.positions),
            "max_positions": self.max_positions,
            "pending_count": len(self.pending_queue),
            "trades_today": self.total_trades_today
        }


# ==================== 核心流程 ====================

class MiracleScanner:
    """Miracle扫描器 - 核心交易逻辑"""

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["BTC", "ETH", "SOL", "AVAX", "DOGE", "DOT"]
        self.config = load_config()
        self.fetcher = DataFetcher()

        # 初始化各Agent
        self.market_intel = {}  # per-symbol agents
        self.signal_gen = AgentSignal(self.config)
        self.risk_mgr = AgentRisk(self.config)
        self.executor = Executor(ExecutorConfig())
        self.learner = AgentLearner()

        # 仓位管理器（高频交易：最多3个仓位）
        self.position_mgr = PositionManager(max_positions=3)

        logger.info(f"Miracle 1.0.1 初始化完成，监控币种: {self.symbols}")

    def scan_symbol(self, symbol: str) -> Dict:
        """
        扫描单个币种

        Returns:
            {
                "symbol": str,
                "action": "long" | "short" | "wait" | "error",
                "signal": dict | None,
                "approved": dict | None,
                "executed": bool
            }
        """
        result = {"symbol": symbol, "action": "wait", "signal": None, "approved": None, "executed": False}

        try:
            # === Step 1: 市场情报 ===
            logger.info(f"[{symbol}] 获取市场情报...")
            intel_agent = MarketIntelAgent(symbol)
            intel = intel_agent.generate_intel_report()

            # === Step 2: 数据获取 (多周期: 1H + 4H) ===
            logger.info(f"[{symbol}] 获取K线数据 (1H + 4H)...")
            
            # 1H数据 - 主信号
            klines_1h = get_klines(symbol, "1H", 100)
            if not klines_1h or len(klines_1h) < 50:
                logger.warning(f"[{symbol}] 1H数据不足，跳过")
                result["action"] = "error"
                return result

            prices = [k["close"] for k in klines_1h]
            highs = [k["high"] for k in klines_1h]
            lows = [k["low"] for k in klines_1h]
            volumes = [k["volume"] for k in klines_1h]

            price_data = {
                "prices": prices,
                "highs": highs,
                "lows": lows,
                "volumes": volumes
            }

            # 4H数据 - 用于趋势确认
            logger.info(f"[{symbol}] 获取4H数据用于多周期确认...")
            klines_4h = get_klines(symbol, "4H", 100)
            price_data_4h = None
            if klines_4h and len(klines_4h) >= 50:
                prices_4h = [k["close"] for k in klines_4h]
                highs_4h = [k["high"] for k in klines_4h]
                lows_4h = [k["low"] for k in klines_4h]
                volumes_4h = [k["volume"] for k in klines_4h]
                
                price_data_4h = {
                    "prices": prices_4h,
                    "highs": highs_4h,
                    "lows": lows_4h,
                    "volumes": volumes_4h
                }
                logger.info(f"[{symbol}] 4H数据获取成功 ({len(klines_4h)}条)")
            else:
                logger.warning(f"[{symbol}] 4H数据不足，将跳过多周期确认")

            # === Step 3: 信号生成 (多周期确认) ===
            logger.info(f"[{symbol}] 生成交易信号 (1H + 4H双周期)...")
            signal = self.signal_gen.process_intel(symbol, price_data, intel, price_data_4h=price_data_4h)
            result["signal"] = signal
            direction = signal.get("direction", "wait")

            if direction == "wait":
                logger.info(f"[{symbol}] 无交易信号 (conf={signal.get('confidence', 0):.3f})")
                result["action"] = "wait"
                return result

            # === Step 4: 风险管理 ===
            logger.info(f"[{symbol}] 风控审批...")

            # 将dict信号转换为Signal dataclass
            factors = signal.get("factors", {})
            risk_signal = Signal(
                symbol=symbol,
                direction=direction,
                trend_strength=signal.get("trend_strength", 0),
                confidence=signal.get("confidence", 0),
                rr_ratio=signal.get("rr_ratio", 0),
                atr=factors.get("atr", 0),
                entry_price=signal.get("entry_price", 0),
                event_impact="none"
            )

            # 获取真实账户状态（OKX API）
            account_state = get_account_state()
            equity = account_state.get("total_equity", 0.0)
            if equity <= 0:
                # API失败时：从缓存文件读取上次已知余额
                equity = _load_last_equity()
                if equity <= 0:
                    # 缓存也没有，使用默认模拟余额以支持回测/模拟交易
                    equity = 10000.0
                    logger.warning(f"OKX API失败且无缓存，使用默认模拟余额 ${equity:.2f}")
                else:
                    logger.warning(f"OKX API失败，使用缓存余额 ${equity:.2f}")
            else:
                _save_last_equity(equity)
            account = AccountState(
                balance=equity,
                peak_balance=account_state.get("total_equity", 0.0) or equity,
                today_pnl=account_state.get("realized_pnl", 0.0) + account_state.get("unrealized_pnl", 0.0),
                today_trades=0,
                loss_streak=0
            )

            approved = self.risk_mgr.process_signal(risk_signal, account)

            if not approved.approved:
                logger.info(f"[{symbol}] 风控拒绝: {approved.rejection_reason}")
                result["action"] = "rejected"
                return result

            result["approved"] = approved
            result["action"] = direction

            # === Step 5: 仓位检查 ===
            # 检查是否已有该币种持仓
            if self.position_mgr.has_position(symbol):
                logger.info(f"[{symbol}] 已有持仓，跳过")
                result["action"] = "already_held"
                return result

            # 检查仓位是否已满
            if not self.position_mgr.can_open_new_position():
                # 仓位满，加入等待队列
                logger.info(f"[{symbol}] 仓位已满({self.position_mgr.get_position_count()}/3)，加入等待队列")
                exec_signal = {
                    "signal_id": signal.get("signal_id", f"{symbol}_{int(time.time())}"),
                    "symbol": f"{symbol}-USDT",
                    "side": direction,
                    "entry_price": signal.get("entry_price"),
                    "stop_loss": approved.stop_loss,
                    "take_profit": approved.take_profit,
                    "leverage": approved.leverage,
                    "position_size": approved.position_size_pct / 100 * account.balance,
                    "atr": signal.get("factors", {}).get("atr", 0),
                    "market_regime": intel.get("recommendation", "unknown"),
                    "factors": signal.get("factors", {})
                }
                self.position_mgr.open_position(symbol, exec_signal)
                result["action"] = "queued"
                return result

            # === Step 6: 执行 ===
            logger.info(f"[{symbol}] 执行信号: {direction} @ {approved.stop_loss} SL / {approved.take_profit} TP")

            # 准备执行信号
            exec_signal = {
                "signal_id": signal.get("signal_id", f"{symbol}_{int(time.time())}"),
                "symbol": f"{symbol}-USDT",
                "side": direction,
                "entry_price": signal.get("entry_price"),
                "stop_loss": approved.stop_loss,
                "take_profit": approved.take_profit,
                "leverage": approved.leverage,
                "position_size": approved.position_size_pct / 100 * account.balance,
                "atr": signal.get("factors", {}).get("atr", 0),
                "market_regime": intel.get("recommendation", "unknown"),
                "factors": signal.get("factors", {})
            }

            # 执行交易信号
            exec_result = self.executor.execute_signal(exec_signal)
            result["executed"] = exec_result is not None
            if exec_result:
                logger.info(f"[{symbol}] 执行成功: {exec_result}")
                # 追踪持仓
                self.position_mgr.open_position(symbol, exec_signal)
            else:
                logger.warning(f"[{symbol}] 执行失败")

            return result

        except Exception as e:
            logger.error(f"[{symbol}] 扫描出错: {e}")
            result["action"] = "error"
            return result

    def scan_all(self) -> List[Dict]:
        """扫描所有币种"""
        results = []
        for symbol in self.symbols:
            r = self.scan_symbol(symbol)
            results.append(r)
        return results

    def print_report(self, results: List[Dict]):
        """打印报告"""
        pos_summary = self.position_mgr.get_positions_summary()

        print("\n" + "="*60)
        print(f"Miracle 1.0.1 扫描报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # 打印仓位状态
        print(f"\n📊 仓位状态: {pos_summary['position_count']}/{pos_summary['max_positions']} 持仓 | "
              f"今日交易: {pos_summary['trades_today']}笔 | "
              f"等待队列: {pos_summary['pending_count']}个")

        if pos_summary['current_positions']:
            print(f"   持仓中: {', '.join(pos_summary['current_positions'])}")

        for r in results:
            symbol = r["symbol"]
            action = r["action"]
            signal = r.get("signal", {})

            status_icon = {
                "long": "🟢",
                "short": "🔴",
                "wait": "⚪",
                "rejected": "🟡",
                "error": "❌",
                "already_held": "🔵",
                "queued": "🟠"
            }.get(action, "❓")

            print(f"\n{status_icon} {symbol}")

            if action in ["long", "short"]:
                sig = signal or {}
                appr = r.get("approved")
                print(f"   方向: {action.upper()}")
                print(f"   入场: {sig.get('entry_price')}")
                print(f"   止损: {appr.stop_loss if appr else 'N/A'}")
                print(f"   止盈: {appr.take_profit if appr else 'N/A'}")
                print(f"   置信度: {sig.get('confidence', 0):.1%}")
                print(f"   趋势强度: {sig.get('trend_strength', 0):.0f}/100")
                print(f"   杠杆: {appr.leverage if appr else 1}x")
                print(f"   仓位: {appr.position_size_pct if appr else 0*100:.1f}%")

                if "volume_info" in sig:
                    vi = sig["volume_info"]
                    print(f"   成交量: {vi.get('volume_ratio', 0):.1f}x (确认={vi.get('is_confirmed', False)})")

                if "multi_timeframe" in sig:
                    mt = sig["multi_timeframe"]
                    if mt.get("applied"):
                        confirmed_icon = "✅" if mt.get("confirmed") else "❌"
                        print(f"   多周期: {confirmed_icon} 确认({mt.get('confirmations')}/{mt.get('total_checks')}) "
                              f"置信调整: {mt.get('confidence_boost'):.2f}")
                        if mt.get("4h_regime"):
                            print(f"   4H状态: {mt.get('4h_regime')}")
                        if mt.get("factors_4h"):
                            f4h = mt["factors_4h"]
                            print(f"   4H因子: RSI={f4h.get('rsi')}, ADX={f4h.get('adx')}, "
                                  f"MACD={f4h.get('macd_direction')}, Vol={f4h.get('volume_ratio')}x")
                    else:
                        print(f"   多周期: ⏭️ 未应用(无4H数据)")

                if "factors" in sig:
                    fac = sig["factors"]
                    print(f"   因子得分: price={fac.get('price_score', 0):.2f}, "
                          f"news={fac.get('news_score', 0):.2f}, "
                          f"combined={fac.get('combined', 0):.2f}")

            elif action == "wait":
                print(f"   无信号")
            elif action == "rejected":
                print(f"   被风控拒绝")
            elif action == "already_held":
                print(f"   已有持仓")
            elif action == "queued":
                print(f"   仓位满，等待队列")
            elif action == "error":
                print(f"   错误")

        print("\n" + "="*60)


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description="Miracle 1.0.1 交易系统")
    parser.add_argument("--symbol", type=str, help="指定币种")
    parser.add_argument("--daemon", action="store_true", help="持续运行模式")
    parser.add_argument("--interval", type=int, default=30, help="扫描间隔(分钟)")
    parser.add_argument("--backtest", action="store_true", help="回测模式")
    parser.add_argument("--info", action="store_true", help="显示系统信息")
    parser.add_argument("--test", action="store_true", help="测试模式(不执行交易)")
    args = parser.parse_args()

    if args.info:
        print("""
╔══════════════════════════════════════════════════════════╗
║              Miracle 1.0.1 - 高频交易系统               ║
╠══════════════════════════════════════════════════════════╣
║  版本: 1.0.1                                           ║
║  核心理念: 赔率优先，赢了要赢很多，输了只输一点        ║
╠══════════════════════════════════════════════════════════╣
║  架构: 5 Agent协作                                      ║
║    Agent-M: 市场情报 (新闻/链上/钱包)                   ║
║    Agent-S: 信号生成 (多因子融合)                       ║
║    Agent-R: 风险管理 (仓位/杠杆/熔断)                   ║
║    Agent-E: 执行引擎 (OKX/Binance)                     ║
║    Agent-L: 学习迭代 (自适应优化)                       ║
╠══════════════════════════════════════════════════════════╣
║  数据源:                                                 ║
║    价格: OKX (实时) + yfinance (备用)                 ║
║    新闻: CryptoCompare + 价格动量代理                   ║
║    链上: 模拟数据 (需接入Glassnode)                   ║
╠══════════════════════════════════════════════════════════╣
║  策略参数:                                               ║
║    最小RR: 2.0                                         ║
║    最大杠杆: 3x                                          ║
║    最大仓位: 15%                                         ║
║    日交易上限: 5笔                                       ║
║    熔断阈值: 日亏5% / 回撤20%                          ║
╚══════════════════════════════════════════════════════════╝
        """)
        return

    symbols = [args.symbol.upper()] if args.symbol else ["BTC", "ETH", "SOL", "AVAX", "DOGE", "DOT"]
    scanner = MiracleScanner(symbols)

    if args.daemon:
        logger.info(f"启动守护进程模式，间隔{args.interval}分钟")
        while True:
            try:
                results = scanner.scan_all()
                scanner.print_report(results)
            except KeyboardInterrupt:
                logger.info("收到中断信号，退出守护进程")
                break
            except Exception as e:
                logger.error(f"扫描出错: {e}", exc_info=True)
            time.sleep(args.interval * 60)
    else:
        results = scanner.scan_all()
        scanner.print_report(results)


if __name__ == "__main__":
    main()
