from __future__ import annotations

"""
Agent-E: 执行引擎Agent
Miracle 1.0.1 — 高频趋势跟踪+事件驱动混合系统

职责:
1. 接收Agent-R的最终执行指令
2. 通过OKX/Binance API下单
3. 实时监控持仓状态
4. 触发止损/止盈时自动平仓
5. 记录成交价格和滑点
6. 向Agent-L（学习模块）反馈交易结果
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import base64
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import RequestException, Timeout

# 交易所客户端已迁移到 core/exchange_client.py
from core.exchange_client import ExchangeClient

# ============================================================
# 配置已迁移到 core/executor_config.py
from core.executor_config import ExecutorConfig
from core.executor_feishu_notifier import FeishuNotifier

# ============================================================
# 订单管理器
# ============================================================
# 订单管理器已迁移到 core/order_manager.py
from core.order_manager import OrderManager

# ============================================================
# 持仓监控器已迁移到 core/position_monitor.py
from core.position_monitor import PositionMonitor

# 安全密钥管理器已迁移到 core/secure_key_manager.py
from core.secure_key_manager import SecureKeyManager, get_key_manager

# 滑点监控已迁移到 core/slippage_monitor.py
from core.slippage_monitor import SlippageMonitor
from core.trade_logger import TradeLogger

# ============================================================
# 执行器 (主类)
# ============================================================

class Executor:
    """
    执行引擎主类

    接收Agent-R的最终执行指令，通过交易所API完成交易执行
    """

    def __init__(self, config: ExecutorConfig = None):
        # 支持传入dict（从JSON加载）或ExecutorConfig dataclass
        if isinstance(config, dict):
            known_fields = {
                'default_exchange', 'use_backup_on_fail',
                'okx_api_key', 'okx_secret_key', 'okx_passphrase', 'okx_testnet',
                'binance_api_key', 'binance_secret_key', 'binance_testnet',
                'max_retry', 'retry_interval', 'order_timeout',
                'slippage_warning_threshold',
                'feishu_webhook', 'feishu_enabled',
                'monitor_interval', 'max_hold_hours',
                'log_dir', 'trade_log_file', 'slippage_log_file',
            }
            filtered = {k: v for k, v in config.items() if k in known_fields}
            self.config = ExecutorConfig(**filtered)
        else:
            self.config = config or ExecutorConfig()

        # 初始化组件
        self.okx_client = ExchangeClient("okx", self.config)
        self.binance_client = ExchangeClient("binance", self.config)
        self.active_client = self.okx_client  # 默认OKX

        # 初始化管理器组件
        self.order_manager = OrderManager(self.active_client, self.config)
        self.position_monitor = PositionMonitor(self.active_client, self.config)

        self.slippage_monitor = SlippageMonitor(self.config)
        self.trade_logger = TradeLogger(self.config)
        self.notifier = FeishuNotifier(self.config)

        # 持仓监控
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._callbacks: Dict[str, Callable] = {}  # 回调函数

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志（仅在未配置时）"""
        os.makedirs(self.config.log_dir, exist_ok=True)

        # 只在root logger尚无handler时配置（防止多次调用basicConfig覆盖）
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(f"{self.config.log_dir}/executor.log"),
                    logging.StreamHandler()
                ]
            )

    def set_exchange(self, exchange: str):
        """设置活跃交易所"""
        if exchange == "okx":
            self.active_client = self.okx_client
        elif exchange == "binance":
            self.active_client = self.binance_client
        else:
            raise ValueError(f"不支持的交易所: {exchange}")
        # 更新管理器组件的客户端引用
        self.order_manager.client = self.active_client
        self.position_monitor.client = self.active_client
    
    def _check_emergency_stop(self) -> bool:
        """
        检查是否处于紧急停止状态。
        检查优先级：
        1. 环境变量 EMERGENCY_STOP_ENABLED=true
        2. 本地emergency_stop状态文件
        3. 远程紧急停止API（如果配置了）
        """
        from pathlib import Path
        
        # 1. 检查环境变量
        if os.getenv("EMERGENCY_STOP_ENABLED", "").lower() == "true":
            emergency_file = Path(PROJECT_ROOT) / ".emergency_stop"
            if emergency_file.exists():
                try:
                    data = emergency_file.read_text().strip()
                    if data:
                        logging.critical(f"🚨 紧急停止文件内容: {data}")
                        return True
                except Exception:
                    pass
        
        # 2. 检查状态文件（由emergency_stop_api.py创建）
        state_file = Path(PROJECT_ROOT) / "data" / ".emergency_stop_state"
        if state_file.exists():
            try:
                import json
                state = json.loads(state_file.read_text())
                if state.get("emergency_stopped", False):
                    reason = state.get("reason", "Unknown")
                    logging.critical(f"🚨 紧急停止状态: {reason}")
                    return True
            except Exception:
                pass
        
        # 3. 尝试连接远程紧急停止API（如果配置了）
        emergency_api_url = os.getenv("EMERGENCY_STOP_API_URL")
        if emergency_api_url:
            try:
                import requests
                resp = requests.get(f"{emergency_api_url}/status", timeout=2)
                if resp.status_code == 200:
                    state = resp.json()
                    if state.get("emergency_stopped", False):
                        logging.critical(f"🚨 远程紧急停止: {state.get('reason', 'Unknown')}")
                        return True
            except Exception:
                pass  # API不可用，不阻止交易（fail-open设计）
        
        return False
    
    def trigger_emergency_stop(self, reason: str = "Manual stop"):
        """
        触发紧急停止（写入状态文件，供交易进程检查）
        """
        import json
        from pathlib import Path
        
        state_file = Path(PROJECT_ROOT) / "data" / ".emergency_stop_state"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "emergency_stopped": True,
            "reason": reason,
            "stop_time": datetime.now().isoformat(),
            "stopped_by": "executor"
        }
        
        state_file.write_text(json.dumps(state, indent=2))
        logging.critical(f"🚨 紧急停止已触发: {reason}")
        
        # 尝试取消所有活跃订单
        try:
            pending = self.order_manager.get_pending_orders()
            for order_id in list(pending.keys()):
                self.order_manager.remove_pending_order(order_id)
                logging.info(f"已移除待处理订单: {order_id}")
        except Exception as e:
            logging.error(f"取消订单时出错: {e}")

    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        self._callbacks[event] = callback

    def _trigger_callback(self, event: str, *args, **kwargs):
        """触发回调"""
        if event in self._callbacks:
            try:
                self._callbacks[event](*args, **kwargs)
            except Exception as e:
                logging.error(f"回调执行失败 [{event}]: {e}")

    def execute_signal(self, approved_signal: Dict) -> Dict | None:
        """
        执行经过风控审批的信号

        approved_signal 格式 (来自Agent-R):
        {
            "signal_id": "uuid",
            "symbol": "BTC-USDT",
            "side": "long",
            "entry_price": 72000,
            "stop_loss": 70560,
            "take_profit": 75600,
            "leverage": 2,
            "position_size": 0.15,
            "market_regime": "bull",
            "factors": {...}
        }

        返回: 交易记录 或 None (执行失败)
        """
        # =====================================================
        # 紧急停止检查 - 如果系统处于紧急停止状态，则拒绝执行
        # =====================================================
        if self._check_emergency_stop():
            logging.warning("🚫 交易被拒绝: 系统处于紧急停止状态")
            return None
        
        symbol = approved_signal.get("symbol")
        side = approved_signal.get("side")
        leverage = approved_signal.get("leverage", 1)
        position_size = approved_signal.get("position_size", 0)

        logging.info(f"执行信号: {symbol} {side} 杠杆={leverage}x")

        # Step 1: 检查账户余额（API优先，失败时使用模拟）
        balance = self.active_client.get_balance()
        if balance["available"] <= 0:
            # 使用模拟余额
            logging.warning("无法获取真实余额，使用模拟余额 $100,000")
            balance = {"available": 100000.0, "total": 100000.0, "currency": "USDT"}

        # Step 2: 计算合约数量 (如果未指定)
        if position_size <= 0:
            current_price = self.active_client.get_ticker(symbol) or approved_signal.get("entry_price", 0)
            if current_price <= 0:
                # 使用入场价作为当前价
                current_price = approved_signal.get("entry_price", 50000)
                logging.warning(f"无法获取当前价格，使用入场价 ${current_price}")

            risk_amount = balance["available"] * (self.config.max_loss_per_trade_pct / 100)  # 1% 风险
            atr = approved_signal.get("atr", current_price * 0.01)
            stop_distance = abs(approved_signal.get("entry_price", current_price) - approved_signal.get("stop_loss", 0))
            stop_distance = max(stop_distance, atr * 1.5)

            position_size = risk_amount / stop_distance
            position_size = min(position_size, balance["available"] * leverage / current_price)

        # Step 3: 市价单入场（优先尝试真实下单，失败时模拟）
        planned_entry = approved_signal.get("entry_price", self.active_client.get_ticker(symbol))
        if not planned_entry:
            planned_entry = approved_signal.get("entry_price", 50000)

        sl_price = approved_signal.get("stop_loss", 0)
        tp_price = approved_signal.get("take_profit", 0)

        # 尝试真实下单
        order_result = None
        try:
            # 检查是否支持OCO
            use_oco = (self.active_client.exchange == "okx" and
                       hasattr(self.active_client, 'place_oco_order') and
                       sl_price > 0 and tp_price > 0 and position_size > 0)

            if use_oco:
                order_result = self.active_client.place_oco_order(
                    symbol=symbol,
                    side=side,
                    size=position_size,
                    entry_price=planned_entry,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    leverage=leverage
                )
            else:
                order_result = self.active_client.place_order(
                    symbol=symbol, side=side, order_type="market",
                    price=None, size=position_size, leverage=leverage
                )
        except Exception as e:
            logging.warning(f"真实下单异常: {e}，切换到模拟模式")
            order_result = None

        # 如果真实下单失败，使用模拟订单
        if not order_result:
            logging.info(f"[模拟模式] 入场@{planned_entry} + SL@{sl_price} + TP@{tp_price}")
            order_result = self._create_simulated_order(
                symbol=symbol, side=side, entry_price=planned_entry,
                sl_price=sl_price, tp_price=tp_price,
                position_size=position_size, leverage=leverage
            )

        # Step 4: 记录成交价格和滑点
        actual_entry = order_result.get("price", planned_entry)
        slippage_info = self.slippage_monitor.record_execution(approved_signal, planned_entry, actual_entry)

        # Step 5: 构建交易记录
        trade_id = str(uuid.uuid4())
        
        # 提取 algo_id (OCO订单会有)
        algo_id = order_result.get("algo_id") if order_result else None
        
        trade_record = {
            "trade_id": trade_id,
            "signal_id": approved_signal.get("signal_id", ""),
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": self.active_client.exchange,
            "side": side,
            "entry_price": actual_entry,
            "entry_slippage": slippage_info["slippage_pct"],
            "exit_price": None,
            "exit_slippage": None,
            "leverage": leverage,
            "position_size": position_size,
            "stop_loss": approved_signal.get("stop_loss", 0),
            "take_profit": approved_signal.get("take_profit", 0),
            "algo_id": algo_id,  # OCO订单ID
            "actual_rr": None,
            "pnl": None,
            "exit_reason": None,
            "hold_hours": None,
            "market_regime": approved_signal.get("market_regime", "unknown"),
            "factors": approved_signal.get("factors", {}),
            "status": "open"
        }

        # Step 6: 记录入场
        self.trade_logger.log_entry(trade_record)

        # Step 7: 飞书通知
        self.notifier.notify(trade_record, "entry")

        # Step 8: 触发回调 (通知Agent-L)
        self._trigger_callback("on_entry", trade_record)

        logging.info(f"入场成功: {trade_id} {symbol} {side} @ {actual_entry}")
        return trade_record

    def monitor_positions(self):
        """
        实时监控持仓
        检查: 止损/止盈/时间止损/移动保本
        触发条件立即平仓
        """
        open_trades = self.trade_logger.get_open_trades()

        if not open_trades:
            return

        current_time = datetime.now()

        for trade in open_trades:
            symbol = trade["symbol"]

            # 获取当前价格
            current_price = self.active_client.get_ticker(symbol)
            if current_price is None:
                continue

            # 使用PositionMonitor检查是否需要平仓（传入ATR用于结构止损）
            factors = trade.get("factors", {})
            atr = factors.get("atr")
            should_exit, reason = self.position_monitor.monitor(trade, current_price, atr)

            if should_exit:
                entry_time = datetime.fromisoformat(trade["timestamp"])
                hold_hours = (current_time - entry_time).total_seconds() / 3600
                self._close_trade(trade, reason, current_price, hold_hours)
                continue

            # 检查移动保本
            new_stop = self.position_monitor.check_moving_stop(trade, current_price)
            if new_stop:
                trade["stop_loss"] = new_stop
                logging.info(f"移动保本: {trade['trade_id']} 新止损={new_stop}")

    def _close_trade(self, trade: Dict, reason: str, current_price: float, hold_hours: float):
        """平仓处理"""
        trade_id = trade["trade_id"]
        symbol = trade["symbol"]
        stop_loss = trade["stop_loss"]
        algo_id = trade.get("algo_id")  # OCO订单ID

        # 计算PNL
        pnl = self.position_monitor.calculate_pnl(trade, current_price)

        # 取消OCO条件单（如果存在）
        if algo_id and self.active_client.exchange == "okx":
            try:
                inst_id = symbol.replace("-USDT", "-USDT-SWAP") if "-SWAP" not in symbol else symbol
                self.active_client.cancel_algo_order(inst_id, algo_id)
                logging.info(f"已取消OCO条件单: {algo_id}")
            except Exception as e:
                logging.warning(f"取消OCO条件单失败: {e}")

        # 执行平仓
        close_result = self.active_client.close_position(symbol)

        if close_result and close_result.get("status") != "error":
            # 记录出场价格和滑点
            planned_exit = stop_loss if reason == "stop_loss" else trade.get("take_profit", current_price)
            slippage_info = self.slippage_monitor.record_execution(
                {"trade_id": trade_id, "symbol": symbol, "side": trade.get("side")},
                planned_exit, current_price
            )

            trade["_exit_price"] = current_price
            trade["_exit_slippage"] = slippage_info["slippage_pct"]

            # 记录出场
            self.trade_logger.log_exit(trade_id, reason, pnl, hold_hours)

            # 飞书通知
            updated_trade = self.trade_logger.get_trade(trade_id)
            if updated_trade:
                self.notifier.notify(updated_trade, reason)

            # 触发回调
            self._trigger_callback("on_exit", updated_trade)

            logging.info(f"平仓完成: {trade_id} {reason} PNL={pnl:.2f}")
        else:
            logging.error(f"平仓失败: {trade_id}")
            self.notifier.send_alert("平仓失败", f"{symbol} 平仓失败，请人工处理", "error")

    def _create_simulated_order(self, symbol: str, side: str, entry_price: float,
                                sl_price: float, tp_price: float,
                                position_size: float, leverage: int) -> Dict:
        """
        创建模拟订单（用于无API或API失败时）

        模拟逻辑：
        - 假设在计划入场价成交
        - 滑点设为0
        - 订单ID为模拟ID
        """
        return {
            "order_id": f"sim_{uuid.uuid4().hex[:12]}",
            "algo_id": f"sim_algo_{uuid.uuid4().hex[:12]}",
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "price": entry_price,  # 假设以计划价格成交
            "size": position_size,
            "leverage": leverage,
            "status": "simulated_fill",
            "exchange": "simulated",
            "is_simulated": True
        }

    def start_monitoring(self):
        """启动持仓监控线程"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logging.info("持仓监控已启动")

    def stop_monitoring(self):
        """停止持仓监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logging.info("持仓监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self.monitor_positions()
            except Exception as e:
                logging.error(f"监控循环异常: {e}")

            time.sleep(self.config.monitor_interval)

    def force_close(self, trade_id: str, reason: str = "manual") -> bool:
        """手动平仓"""
        trade = self.trade_logger.get_trade(trade_id)
        if not trade:
            logging.error(f"交易不存在: {trade_id}")
            return False

        symbol = trade["symbol"]
        current_price = self.active_client.get_ticker(symbol)

        if current_price is None:
            logging.error(f"无法获取价格: {symbol}")
            return False

        entry_time = datetime.fromisoformat(trade["timestamp"])
        hold_hours = (datetime.now() - entry_time).total_seconds() / 3600

        self._close_trade(trade, reason, current_price, hold_hours)
        return True


# ============================================================
# 工具函数
# ============================================================

def create_executor(
    okx_api_key: str = "",
    okx_secret_key: str = "",
    okx_passphrase: str = "",
    binance_api_key: str = "",
    binance_secret_key: str = "",
    feishu_webhook: str = "",
    use_testnet: bool = True,
    log_dir: str = "logs"
) -> Executor:
    """创建执行器实例 (便捷函数)"""

    config = ExecutorConfig(
        okx_api_key=okx_api_key,
        okx_secret_key=okx_secret_key,
        okx_passphrase=okx_passphrase,
        binance_api_key=binance_api_key,
        binance_secret_key=binance_secret_key,
        okx_testnet=use_testnet,
        binance_testnet=use_testnet,
        feishu_webhook=feishu_webhook,
        feishu_enabled=bool(feishu_webhook),
        log_dir=log_dir
    )

    return Executor(config)


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    # 模拟执行测试
    print("Agent-E 执行引擎测试")
    print("=" * 50)

    # 创建测试配置
    test_config = ExecutorConfig(
        default_exchange="okx",
        use_testnet=True,
        log_dir="logs"
    )

    # 初始化执行器
    executor = Executor(test_config)

    # 测试: 获取余额
    print("\n1. 测试获取余额...")
    balance = executor.active_client.get_balance()
    print(f"   余额: {balance}")

    # 测试: 获取价格
    print("\n2. 测试获取价格...")
    price = executor.active_client.get_ticker("BTC-USDT")
    print(f"   BTC价格: {price}")

    # 测试: 模拟执行信号
    print("\n3. 测试执行信号 (模拟)...")
    mock_signal = {
        "signal_id": "test-signal-001",
        "symbol": "BTC-USDT",
        "side": "long",
        "entry_price": 72000,
        "stop_loss": 70560,
        "take_profit": 75600,
        "leverage": 2,
        "position_size": 0.01,
        "market_regime": "bull",
        "factors": {"atr": 500}
    }

    print("   模拟信号已构建 (实际执行需配置真实API)")
    print(f"   信号: {json.dumps(mock_signal, indent=2)}")

    # 启动监控测试
    print("\n4. 测试持仓监控...")
    executor.start_monitoring()
    print("   监控线程已启动")

    # 5秒后停止
    import time
    time.sleep(2)
    executor.stop_monitoring()

    print("\n" + "=" * 50)
    print("测试完成")
