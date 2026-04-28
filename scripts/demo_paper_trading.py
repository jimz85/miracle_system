from __future__ import annotations

"""
Miracle 2.0 Paper Trading Demo
演示完整的Paper Trading流程

运行方式:
    python demo_paper_trading.py --symbol BTC-USDT --balance 10000 --periods 100
"""
import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.exchange_adapter import ExchangeType, OKXAdapter, create_exchange_adapter
from core.logging_config import get_trade_logger, setup_root_logger
from core.risk_management import (
    CrossCurrencyRiskMonitor,
    DynamicPositionSizer,
    Position,
    SlippageFeeSimulator,
    create_risk_manager,
)
from plugins import PluginHook, PluginManager, create_plugin

# ========================
# 市场数据生成器
# ========================

class MarketDataGenerator:
    """模拟市场数据生成器"""
    
    def __init__(self, symbol: str, initial_price: float = 50000):
        self.symbol = symbol
        self.current_price = initial_price
        self.prices = [initial_price]
        self.rsi_values = [50]
        self.macd_values = [0]
        self.timestamp = datetime.now()
    
    def next_tick(self) -> Dict:
        """生成下一个Tick"""
        # 随机价格变动
        change_pct = random.uniform(-0.02, 0.025)
        self.current_price *= (1 + change_pct)
        self.prices.append(self.current_price)
        
        # 简化的RSI计算
        if len(self.prices) > 14:
            gains = sum(max(self.prices[i] - self.prices[i-1], 0) for i in range(-14, 0))
            losses = sum(max(self.prices[i-1] - self.prices[i], 0) for i in range(-14, 0))
            avg_gain = gains / 14
            avg_loss = losses / 14
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50
        
        self.rsi_values.append(rsi)
        
        # 简化的MACD
        if len(self.prices) > 26:
            ema12 = sum(self.prices[-12:]) / 12
            ema26 = sum(self.prices[-26:]) / 26
            macd = ema12 - ema26
            signal = macd * 0.8  # 简化信号线
            self.macd_values.append(macd - signal)
        else:
            macd = 0
            self.macd_values.append(0)
        
        self.timestamp += timedelta(minutes=1)
        
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.prices[-2] if len(self.prices) > 1 else self.current_price,
            "high": max(self.prices[-2:]),
            "low": min(self.prices[-2:]),
            "close": self.current_price,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd * 0.8,
            "macd_hist": self.macd_values[-1]
        }


# ========================
# Paper Trading引擎
# ========================

class PaperTradingEngine:
    """Paper Trading引擎"""
    
    def __init__(
        self,
        symbol: str,
        initial_balance: float = 10000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005
    ):
        self.symbol = symbol
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: List[Dict] = []
        self.closed_positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # 风险管理
        self.position_sizer = DynamicPositionSizer(
            account_balance=initial_balance,
            risk_percent=0.02,
            stop_multiplier=2.0
        )
        self.risk_monitor = CrossCurrencyRiskMonitor(
            max_total_exposure=1.0,
            max_single_exposure=0.3,
            account_balance=initial_balance
        )
        
        # 日志
        setup_root_logger()
        self.trade_logger = get_trade_logger()
        
        # 插件
        self.plugin_manager = PluginManager()
        self._setup_plugins()
    
    def _setup_plugins(self):
        """设置插件"""
        # 内置信号插件
        rsi = create_plugin("rsi", {"oversold": 30, "overbought": 70})
        macd = create_plugin("macd")
        
        # 风险插件
        risk = create_plugin("max_position", {"max_position_pct": 0.2})
        
        self.plugin_manager.load_plugin(rsi)
        self.plugin_manager.load_plugin(macd)
        self.plugin_manager.load_plugin(risk)
    
    def generate_signals(self, market_data: Dict) -> List[Dict]:
        """生成信号"""
        signals = []
        
        for plugin in self.plugin_manager.get_signal_plugins():
            signal = plugin.generate_signal(market_data)
            if signal:
                signals.append(signal)
        
        return signals
    
    def check_risk(self, signal: Dict) -> bool:
        """风险检查"""
        for plugin in self.plugin_manager.get_risk_plugins():
            can_proceed, reason = plugin.check_risk(
                signal,
                {"balance": self.balance, "total_exposure": self._get_total_exposure()}
            )
            if not can_proceed:
                return False
        return True
    
    def calculate_position_size(self, market_data: Dict, direction: str) -> float:
        """计算仓位"""
        high = market_data.get("high", market_data["close"])
        low = market_data.get("low", market_data["close"])
        close = market_data["close"]
        entry = close
        
        result = self.position_sizer.calculate_position(
            high=high, low=low, close=close,
            entry_price=entry, direction=direction
        )
        
        return result["position_size"]
    
    def _get_total_exposure(self) -> float:
        """获取总敞口"""
        return sum(
            pos["size"] * pos["entry_price"]
            for pos in self.positions
        )
    
    def execute_buy(self, market_data: Dict, size: float) -> bool:
        """执行买入"""
        price = market_data["close"]
        
        # 滑点
        execution_price = price * (1 + self.slippage_rate)
        
        # 手续费
        commission = size * execution_price * self.commission_rate
        total_cost = size * execution_price + commission
        
        if total_cost > self.balance:
            return False
        
        # 执行
        self.balance -= total_cost
        
        position = {
            "side": "long",
            "size": size,
            "entry_price": execution_price,
            "entry_time": market_data["timestamp"],
            "stop_loss": execution_price * 0.98,
            "take_profit": execution_price * 1.05
        }
        self.positions.append(position)
        
        # 记录
        self.trade_logger.info(f"BUY {size} @ {execution_price:.2f}, Cost: {total_cost:.2f}")
        
        self.trades.append({
            "action": "buy",
            "symbol": self.symbol,
            "size": size,
            "price": execution_price,
            "commission": commission,
            "balance": self.balance,
            "timestamp": market_data["timestamp"]
        })
        
        return True
    
    def execute_sell(self, market_data: Dict, position_idx: int) -> bool:
        """执行卖出"""
        position = self.positions[position_idx]
        price = market_data["close"]
        
        # 滑点
        execution_price = price * (1 - self.slippage_rate)
        
        # 手续费
        pnl = (execution_price - position["entry_price"]) * position["size"]
        commission = position["size"] * execution_price * self.commission_rate
        
        # 执行
        self.balance += position["size"] * execution_price - commission
        
        closed = {
            **position,
            "exit_price": execution_price,
            "exit_time": market_data["timestamp"],
            "pnl": pnl - commission,
            "pnl_pct": (execution_price - position["entry_price"]) / position["entry_price"] * 100
        }
        self.closed_positions.append(closed)
        self.positions.pop(position_idx)
        
        # 记录
        self.trade_logger.info(
            f"SELL {closed['size']} @ {execution_price:.2f}, "
            f"PnL: {closed['pnl']:.2f} ({closed['pnl_pct']:.2f}%)"
        )
        
        self.trades.append({
            "action": "sell",
            "symbol": self.symbol,
            "size": closed["size"],
            "price": execution_price,
            "pnl": closed["pnl"],
            "commission": commission,
            "balance": self.balance,
            "timestamp": market_data["timestamp"]
        })
        
        return True
    
    def check_stop_loss(self, market_data: Dict) -> bool:
        """检查止损"""
        price = market_data["close"]
        stopped = False
        
        for i, pos in enumerate(self.positions[:]):
            if pos["side"] == "long" and price < pos["stop_loss"]:
                self.execute_sell(market_data, i)
                stopped = True
            elif pos["side"] == "short" and price > pos["stop_loss"]:
                self.execute_sell(market_data, i)
                stopped = True
        
        return stopped
    
    def check_take_profit(self, market_data: Dict) -> bool:
        """检查止盈"""
        price = market_data["close"]
        taken = False
        
        for i, pos in enumerate(self.positions[:]):
            if pos["side"] == "long" and price > pos["take_profit"]:
                self.execute_sell(market_data, i)
                taken = True
            elif pos["side"] == "short" and price < pos["take_profit"]:
                self.execute_sell(market_data, i)
                taken = True
        
        return taken
    
    def run(self, market_data_generator: MarketDataGenerator, periods: int = 100):
        """运行Paper Trading"""
        print(f"\n{'='*60}")
        print("Miracle 2.0 Paper Trading Demo")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Periods: {periods}")
        print(f"{'='*60}\n")
        
        for i in range(periods):
            # 获取市场数据
            market_data = market_data_generator.next_tick()
            
            # 检查止损/止盈
            if self.positions:
                self.check_stop_loss(market_data)
                self.check_take_profit(market_data)
            
            # 生成信号
            if not self.positions:
                signals = self.generate_signals(market_data)
                
                for signal in signals:
                    # 风险检查
                    if not self.check_risk(signal):
                        continue
                    
                    direction = signal["direction"]
                    confidence = signal["confidence"]
                    
                    if confidence < 0.5:
                        continue
                    
                    # 计算仓位
                    size = self.calculate_position_size(market_data, direction)
                    
                    if direction == "buy":
                        self.execute_buy(market_data, size)
                        break
            
            # 定期报告
            if (i + 1) % 20 == 0:
                self._print_status(i + 1, market_data)
        
        # 平仓
        if self.positions:
            final_data = market_data_generator.next_tick()
            for i in range(len(self.positions)):
                self.execute_sell(final_data, 0)
        
        self._print_final_report()
    
    def _print_status(self, step: int, market_data: Dict):
        """打印状态"""
        total_value = self.balance + sum(
            pos["size"] * market_data["close"]
            for pos in self.positions
        )
        pnl_pct = (total_value - self.initial_balance) / self.initial_balance * 100
        
        print(f"[{step:3d}] Price: ${market_data['close']:,.2f} | "
              f"Balance: ${self.balance:,.2f} | "
              f"Positions: {len(self.positions)} | "
              f"Total: ${total_value:,.2f} ({pnl_pct:+.2f}%)")
    
    def _print_final_report(self):
        """打印最终报告"""
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        win_count = sum(1 for p in self.closed_positions if p["pnl"] > 0)
        loss_count = len(self.closed_positions) - win_count
        win_rate = win_count / len(self.closed_positions) * 100 if self.closed_positions else 0
        
        total_pnl = sum(p["pnl"] for p in self.closed_positions)
        max_pnl = max((p["pnl"] for p in self.closed_positions), default=0)
        min_pnl = min((p["pnl"] for p in self.closed_positions), default=0)
        
        print(f"\n{'='*60}")
        print("Final Report")
        print(f"{'='*60}")
        print(f"Initial Balance:  ${self.initial_balance:,.2f}")
        print(f"Final Balance:    ${self.balance:,.2f}")
        print(f"Total Return:      {total_return:+.2f}%")
        print(f"{'='*60}")
        print(f"Total Trades:     {len(self.trades)}")
        print(f"Closed Positions:  {len(self.closed_positions)}")
        print(f"Open Positions:    {len(self.positions)}")
        print(f"Win/Loss:         {win_count}W / {loss_count}L")
        print(f"Win Rate:         {win_rate:.1f}%")
        print(f"{'='*60}")
        print(f"Total PnL:        ${total_pnl:,.2f}")
        print(f"Best Trade:       ${max_pnl:,.2f}")
        print(f"Worst Trade:      ${min_pnl:,.2f}")
        print(f"{'='*60}\n")


# ========================
# 主函数
# ========================

def main():
    parser = argparse.ArgumentParser(description="Miracle 2.0 Paper Trading Demo")
    parser.add_argument("--symbol", default="BTC-USDT", help="Trading symbol")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--periods", type=int, default=100, help="Number of periods")
    parser.add_argument("--initial-price", type=float, default=50000, help="Initial price")
    
    args = parser.parse_args()
    
    # 创建市场数据生成器
    generator = MarketDataGenerator(args.symbol, args.initial_price)
    
    # 创建并运行交易引擎
    engine = PaperTradingEngine(
        symbol=args.symbol,
        initial_balance=args.balance
    )
    
    engine.run(generator, args.periods)


if __name__ == "__main__":
    main()
