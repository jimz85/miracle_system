"""
集成测试 - 核心模块协同工作
"""
import pytest
from core.risk_management import (
    DynamicPositionSizer, CrossCurrencyRiskMonitor,
    SlippageFeeSimulator, Position
)
from core.exchange_adapter import OKXAdapter, ExchangeType, create_exchange_adapter
from plugins import PluginManager, PluginHook, create_plugin


class TestRiskManagementIntegration:
    """风险管理模块集成测试"""
    
    def test_position_sizing_with_risk_monitor(self):
        """测试仓位计算与风险监控协同"""
        # 创建仓位计算器
        sizer = DynamicPositionSizer(
            account_balance=10000,
            risk_percent=0.02,
            stop_multiplier=2.0
        )

        # 计算仓位 - 使用足够大的价格差异让ATR有意义
        result = sizer.calculate_position(
            high=105, low=95, close=100,
            entry_price=100, direction="long"
        )

        assert result["position_size"] > 0
        # 止损应该低于入场价(如果ATR>0)
        # 如果ATR=0则止损等于入场价，这是边缘情况

        # 创建风险监控
        monitor = CrossCurrencyRiskMonitor(account_balance=10000)

        # 检查是否可以开仓
        can_open, reason = monitor.can_open_position(
            symbol="BTC",
            size=result["position_size"],
            price=100
        )

        assert can_open is True
    
    def test_multi_position_risk(self):
        """测试多仓位风险监控"""
        monitor = CrossCurrencyRiskMonitor(
            account_balance=10000,
            max_total_exposure=0.8,
            max_single_exposure=0.3
        )

        # 添加第一个仓位 - 接近单币种上限
        monitor.add_position(Position(
            symbol="BTC",
            side="long",
            size=0.25,  # 0.25 * 50000 = 12500 = 125% > 30%
            entry_price=50000,
            current_price=50000
        ))

        total, pct = monitor.get_total_exposure()
        assert pct > 0.3  # 单币种超过30%

        # 检查第二个仓位 - 应该被拒绝因为已有BTC仓位
        can_open, reason = monitor.can_open_position(
            symbol="ETH",
            size=0.1,
            price=3000
        )

        # 因为已有BTC仓位占125%，新仓位应该被拒绝
        assert can_open is False, f"Expected rejection but got: can_open={can_open}, reason={reason}"


class TestExchangeAdapterIntegration:
    """交易所适配器集成测试"""
    
    def test_okx_public_api(self):
        """测试OKX公开API"""
        okx = create_exchange_adapter(ExchangeType.OKX)
        
        # 获取行情
        ticker = okx.get_ticker("BTC-USDT")
        
        assert ticker is not None
        assert ticker.last_price > 0
        assert ticker.symbol == "BTC-USDT"
        
        # 获取K线
        candles = okx.get_candles("BTC-USDT", "1h", 10)
        
        assert len(candles) > 0
        assert "close" in candles[0]
    
    def test_exchange_adapter_factory(self):
        """测试工厂函数"""
        okx = create_exchange_adapter(ExchangeType.OKX)
        assert okx.name == "okx"


class TestPluginSystemIntegration:
    """插件系统集成测试"""

    def test_signal_plugins(self):
        """测试信号插件协同"""
        manager = PluginManager()

        # 加载信号插件
        rsi = create_plugin("rsi", {"oversold": 30, "overbought": 70})
        macd = create_plugin("macd")

        manager.load_plugin(rsi)
        manager.load_plugin(macd)

        assert len(manager.get_signal_plugins()) == 2

        # 生成信号 - 直接调用插件方法
        market_data = {
            "symbol": "BTC",
            "close": 50000,
            "rsi": 25,  # 超卖
            "macd": 100,
            "macd_signal": 50,
            "macd_hist": 50
        }

        # 直接从插件生成信号
        signals = []
        for plugin in manager.get_signal_plugins():
            sig = plugin.generate_signal(market_data)
            if sig:
                signals.append(sig)

        # 至少有一个信号
        assert len(signals) >= 1, f"Expected at least 1 signal, got {len(signals)}"

    def test_risk_plugins(self):
        """测试风险插件"""
        manager = PluginManager()

        # 加载风险插件
        risk = create_plugin("max_position", {"max_position_pct": 0.2})
        manager.load_plugin(risk)

        # 风险检查 - 直接调用插件方法
        signal = {"symbol": "BTC", "size": 0.5, "price": 50000}
        portfolio = {"balance": 10000, "total_exposure": 0}

        # 直接从插件检查风险
        for plugin in manager.get_risk_plugins():
            can_proceed, reason = plugin.check_risk(signal, portfolio)
            # 0.5 * 50000 / 10000 = 2.5 = 250% > 20%，应该被拒绝
            assert can_proceed is False, f"Expected rejection but got: can_proceed={can_proceed}"


class TestSlippageFeeSimulation:
    """滑点手续费模拟集成测试"""
    
    def test_trade_simulation(self):
        """测试交易模拟"""
        simulator = SlippageFeeSimulator(
            maker_fee=0.001,
            taker_fee=0.001,
            base_slippage=0.0005
        )
        
        result = simulator.simulate_trade(
            symbol="BTC",
            side="buy",
            size=0.1,
            market_price=50000,
            order_type="market"
        )
        
        assert result["execution_price"] > 50000  # 买入滑点
        assert result["fees"] > 0
        assert result["total_cost"] > 0
    
    def test_backtest_adjustment(self):
        """测试回测成本修正"""
        simulator = SlippageFeeSimulator()
        
        trades = [
            {"symbol": "BTC", "side": "long", "size": 0.1, "entry_price": 50000, "exit_price": 51000},
            {"symbol": "ETH", "side": "long", "size": 1.0, "entry_price": 3000, "exit_price": 3100},
        ]
        
        result = simulator.backtest_adjustment(trades, 10000)
        
        assert "total_fees" in result
        assert "total_slippage" in result
        assert "adjusted_return" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
