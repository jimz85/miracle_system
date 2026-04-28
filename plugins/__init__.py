from __future__ import annotations

"""
Plugin System - 可扩展插件架构
支持信号生成器、策略、风险管理器的热插拔
"""
import importlib
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ========================
# 插件元数据
# ========================

@dataclass
class PluginMetadata:
    """插件元数据"""
    name: str
    version: str
    author: str
    description: str
    hooks: List[str] = field(default_factory=list)
    config_schema: Dict | None = None


# ========================
# 插件钩子枚举
# ========================

class PluginHook(Enum):
    """支持的插件钩子"""
    # 信号生成
    ON_SIGNAL_GENERATE = "on_signal_generate"           # 生成交易信号
    ON_SIGNAL_VALIDATE = "on_signal_validate"           # 验证信号
    ON_SIGNAL_FILTER = "on_signal_filter"               # 过滤信号
    
    # 策略执行
    ON_STRATEGY_INIT = "on_strategy_init"               # 策略初始化
    ON_STRATEGY_UPDATE = "on_strategy_update"           # 策略更新
    ON_STRATEGY_SELECT = "on_strategy_select"           # 选择策略
    
    # 风险管理
    ON_RISK_CHECK = "on_risk_check"                     # 风险检查
    ON_RISK_CALCULATE_SIZE = "on_risk_calculate_size"   # 计算仓位
    ON_POSITION_OPEN = "on_position_open"                # 开仓前
    ON_POSITION_CLOSE = "on_position_close"              # 平仓前
    
    # 交易执行
    ON_ORDER_PLACE = "on_order_place"                   # 下单前
    ON_ORDER_FILLED = "on_order_filled"                 # 订单成交
    ON_ORDER_CANCEL = "on_order_cancel"                 # 订单取消
    
    # 系统
    ON_TICK = "on_tick"                                 # 每Tick
    ON_CANDLE = "on_candle"                             # K线完成
    ON_BACKTEST_START = "on_backtest_start"             # 回测开始
    ON_BACKTEST_END = "on_backtest_end"                 # 回测结束


# ========================
# 基础插件类
# ========================

class BasePlugin:
    """插件基类"""
    
    metadata: PluginMetadata
    enabled: bool = True
    config: Dict[str, Any] = {}
    
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self._hooks: Dict[PluginHook, List[Callable]] = {}
    
    def enable(self):
        """启用插件"""
        self.enabled = True
        logger.info(f"Plugin enabled: {self.metadata.name}")
    
    def disable(self):
        """禁用插件"""
        self.enabled = False
        logger.info(f"Plugin disabled: {self.metadata.name}")
    
    def register_hook(self, hook: PluginHook, callback: Callable):
        """注册钩子回调"""
        if hook not in self._hooks:
            self._hooks[hook] = []
        self._hooks[hook].append(callback)
    
    def unregister_hook(self, hook: PluginHook, callback: Callable):
        """取消注册钩子"""
        if hook in self._hooks and callback in self._hooks[hook]:
            self._hooks[hook].remove(callback)
    
    def call_hooks(self, hook: PluginHook, *args, **kwargs) -> Any:
        """调用钩子"""
        if not self.enabled:
            return None
        
        results = []
        if hook in self._hooks:
            for callback in self._hooks[hook]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook {hook.value} failed in {self.metadata.name}: {e}")
        
        return results if results else None
    
    def initialize(self):
        """初始化插件"""
        pass
    
    def shutdown(self):
        """关闭插件"""
        pass


# ========================
# 信号插件
# ========================

class SignalPlugin(BasePlugin):
    """信号生成插件基类"""
    
    def generate_signal(self, market_data: Dict) -> Dict | None:
        """生成信号 - 子类实现"""
        raise NotImplementedError
    
    def validate_signal(self, signal: Dict) -> bool:
        """验证信号"""
        required = ["symbol", "direction", "confidence"]
        return all(k in signal for k in required)


# ========================
# 策略插件
# ========================

class StrategyPlugin(BasePlugin):
    """策略插件基类"""
    
    def select_signal(self, signals: List[Dict], context: Dict) -> Dict | None:
        """从多个信号中选择"""
        if not signals:
            return None
        # 默认返回置信度最高的
        return max(signals, key=lambda s: s.get("confidence", 0))


# ========================
# 风险管理插件
# ========================

class RiskManagementPlugin(BasePlugin):
    """风险管理插件基类"""
    
    def check_risk(self, signal: Dict, portfolio: Dict) -> tuple[bool, str]:
        """检查风险 - 返回(can_proceed, reason)"""
        return True, "OK"
    
    def calculate_position_size(self, signal: Dict, portfolio: Dict) -> float:
        """计算仓位大小"""
        return signal.get("size", 0)


# ========================
# 插件管理器
# ========================

class PluginRegistry:
    """全局插件注册表"""
    
    _instance = None
    _plugins: Dict[str, BasePlugin] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, plugin: BasePlugin) -> None:
        """注册插件"""
        name = plugin.metadata.name
        cls._plugins[name] = plugin
        logger.info(f"Registered plugin: {name} v{plugin.metadata.version}")
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """注销插件"""
        if name in cls._plugins:
            del cls._plugins[name]
            logger.info(f"Unregistered plugin: {name}")
    
    @classmethod
    def get(cls, name: str) -> BasePlugin | None:
        """获取插件"""
        return cls._plugins.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, BasePlugin]:
        """获取所有插件"""
        return cls._plugins.copy()
    
    @classmethod
    def get_by_type(cls, plugin_type: Type) -> List[BasePlugin]:
        """按类型获取插件"""
        return [p for p in cls._plugins.values() if isinstance(p, plugin_type)]


class PluginManager:
    """插件生命周期管理器"""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self._loaded_plugins: List[BasePlugin] = []
        self._hook_subscribers: Dict[PluginHook, List[BasePlugin]] = {
            hook: [] for hook in PluginHook
        }
    
    def load_plugin(self, plugin: BasePlugin) -> bool:
        """加载插件"""
        try:
            plugin.initialize()
            self.registry.register(plugin)
            self._loaded_plugins.append(plugin)
            
            # 注册钩子
            for hook in PluginHook:
                if hasattr(plugin, hook.value.replace("on_", "")):
                    method = getattr(plugin, hook.value.replace("on_", ""))
                    if callable(method):
                        self._hook_subscribers[hook].append(plugin)
            
            logger.info(f"Loaded plugin: {plugin.metadata.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin.metadata.name}: {e}")
            return False
    
    def unload_plugin(self, name: str) -> bool:
        """卸载插件"""
        plugin = self.registry.get(name)
        if not plugin:
            return False
        
        try:
            plugin.shutdown()
            self.registry.unregister(name)
            self._loaded_plugins = [p for p in self._loaded_plugins if p.metadata.name != name]
            
            # 取消注册钩子
            for hook in self._hook_subscribers:
                self._hook_subscribers[hook] = [
                    p for p in self._hook_subscribers[hook] if p.metadata.name != name
                ]
            
            return True
        except Exception as e:
            logger.error(f"Failed to unload plugin {name}: {e}")
            return False
    
    def execute_hook(self, hook: PluginHook, *args, **kwargs) -> List[Any]:
        """执行钩子"""
        results = []
        for plugin in self._hook_subscribers.get(hook, []):
            if plugin.enabled:
                try:
                    method_name = hook.value.replace("on_", "")
                    if hasattr(plugin, method_name):
                        method = getattr(plugin, method_name)
                        if callable(method):
                            result = method(*args, **kwargs)
                            results.append(result)
                except Exception as e:
                    logger.error(f"Hook {hook.value} failed in {plugin.metadata.name}: {e}")
        return results
    
    def get_signal_plugins(self) -> List[SignalPlugin]:
        """获取信号插件"""
        return self.registry.get_by_type(SignalPlugin)
    
    def get_strategy_plugins(self) -> List[StrategyPlugin]:
        """获取策略插件"""
        return self.registry.get_by_type(StrategyPlugin)
    
    def get_risk_plugins(self) -> List[RiskManagementPlugin]:
        """获取风险管理插件"""
        return self.registry.get_by_type(RiskManagementPlugin)


# ========================
# 内置插件
# ========================

class RSISignalPlugin(SignalPlugin):
    """RSI信号插件"""
    
    metadata = PluginMetadata(
        name="rsi_signal",
        version="1.0.0",
        author="Miracle Team",
        description="RSI指标信号生成器",
        hooks=["on_signal_generate"]
    )
    
    def __init__(self, config: Dict | None = None):
        super().__init__(config)
        self.period = self.config.get("period", 14)
        self.oversold = self.config.get("oversold", 30)
        self.overbought = self.config.get("overbought", 70)
    
    def generate_signal(self, market_data: Dict) -> Dict | None:
        """生成RSI信号"""
        rsi = market_data.get("rsi", 50)
        market_data.get("close", 0)
        
        if rsi < self.oversold:
            return {
                "symbol": market_data.get("symbol", "UNKNOWN"),
                "direction": "buy",
                "confidence": (self.oversold - rsi) / self.oversold,
                "indicator": "RSI",
                "value": rsi
            }
        elif rsi > self.overbought:
            return {
                "symbol": market_data.get("symbol", "UNKNOWN"),
                "direction": "sell",
                "confidence": (rsi - self.overbought) / (100 - self.overbought),
                "indicator": "RSI",
                "value": rsi
            }
        return None


class MACDSignalPlugin(SignalPlugin):
    """MACD信号插件"""
    
    metadata = PluginMetadata(
        name="macd_signal",
        version="1.0.0",
        author="Miracle Team",
        description="MACD指标信号生成器",
        hooks=["on_signal_generate"]
    )
    
    def generate_signal(self, market_data: Dict) -> Dict | None:
        """生成MACD信号"""
        macd = market_data.get("macd", 0)
        signal = market_data.get("macd_signal", 0)
        
        if macd > signal and market_data.get("macd_hist", 0) > 0:
            return {
                "symbol": market_data.get("symbol", "UNKNOWN"),
                "direction": "buy",
                "confidence": 0.7,
                "indicator": "MACD",
                "value": macd - signal
            }
        elif macd < signal and market_data.get("macd_hist", 0) < 0:
            return {
                "symbol": market_data.get("symbol", "UNKNOWN"),
                "direction": "sell",
                "confidence": 0.7,
                "indicator": "MACD",
                "value": signal - macd
            }
        return None


class MaxPositionRiskPlugin(RiskManagementPlugin):
    """最大仓位风险管理"""
    
    metadata = PluginMetadata(
        name="max_position_risk",
        version="1.0.0",
        author="Miracle Team",
        description="限制最大仓位和总敞口",
        hooks=["on_risk_check"]
    )
    
    def __init__(self, config: Dict | None = None):
        super().__init__(config)
        self.max_position_pct = self.config.get("max_position_pct", 0.3)  # 单币种30%
        self.max_total_pct = self.config.get("max_total_pct", 1.0)         # 总仓位100%
    
    def check_risk(self, signal: Dict, portfolio: Dict) -> tuple[bool, str]:
        """检查仓位风险"""
        balance = portfolio.get("balance", 10000)
        position_value = signal.get("size", 0) * signal.get("price", 0)
        
        # 单币种检查
        position_pct = position_value / balance
        if position_pct > self.max_position_pct:
            return False, f"单币种仓位 {position_pct:.1%} 超过上限 {self.max_position_pct:.1%}"
        
        # 总敞口检查
        current_exposure = portfolio.get("total_exposure", 0)
        new_exposure = current_exposure + position_value
        
        if new_exposure / balance > self.max_total_pct:
            return False, f"总敞口 {new_exposure/balance:.1%} 超过上限 {self.max_total_pct:.1%}"
        
        return True, "OK"


# ========================
# 装饰器
# ========================

def signal_plugin(config: Dict | None = None):
    """信号插件装饰器"""
    def decorator(cls):
        cls.metadata = getattr(cls, "metadata", PluginMetadata(
            name=cls.__name__,
            version="1.0.0",
            author="Anonymous",
            description=cls.__doc__ or ""
        ))
        return cls
    return decorator


def strategy_plugin(config: Dict | None = None):
    """策略插件装饰器"""
    def decorator(cls):
        cls.metadata = getattr(cls, "metadata", PluginMetadata(
            name=cls.__name__,
            version="1.0.0",
            author="Anonymous",
            description=cls.__doc__ or ""
        ))
        return cls
    return decorator


def risk_plugin(config: Dict | None = None):
    """风险管理插件装饰器"""
    def decorator(cls):
        cls.metadata = getattr(cls, "metadata", PluginMetadata(
            name=cls.__name__,
            version="1.0.0",
            author="Anonymous",
            description=cls.__doc__ or ""
        ))
        return cls
    return decorator


# ========================
# 工厂函数
# ========================

def create_plugin(plugin_type: str, config: Dict | None = None) -> BasePlugin | None:
    """创建内置插件"""
    plugins = {
        "rsi": RSISignalPlugin,
        "macd": MACDSignalPlugin,
        "max_position": MaxPositionRiskPlugin,
    }
    
    plugin_class = plugins.get(plugin_type.lower())
    if plugin_class:
        return plugin_class(config)
    return None


def load_plugins_from_directory(directory: str) -> List[BasePlugin]:
    """从目录加载插件"""
    plugins = []
    path = Path(directory)
    
    if not path.exists():
        logger.warning(f"Plugin directory not found: {directory}")
        return plugins
    
    for file in path.glob("*.py"):
        if file.name.startswith("_"):
            continue
        
        try:
            # 动态导入
            spec = importlib.util.spec_from_file_location(file.stem, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[file.stem] = module
            spec.loader.exec_module(module)
            
            # 查找插件类
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj != BasePlugin:
                    plugins.append(obj())
        except Exception as e:
            logger.error(f"Failed to load plugin from {file}: {e}")
    
    return plugins


if __name__ == "__main__":
    print("=== Plugin System Test ===\n")
    
    # 创建管理器
    manager = PluginManager()
    
    # 加载内置插件
    rsi_plugin = create_plugin("rsi", {"period": 14, "oversold": 30, "overbought": 70})
    macd_plugin = create_plugin("macd")
    risk_plugin = create_plugin("max_position", {"max_position_pct": 0.2})
    
    manager.load_plugin(rsi_plugin)
    manager.load_plugin(macd_plugin)
    manager.load_plugin(risk_plugin)
    
    # 测试信号生成
    market_data = {
        "symbol": "BTC",
        "close": 50000,
        "rsi": 25,  # 超卖
        "macd": 100,
        "macd_signal": 50,
        "macd_hist": 50
    }
    
    signals = manager.execute_hook(PluginHook.ON_SIGNAL_GENERATE, market_data)
    print(f"Generated signals: {len(signals) if signals else 0}")
    if signals:
        for sig in signals:
            if sig:
                print(f"  Signal: {sig}")
    
    # 测试风险检查
    portfolio = {"balance": 10000, "total_exposure": 0}
    signal = {"symbol": "BTC", "size": 0.5, "price": 50000}
    
    risk_results = manager.execute_hook(PluginHook.ON_RISK_CHECK, signal, portfolio)
    print(f"\nRisk check: {risk_results}")
    
    print("\n=== Plugin System Ready ===")
