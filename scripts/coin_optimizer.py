#!/usr/bin/env python3
from __future__ import annotations

"""
coin_optimizer.py - 每币种参数优化模块
=====================================

参考 Kronos coin_strategy_map.json 实现的功能：
1. 每币种独立的策略参数配置
2. Walk-Forward 验证支持
3. 与 Miracle 自适应学习系统集成

功能：
- 加载和管理 coin_params.json 配置
- 根据市场环境和币种特性选择最优参数
- 与 backtest.py 集成进行参数优化
- 与 adaptive_learner.py 集成进行动态参数调整

Author: Miracle System
Version: 1.0.0
"""

import copy
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("miracle.coin_optimizer")

# ===== 数据结构 =====

class StrategyType(Enum):
    """策略类型枚举"""
    RSI_MR = "RSI_MR"      # RSI均值回归
    RSI_EMAn = "RSI_EMAn"  # RSI+EMA共振
    RSI_VOL = "RSI_VOL"    # RSI+成交量
    VOL_BRK = "VOL_BRK"    # 成交量突破
    BB_TREND = "BB_TREND"  # 布林带趋势


@dataclass
class CoinParams:
    """币种参数配置"""
    symbol: str
    enabled: bool
    optimal_strategy: str
    timeframe: str
    confidence: float
    signal_params: Dict[str, float]
    exit: Dict[str, Any]
    position: Dict[str, float]
    performance: Dict[str, Any]
    notes: str
    excluded: bool
    excluded_reason: str | None

    @classmethod
    def from_dict(cls, data: Dict) -> CoinParams:
        return cls(
            symbol=data['symbol'],
            enabled=data.get('enabled', True),
            optimal_strategy=data.get('optimal_strategy', 'RSI_MR'),
            timeframe=data.get('timeframe', '1H'),
            confidence=data.get('confidence', 0.5),
            signal_params=data.get('signal_params', {}),
            exit=data.get('exit', {}),
            position=data.get('position', {}),
            performance=data.get('performance', {}),
            notes=data.get('notes', ''),
            excluded=data.get('excluded', False),
            excluded_reason=data.get('excluded_reason')
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OptimizationResult:
    """优化结果"""
    symbol: str
    strategy: str
    sharpe: float
    winrate: float
    max_drawdown: float
    trades: int
    params: Dict[str, float]
    ic_score: float | None = None
    walk_forward_passed: bool = False
    notes: str = ""


# ===== 核心类 =====

class CoinParameterOptimizer:
    """
    每币种参数优化器
    
    功能：
    - 管理币种参数配置（coin_params.json）
    - 根据市场环境动态选择参数
    - 支持参数网格搜索优化
    - 与回测系统集成验证参数有效性
    """
    
    # 默认参数文件路径
    DEFAULT_PARAMS_PATH = Path(__file__).parent / "coin_params.json"
    
    # Kronos 参考文件路径（用于对比）
    KRONOS_COIN_STRATEGY_MAP = Path.home() / "kronos" / "coin_strategy_map.json"
    
    def __init__(self, params_path: str = None, kronos_compare: bool = True):
        """
        初始化参数优化器
        
        Args:
            params_path: coin_params.json 路径，默认使用内置路径
            kronos_compare: 是否加载 Kronos 配置进行对比
        """
        self.params_path = Path(params_path) if params_path else self.DEFAULT_PARAMS_PATH
        self.kronos_compare = kronos_compare
        
        # 币种参数缓存
        self._coin_params: Dict[str, CoinParams] = {}
        self._kronos_params: Dict[str, Dict] = {}
        
        # 加载配置
        self._load_params()
        if kronos_compare:
            self._load_kronos_params()
        
        logger.info(f"CoinParameterOptimizer initialized with {len(self._coin_params)} coins")
    
    def _load_params(self):
        """加载币种参数配置"""
        if not self.params_path.exists():
            logger.warning(f"参数文件不存在: {self.params_path}，使用空配置")
            return
        
        try:
            with open(self.params_path, encoding='utf-8') as f:
                data = json.load(f)
            
            coins_data = data.get('coins', [])
            for coin_data in coins_data:
                params = CoinParams.from_dict(coin_data)
                self._coin_params[params.symbol] = params
            
            logger.info(f"加载 {len(self._coin_params)} 个币种参数")
            
        except Exception as e:
            logger.error(f"加载参数文件失败: {e}")
    
    def _load_kronos_params(self):
        """加载 Kronos 币种策略配置用于对比"""
        if not self.KRONOS_COIN_STRATEGY_MAP.exists():
            logger.warning(f"Kronos配置不存在: {self.KRONOS_COIN_STRATEGY_MAP}")
            return
        
        try:
            with open(self.KRONOS_COIN_STRATEGY_MAP, encoding='utf-8') as f:
                data = json.load(f)
            
            for coin_data in data.get('coins', []):
                symbol = coin_data.get('symbol')
                if symbol:
                    self._kronos_params[symbol] = coin_data
            
            logger.info(f"加载 {len(self._kronos_params)} 个 Kronos 币种配置")
            
        except Exception as e:
            logger.error(f"加载Kronos配置失败: {e}")
    
    def get_coin_params(self, symbol: str) -> CoinParams | None:
        """
        获取指定币种的参数配置
        
        Args:
            symbol: 币种符号（如 BTC, ETH）
            
        Returns:
            CoinParams 或 None
        """
        return self._coin_params.get(symbol.upper())
    
    def get_signal_params(self, symbol: str) -> Dict[str, float]:
        """
        获取指定币种的信号参数
        
        Args:
            symbol: 币种符号
            
        Returns:
            信号参数字典
        """
        params = self.get_coin_params(symbol)
        if params:
            return params.signal_params
        return self._get_default_signal_params()
    
    def _get_default_signal_params(self) -> Dict[str, float]:
        """获取默认信号参数"""
        return {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "adx_min": 15,
            "adx_strong": 25,
            "atr_period": 14,
            "atr_stop_multiplier": 3.0,
            "time_exit_hours": 24
        }
    
    def get_position_params(self, symbol: str, trend_strength: str = "medium") -> Dict[str, float]:
        """
        获取指定币种的仓位参数
        
        Args:
            symbol: 币种符号
            trend_strength: 趋势强度 (weak/medium/strong)
            
        Returns:
            仓位参数字典
        """
        params = self.get_coin_params(symbol)
        if not params:
            return self._get_default_position_params()
        
        pos = params.position
        leverage_key = "leverage_strong_trend" if trend_strength == "strong" else "leverage"
        
        return {
            "base_position_pct": pos.get("base_position_pct", 2.0),
            "max_position_pct": pos.get("max_position_pct", 10.0),
            "leverage": pos.get(leverage_key, pos.get("leverage", 2)),
        }
    
    def _get_default_position_params(self) -> Dict[str, float]:
        """获取默认仓位参数"""
        return {
            "base_position_pct": 2.0,
            "max_position_pct": 10.0,
            "leverage": 2
        }
    
    def get_enabled_coins(self) -> List[str]:
        """获取所有已启用的币种列表"""
        return [
            symbol for symbol, params in self._coin_params.items()
            if params.enabled and not params.excluded
        ]
    
    def get_all_symbols(self) -> List[str]:
        """获取所有配置的币种列表"""
        return list(self._coin_params.keys())
    
    def get_kronos_comparison(self, symbol: str) -> Dict | None:
        """
        获取 Kronos 对比数据
        
        Args:
            symbol: 币种符号
            
        Returns:
            Kronos 配置字典或 None
        """
        return self._kronos_params.get(symbol.upper())
    
    def compare_with_kronos(self, symbol: str) -> Dict[str, Any]:
        """
        对比 Miracle 与 Kronos 的币种配置
        
        Args:
            symbol: 币种符号
            
        Returns:
            对比结果字典
        """
        miracle_params = self.get_coin_params(symbol)
        kronos_params = self.get_kronos_comparison(symbol)
        
        if not miracle_params:
            return {"error": f"Miracle中未配置{symbol}"}
        if not kronos_params:
            return {"error": f"Kronos中未配置{symbol}"}
        
        comparison = {
            "symbol": symbol,
            "miracle": {
                "strategy": miracle_params.optimal_strategy,
                "timeframe": miracle_params.timeframe,
                "confidence": miracle_params.confidence,
                "excluded": miracle_params.excluded,
                "performance": miracle_params.performance
            },
            "kronos": {
                "strategy": kronos_params.get("optimal_strategy"),
                "timeframe": kronos_params.get("timeframe"),
                "confidence": kronos_params.get("confidence"),
                "excluded": kronos_params.get("excluded", False),
                "performance": {
                    "sharpe": kronos_params.get("sharpe"),
                    "winrate": kronos_params.get("winrate"),
                    "max_drawdown": kronos_params.get("max_drawdown")
                }
            },
            "differences": []
        }
        
        # 检查差异
        if miracle_params.optimal_strategy != kronos_params.get("optimal_strategy"):
            comparison["differences"].append({
                "field": "strategy",
                "miracle": miracle_params.optimal_strategy,
                "kronos": kronos_params.get("optimal_strategy")
            })
        
        if miracle_params.timeframe != kronos_params.get("timeframe"):
            comparison["differences"].append({
                "field": "timeframe",
                "miracle": miracle_params.timeframe,
                "kronos": kronos_params.get("timeframe")
            })
        
        return comparison
    
    def update_coin_params(self, symbol: str, updates: Dict) -> bool:
        """
        更新币种参数（仅内存中，不写入文件）
        
        Args:
            symbol: 币种符号
            updates: 要更新的参数字典
            
        Returns:
            是否成功
        """
        if symbol.upper() not in self._coin_params:
            logger.warning(f"币种 {symbol} 不存在于配置中")
            return False
        
        params = self._coin_params[symbol.upper()]
        params_dict = params.to_dict()
        
        # 更新字段
        for key, value in updates.items():
            if key in params_dict:
                if isinstance(value, dict) and isinstance(params_dict[key], dict):
                    params_dict[key].update(value)
                else:
                    params_dict[key] = value
        
        # 重新创建 CoinParams
        self._coin_params[symbol.upper()] = CoinParams.from_dict(params_dict)
        return True
    
    def save_params(self) -> bool:
        """
        保存参数到文件
        
        Returns:
            是否成功
        """
        try:
            data = {
                "_comment": "Miracle System 币种参数配置 v1.0 | 参考Kronos coin_strategy_map.json | 自动更新",
                "_last_updated": datetime.now().isoformat(),
                "coins": [params.to_dict() for params in self._coin_params.values()],
                "strategy_descriptions": {
                    "RSI_MR": "RSI均值回归策略",
                    "RSI_EMAn": "RSI+EMA共振策略",
                    "RSI_VOL": "RSI+成交量混合策略",
                    "VOL_BRK": "成交量突破策略",
                    "BB_TREND": "布林带趋势跟踪策略"
                }
            }
            
            with open(self.params_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"参数已保存到 {self.params_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存参数失败: {e}")
            return False
    
    def generate_optimization_report(self) -> str:
        """
        生成优化报告
        
        Returns:
            格式化的报告字符串
        """
        lines = [
            "=" * 60,
            "Miracle System - 每币种参数优化报告",
            "=" * 60,
            f"生成时间: {datetime.now().isoformat()}",
            f"总币种数: {len(self._coin_params)}",
            f"已启用: {len(self.get_enabled_coins())}",
            ""
        ]
        
        # 按 sharpe 排序
        sorted_params = sorted(
            self._coin_params.values(),
            key=lambda x: x.performance.get('sharpe', 0),
            reverse=True
        )
        
        lines.append("币种参数详情:")
        lines.append("-" * 60)
        
        for params in sorted_params:
            status = "❌ 排除" if params.excluded else "✅ 启用" if params.enabled else "⏸ 禁用"
            lines.append(f"\n{params.symbol} {status}")
            lines.append(f"  策略: {params.optimal_strategy} ({params.timeframe})")
            lines.append(f"  置信度: {params.confidence:.0%}")
            
            perf = params.performance
            lines.append(f"  Sharpe: {perf.get('sharpe', 'N/A')}")
            lines.append(f"  胜率: {perf.get('winrate', 'N/A')}")
            lines.append(f"  最大回撤: {perf.get('max_drawdown', 'N/A')}")
            
            if params.excluded_reason:
                lines.append(f"  排除原因: {params.excluded_reason}")
        
        # Kronos 对比摘要
        if self._kronos_params:
            lines.append("\n" + "=" * 60)
            lines.append("Kronos 对比:")
            lines.append("-" * 60)
            
            for symbol in self.get_enabled_coins():
                kronos = self.get_kronos_comparison(symbol)
                if kronos:
                    lines.append(f"\n{symbol}:")
                    lines.append(f"  Kronos: {kronos.get('optimal_strategy')} ({kronos.get('timeframe')})")
                    lines.append(f"  Miracle: {self.get_coin_params(symbol).optimal_strategy}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ===== 与回测系统集成的参数生成器 =====

class CoinSignalGenerator:
    """
    币种信号生成器 - 使用 coin_params.json 生成交易信号
    
    与 backtest.py 的 signal_func 兼容
    """
    
    def __init__(self, optimizer: CoinParameterOptimizer = None):
        self.optimizer = optimizer or CoinParameterOptimizer()
    
    def generate_signal(self, symbol: str, prices: List[float], highs: List[float], 
                       lows: List[float], index: int) -> Dict | None:
        """
        生成交易信号（与 backtest.py 兼容）
        
        Args:
            symbol: 币种符号
            prices: 所有收盘价
            highs: 所有最高价
            lows: 所有最低价
            index: 当前索引
            
        Returns:
            信号字典或 None
        """
        from miracle_core import calc_adx, calc_atr, calc_macd, calc_rsi
        
        params = self.optimizer.get_coin_params(symbol)
        if not params or not params.enabled or params.excluded:
            return None
        
        signal_params = params.signal_params
        
        # 计算指标
        rsi = calc_rsi(prices, period=14)
        adx, plus_di, minus_di = calc_adx(highs, lows, prices, period=14)
        atr = calc_atr(highs, lows, prices, period=signal_params.get('atr_period', 14))
        
        # RSI 策略
        strategy = params.optimal_strategy
        
        if strategy == "RSI_MR":
            return self._generate_rsi_mr_signal(
                symbol, prices, highs, lows, index,
                rsi, adx, atr, signal_params, params
            )
        elif strategy == "RSI_EMAn":
            return self._generate_rsi_eman_signal(
                symbol, prices, highs, lows, index,
                rsi, adx, atr, signal_params, params
            )
        elif strategy == "RSI_VOL":
            return self._generate_rsi_vol_signal(
                symbol, prices, highs, lows, index,
                rsi, adx, atr, signal_params, params
            )
        elif strategy == "VOL_BRK":
            return self._generate_vol_brk_signal(
                symbol, prices, highs, lows, index,
                rsi, adx, atr, signal_params, params
            )
        else:
            return self._generate_rsi_mr_signal(
                symbol, prices, highs, lows, index,
                rsi, adx, atr, signal_params, params
            )
    
    def _generate_rsi_mr_signal(self, symbol, prices, highs, lows, index,
                               rsi, adx, atr, signal_params, params) -> Dict | None:
        """RSI 均值回归策略"""
        rsi_oversold = signal_params.get('rsi_oversold', 30)
        adx_min = signal_params.get('adx_min', 15)
        
        # 入场条件
        if rsi < rsi_oversold and adx > adx_min:
            entry_price = prices[index]
            stop_loss = entry_price * (1 - signal_params.get('atr_stop_multiplier', 3.0) * atr / entry_price)
            take_profit = params.exit.get('rsi_exit', 70)
            
            return {
                "direction": "long",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "leverage": params.position.get('leverage', 2),
                "factors": {"rsi": rsi, "adx": adx},
                "confidence": params.confidence,
                "strategy": "RSI_MR"
            }
        
        return None
    
    def _generate_rsi_eman_signal(self, symbol, prices, highs, lows, index,
                                 rsi, adx, atr, signal_params, params) -> Dict | None:
        """RSI + EMA 共振策略"""
        from miracle_core import calc_macd
        
        rsi_oversold = signal_params.get('rsi_oversold', 35)
        adx_min = signal_params.get('adx_min', 15)
        
        macd, signal, hist = calc_macd(prices)
        
        # 入场条件: RSI < 35 AND MACD > Signal AND ADX > 15
        if rsi < rsi_oversold and macd > signal and adx > adx_min:
            entry_price = prices[index]
            stop_loss = entry_price * (1 - signal_params.get('atr_stop_multiplier', 2.5) * atr / entry_price)
            take_profit = params.exit.get('rsi_exit', 65)
            
            return {
                "direction": "long",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "leverage": params.position.get('leverage', 2),
                "factors": {"rsi": rsi, "adx": adx, "macd_hist": hist},
                "confidence": params.confidence,
                "strategy": "RSI_EMAn"
            }
        
        return None
    
    def _generate_rsi_vol_signal(self, symbol, prices, highs, lows, index,
                                rsi, adx, atr, signal_params, params) -> Dict | None:
        """RSI + 成交量策略"""
        # 需要成交量数据，这里简化处理
        return self._generate_rsi_mr_signal(symbol, prices, highs, lows, index,
                                           rsi, adx, atr, signal_params, params)
    
    def _generate_vol_brk_signal(self, symbol, prices, highs, lows, index,
                                rsi, adx, atr, signal_params, params) -> Dict | None:
        """成交量突破策略"""
        # 需要成交量数据，这里简化处理
        return None


# ===== 便捷函数 =====

def get_coin_optimizer() -> CoinParameterOptimizer:
    """获取全局 CoinParameterOptimizer 实例"""
    global _coin_optimizer_instance
    if '_coin_optimizer_instance' not in globals():
        _coin_optimizer_instance = CoinParameterOptimizer()
    return _coin_optimizer_instance


def get_coin_signal_generator() -> CoinSignalGenerator:
    """获取全局 CoinSignalGenerator 实例"""
    global _coin_signal_generator_instance
    if '_coin_signal_generator_instance' not in globals():
        _coin_signal_generator_instance = CoinSignalGenerator()
    return _coin_signal_generator_instance


# ===== 测试代码 =====

if __name__ == "__main__":
    print("Testing CoinParameterOptimizer for Miracle System...")
    print("=" * 60)
    
    # 初始化
    optimizer = CoinParameterOptimizer()
    
    # 获取可用币种
    enabled_coins = optimizer.get_enabled_coins()
    print(f"\n启用的币种: {enabled_coins}")
    
    # 查看 BTC 参数
    btc_params = optimizer.get_coin_params("BTC")
    if btc_params:
        print("\nBTC 参数:")
        print(f"  策略: {btc_params.optimal_strategy}")
        print(f"  周期: {btc_params.timeframe}")
        print(f"  置信度: {btc_params.confidence}")
        print(f"  RSI 参数: {btc_params.signal_params}")
    
    # Kronos 对比
    print("\n" + "-" * 60)
    print("Kronos 对比:")
    for symbol in ["BTC", "ETH", "SOL", "DOGE"]:
        comparison = optimizer.compare_with_kronos(symbol)
        if "error" not in comparison:
            print(f"\n{symbol}:")
            print(f"  Miracle: {comparison['miracle']['strategy']} ({comparison['miracle']['timeframe']})")
            print(f"  Kronos:  {comparison['kronos']['strategy']} ({comparison['kronos']['timeframe']})")
            if comparison['differences']:
                print(f"  差异: {comparison['differences']}")
    
    # 生成报告
    print("\n" + optimizer.generate_optimization_report())
