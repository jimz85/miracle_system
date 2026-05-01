from __future__ import annotations

"""
Backtest Main Interface (P2.1)
===============================
回测框架主入口

功能:
1. Walk-Forward滚动窗口验证
2. 多币种批量扫描
3. 均值回归 vs 趋势跟踪对比
4. 结果持久化到JSON
5. 集成IC权重系统

Usage:
    from backtest import BacktestRunner, run_backtest, load_klines_from_csv
    
    # 方式1: 使用便捷函数
    result = run_backtest("BTC", klines)
    
    # 方式2: 使用完整Runner
    runner = BacktestRunner(config)
    runner.load_data("BTC", klines)
    runner.run_walkforward(strategy="both")
    runner.save_results("output.json")
    
    # 方式3: 多币种批量
    runner = BacktestRunner()
    runner.load_multi_coins({"BTC": btc_klines, "ETH": eth_klines})
    runner.run_all()
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("miracle.backtest")

# ==================== 配置 ====================

DEFAULT_BACKTEST_CONFIG = {
    "initial_balance": 100000,      # 初始资金
    "commission_rate": 0.0005,      # 0.05% 手续费
    "slippage_rate": 0.0002,        # 0.02% 滑点
    "leverage": 2,                  # 默认杠杆
    # Walk-Forward配置
    "wf_train_days": 90,            # 3个月训练窗口
    "wf_test_days": 30,             # 1个月测试窗口
    "wf_step_days": 15,             # 滚动步长
    # 输出配置
    "output_dir": "~/.miracle_backtest",
    "save_trades": True,
    "save_equity": False,
}


# ==================== 数据结构 ====================

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_balance: float = 100000
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0002
    per_coin_slippage: Dict[str, float] = field(default_factory=dict)
    leverage: float = 2
    wf_train_days: int = 90
    wf_test_days: int = 30
    wf_step_days: int = 15
    output_dir: str = "~/.miracle_backtest"
    save_trades: bool = True
    save_equity: bool = False


@dataclass
class CoinResult:
    """单个币种回测结果"""
    symbol: str
    success: bool
    error: str = ""
    mean_reversion: Dict | None = None
    trend_following: Dict | None = None
    comparison: Dict | None = None
    best_strategy: str = ""
    best_return: float = 0


@dataclass
class BatchResult:
    """批量回测结果"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    num_coins: int = 0
    total_windows: int = 0
    results: Dict[str, CoinResult] = field(default_factory=dict)
    summary: Dict = field(default_factory=dict)


# ==================== 主Runner ====================

class BacktestRunner:
    """
    回测运行器
    
    提供完整的回测工作流:
    1. 加载数据
    2. 运行Walk-Forward验证
    3. 多币种批量扫描
    4. 结果持久化
    """
    
    def __init__(self, config: Dict = None):
        self.config = BacktestConfig(**{**DEFAULT_BACKTEST_CONFIG, **(config or {})})
        self.data: Dict[str, List[Dict]] = {}  # symbol -> klines
        self.results: Dict[str, CoinResult] = {}
        self.batch_result: BatchResult | None = None
        
        # 如果per_coin_slippage未传入，从miracle_config.json自动加载
        if not self.config.per_coin_slippage:
            self._load_per_coin_slippage()
        
        # 延迟导入walkforward模块，避免循环导入
        from backtest.walkforward import WalkForwardValidator
        
        # 创建Walk-Forward验证器
        wf_config = {
            "train_days": self.config.wf_train_days,
            "test_days": self.config.wf_test_days,
            "step_days": self.config.wf_step_days,
            "slippage_rate": self.config.slippage_rate,
        }
        self.validator = WalkForwardValidator(wf_config)
        
        # 确保输出目录存在
        self.output_dir = os.path.expanduser(self.config.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_per_coin_slippage(self) -> None:
        """从miracle_config.json加载每币种滑点配置"""
        try:
            cfg_path = Path(__file__).parent.parent / "miracle_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                per_coin = cfg.get("fee", {}).get("per_coin_slippage", {})
                if per_coin:
                    self.config.per_coin_slippage = per_coin
                    logger.info(f"加载了 {len(per_coin)} 个币种的专属滑点配置")
        except Exception as e:
            logger.warning(f"加载per_coin_slippage失败: {e}")
    
    def load_data(self, symbol: str, klines: List[Dict]) -> None:
        """加载单个币种数据"""
        if not klines:
            raise ValueError(f"数据不能为空: {symbol}")
        
        # 验证数据格式
        required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
        first = klines[0]
        missing = [f for f in required_fields if f not in first and f != "volume"]
        if missing:
            # 尝试兼容格式
            if "ts" in first:
                klines = [{**k, "timestamp": k.get("ts", k.get("timestamp", 0))} for k in klines]
        
        self.data[symbol] = klines
        logger.info(f"加载 {symbol}: {len(klines)} 条K线")
    
    def load_multi_coins(self, coin_data: Dict[str, List[Dict]]) -> None:
        """批量加载多个币种数据"""
        for symbol, klines in coin_data.items():
            try:
                self.load_data(symbol, klines)
            except Exception as e:
                logger.error(f"加载 {symbol} 失败: {e}")
    
    def load_from_csv(self, csv_path: str, symbol: str = None) -> None:
        """从CSV文件加载数据"""
        import csv
        
        if symbol is None:
            # 从文件名推断
            basename = os.path.basename(csv_path)
            symbol = basename.replace("klines_", "").replace(".csv", "").split("_")[0]
        
        klines = []
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                klines.append({
                    "timestamp": int(row.get("ts", row.get("timestamp", 0))),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })
        
        self.load_data(symbol, klines)
    
    def load_default_data(self) -> None:
        """加载默认数据 (BTC, ETH, SOL 15m)"""
        import glob
        
        data_dir = os.path.expanduser("~/.hermes/cron/output")
        patterns = [
            f"{data_dir}/klines_BTC_15m.csv",
            f"{data_dir}/klines_ETH_15m.csv", 
            f"{data_dir}/klines_SOL_15m.csv",
        ]
        
        for pattern in patterns:
            for csv_path in glob.glob(pattern):
                try:
                    self.load_from_csv(csv_path)
                except Exception as e:
                    logger.warning(f"加载 {csv_path} 失败: {e}")
    
    def run_walkforward(
        self, 
        symbol: str = None,
        strategy: str = "both",
        leverage: float = None
    ) -> CoinResult:
        """
        运行Walk-Forward验证
        
        Args:
            symbol: 币种 (默认使用第一个加载的)
            strategy: "mean_reversion", "trend_following", 或 "both"
            leverage: 杠杆倍数
            
        Returns:
            CoinResult
        """
        if symbol is None:
            if not self.data:
                raise ValueError("未加载数据")
            symbol = list(self.data.keys())[0]
        
        if symbol not in self.data:
            raise ValueError(f"未找到数据: {symbol}")
        
        klines = self.data[symbol]
        leverage = leverage or self.config.leverage

        # 按币种覆盖滑点率(优先使用per_coin_slippage配置)
        coin_slippage = self.config.per_coin_slippage.get(symbol.upper())
        if coin_slippage is not None:
            self.validator.slippage_rate = coin_slippage
            logger.info(f"  {symbol}: 使用币种专属滑点 {coin_slippage:.4f} ({coin_slippage*100:.2f}%)")
        else:
            self.validator.slippage_rate = self.config.slippage_rate
        
        logger.info(f"运行 Walk-Forward: {symbol} ({strategy})")
        
        result = CoinResult(symbol=symbol, success=True)
        
        try:
            if strategy == "mean_reversion":
                wf_result = self.validator.run_mean_reversion(klines, leverage)
                result.mean_reversion = wf_result.to_dict()
                result.best_strategy = "mean_reversion"
                result.best_return = wf_result.test_avg_return
                
            elif strategy == "trend_following":
                wf_result = self.validator.run_trend_following(klines, leverage)
                result.trend_following = wf_result.to_dict()
                result.best_strategy = "trend_following"
                result.best_return = wf_result.test_avg_return
                
            else:  # "both"
                mr_result, tf_result = self.validator.run_both(klines, leverage)
                
                result.mean_reversion = mr_result.to_dict()
                result.trend_following = tf_result.to_dict()
                result.comparison = {
                    "winner": "mean_reversion" if mr_result.test_avg_return > tf_result.test_avg_return else "trend_following",
                    "mr_return": mr_result.test_avg_return,
                    "tf_return": tf_result.test_avg_return,
                    "mr_sharpe": mr_result.test_avg_sharpe,
                    "tf_sharpe": tf_result.test_avg_sharpe,
                    "return_diff": abs(mr_result.test_avg_return - tf_result.test_avg_return),
                    "sharpe_diff": abs(mr_result.test_avg_sharpe - tf_result.test_avg_sharpe),
                }
                result.best_strategy = result.comparison["winner"]
                result.best_return = max(mr_result.test_avg_return, tf_result.test_avg_return)
            
            self.results[symbol] = result
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Walk-Forward 运行失败: {symbol} - {e}")
        
        return result
    
    def run_all(self, strategy: str = "both") -> BatchResult:
        """
        对所有加载的币种运行回测
        
        Args:
            strategy: "mean_reversion", "trend_following", 或 "both"
            
        Returns:
            BatchResult
        """
        self.results = {}
        total_windows = 0
        
        for symbol in self.data:
            logger.info(f"处理 {symbol}...")
            result = self.run_walkforward(symbol, strategy)
            self.results[symbol] = result
            
            if result.success:
                if result.mean_reversion:
                    total_windows += result.mean_reversion.get("num_windows", 0)
                if result.trend_following:
                    total_windows += result.trend_following.get("num_windows", 0)
        
        # 生成汇总
        self.batch_result = self._summarize_batch_result(total_windows)
        return self.batch_result
    
    def _summarize_batch_result(self, total_windows: int) -> BatchResult:
        """汇总批量结果"""
        successful = [r for r in self.results.values() if r.success]
        
        summary = {
            "total_coins": len(self.data),
            "successful_coins": len(successful),
            "total_windows": total_windows,
            "best_performer": None,
            "avg_return": 0,
            "strategy_wins": {"mean_reversion": 0, "trend_following": 0},
        }
        
        if successful:
            best_return = -999
            best_symbol = None
            
            for r in successful:
                if r.best_return > best_return:
                    best_return = r.best_return
                    best_symbol = r.symbol
                
                if r.comparison:
                    summary["strategy_wins"][r.comparison["winner"]] += 1
            
            summary["best_performer"] = {
                "symbol": best_symbol,
                "return": best_return,
                "strategy": self.results[best_symbol].best_strategy,
            }
            summary["avg_return"] = sum(r.best_return for r in successful) / len(successful)
        
        return BatchResult(
            timestamp=datetime.now().isoformat(),
            num_coins=len(self.data),
            total_windows=total_windows,
            results={k: asdict(v) for k, v in self.results.items()},
            summary=summary,
        )
    
    def save_results(self, filename: str = None) -> str:
        """
        保存结果到JSON文件
        
        Args:
            filename: 文件名 (默认: backtest_{timestamp}.json)
            
        Returns:
            保存的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if self.batch_result:
            data = asdict(self.batch_result)
        else:
            data = {
                "timestamp": datetime.now().isoformat(),
                "results": {k: asdict(v) for k, v in self.results.items()},
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存: {filepath}")
        return filepath
    
    def get_summary_report(self) -> str:
        """生成汇总报告"""
        if not self.batch_result:
            return "无回测结果"
        
        s = self.batch_result.summary
        lines = [
            "=" * 60,
            "Walk-Forward 回测汇总报告",
            "=" * 60,
            f"时间: {self.batch_result.timestamp}",
            f"币种数量: {s['total_coins']} (成功: {s['successful_coins']})",
            f"总窗口数: {s['total_windows']}",
            f"平均收益: {s['avg_return']:.2f}%",
            "",
            "策略胜出统计:",
            f"  均值回归: {s['strategy_wins']['mean_reversion']} 次",
            f"  趋势跟踪: {s['strategy_wins']['trend_following']} 次",
            "",
        ]
        
        if s['best_performer']:
            bp = s['best_performer']
            lines.append(f"最佳表现: {bp['symbol']} ({bp['strategy']}) - {bp['return']:.2f}%")
        
        lines.append("=" * 60)
        
        # 各币种详情
        for symbol, result in self.results.items():
            if result.success:
                mr = result.mean_reversion
                tf = result.trend_following
                
                lines.append(f"\n{symbol}:")
                if mr:
                    lines.append(f"  均值回归: 收益={mr['test_avg_return']:.2f}%, 夏普={mr['test_avg_sharpe']:.2f}")
                if tf:
                    lines.append(f"  趋势跟踪: 收益={tf['test_avg_return']:.2f}%, 夏普={tf['test_avg_sharpe']:.2f}")
                if result.comparison:
                    lines.append(f"  胜出: {result.comparison['winner']} (差异: {result.comparison['return_diff']:.2f}%)")
            else:
                lines.append(f"\n{symbol}: 失败 - {result.error}")
        
        return "\n".join(lines)


# ==================== 便捷函数 ====================

def load_klines_from_csv(csv_path: str) -> Tuple[str, List[Dict]]:
    """
    从CSV加载K线数据
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        (symbol, klines) 元组
    """
    import csv
    
    # 从文件名推断symbol
    basename = os.path.basename(csv_path)
    symbol = basename.replace("klines_", "").replace(".csv", "").split("_")[0]
    
    klines = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            klines.append({
                "timestamp": int(row.get("ts", row.get("timestamp", 0))),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
    
    return symbol, klines


def run_backtest(
    symbol: str,
    klines: List[Dict],
    strategy: str = "both",
    leverage: float = 2,
    output_file: str = None
) -> Dict:
    """
    运行回测的便捷函数
    
    Args:
        symbol: 币种名称
        klines: K线数据
        strategy: "mean_reversion", "trend_following", 或 "both"
        leverage: 杠杆倍数
        output_file: 输出文件路径
        
    Returns:
        回测结果字典
    """
    runner = BacktestRunner()
    runner.load_data(symbol, klines)
    
    if strategy == "both":
        runner.run_walkforward(strategy="both")
    elif strategy == "mean_reversion":
        runner.run_walkforward(strategy="mean_reversion")
    else:
        runner.run_walkforward(strategy="trend_following")
    
    result = runner.results.get(symbol)
    
    if output_file:
        runner.save_results(output_file)
    
    return asdict(result) if result else {}


def run_multi_coin_backtest(
    coin_data: Dict[str, List[Dict]],
    strategy: str = "both",
    output_dir: str = None
) -> BatchResult:
    """
    运行多币种批量回测
    
    Args:
        coin_data: 币种数据字典 {"BTC": [...], "ETH": [...]}
        strategy: "mean_reversion", "trend_following", 或 "both"
        output_dir: 输出目录
        
    Returns:
        BatchResult
    """
    runner = BacktestRunner({"output_dir": output_dir} if output_dir else {})
    runner.load_multi_coins(coin_data)
    return runner.run_all(strategy)


# ==================== 自检 ====================

if __name__ == "__main__":
    import random
    import sys
    from pathlib import Path
    
    # 添加项目根目录到path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("=== Backtest 自检 ===\n")
    
    # 生成模拟数据
    base_price = 50000
    klines = []
    price = base_price
    start_ts = int(datetime.now().timestamp() * 1000) - 180 * 24 * 3600 * 1000
    
    for i in range(1000):
        price = price * (1 + random.uniform(-0.015, 0.02))
        klines.append({
            "timestamp": start_ts + i * 3600 * 1000,
            "open": price * 0.99,
            "high": price * 1.02,
            "low": price * 0.97,
            "close": price,
            "volume": random.uniform(100, 1000),
        })
    
    print(f"生成模拟数据: {len(klines)} 条K线\n")
    
    # 方式1: 便捷函数
    print("--- 方式1: run_backtest ---")
    result = run_backtest("BTC", klines, strategy="both")
    print(f"成功: {result.get('success')}")
    print(f"最佳策略: {result.get('best_strategy')}")
    print(f"最佳收益: {result.get('best_return', 0):.2f}%")
    
    # 方式2: 完整Runner
    print("\n--- 方式2: BacktestRunner ---")
    runner = BacktestRunner()
    runner.load_data("ETH", klines)
    runner.run_walkforward(strategy="both")
    print(runner.get_summary_report())
    
    # 保存结果
    output_path = runner.save_results()
    print(f"\n结果已保存: {output_path}")
    
    print("\n=== 自检完成 ===")
