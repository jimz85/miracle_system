"""
Multi-Strategy Portfolio Manager - 多策略组合管理器
支持多个策略组合、动态权重调整、蒙特卡洛压力测试
"""
import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ========================
# 枚举和配置
# ========================

class AllocationMethod(Enum):
    """资金分配方法"""
    EQUAL = "equal"                        # 均等分配
    VOLATILITY_WEIGHTED = "vol_weighted"  # 波动率加权
    PERFORMANCE_WEIGHTED = "perf_weighted"  # 绩效加权
    RISK_PARITY = "risk_parity"           # 风险平价


class WeightUpdateMethod(Enum):
    """权重更新方法"""
    FIXED = "fixed"               # 固定权重
    MEAN_REVERSION = "mean_rev"   # 均值回归
    MOMENTUM = "momentum"         # 动量
    TRAILING_STOP = "trailing"    # 追踪止损


@dataclass
class StrategyResult:
    """策略结果"""
    strategy_name: str
    period_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    volatility: float


@dataclass
class PortfolioAllocation:
    """组合配置"""
    strategy_name: str
    weight: float
    enabled: bool = True
    max_drawdown_limit: float = 0.2  # 20%最大回撤限制


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario: str
    var_95: float          # 95% VaR
    cvar_95: float         # 95% CVaR
    max_drawdown: float
    prob_loss: float       # 亏损概率
    expected_return: float
    sharpe_ratio: float
    simulation_paths: int = 0


# ========================
# 多策略组合管理器
# ========================

class MultiStrategyPortfolio:
    """
    多策略组合管理器
    
    功能:
    - 多个策略同时运行
    - 动态权重调整
    - 自动再平衡
    - 相关性风险管理
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        allocation_method: AllocationMethod = AllocationMethod.EQUAL,
        weight_update_method: WeightUpdateMethod = WeightUpdateMethod.FIXED,
        rebalance_threshold: float = 0.05  # 5%偏移触发再平衡
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.allocation_method = allocation_method
        self.weight_update_method = weight_update_method
        self.rebalance_threshold = rebalance_threshold
        
        self.strategies: Dict[str, Dict] = {}
        self.allocations: Dict[str, PortfolioAllocation] = {}
        self.performance_history: List[Dict] = []
        
        # 相关性矩阵
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
    
    def add_strategy(
        self,
        name: str,
        weight: float = 0.2,
        max_drawdown_limit: float = 0.2,
        enabled: bool = True
    ) -> None:
        """添加策略到组合"""
        self.strategies[name] = {
            "name": name,
            "capital": self.current_capital * weight,
            "enabled": enabled,
            "trades": [],
            "equity_curve": [self.current_capital * weight]
        }
        
        self.allocations[name] = PortfolioAllocation(
            strategy_name=name,
            weight=weight,
            enabled=enabled,
            max_drawdown_limit=max_drawdown_limit
        )
        
        logger.info(f"Added strategy '{name}' with weight {weight:.1%}")
    
    def remove_strategy(self, name: str) -> bool:
        """从组合移除策略"""
        if name in self.strategies:
            del self.strategies[name]
            del self.allocations[name]
            return True
        return False
    
    def get_strategy_capital(self, name: str) -> float:
        """获取策略分配的资本"""
        if name not in self.strategies:
            return 0
        return self.strategies[name]["capital"]
    
    def update_weights(self, results: Dict[str, StrategyResult]) -> Dict[str, float]:
        """
        根据绩效更新策略权重
        
        Args:
            results: 各策略的绩效结果
        
        Returns:
            新权重字典
        """
        if not results:
            return {name: alloc.weight for name, alloc in self.allocations.items()}
        
        if self.weight_update_method == WeightUpdateMethod.FIXED:
            return {name: alloc.weight for name, alloc in self.allocations.items()}
        
        # 计算新权重
        new_weights = {}
        total_score = 0
        scores = {}
        
        for name, result in results.items():
            if name not in self.allocations:
                continue
            
            # 计算策略得分
            if self.weight_update_method == WeightUpdateMethod.MEAN_REVERSION:
                # 均值回归: 低回撤高得分
                score = result.sharpe_ratio * (1 - result.max_drawdown)
            elif self.weight_update_method == WeightUpdateMethod.MOMENTUM:
                # 动量: 高收益高得分
                score = result.period_return * result.sharpe_ratio
            elif self.weight_update_method == WeightUpdateMethod.TRAILING_STOP:
                # 追踪止损: 考虑最大回撤限制
                if result.max_drawdown > self.allocations[name].max_drawdown_limit:
                    score = 0  # 超限则清零
                else:
                    score = result.sharpe_ratio
            else:
                score = result.sharpe_ratio
            
            scores[name] = max(score, 0.01)  # 避免零权重
            total_score += scores[name]
        
        # 归一化权重
        for name, score in scores.items():
            new_weights[name] = score / total_score if total_score > 0 else 0
        
        return new_weights
    
    def rebalance_if_needed(self, new_weights: Dict[str, float]) -> bool:
        """
        检查是否需要再平衡
        
        Returns:
            是否执行了再平衡
        """
        rebalanced = False
        
        for name, new_weight in new_weights.items():
            if name not in self.allocations:
                continue
            
            old_weight = self.allocations[name].weight
            drift = abs(new_weight - old_weight)
            
            if drift > self.rebalance_threshold:
                # 再平衡
                self.allocations[name].weight = new_weight
                self.strategies[name]["capital"] = self.current_capital * new_weight
                rebalanced = True
                logger.info(f"Rebalanced '{name}': {old_weight:.2%} → {new_weight:.2%}")
        
        return rebalanced
    
    def should_disable_strategy(self, name: str, result: StrategyResult) -> bool:
        """检查策略是否应该被禁用"""
        if name not in self.allocations:
            return False
        
        alloc = self.allocations[name]
        
        # 检查最大回撤限制
        if result.max_drawdown > alloc.max_drawdown_limit:
            logger.warning(f"Strategy '{name}' exceeded max drawdown limit: "
                         f"{result.max_drawdown:.1%} > {alloc.max_drawdown_limit:.1%}")
            return True
        
        # 检查连续亏损
        recent_trades = self.strategies[name].get("trades", [])[-5:]
        if len(recent_trades) >= 5:
            losses = sum(1 for t in recent_trades if t.get("pnl", 0) < 0)
            if losses >= 4:
                logger.warning(f"Strategy '{name}' has {losses} losses in last 5 trades")
                return True
        
        return False
    
    def record_trade(self, strategy_name: str, trade: Dict) -> None:
        """记录交易"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name]["trades"].append(trade)
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """获取组合统计"""
        total_value = sum(s["capital"] for s in self.strategies.values())
        total_pnl = total_value - self.initial_capital
        
        returns = []
        for s in self.strategies.values():
            if len(s["equity_curve"]) >= 2:
                start = s["equity_curve"][0]
                end = s["equity_curve"][-1]
                if start > 0:
                    returns.append((end - start) / start)
        
        avg_return = np.mean(returns) if returns else 0
        portfolio_vol = np.std(returns) if len(returns) > 1 else 0
        
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "return_pct": (total_value - self.initial_capital) / self.initial_capital,
            "num_strategies": len(self.strategies),
            "avg_return": avg_return,
            "portfolio_volatility": portfolio_vol,
            "sharpe_ratio": (avg_return / portfolio_vol) if portfolio_vol > 0 else 0
        }
    
    def export_allocations(self) -> List[Dict]:
        """导出配置"""
        return [
            {
                "strategy_name": name,
                "weight": alloc.weight,
                "enabled": alloc.enabled,
                "max_drawdown_limit": alloc.max_drawdown_limit
            }
            for name, alloc in self.allocations.items()
        ]


# ========================
# 蒙特卡洛压力测试
# ========================

class MonteCarloStressTest:
    """
    蒙特卡洛压力测试
    
    功能:
    - GBM价格路径模拟
    - VaR/CVaR计算
    - 黑天鹅场景测试
    - 多策略相关性分析
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        num_simulations: int = 10000,
        time_horizon_days: int = 252
    ):
        self.initial_capital = initial_capital
        self.num_simulations = num_simulations
        self.time_horizon_days = time_horizon_days
    
    def simulate_gbm(
        self,
        mu: float = 0.0,      # 年化收益率
        sigma: float = 0.2,   # 年化波动率
        initial_price: float = 1.0,
        dt: float = 1/252     # 日步长
    ) -> np.ndarray:
        """
        几何布朗运动模拟
        
        Returns:
            (num_simulations, time_steps) 的价格路径数组
        """
        num_steps = self.time_horizon_days
        dt = dt
        
        # 生成随机增量
        Z = np.random.standard_normal((self.num_simulations, num_steps))
        
        # 累积路径
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # 计算对数收益
        log_returns = drift + diffusion
        
        # 累积收益
        log_prices = np.cumsum(log_returns, axis=1)
        
        # 添加初始价格
        log_prices += np.log(initial_price)
        
        return np.exp(log_prices)
    
    def simulate_jump_diffusion(
        self,
        mu: float = 0.0,
        sigma: float = 0.2,
        lam: float = 0.5,       # 跳频率
        mu_j: float = -0.1,    # 跳均值
        sigma_j: float = 0.3,  # 跳波动率
        initial_price: float = 1.0
    ) -> np.ndarray:
        """
        Merton跳扩散模型 (更真实的极端场景)
        """
        num_steps = self.time_horizon_days
        dt = 1/252
        
        # GBM部分
        Z = np.random.standard_normal((self.num_simulations, num_steps))
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # 跳跃部分
        J = np.random.poisson(lam * dt, (self.num_simulations, num_steps))
        jump_sizes = np.random.normal(mu_j, sigma_j, (self.num_simulations, num_steps))
        jumps = J * jump_sizes
        
        # 组合
        log_returns = drift + diffusion + jumps
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices += np.log(initial_price)
        
        return np.exp(log_prices)
    
    def calculate_var_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        计算VaR和CVaR
        
        Args:
            returns: 收益率数组
            confidence: 置信度
        
        Returns:
            (VaR, CVaR)
        """
        sorted_returns = np.sort(returns)
        var_index = int((1 - confidence) * len(sorted_returns))
        var = sorted_returns[var_index]
        
        # CVaR是VaR左侧尾巴的平均
        cvar = np.mean(sorted_returns[:var_index])
        
        return var, cvar
    
    def run_stress_test(
        self,
        strategy_configs: List[Dict],
        scenario: str = "normal"
    ) -> StressTestResult:
        """
        运行压力测试
        
        Args:
            strategy_configs: 策略配置列表
                [{"name": str, "weight": float, "mu": float, "sigma": float}]
            scenario: 场景名称
        """
        # 设置参数
        if scenario == "normal":
            base_mu, base_sigma = 0.1, 0.2
        elif scenario == "bull":
            base_mu, base_sigma = 0.3, 0.15
        elif scenario == "crash":
            base_mu, base_sigma = -0.2, 0.4
        elif scenario == "black_swan":
            base_mu, base_sigma = -0.5, 0.8
        elif scenario == "volatile":
            base_mu, base_sigma = 0.0, 0.5
        else:
            base_mu, base_sigma = 0.1, 0.2
        
        # 模拟组合价值路径
        num_steps = self.time_horizon_days
        
        for config in strategy_configs:
            weight = config.get("weight", 1.0)
            mu = config.get("mu", base_mu)
            sigma = config.get("sigma", base_sigma)
            
            if config.get("jump_risk", False):
                price_paths = self.simulate_jump_diffusion(mu, sigma)
            else:
                price_paths = self.simulate_gbm(mu, sigma)
            
            # 策略价值路径: 初始资本 * 权重 * 价格比例
            strategy_capital = self.initial_capital * weight
            strategy_value = strategy_capital * price_paths  # [sims, 252]
            
            # 累积到组合
            if config == strategy_configs[0]:
                portfolio_paths = strategy_value
            else:
                portfolio_paths += strategy_value[:, :num_steps]
        
        # 计算每日收益率 (修正形状)
        portfolio_returns = np.diff(portfolio_paths, axis=1) / np.maximum(portfolio_paths[:, :-1], 1e-10)
        
        # 最终收益率
        final_returns = (portfolio_paths[:, -1] - self.initial_capital) / self.initial_capital
        
        # VaR/CVaR
        var_95, cvar_95 = self.calculate_var_cvar(final_returns, 0.95)
        
        # 最大回撤
        cumulative = portfolio_paths / self.initial_capital
        running_max = np.maximum.accumulate(cumulative, axis=1)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdowns = np.min(drawdowns, axis=1)
        
        # 统计
        prob_loss = np.mean(final_returns < 0)
        expected_return = np.mean(final_returns)
        sharpe = expected_return / np.std(final_returns) if np.std(final_returns) > 0 else 0
        
        return StressTestResult(
            scenario=scenario,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=np.mean(max_drawdowns),
            prob_loss=prob_loss,
            expected_return=expected_return,
            sharpe_ratio=sharpe,
            simulation_paths=self.num_simulations
        )
    
    def run_multi_scenario(self, strategy_configs: List[Dict]) -> List[StressTestResult]:
        """运行多种场景"""
        scenarios = ["normal", "bull", "crash", "black_swan", "volatile"]
        return [self.run_stress_test(strategy_configs, s) for s in scenarios]
    
    def generate_report(self, results: List[StressTestResult]) -> str:
        """生成压力测试报告"""
        report = ["=" * 60]
        report.append("Monte Carlo Stress Test Report")
        report.append("=" * 60)
        report.append(f"Simulations: {self.num_simulations:,}")
        report.append(f"Time Horizon: {self.time_horizon_days} days ({self.time_horizon_days/252:.1f} years)")
        report.append("")
        
        for result in results:
            report.append(f"\n{'-'*40}")
            report.append(f"Scenario: {result.scenario.upper()}")
            report.append(f"{'-'*40}")
            report.append(f"  Expected Return:    {result.expected_return:+.2%}")
            report.append(f"  VaR (95%):          {result.var_95:+.2%}")
            report.append(f"  CVaR (95%):         {result.cvar_95:+.2%}")
            report.append(f"  Max Drawdown:       {result.max_drawdown:+.2%}")
            report.append(f"  Probability of Loss: {result.prob_loss:.2%}")
            report.append(f"  Sharpe Ratio:        {result.sharpe_ratio:.2f}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


# ========================
# 工厂函数
# ========================

def create_portfolio(config: Dict) -> MultiStrategyPortfolio:
    """创建组合管理器"""
    portfolio = MultiStrategyPortfolio(
        initial_capital=config.get("initial_capital", 100000),
        allocation_method=AllocationMethod(config.get("allocation_method", "equal")),
        weight_update_method=WeightUpdateMethod(config.get("weight_update_method", "fixed")),
        rebalance_threshold=config.get("rebalance_threshold", 0.05)
    )
    
    for strategy in config.get("strategies", []):
        portfolio.add_strategy(
            name=strategy["name"],
            weight=strategy.get("weight", 0.2),
            max_drawdown_limit=strategy.get("max_drawdown_limit", 0.2),
            enabled=strategy.get("enabled", True)
        )
    
    return portfolio


def run_quick_stress_test(
    initial_capital: float = 100000,
    num_simulations: int = 1000
) -> List[StressTestResult]:
    """快速压力测试"""
    test = MonteCarloStressTest(
        initial_capital=initial_capital,
        num_simulations=num_simulations
    )
    
    strategy_configs = [
        {"name": "trend_following", "weight": 0.4, "mu": 0.15, "sigma": 0.25},
        {"name": "mean_reversion", "weight": 0.3, "mu": 0.08, "sigma": 0.15},
        {"name": "momentum", "weight": 0.3, "mu": 0.12, "sigma": 0.30},
    ]
    
    return test.run_multi_scenario(strategy_configs)


if __name__ == "__main__":
    print("=== Multi-Strategy Portfolio Test ===\n")
    
    # 测试组合管理器
    portfolio = MultiStrategyPortfolio(
        initial_capital=100000,
        allocation_method=AllocationMethod.VOLATILITY_WEIGHTED
    )
    
    portfolio.add_strategy("Trend Following", weight=0.4, max_drawdown_limit=0.15)
    portfolio.add_strategy("Mean Reversion", weight=0.35, max_drawdown_limit=0.10)
    portfolio.add_strategy("Momentum", weight=0.25, max_drawdown_limit=0.20)
    
    print(f"Strategies: {len(portfolio.strategies)}")
    print(f"Allocations: {portfolio.export_allocations()}")
    print(f"Stats: {portfolio.get_portfolio_stats()}")
    
    # 测试蒙特卡洛
    print("\n=== Monte Carlo Stress Test ===\n")
    
    test = MonteCarloStressTest(
        initial_capital=100000,
        num_simulations=1000
    )
    
    strategy_configs = [
        {"name": "trend", "weight": 0.5, "mu": 0.15, "sigma": 0.25},
        {"name": "mean_rev", "weight": 0.3, "mu": 0.08, "sigma": 0.15},
        {"name": "momentum", "weight": 0.2, "mu": 0.12, "sigma": 0.30},
    ]
    
    results = test.run_multi_scenario(strategy_configs)
    print(test.generate_report(results))
    
    print("\n=== Test Complete ===")
