#!/usr/bin/env python3
from __future__ import annotations

"""
Miracle Autonomous 2.0 - 自主研究循环主程序

整合数据收集→假设生成→回测验证→反思改进的完整闭环。
作为Miracle-Kronos v2架构的核心研究引擎。

核心特性:
1. 数据收集: 多源数据聚合 (OKX/Kronos数据 + Miracle情报)
2. 假设生成: 智能参数变异 (基于历史表现的趋势外推)
3. 回测验证: Walk-Forward多窗口验证 + 多币种综合评估
4. 反思改进: IC权重动态调整 + 策略质量评分

参考: Karpathy Autoresearch Keep/Discard循环模式
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import numpy as np
import pandas as pd

from autoresearch.backtest_engine import run_single_backtest, run_walkforward
from autoresearch.data_loader import COINS, compute_indicators, load_timeframe_data
from autoresearch.experiment_logger import (
    RESULTS_DIR,
    ExperimentRecord,
    ensure_dirs,
    generate_experiment_id,
    get_best_result,
    get_git_commit,
    init_results_tsv,
    print_summary,
    write_experiment,
)
from autoresearch.strategy_config import BacktestResult, StrategyConfig
from core.regime_classifier import RegimeClassifier

# 工作目录
WORKSPACE = Path(os.environ.get("MIRACLE_WORKSPACE", str(Path.home())))

# ===== 日志配置 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(WORKSPACE / "miracle_autonomous.log")
    ]
)
logger = logging.getLogger(__name__)


# ===== 搜索空间定义 =====
SEARCH_SPACE = {
    "rsi_oversold": {"min": 25.0, "max": 40.0, "type": "float"},
    "rsi_overbought": {"min": 60.0, "max": 75.0, "type": "float"},
    "adx_threshold": {"min": 10.0, "max": 30.0, "type": "float"},
    "sl_atr_mult": {"min": 0.8, "max": 3.0, "type": "float"},
    "tp1_atr_mult": {"min": 1.5, "max": 5.0, "type": "float"},
    "tp2_atr_mult": {"min": 3.0, "max": 10.0, "type": "float"},
    "vol_ratio_threshold": {"min": 0.8, "max": 2.0, "type": "float"},
    "position_size_pct": {"min": 0.05, "max": 0.15, "type": "float"},
    "bb_position_oversold": {"min": 10.0, "max": 35.0, "type": "float"},
    "bb_position_overbought": {"min": 65.0, "max": 90.0, "type": "float"},
    "atr_percentile_max": {"min": 40.0, "max": 90.0, "type": "float"},
    "atr_percentile_period": {"min": 20, "max": 100, "type": "int"},
}

# 默认币种列表
DEFAULT_COINS = ["BTC", "ETH", "ADA", "DOGE", "AVAX", "DOT", "BCH", "ETC", "ALGO", "CRO", "BNT", "CVC", "GAS"]

# Walk-Forward参数
N_WINDOWS = 8
TRAIN_RATIO = 0.7
INITIAL_CAPITAL = 100000.0


# ===== 阶段枚举 =====
class LoopStage(Enum):
    DATA_COLLECTION = "data_collection"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    BACKTEST_VALIDATION = "backtest_validation"
    REFLECTION_IMPROVEMENT = "reflection_improvement"


# ===== 假设类型 =====
@dataclass
class Hypothesis:
    """交易策略假设"""
    id: str
    description: str
    mutations: Dict[str, float]  # 参数名 -> 新值
    confidence: float = 0.5  # 置信度
    source: str = "random"  # random / trend / analysis
    parent_id: str | None = None  # 父假设ID


@dataclass
class LoopState:
    """自主研究循环状态"""
    stage: LoopStage = LoopStage.DATA_COLLECTION
    iteration: int = 0
    best_sharpe: float = 0.0
    best_config: StrategyConfig | None = None
    total_experiments: int = 0
    kept_count: int = 0
    discarded_count: int = 0
    crashed_count: int = 0
    data_loaded: bool = False
    market_regime: str = "unknown"  # bull / bear / sideways
    strategy_quality_score: float = 0.5  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "stage": self.stage.value,
            "iteration": self.iteration,
            "best_sharpe": self.best_sharpe,
            "total_experiments": self.total_experiments,
            "kept": self.kept_count,
            "discarded": self.discarded_count,
            "crashed": self.crashed_count,
            "data_loaded": self.data_loaded,
            "market_regime": self.market_regime,
            "strategy_quality_score": self.strategy_quality_score,
        }
        if self.best_config is not None:
            result["best_config"] = self.best_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoopState":
        state = cls()
        state.stage = LoopStage(d.get("stage", "data_collection"))
        state.iteration = d.get("iteration", 0)
        state.best_sharpe = d.get("best_sharpe", 0.0)
        state.total_experiments = d.get("total_experiments", 0)
        state.kept_count = d.get("kept", 0)
        state.discarded_count = d.get("discarded", 0)
        state.crashed_count = d.get("crashed", 0)
        state.data_loaded = d.get("data_loaded", False)
        state.market_regime = d.get("market_regime", "unknown")
        state.strategy_quality_score = d.get("strategy_quality_score", 0.5)
        if "best_config" in d and d["best_config"]:
            state.best_config = StrategyConfig.from_dict(d["best_config"])
        return state


# ===== 数据收集模块 =====
class DataCollector:
    """多源数据收集器"""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or WORKSPACE / ".hermes" / "cron" / "output"
        self.cache: Dict[str, pd.DataFrame] = {}
        self._regime_classifier = RegimeClassifier()
        self._bear_lookback_days = 180  # 熊市分析窗口长度（天）

    def load_coin_data(self, coin: str, timeframe: str = "1h",
                       bear_only: bool = False) -> pd.DataFrame | None:
        """加载单个币种数据"""
        cache_key = f"{coin}_{timeframe}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        df = load_timeframe_data(coin, timeframe, data_dir=self.data_dir)
        if df is None or len(df) < 500:
            logger.warning(f"[DataCollector] {coin}: insufficient data")
            return None

        # 熊市过滤 (仅 bear_only=True 时启用)
        # 使用动态滚动窗口，避免硬编码日期导致的数据窥视(data snooping)
        if bear_only:
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
            dynamic_end = df["datetime_utc"].max()
            # 用数据末尾向前滚动 N 天作为分析窗口
            # 旧代码硬编码了 "2025-08-01"，导致数据窥视偏差
            dynamic_start = dynamic_end - timedelta(days=self._bear_lookback_days)
            mask = (df["datetime_utc"] >= dynamic_start.strftime("%Y-%m-%d")) & \
                   (df["datetime_utc"] <= dynamic_end.strftime("%Y-%m-%d"))
            df = df[mask].copy()
        else:
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

        if len(df) < 200:
            logger.warning(f"[DataCollector] {coin}: bear period has only {len(df)} rows")
            return None

        # 计算指标
        df_ind = compute_indicators(df)
        self.cache[cache_key] = df_ind
        return df_ind

    def load_all_data(self, coins: List[str], timeframe: str = "1h",
                      bear_only: bool = True) -> Dict[str, pd.DataFrame]:
        """加载所有币种数据"""
        data = {}
        logger.info(f"[DataCollector] Loading {len(coins)} coins from {self.data_dir}")
        for coin in coins:
            df = self.load_coin_data(coin, timeframe, bear_only)
            if df is not None:
                data[coin] = df
                logger.info(f"  {coin}: {len(df)} rows with indicators")
        logger.info(f"[DataCollector] Loaded {len(data)} coins successfully")
        return data

    def get_market_regime(self, data: Dict[str, pd.DataFrame]) -> str:
        """分析市场状态 (基于BTC走势, 使用RegimeClassifier)"""
        if "BTC" not in data:
            return "unknown"

        btc_df = data["BTC"]
        if len(btc_df) < 50:
            return "unknown"

        # 使用RegimeClassifier进行ADX+DMI+ATR分析
        try:
            # 构建分类器需要的DataFrame (high, low, close)
            regime_df = pd.DataFrame({
                'high': btc_df['high'].values,
                'low': btc_df['low'].values,
                'close': btc_df['close'].values,
            })
            regime, confidence, metrics = self._regime_classifier.classify(regime_df)
            return regime.value  # 'bull' / 'bear' / 'sideways'
        except Exception as e:
            logger.warning(f"[DataCollector] RegimeClassifier failed: {e}")
            return "unknown"


# ===== 假设生成器 =====
class HypothesisGenerator:
    """智能假设生成器 - LLM推理 + 历史趋势 + 定向优化"""

    def __init__(self, search_space: Dict = SEARCH_SPACE):
        self.search_space = search_space
        self._llm_available = None  # lazy check

    def _check_ollama(self) -> bool:
        """检查Ollama是否可用"""
        if self._llm_available is not None:
            return self._llm_available
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            self._llm_available = r.status_code == 200
            return self._llm_available
        except Exception:
            self._llm_available = False
            return False

    def _ollama_invoke(self, prompt: str, model: str = "gemma4-2b-heretic:latest",
                       max_tokens: int = 512) -> Optional[str]:
        """调用Ollama模型"""
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.7},
                },
                timeout=60,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            logger.warning("Ollama调用失败: %s", e)
            return None

    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """从模型响应中提取JSON（处理thinking模型输出格式）"""
        # 尝试先找```json ... ```块
        json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试找{...}块
        brace_block = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_block:
            try:
                return json.loads(brace_block.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def generate_llm_hypothesis(self, base_config: StrategyConfig,
                                 last_results: List[ExperimentRecord] = None,
                                 weak_metrics: Dict[str, float] = None,
                                 regime: str = "unknown") -> Optional[Hypothesis]:
        """
        LLM驱动的策略假设生成。
        
        使用本地Gemma4模型分析当前策略表现 + 市场状态，
        推理出有逻辑依据的参数调整建议，而非随机扰动。
        """
        if not self._check_ollama():
            logger.info("Ollama不可用，回退到参数微调")
            return None

        config_dict = base_config.to_dict()

        # 构建提示词
        prompt_parts = [
            "你是一位加密货币量化交易策略专家。分析当前策略表现并提出改进方案。",
            "",
            "## 当前策略参数",
        ]
        for k, v in sorted(config_dict.items()):
            space = self.search_space.get(k)
            if space:
                prompt_parts.append(f"  {k} = {v} (范围: {space['min']}~{space['max']})")

        if last_results:
            prompt_parts.extend(["\n## 最近实验结果"])
            for r in last_results[-5:]:
                if r.status == "keep":
                    prompt_parts.append(f"  ✅ KEEP: sharpe={r.sharpe:.2f} return={r.total_return:.1f}% "
                                        f"win_rate={r.win_rate:.1%} dd={r.max_drawdown:.1f}%")
                else:
                    prompt_parts.append(f"  ❌ DISCARD: sharpe={r.sharpe:.2f} return={r.total_return:.1f}%")

        if weak_metrics:
            prompt_parts.extend(["\n## 弱项指标"])
            for k, v in weak_metrics.items():
                prompt_parts.append(f"  {k} = {v}")

        prompt_parts.extend([
            f"\n## 市场状态: {regime}",
            "",
            "## 任务",
            "分析上述信息，找出策略弱点，最多修改3个参数。",
            "返回JSON格式：",
            """{
                "reasoning": "你的分析逻辑...",
                "mutations": {"参数名": 新数值},
                "confidence": 0.0-1.0,
                "estimated_impact": "你的预期效果"
            }""",
            "只返回JSON，不要其他内容。"
        ])
        prompt = "\n".join(prompt_parts)

        response = self._ollama_invoke(prompt)
        if not response:
            return None

        result = self._extract_json_from_response(response)
        if not result:
            logger.warning("LLM响应解析失败: %s...", response[:100])
            return None

        mutations = result.get("mutations", {})
        if not mutations:
            return None

        # 验证并裁剪突变
        valid_mutations = {}
        for k, v in mutations.items():
            space = self.search_space.get(k)
            if space:
                if space["type"] == "float":
                    v = round(max(space["min"], min(space["max"], float(v))), 2)
                elif space["type"] == "int":
                    v = int(max(space["min"], min(space["max"], int(v))))
                valid_mutations[k] = v

        if not valid_mutations:
            return None

        # 应用突变
        new_config = config_dict.copy()
        for k, v in valid_mutations.items():
            new_config[k] = v

        reasoning = result.get("reasoning", "LLM分析")[:120]
        confidence = max(0.3, min(0.95, result.get("confidence", 0.5)))
        hyp_id = f"hyp_{uuid.uuid4().hex[:12]}"

        logger.info("LLM假设: %s | 突变: %s | 置信度: %.2f", reasoning, valid_mutations, confidence)
        return Hypothesis(
            id=hyp_id,
            description=f"llm_analysis({reasoning})",
            mutations=valid_mutations,
            confidence=confidence,
            source="llm"
        )

    def generate_random_mutation(self, base_config: StrategyConfig,
                                  n_mutations: int = None) -> Hypothesis:
        """随机变异假设"""
        if n_mutations is None:
            n_mutations = random.randint(1, min(3, len(self.search_space)))

        keys = list(self.search_space.keys())
        selected_keys = random.sample(keys, n_mutations)
        config_dict = base_config.to_dict()
        mutations = {}

        for key in selected_keys:
            space = self.search_space[key]
            current = config_dict.get(key)
            if current is None:
                continue

            if space["type"] == "float":
                range_size = space["max"] - space["min"]
                step = range_size * random.uniform(0.15, 0.25)
                direction = random.choice([-1, 1])
                new_val = current + direction * step
                new_val = max(space["min"], min(space["max"], new_val))
                mutations[key] = round(new_val, 2)
            elif space["type"] == "int":
                range_size = space["max"] - space["min"]
                step = int(range_size * random.uniform(0.15, 0.25))
                direction = random.choice([-1, 1])
                new_val = current + direction * step
                new_val = max(space["min"], min(space["max"], new_val))
                mutations[key] = int(new_val)

        # 构建变异后的配置
        for k, v in mutations.items():
            config_dict[k] = v

        # 自动启用ATR过滤器
        if "atr_percentile_max" in mutations or "atr_percentile_period" in mutations:
            config_dict["atr_filter_enabled"] = True

        desc = f"random_mutate({', '.join(f'{k}={v}' for k, v in mutations.items())})"
        return Hypothesis(
            id=f"hyp_{uuid.uuid4().hex[:12]}",
            description=desc,
            mutations=mutations,
            confidence=0.5,
            source="random"
        )

    def generate_trend_mutation(self, base_config: StrategyConfig,
                                 last_results: List[ExperimentRecord] = None) -> Hypothesis:
        """趋势外推假设 - 基于最近成功的参数变化方向"""
        if last_results is None or len(last_results) < 3:
            return self.generate_random_mutation(base_config)

        # 分析最近keep实验的参数变化趋势
        keep_results = [r for r in last_results[-10:] if r.status == "keep"]
        if len(keep_results) < 2:
            return self.generate_random_mutation(base_config)

        config_dict = base_config.to_dict()
        mutations = {}
        trend_keys = ["rsi_oversold", "rsi_overbought", "adx_threshold", "sl_atr_mult", "tp2_atr_mult"]

        for key in trend_keys:
            if key not in self.search_space:
                continue

            # 收集该参数的变化趋势
            changes = []
            for r in keep_results:
                if r.config and key in r.config:
                    changes.append(r.config[key])

            if len(changes) >= 2:
                # 趋势外推
                space = self.search_space[key]
                current = config_dict.get(key)

                # 计算平均变化
                avg_change = (changes[-1] - changes[0]) / len(changes) if len(changes) > 1 else 0

                if space["type"] == "float":
                    new_val = current + avg_change * 0.5  # 减弱趋势以避免过拟合
                    new_val = max(space["min"], min(space["max"], new_val))
                    mutations[key] = round(new_val, 2)
                elif space["type"] == "int":
                    new_val = current + int(avg_change * 0.5)
                    new_val = max(space["min"], min(space["max"], new_val))
                    mutations[key] = int(new_val)

        # 如果没有足够的趋势数据，回退到随机
        if not mutations:
            return self.generate_random_mutation(base_config)

        # 应用变异
        for k, v in mutations.items():
            config_dict[k] = v

        desc = f"trend_extrapolate({', '.join(f'{k}={v}' for k, v in mutations.items())})"
        return Hypothesis(
            id=f"hyp_{uuid.uuid4().hex[:12]}",
            description=desc,
            mutations=mutations,
            confidence=0.7,
            source="trend"
        )

    def generate_focused_mutation(self, base_config: StrategyConfig,
                                   weak_metrics: Dict[str, float] = None) -> Hypothesis:
        """聚焦变异假设 - 针对弱指标定向优化"""
        config_dict = base_config.to_dict()
        mutations = {}

        # 根据弱指标选择聚焦参数
        if weak_metrics:
            if weak_metrics.get("win_rate", 0.5) < 0.45:
                # 胜率低: 调整RSI阈值
                mutations["rsi_oversold"] = random.uniform(30, 35)
                mutations["rsi_overbought"] = random.uniform(65, 70)
            if weak_metrics.get("avg_dd", 0) > 15:
                # 回撤大: 收紧止损
                mutations["sl_atr_mult"] = random.uniform(1.0, 1.5)
            if weak_metrics.get("wlr", 1) < 1.2:
                # 盈亏比低: 调整止盈
                mutations["tp2_atr_mult"] = random.uniform(4.0, 6.0)

        if not mutations:
            return self.generate_random_mutation(base_config)

        for k, v in mutations.items():
            if k in self.search_space:
                space = self.search_space[k]
                if space["type"] == "float":
                    mutations[k] = round(max(space["min"], min(space["max"], v)), 2)
                elif space["type"] == "int":
                    mutations[k] = int(max(space["min"], min(space["max"], v)))

        for k, v in mutations.items():
            config_dict[k] = v

        desc = f"focused_optimize({', '.join(f'{k}={v}' for k, v in mutations.items())})"
        return Hypothesis(
            id=f"hyp_{uuid.uuid4().hex[:12]}",
            description=desc,
            mutations=mutations,
            confidence=0.6,
            source="analysis"
        )

    def generate(self, base_config: StrategyConfig,
                 last_results: List[ExperimentRecord] = None,
                 weak_metrics: Dict[str, float] = None,
                 regime: str = "unknown",
                 mode: str = "mixed") -> Hypothesis:
        """
        生成假设

        mode:
        - llm: LLM推理分析（需要Ollama）
        - random: 完全随机
        - trend: 趋势外推
        - focused: 聚焦优化
        - mixed: 混合模式 (40% trend + 30% focused + 20% LLM + 10% random)
        """
        if mode == "llm":
            llm_hyp = self.generate_llm_hypothesis(base_config, last_results, weak_metrics, regime)
            if llm_hyp:
                return llm_hyp
            logger.info("LLM假设生成失败，回退到trend")
            return self.generate_trend_mutation(base_config, last_results)
        elif mode == "random":
            return self.generate_random_mutation(base_config)
        elif mode == "trend":
            return self.generate_trend_mutation(base_config, last_results)
        elif mode == "focused":
            return self.generate_focused_mutation(base_config, weak_metrics)
        else:  # mixed
            r = random.random()
            if r < 0.4:
                return self.generate_trend_mutation(base_config, last_results)
            elif r < 0.7:
                return self.generate_focused_mutation(base_config, weak_metrics)
            elif r < 0.9:
                # LLM推理(20%)
                llm_hyp = self.generate_llm_hypothesis(base_config, last_results, weak_metrics, regime)
                if llm_hyp:
                    return llm_hyp
                # LLM失败时回退到trend
                return self.generate_trend_mutation(base_config, last_results)
            else:
                return self.generate_random_mutation(base_config)


# ===== 回测验证器 =====
class BacktestValidator:
    """回测验证器 - Walk-Forward多窗口验证"""

    def __init__(self, n_windows: int = N_WINDOWS, train_ratio: float = TRAIN_RATIO):
        self.n_windows = n_windows
        self.train_ratio = train_ratio

    def validate(self, config: StrategyConfig, data: Dict[str, pd.DataFrame],
                 coins: List[str], wf_mode: str = "expanding") -> Tuple[Dict[str, BacktestResult], Dict[str, float]]:
        """
        在多个币种上运行Walk-Forward验证
        返回: (结果字典, 汇总指标)
        """
        results = {}
        for coin in coins:
            if coin not in data:
                continue
            df = data[coin]
            result = run_walkforward(
                df=df,
                config=config,
                n_windows=self.n_windows,
                train_ratio=self.train_ratio,
                initial_capital=INITIAL_CAPITAL,
                wf_mode=wf_mode,
            )
            results[coin] = result

        # 汇总
        metrics = self.aggregate_results(results)
        return results, metrics

    def aggregate_results(self, results: Dict[str, BacktestResult]) -> Dict[str, float]:
        """汇总多币种结果"""
        if not results:
            return {
                "avg_return": 0.0,
                "avg_sharpe": 0.0,
                "avg_dd": 100.0,
                "avg_win_rate": 0.0,
                "avg_wlr": 0.0,
                "total_trades": 0,
                "n_coins": 0,
            }

        valid_results = [r for r in results.values() if r.total_trades > 0]
        if not valid_results:
            return {
                "avg_return": 0.0,
                "avg_sharpe": 0.0,
                "avg_dd": 100.0,
                "avg_win_rate": 0.0,
                "avg_wlr": 0.0,
                "total_trades": sum(r.total_trades for r in results.values()),
                "n_coins": 0,
            }

        return {
            "avg_sharpe": np.mean([r.sharpe_ratio for r in valid_results]),
            "avg_return": np.mean([r.total_return for r in valid_results]),
            "avg_dd": np.mean([r.max_drawdown for r in valid_results]),
            "avg_wlr": np.mean([r.win_loss_ratio for r in valid_results]),
            "avg_win_rate": np.mean([r.win_rate for r in valid_results]),
            "total_trades": sum(r.total_trades for r in results.values()),
            "n_coins": len(valid_results),
        }


# ===== 反思改进器 =====
class ReflectionImprover:
    """反思改进器 - 分析结果并调整策略"""

    def __init__(self):
        self.ic_weights_path = WORKSPACE / ".hermes" / "kronos_ic_weights.json"

    def analyze_results(self, metrics: Dict[str, float],
                        last_n_records: List[ExperimentRecord] = None) -> Dict[str, float]:
        """分析回测结果，返回弱指标"""
        weak_metrics = {}

        if metrics.get("avg_win_rate", 0) < 0.40:
            weak_metrics["win_rate"] = metrics["avg_win_rate"]
        if metrics.get("avg_dd", 0) > 20:
            weak_metrics["avg_dd"] = metrics["avg_dd"]
        if metrics.get("avg_wlr", 0) < 1.0:
            weak_metrics["wlr"] = metrics["avg_wlr"]

        return weak_metrics

    def calculate_quality_score(self, metrics: Dict[str, float],
                                  last_n_records: List[ExperimentRecord] = None) -> float:
        """计算策略质量评分 (0-1)"""
        score = 0.5

        # Sharpe贡献 (最高+0.3)
        sharpe = metrics.get("avg_sharpe", 0)
        if not np.isnan(sharpe):
            score += min(0.3, sharpe * 0.1)

        # 胜率贡献 (最高+0.15)
        wr = metrics.get("avg_win_rate", 0)
        if wr > 0.50:  # 50% as fraction
            score += min(0.15, (wr - 0.50) * 0.3)  # 30% improvement → +0.15 max bonus

        # 回撤惩罚 (-0.2以内)
        dd = metrics.get("avg_dd", 0)
        if dd > 10:
            score -= min(0.2, (dd - 10) / 100)

        # 最近趋势惩罚
        if last_n_records:
            recent_returns = [r.val_return for r in last_n_records[-5:] if hasattr(r, 'val_return')]
            if recent_returns and sum(recent_returns) < -10:
                score -= 0.1

        return max(0.0, min(1.0, score))

    def update_ic_weights(self, metrics: Dict[str, float],
                          last_n_records: List[ExperimentRecord] = None):
        """更新IC权重 - 基于策略质量反馈"""
        try:
            if not self.ic_weights_path.exists():
                logger.info("[ReflectionImprover] IC weights file not found, skipping")
                return

            with open(self.ic_weights_path) as f:
                ic_data = json.load(f)

            weights = ic_data.get('weights', {})

            # 读取最近实验结果
            results_path = RESULTS_DIR / "results.tsv"
            if not results_path.exists():
                return

            lines = results_path.read_text().strip().split('\n')
            if len(lines) < 3:
                return

            # 解析TSV，使用header映射列位置，避免硬编码索引
            header = lines[1] if len(lines) > 1 else ""
            col_map = {}
            if header:
                for idx, col_name in enumerate(header.split('\t')):
                    col_map[col_name.strip()] = idx
            # 查找return和sharpe列
            ret_col = col_map.get('return') or col_map.get('total_return') or col_map.get('avg_return')
            sh_col = col_map.get('sharpe') or col_map.get('sharpe_ratio')
            if ret_col is None or sh_col is None:
                logger.debug(f"TSV header缺少return/sharpe列，跳过IC权重更新")
                return
            sharpes, returns = [], []
            for line in lines[2:][-200:]:  # skip 2 header rows, use last 200 data rows
                cols = line.split('\t')
                if len(cols) <= max(ret_col, sh_col):
                    continue
                try:
                    sharpes.append(float(cols[sh_col]))
                    returns.append(float(cols[ret_col]))
                except Exception:
                    continue

            if not sharpes:
                return

            avg_sharpe = sum(sharpes) / len(sharpes)
            avg_return = sum(returns) / len(returns)
            keep_rate = sum(1 for s in sharpes if s > 0) / len(sharpes)
            keep_returns = [returns[i] for i, s in enumerate(sharpes) if s > 0]
            keep_avg_return = sum(keep_returns) / len(keep_returns) if keep_returns else 0.0

            # 策略质量反馈
            if avg_return < -3:
                decay = 0.5
            elif avg_return < -1:
                decay = 0.3
            elif avg_return < 0:
                decay = 0.15
            elif keep_rate < 0.3:
                decay = 0.2
            elif len(keep_returns) >= 3 and keep_avg_return > 5:
                boost = min(0.25, keep_avg_return / 100.0)
                decay = -boost
            else:
                decay = 0.0

            coin_factors = ['RSI', 'ADX', 'Bollinger', 'Vol', 'MACD']

            if decay > 0:
                # 负反馈
                for factor in coin_factors:
                    if factor in weights and weights[factor] > 0:
                        weights[factor] = max(0.01, weights[factor] * (1 - decay))
            elif decay < 0:
                # 正反馈
                boost = abs(decay)
                for factor in coin_factors:
                    if factor in weights and weights[factor] > 0:
                        weights[factor] = min(weights[factor] * (1 + boost * 2), 0.25)

            # 归一化
            total = sum(weights.values())
            if total > 0:
                for k in weights:
                    weights[k] /= total

            ic_data['weights'] = weights
            ic_data['last_update'] = datetime.now().isoformat()
            ic_data['strategy_quality'] = {
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'keep_rate': keep_rate,
                'keep_avg_return': keep_avg_return,
                'decay_applied': decay,
                'source': 'miracle_autonomous'
            }

            with open(self.ic_weights_path, 'w') as f:
                json.dump(ic_data, f, indent=2)

            logger.info(f"[ReflectionImprover] IC weights updated: decay={decay}")

        except Exception as e:
            logger.error(f"[ReflectionImprover] IC weight update failed: {e}")


# ===== 自主研究循环 =====
class AutonomousLoop:
    """
    自主研究循环 - 完整闭环

    流程: 数据收集 → 假设生成 → 回测验证 → 反思改进
    """

    def __init__(self, coins: List[str] = None, timeframe: str = "1h",
                 data_dir: Path = None, wf_mode: str = "expanding"):
        self.coins = coins or DEFAULT_COINS
        self.timeframe = timeframe
        self.data_dir = data_dir or WORKSPACE / ".hermes" / "cron" / "output"
        self.wf_mode = wf_mode

        # 从coin_params.json加载启用状态，过滤掉被禁用的币种
        self._apply_coin_params_filter()

        # 初始化组件
        self.data_collector = DataCollector(self.data_dir)
        self.hyp_generator = HypothesisGenerator()
        self.validator = BacktestValidator()
        self.improver = ReflectionImprover()

        # 状态
        self.state = LoopState()
        self.data: Dict[str, pd.DataFrame] = {}
        self.last_results: List[ExperimentRecord] = []

        # 配置
        ensure_dirs()
        init_results_tsv()
    
    def _apply_coin_params_filter(self) -> None:
        """从coin_params.json加载enabled状态，过滤掉被禁用的币种"""
        try:
            cp_path = Path(__file__).parent / "coin_params.json"
            if not cp_path.exists():
                return
            with open(cp_path) as f:
                data = json.load(f)
            coins_list = data.get("coins", [])
            # 收集enabled=true的币种
            enabled_symbols = {c["symbol"].upper() for c in coins_list if c.get("enabled", True)}
            if not enabled_symbols:
                return
            # 记录被过滤的币种
            filtered = [c for c in self.coins if c.upper() not in enabled_symbols]
            self.coins = [c for c in self.coins if c.upper() in enabled_symbols]
            if filtered:
                logger.info(f"[AutonomousLoop] coin_params过滤: {filtered} 被禁用(enabled=false), 已从交易列表移除")
            if not self.coins:
                logger.warning("[AutonomousLoop] 所有币种均被禁用! 使用DEFAULT_COINS")
                self.coins = DEFAULT_COINS
        except Exception as e:
            logger.warning(f"[AutonomousLoop] 加载coin_params.json失败: {e}")
    
    def _load_data(self) -> bool:
        """阶段1: 数据收集"""
        logger.info("[Stage 1/4] DATA COLLECTION")
        self.state.stage = LoopStage.DATA_COLLECTION

        self.data = self.data_collector.load_all_data(
            self.coins, self.timeframe, bear_only=False
        )

        if not self.data:
            logger.error("[AutonomousLoop] No data loaded!")
            return False

        self.state.market_regime = self.data_collector.get_market_regime(self.data)
        logger.info(f"[AutonomousLoop] Market regime: {self.state.market_regime}")
        self.state.data_loaded = True
        return True

    def _generate_hypothesis(self, baseline_config: StrategyConfig) -> Hypothesis:
        """阶段2: 假设生成"""
        self.state.stage = LoopStage.HYPOTHESIS_GENERATION
        logger.info("[Stage 2/4] HYPOTHESIS GENERATION")

        # 分析弱指标
        weak_metrics = self.improver.analyze_results(
            {}, self.last_results
        )

        # 选择生成模式
        mode = "mixed"
        if self.state.iteration < 3:
            mode = "random"  # 前期随机探索
        elif weak_metrics:
            mode = "focused"  # 有弱指标时聚焦优化

        hypothesis = self.hyp_generator.generate(
            baseline_config, self.last_results, weak_metrics,
            regime=self.state.market_regime, mode=mode
        )

        logger.info(f"[AutonomousLoop] Generated: {hypothesis.description} (confidence={hypothesis.confidence})")
        return hypothesis

    def _validate_hypothesis(self, hypothesis: Hypothesis,
                              baseline_config: StrategyConfig) -> Tuple[Dict[str, Any], bool]:
        """阶段3: 回测验证"""
        self.state.stage = LoopStage.BACKTEST_VALIDATION
        logger.info("[Stage 3/4] BACKTEST VALIDATION")

        # 应用变异到配置
        config_dict = baseline_config.to_dict()
        for k, v in hypothesis.mutations.items():
            config_dict[k] = v
        test_config = StrategyConfig.from_dict(config_dict)

        # 运行验证
        try:
            results, metrics = self.validator.validate(
                test_config, self.data, self.coins, self.wf_mode
            )
            return {
                "config": test_config,
                "results": results,
                "metrics": metrics,
                "hypothesis": hypothesis,
            }, True
        except Exception as e:
            logger.error(f"[AutonomousLoop] Backtest failed: {e}")
            traceback.print_exc()
            return {}, False

    def _reflect_and_improve(self, validation_result: Dict[str, Any]) -> bool:
        """阶段4: 反思改进"""
        self.state.stage = LoopStage.REFLECTION_IMPROVEMENT
        logger.info("[Stage 4/4] REFLECTION & IMPROVEMENT")

        if not validation_result:
            self.state.crashed_count += 1
            return False

        metrics = validation_result.get("metrics", {})
        test_config = validation_result.get("config")
        hypothesis = validation_result.get("hypothesis")

        current_sharpe = metrics.get("avg_sharpe", 0)
        current_win_rate = metrics.get("avg_win_rate", 0)
        current_max_dd = metrics.get("avg_dd", 0)
        improvement = current_sharpe - self.state.best_sharpe

        # 判断是否keep — 三重条件防过拟合
        # 1. Sharpe必须明显改善(>0.1)
        # 2. 胜率 > 40%（排除随机水平）
        # 3. 最大回撤 < 20%（防止高风险策略）
        improvement_threshold = 0.1
        passes_sharpe = improvement > improvement_threshold
        passes_win_rate = current_win_rate > 0.40
        passes_drawdown = current_max_dd < 20.0

        if passes_sharpe and passes_win_rate and passes_drawdown:
            status = "keep"
            self.state.kept_count += 1
            if current_sharpe > self.state.best_sharpe:
                self.state.best_sharpe = current_sharpe
                self.state.best_config = test_config
        else:
            status = "discard"
            self.state.discarded_count += 1

        # 计算质量评分
        quality_score = self.improver.calculate_quality_score(metrics, self.last_results)
        self.state.strategy_quality_score = quality_score

        # 记录实验
        exp_id = f"auto_{datetime.now().strftime('%m%d_%H%M%S')}_{self.state.iteration}"
        record = ExperimentRecord(
            commit=get_git_commit(),
            experiment_id=exp_id,
            coins=",".join(self.coins),
            val_return=metrics.get("avg_return", 0),
            sharpe=current_sharpe,
            max_dd=metrics.get("avg_dd", 0),
            win_rate=metrics.get("avg_win_rate", 0),
            wlr=metrics.get("avg_wlr", 0),
            total_trades=int(metrics.get("total_trades", 0)),
            status=status,
            description=hypothesis.description if hypothesis else "unknown",
            timestamp=datetime.now().isoformat(),
            config=test_config.to_dict() if test_config else {},
            details={},
        )

        write_experiment(record)
        self.last_results.append(record)

        # 更新IC权重
        self.improver.update_ic_weights(metrics, self.last_results)

        logger.info(
            f"[AutonomousLoop] Result: Sharpe={current_sharpe:.3f} (Δ{improvement:+.3f}) "
            f"[{status.upper()}] Quality={quality_score:.2f}"
        )

        return status == "keep"

    def run_iteration(self, baseline_config: StrategyConfig) -> bool:
        """运行单次迭代"""
        self.state.iteration += 1
        self.state.total_experiments += 1

        logger.info(f"\n{'='*60}")
        logger.info(f"[ITERATION {self.state.iteration}] Regime: {self.state.market_regime}")
        logger.info(f"{'='*60}")

        # 生成假设
        hypothesis = self._generate_hypothesis(baseline_config)

        # 验证假设
        validation_result, success = self._validate_hypothesis(hypothesis, baseline_config)
        if not success:
            return False

        # 反思改进
        kept = self._reflect_and_improve(validation_result)

        # 如果keep了，更新baseline
        if kept:
            return True
        return False

    def run(self, n_iterations: int = 50, max_time_minutes: int = 480):
        """运行自主研究循环"""
        logger.info(f"\n{'#'*70}")
        logger.info("# MIRACLE AUTONOMOUS 2.0 - 自主研究循环启动")
        logger.info(f"# 币种: {self.coins}")
        logger.info(f"# 时间周期: {self.timeframe}")
        logger.info(f"# 最大迭代: {n_iterations}")
        logger.info(f"# 最大运行时间: {max_time_minutes}分钟")
        logger.info(f"{'#'*70}\n")

        start_time = time.time()

        # 阶段1: 数据收集
        if not self._load_data():
            logger.error("[AutonomousLoop] 数据收集失败，退出")
            return None

        # 获取baseline
        if self.state.best_config is None:
            best_result = get_best_result()
            if best_result and best_result.config:
                self.state.best_config = StrategyConfig.from_dict(best_result.config)
                self.state.best_sharpe = best_result.sharpe
                logger.info(f"[AutonomousLoop] 从历史最佳恢复: Sharpe={self.state.best_sharpe:.3f}")
            else:
                self.state.best_config = StrategyConfig.baseline()
                logger.info("[AutonomousLoop] 使用默认baseline配置")

        baseline_config = self.state.best_config

        # 迭代循环
        for i in range(n_iterations):
            # 时间限制检查
            elapsed_min = (time.time() - start_time) / 60
            if elapsed_min > max_time_minutes:
                logger.info(f"[AutonomousLoop] 时间限制到达 ({max_time_minutes}min)")
                break

            # ε-greedy探索: 10%概率随机探索以避免局部最优
            if random.random() < 0.1:
                hyp = self.hyp_generator.generate_random_mutation(baseline_config)
                mutated_dict = baseline_config.to_dict()
                for k, v in hyp.mutations.items():
                    mutated_dict[k] = v
                baseline_config = StrategyConfig.from_dict(mutated_dict)
                logger.info(f"ε-greedy: 从随机基线探索 ({hyp.description})")

            # 运行迭代
            kept = self.run_iteration(baseline_config)

            # 如果keep了，更新baseline用于下次变异
            if kept and self.state.best_config:
                baseline_config = self.state.best_config

            # 定期保存状态
            if (i + 1) % 10 == 0:
                self._save_state()

        # 最终总结
        total_time = (time.time() - start_time) / 60
        logger.info(f"\n{'='*60}")
        logger.info("[AUTONOMOUS LOOP COMPLETE]")
        logger.info(f"总迭代: {self.state.iteration}")
        logger.info(f"运行时间: {total_time:.1f}分钟")
        logger.info(f"Keep: {self.state.kept_count} | Discard: {self.state.discarded_count} | Crash: {self.state.crashed_count}")
        logger.info(f"最佳Sharpe: {self.state.best_sharpe:.3f}")
        logger.info(f"策略质量评分: {self.state.strategy_quality_score:.2f}")
        logger.info(f"{'='*60}")

        # 保存最终状态
        self._save_state()

        return self.state.best_config

    def _save_state(self):
        """保存循环状态"""
        state_file = RESULTS_DIR / "autonomous_loop_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)


# ===== 主程序入口 =====
def main():
    parser = argparse.ArgumentParser(description="Miracle Autonomous 2.0 - 自主研究循环")
    parser.add_argument("--experiments", type=int, default=50, help="最大实验次数")
    parser.add_argument("--coins", type=str, default="BTC,ETH,ADA,DOGE,AVAX,DOT",
                        help="交易的币种列表")
    parser.add_argument("--timeframe", type=str, default="1h", help="K线时间周期")
    parser.add_argument("--max-time", type=int, default=480, help="最大运行时间(分钟)")
    parser.add_argument("--wf-mode", type=str, default="expanding",
                        choices=["expanding", "rolling", "rolling_recent"],
                        help="Walk-Forward模式")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--resume", action="store_true", help="从上次状态恢复")

    args = parser.parse_args()

    coins = args.coins.split(",")
    data_dir = Path(args.data_dir) if args.data_dir else None

    # 创建并运行自主循环
    loop = AutonomousLoop(
        coins=coins,
        timeframe=args.timeframe,
        data_dir=data_dir,
        wf_mode=args.wf_mode,
    )

    if args.resume:
        state_file = RESULTS_DIR / "autonomous_loop_state.json"
        if state_file.exists():
            with open(state_file) as f:
                saved_state = json.load(f)
            # Use from_dict for complete restoration
            restored = LoopState.from_dict(saved_state)
            loop.state.iteration = restored.iteration
            loop.state.best_sharpe = restored.best_sharpe
            loop.state.kept_count = restored.kept_count
            loop.state.discarded_count = restored.discarded_count
            loop.state.crashed_count = restored.crashed_count
            loop.state.market_regime = restored.market_regime
            loop.state.strategy_quality_score = restored.strategy_quality_score
            loop.state.stage = restored.stage
            loop.state.data_loaded = restored.data_loaded
            loop.state.best_config = restored.best_config
            logger.info(f"[main] 恢复状态: iteration={loop.state.iteration}, best_sharpe={loop.state.best_sharpe}")

    best_config = loop.run(
        n_iterations=args.experiments,
        max_time_minutes=args.max_time,
    )

    if best_config:
        # 保存最佳配置
        config_path = RESULTS_DIR / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(best_config.to_dict(), f, indent=2)
        logger.info(f"[main] 最佳配置已保存到 {config_path}")

        # 打印最终配置
        logger.info("\n[main] 最佳策略配置:")
        for k, v in best_config.to_dict().items():
            logger.info(f"  {k}: {v}")
    else:
        logger.warning("[main] 未能找到有效配置")


if __name__ == "__main__":
    main()
