"""
Autoresearch Loop - 核心研究循环

参考 Karpathy Autoresearch 的 Keep/Discard 循环，
应用于量化交易策略参数优化。

核心流程:
1. 从baseline或历史最优参数出发
2. 随机扰动1-3个关键参数
3. Walk-Forward验证 (8窗口)
4. 按Sharpe排序 → 改善则keep，变差则discard
5. 记录到results.tsv，循环
"""
import os
import sys
import json
import time
import random
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from autoresearch.strategy_config import StrategyConfig, BacktestResult
from autoresearch.backtest_engine import run_walkforward, run_single_backtest
from autoresearch.data_loader import load_timeframe_data, compute_indicators, COINS
from autoresearch.experiment_logger import (
    ExperimentRecord, write_experiment, get_best_result,
    get_git_commit, generate_experiment_id, init_results_tsv,
    print_summary, ensure_dirs, RESULTS_DIR
)


# ===== 参数搜索空间定义 =====
SEARCH_SPACE = {
    "rsi_oversold": {"min": 25.0, "max": 40.0, "type": "float"},
    "rsi_overbought": {"min": 60.0, "max": 75.0, "type": "float"},
    "adx_threshold": {"min": 10.0, "max": 30.0, "type": "float"},
    "sl_atr_mult": {"min": 0.8, "max": 3.0, "type": "float"},
    "tp1_atr_mult": {"min": 1.5, "max": 5.0, "type": "float"},
    "tp2_atr_mult": {"min": 3.0, "max": 10.0, "type": "float"},
    "vol_ratio_threshold": {"min": 0.8, "max": 2.0, "type": "float"},
    "position_size_pct": {"min": 0.05, "max": 0.20, "type": "float"},
    "bb_position_oversold": {"min": 10.0, "max": 35.0, "type": "float"},
    "bb_position_overbought": {"min": 65.0, "max": 90.0, "type": "float"},
    # New factors
    "atr_percentile_max": {"min": 40.0, "max": 90.0, "type": "float"},  # Lower = stricter vol filter
    "atr_percentile_period": {"min": 20, "max": 100, "type": "int"},  # ATR percentile lookback period
}

# 默认搜索的币种
DEFAULT_COINS = ["BTC", "ETH", "ADA", "DOGE", "AVAX", "DOT", "BCH", "ETC", "ALGO", "CRO", "BNT", "CVC", "GAS"]

# Walk-Forward参数
N_WINDOWS = 8
TRAIN_RATIO = 0.7
INITIAL_CAPITAL = 100000.0


def load_all_data(timeframe: str = "1h", bear_only: bool = True, coins=None, data_dir: Path = None) -> Dict[str, pd.DataFrame]:
    """加载所有币种数据并计算指标"""
    import pandas as pd
    data = {}
    coins = coins or DEFAULT_COINS
    for coin in coins:
        print(f"  Loading {coin}...")
        df = load_timeframe_data(coin, timeframe, data_dir=data_dir)
        if df is not None and len(df) > 500:
            # 过滤到熊市时间段（动态结束日期）
            if bear_only:
                df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
                # 动态结束日期：使用数据中的最新日期
                dynamic_end = df["datetime_utc"].max().strftime("%Y-%m-%d")
                # 动态熊市窗口：使用最近18个月数据（代替硬编码日期）
                # 防止时间推移后硬编码日期逐渐失效
                dynamic_start = (df["datetime_utc"].max() - pd.Timedelta(days=548)).strftime("%Y-%m-%d")
                mask = (df["datetime_utc"] >= dynamic_start) & (df["datetime_utc"] <= dynamic_end)
                df = df[mask].copy()
                if len(df) < 200:
                    print(f"    {coin}: SKIP (bear period has only {len(df)} rows)")
                    continue

            df_ind = compute_indicators(df)
            data[coin] = df_ind
            print(f"    {coin}: {len(df_ind)} rows with indicators")
        else:
            print(f"    {coin}: SKIP (insufficient data)")
    return data


def mutate_config(base_config: StrategyConfig, seed: int = None) -> StrategyConfig:
    """
    从base_config随机扰动1-3个参数
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 随机选择1-3个参数扰动
    keys = list(SEARCH_SPACE.keys())
    n_mutations = random.randint(1, min(3, len(keys)))
    selected_keys = random.sample(keys, n_mutations)

    config_dict = base_config.to_dict()

    for key in selected_keys:
        space = SEARCH_SPACE[key]
        current = config_dict.get(key)
        if current is None:
            continue

        # 随机步长
        if space["type"] == "float":
            range_size = space["max"] - space["min"]
            # ±15-25%扰动
            step = range_size * random.uniform(0.15, 0.25)
            direction = random.choice([-1, 1])
            new_val = current + direction * step
            new_val = max(space["min"], min(space["max"], new_val))
            config_dict[key] = round(new_val, 2)
        elif space["type"] == "int":
            range_size = space["max"] - space["min"]
            step = int(range_size * random.uniform(0.15, 0.25))
            direction = random.choice([-1, 1])
            new_val = current + direction * step
            new_val = max(space["min"], min(space["max"], new_val))
            config_dict[key] = int(new_val)

    # 自动启用被变异参数对应的过滤器
    if "atr_percentile_max" in selected_keys or "atr_percentile_period" in selected_keys:
        config_dict["atr_filter_enabled"] = True

    return StrategyConfig.from_dict(config_dict)


def run_experiment(
    config: StrategyConfig,
    data: Dict[str, pd.DataFrame],
    coins: List[str],
    n_windows: int = N_WINDOWS,
    wf_mode: str = "expanding",
) -> Dict[str, Any]:
    """
    在多个币种上运行Walk-Forward实验
    返回: {coin: BacktestResult, ...}
    """
    results = {}
    for coin in coins:
        if coin not in data:
            continue
        df = data[coin]
        result = run_walkforward(
            df=df,
            config=config,
            n_windows=n_windows,
            train_ratio=TRAIN_RATIO,
            initial_capital=INITIAL_CAPITAL,
            wf_mode=wf_mode,
        )
        results[coin] = result
    return results


def aggregate_results(results: Dict[str, BacktestResult]) -> Dict[str, float]:
    """
    汇总多币种结果 → 返回综合指标
    """
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

    sharpes = [r.sharpe_ratio for r in results.values() if r.total_trades > 0]
    returns = [r.total_return for r in results.values() if r.total_trades > 0]
    dds = [r.max_drawdown for r in results.values() if r.total_trades > 0]
    wlr = [r.win_loss_ratio for r in results.values() if r.total_trades > 0]
    wr = [r.win_rate for r in results.values() if r.total_trades > 0]
    trades = [r.total_trades for r in results.values()]

    return {
        "avg_sharpe": np.mean(sharpes) if sharpes else 0.0,
        "avg_return": np.mean(returns) if returns else 0.0,
        "avg_dd": np.mean(dds) if dds else 100.0,
        "avg_wlr": np.mean(wlr) if wlr else 0.0,
        "avg_win_rate": np.mean(wr) if wr else 0.0,
        "total_trades": sum(trades),
        "n_coins": len([r for r in results.values() if r.total_trades > 0]),
        # 原始结果
        "_results": {coin: r for coin, r in results.items()},
    }


def build_description(config: StrategyConfig, mutation: Dict[str, float]) -> str:
    """生成实验描述"""
    changed = [f"{k}={v:.2f}" for k, v in mutation.items()]
    return f"mutate({' '.join(changed)})"


def run_autoresearch(
    n_experiments: int = 50,
    coins: List[str] = None,
    timeframe: str = "1h",
    baseline_config: StrategyConfig = None,
    start_from_best: bool = True,
    improvement_threshold: float = 0.1,  # Sharpe改善 > 0.1 才keep（防过拟合）
    max_time_minutes: int = 480,  # 最多运行8小时
    wf_mode: str = "expanding",  # walkforward模式
    bear_only: bool = True,  # 只用熊市数据
    data_dir: Path = None,  # 数据目录
    oos_ratio: float = 0.2,  # OOS验证保留比例（最后20%不参与任何实验）
):
    """
    主Autoresearch循环
    
    OOS验证: 保留最后 oos_ratio 比例的数据完全不参与实验，
    在找到最优参数后用OOS数据验证，防止过拟合。
    """
    coins = coins or DEFAULT_COINS
    ensure_dirs()
    init_results_tsv()

    print(f"\n{'='*70}")
    print(f"KRONOS AUTORESEARCH | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Coins: {coins} | Timeframe: {timeframe} | WF mode: {wf_mode} | Bear only: {bear_only}")
    print(f"Max experiments: {n_experiments} | Max time: {max_time_minutes}min")
    print(f"OOS holdout: {oos_ratio*100:.0f}% (last {oos_ratio*100:.0f}% untouched)")
    print(f"{'='*70}\n")

    # Step 1: 加载数据
    print("[1/5] Loading data...")
    start_load = time.time()
    all_data = load_all_data(timeframe, bear_only=bear_only, coins=coins, data_dir=data_dir)
    print(f"Loaded {len(all_data)} coins in {time.time()-start_load:.1f}s\n")

    if not all_data:
        print("ERROR: No data loaded. Check data paths.")
        Path("/tmp/autoresearch_last_failure").write_text(f"{datetime.now().isoformat()}: No data loaded for coins={coins}")
        return

    # OOS分割：保留最后 oos_ratio 数据作为完全没见过的验证集
    data = {}
    oos_data = {}
    for coin, df in all_data.items():
        # 确保按时间排序
        if 'datetime_utc' in df.columns:
            df = df.sort_values('datetime_utc').reset_index(drop=True)
        split_idx = int(len(df) * (1 - oos_ratio))
        if split_idx < 200:
            print(f"  {coin}: SKIP OOS (train set too small: {split_idx} rows)")
            data[coin] = df
            continue
        data[coin] = df.iloc[:split_idx].copy()
        oos_df = df.iloc[split_idx:].copy()
        if len(oos_df) >= 30:
            oos_data[coin] = oos_df
        print(f"  {coin}: {len(data[coin])} train + {len(oos_df)} OOS holdout")

    if not data:
        print("ERROR: No training data after OOS split. Aborting.")
        return

    print(f"\nOOS holdout coins: {list(oos_data.keys())}\n")

    # Step 2: 建立baseline
    if baseline_config is None:
        if start_from_best:
            best = get_best_result()
            if best and best.config:
                baseline_config = StrategyConfig.from_dict(best.config)
                print(f"Starting from best result: [{best.experiment_id}] Sharpe={best.sharpe:.3f}")
            else:
                baseline_config = StrategyConfig.baseline()
                print("No previous best found. Using baseline config.")
        else:
            baseline_config = StrategyConfig.baseline()
            print("Starting from baseline config.")

    # Step 3: 获取当前最佳Sharpe
    best = get_best_result()
    best_sharpe = best.sharpe if best else 0.0
    print(f"Current best Sharpe: {best_sharpe:.3f}\n")

    # Step 4: 运行实验（仅在训练集上）
    print(f"[2/5] Running {n_experiments} experiments on training data ({len(data)} coins)...\n")
    start_time = time.time()
    kept_count = 0
    discarded_count = 0
    crashed_count = 0

    for exp_i in range(n_experiments):
        # 时间限制
        elapsed_min = (time.time() - start_time) / 60
        if elapsed_min > max_time_minutes:
            print(f"\n[TIME LIMIT] {max_time_minutes}min reached. Stopping.")
            break

        exp_start = time.time()
        exp_id = f"exp_{datetime.now().strftime('%m%d_%H%M%S')}_{exp_i+1}"

        # 变异配置
        mutation = {}
        if exp_i == 0:
            # 第一个实验：运行baseline
            test_config = baseline_config
            mutation_desc = "baseline"
        else:
            test_config = mutate_config(baseline_config)
            # 记录变异
            for k in SEARCH_SPACE.keys():
                old_v = getattr(baseline_config, k, None)
                new_v = getattr(test_config, k, None)
                if old_v != new_v:
                    mutation[k] = new_v
            mutation_desc = build_description(test_config, mutation)

        # 运行实验
        try:
            results = run_experiment(test_config, data, coins, wf_mode=wf_mode)
            metrics = aggregate_results(results)
        except Exception as e:
            print(f"  [CRASH {exp_i+1}] {exp_id}: {e}")
            traceback.print_exc()
            crashed_count += 1
            continue

        # 判断是否keep
        current_sharpe = metrics["avg_sharpe"]
        improvement = current_sharpe - best_sharpe

        if improvement > improvement_threshold:
            status = "keep"
            baseline_config = test_config  # 更新baseline
            best_sharpe = current_sharpe
            kept_count += 1
        else:
            status = "discard"
            discarded_count += 1

        # 记录结果
        record = ExperimentRecord(
            commit=get_git_commit(),
            experiment_id=exp_id,
            coins=",".join(coins),
            val_return=metrics["avg_return"],
            sharpe=current_sharpe,
            max_dd=metrics["avg_dd"],
            win_rate=metrics["avg_win_rate"],
            wlr=metrics["avg_wlr"],
            total_trades=metrics["total_trades"],
            status=status,
            description=mutation_desc,
            timestamp=datetime.now().isoformat(),
            config=test_config.to_dict(),
            details={
                k: r.to_dict() if hasattr(r, 'to_dict') else str(r)
                for k, r in results.items()
            } if isinstance(results, dict) else {},
        )

        write_experiment(record)

        exp_time = time.time() - exp_start
        print(f"  [{exp_i+1}/{n_experiments}] {exp_id} | "
              f"Sharpe={current_sharpe:.3f} (Δ{improvement:+.3f}) | "
              f"Return={metrics['avg_return']:.1f}% DD={metrics['avg_dd']:.1f}% | "
              f"WR={metrics['avg_win_rate']:.1f}% WLR={metrics['avg_wlr']:.2f} | "
              f"Trades={metrics['total_trades']} [{status.upper()}] "
              f"({exp_time:.1f}s) | {mutation_desc}")

    # Step 5: 打印汇总
    total_time = (time.time() - start_time) / 60
    print(f"\n[DONE] {n_experiments} experiments in {total_time:.1f}min")
    print(f"  Keep: {kept_count} | Discard: {discarded_count} | Crash: {crashed_count}")

    from experiment_logger import read_results
    print_summary(read_results())

    # Step 6: OOS验证 — 用完全没见过的数据验证最佳策略
    print(f"\n{'='*70}")
    print("🔬 OOS VALIDATION — 样本外验证")
    print(f"{'='*70}")
    if oos_data and best and best.config:
        try:
            best_config = StrategyConfig.from_dict(best.config)
            oos_results = run_experiment(best_config, oos_data, list(oos_data.keys()),
                                        n_windows=min(4, n_windows), wf_mode=wf_mode)
            oos_metrics = aggregate_results(oos_results)

            oos_sharpe = oos_metrics["avg_sharpe"]
            oos_return = oos_metrics["avg_return"]
            oos_dd = oos_metrics["avg_dd"]
            oos_wr = oos_metrics["avg_win_rate"]
            oos_trades = oos_metrics["total_trades"]

            print(f"\n  Training Sharpe: {best.sharpe:.3f}")
            print(f"  OOS Sharpe:      {oos_sharpe:.3f}")
            print(f"  OOS Return:      {oos_return:.1f}%")
            print(f"  OOS Max DD:      {oos_dd:.1f}%")
            print(f"  OOS Win Rate:    {oos_wr:.1f}%")
            print(f"  OOS Trades:      {oos_trades}")

            # 判定标准
            oos_passed = True
            if oos_trades < 10:
                print(f"\n  ⚠️  OOS交易太少({oos_trades})，结论不可靠")
                oos_passed = False
            if oos_sharpe < 0:
                print(f"\n  ❌ OOS Sharpe为负！策略在样本外亏损")
                oos_passed = False
            if best.sharpe > 0 and oos_sharpe < best.sharpe * 0.3:
                print(f"\n  ❌ OOS Sharpe严重退化({oos_sharpe:.3f} < {best.sharpe*0.3:.3f})，强烈过拟合信号")
                oos_passed = False
            if oos_dd > 25:
                print(f"\n  ❌ OOS最大回撤{oos_dd:.1f}% > 25%，风险过高")
                oos_passed = False

            if oos_passed:
                print(f"\n  ✅ OOS验证通过 — 策略具备泛化能力")
            else:
                print(f"\n  ⚠️  OOS验证未完全通过 — 最佳策略可能存在过拟合")
                print(f"     建议检查参数范围或增加OOS数据量")
        except Exception as e:
            print(f"\n  ⚠️  OOS验证异常: {e}")
            traceback.print_exc()
    else:
        print(f"\n  ⏭️  跳过OOS验证 (oos_data={len(oos_data) if oos_data else 0}, best_config={'yes' if best and best.config else 'no'})")
    print(f"{'='*70}\n")

    # 保存最优配置
    best = get_best_result()
    if best:
        config_path = RESULTS_DIR / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(best.config, f, indent=2)
        print(f"\nBest config saved to {config_path}")

    # 反馈闭环：写成功标记，供gemma4检查IC权重新鲜度
    marker_path = Path("/tmp/autoresearch_last_success")
    marker_path.write_text(datetime.now().isoformat())
    # 清除失败标记（成功时清理）
    failure_marker = Path("/tmp/autoresearch_last_failure")
    if failure_marker.exists():
        failure_marker.unlink()
    print(f"[FEEDBACK] Success marker written: {marker_path}")

    # 反馈闭环Step2：用autoresearch结果更新IC权重
    _update_ic_weights_from_autoresearch()

    return best


def _update_ic_weights_from_autoresearch():
    """
    反馈闭环核心：从autoresearch结果计算策略质量，动态调整IC权重。

    逻辑：
    - 读取最近N个实验的Sharpe均值
    - 如果Sharpe<0：说明当前市场无有效策略，降低coin-specific因子权重
    - BTC/Gemma不受影响（是外部信号）
    """
    import json as _json
    from pathlib import Path

    results_path = RESULTS_DIR / "results.tsv"
    if not results_path.exists():
        return

    # 读取最近100个实验（今天的批次）
    try:
        lines = results_path.read_text().strip().split('\n')
        if len(lines) < 3:
            return
        header = lines[0].split('\t')
        data_lines = lines[-200:]  # 最近200个
        # 读取结果：使用val_return(第4列)而非sharpe(第5列)
        # 因为DISCARD实验sharpe=0但return是负数
        shames = []
        returns = []
        for line in data_lines:
            cols = line.split('\t')
            if len(cols) < 5:
                continue
            try:
                shames.append(float(cols[4]))  # sharpe
                returns.append(float(cols[3]))  # val_return
            except (ValueError, IndexError):
                continue

        if not shames:
            return

        avg_sharpe = sum(shames) / len(shames)
        avg_return = sum(returns) / len(returns)
        keep_rate = sum(1 for s in shames if s > 0) / len(shames)
        
        # 额外统计：只看KEEP实验的平均回报（更能反映策略真实质量）
        keep_returns = [returns[i] for i, s in enumerate(shames) if s > 0]
        keep_avg_return = sum(keep_returns) / len(keep_returns) if keep_returns else 0.0
        n_keep = len(keep_returns)
        print(f"[FEEDBACK] 最近{len(shames)}实验: 平均Sharpe={avg_sharpe:.4f} 平均回报={avg_return:+.2f}% 正收益={keep_rate:.1%}")
        print(f"[FEEDBACK] KEEP实验{n_keep}个: 平均回报={keep_avg_return:+.2f}%")

        # 读取当前IC权重
        ic_file = Path.home() / ".hermes/kronos_ic_weights.json"
        if ic_file.exists():
            with open(ic_file) as f:
                ic_data = _json.load(f)
        else:
            ic_data = {'weights': {}, 'last_update': None}

        weights = ic_data.get('weights', {})
        original = dict(weights)

        # 策略质量反馈因子：用avg_return和keep_rate双向调节
        # 负反馈：市场无有效策略时降低技术因子权重
        # 正反馈：策略有效时boost技术因子权重（之前缺失的关键环节！）
        if avg_return < -3:
            decay = 0.5  # 强信号：平均亏损>3%
            feedback_type = 'decay'
        elif avg_return < -1:
            decay = 0.3  # 中等信号：平均亏损>1%
            feedback_type = 'decay'
        elif avg_return < 0:
            decay = 0.15  # 轻微信号：平均亏损
            feedback_type = 'decay'
        elif keep_rate < 0.3:
            decay = 0.2  # 胜率<30%也降权
            feedback_type = 'decay'
        elif n_keep >= 3 and keep_avg_return > 5:
            # 正反馈：当有≥3个keep实验且平均回报>5%时，boost技术因子
            # boost幅度与keep_avg_return成正比
            boost = min(0.25, keep_avg_return / 100.0)  # 最多boost 25%
            decay = -boost  # 用负值表示boost
            feedback_type = 'boost'
        elif n_keep >= 5 and keep_avg_return > 2:
            # 中等正反馈
            boost = min(0.15, keep_avg_return / 150.0)
            decay = -boost
            feedback_type = 'boost'
        else:
            decay = 0.0
            feedback_type = 'neutral'

        BTC_GEMMA_MAX = 0.20
        TECH_MIN = 0.03  # 技术因子最低权重
        coin_factors = ['RSI', 'ADX', 'Bollinger', 'Vol', 'MACD']

        if decay > 0:
            # === 负反馈：降低技术因子权重 ===
            for factor in coin_factors:
                if factor in weights and weights[factor] > 0:
                    weights[factor] = max(0.01, weights[factor] * (1 - decay))
                    print(f"  [{factor}] {original[factor]:.2%} → {weights[factor]:.2%} (策略衰减-{decay:.0%})")

            # BTC/Gemma权重上限，防止无限增长彻底挤出技术因子
            for cap_factor in ('BTC', 'Gemma'):
                if cap_factor in weights and weights[cap_factor] > BTC_GEMMA_MAX:
                    excess = weights[cap_factor] - BTC_GEMMA_MAX
                    weights[cap_factor] = BTC_GEMMA_MAX
                    tech_fs = [f for f in weights if f not in ('BTC', 'Gemma')]
                    if tech_fs:
                        boost = excess / len(tech_fs)
                        for f in tech_fs:
                            weights[f] = min(weights.get(f, 0) + boost, TECH_MIN * 3)
                    print(f"  [{cap_factor}] 上限触达{BTC_GEMMA_MAX:.0%}，超额{excess:.1%}分配给技术因子")

            # 归一化到100%
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
                'n_keep': n_keep,
                'decay_applied': decay,
                'feedback_type': feedback_type,
                'source': 'autoresearch'
            }

            with open(ic_file, 'w') as f:
                _json.dump(ic_data, f, indent=2)
            print(f"[FEEDBACK] IC权重已更新({feedback_type}): {ic_file}")

        elif decay < 0:
            # === 正反馈：Boost技术因子权重（之前缺失的闭环环节） ===
            boost = abs(decay)
            print(f"[FEEDBACK] 正反馈触发：boost技术因子+{boost:.0%}")

            for factor in coin_factors:
                if factor in weights and weights[factor] > 0:
                    old_w = weights[factor]
                    # Boost幅度：技术因子权重增加，同时BTC/Gemma等权减少以保持归一化
                    weights[factor] = min(old_w * (1 + boost * 2), 0.25)  # 单因子最多25%
                    print(f"  [{factor}] {old_w:.2%} → {weights[factor]:.2%} (+{boost:.0%} boost)")

            # 归一化BTC/Gemma：降低BTC/Gemma，补给被boost的技术因子
            btc_gemma_total = sum(weights.get(f, 0) for f in ('BTC', 'Gemma') if f in weights)
            excess = btc_gemma_total - BTC_GEMMA_MAX * 2  # 期望BTC+Gemma=40%
            if excess > 0:
                # BTC/Gemma超出上限，从技术因子中扣除
                for cap_factor in ('BTC', 'Gemma'):
                    weights[cap_factor] = BTC_GEMMA_MAX
                tech_fs = [f for f in weights if f not in ('BTC', 'Gemma')]
                if tech_fs:
                    deduct = excess / len(tech_fs)
                    for f in tech_fs:
                        weights[f] = max(weights.get(f, 0) - deduct, TECH_MIN)
                print(f"  [BTC/Gemma] 权重重分配: 超出部分→技术因子")
            else:
                # BTC/Gemma未超上限，直接降低BTC/Gemma以平衡技术因子增长
                reduce_each = (boost * 0.5) if btc_gemma_total > 0.01 else 0
                for cap_factor in ('BTC', 'Gemma'):
                    if cap_factor in weights:
                        weights[cap_factor] = max(weights[cap_factor] - reduce_each, 0.05)
                # 技术因子boost需要从BTC/Gemma转移
                total_tech_boost = sum(weights.get(f, 0) - original.get(f, 0) for f in coin_factors if f in weights)
                if total_tech_boost > 0 and btc_gemma_total > 0:
                    transfer = min(total_tech_boost * 0.5, btc_gemma_total * 0.3)
                    for cap_factor in ('BTC', 'Gemma'):
                        weights[cap_factor] = max(weights.get(cap_factor, 0) - transfer / 2, 0.05)

            # 归一化到100%
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
                'n_keep': n_keep,
                'decay_applied': boost,
                'feedback_type': feedback_type,
                'source': 'autoresearch'
            }

            with open(ic_file, 'w') as f:
                _json.dump(ic_data, f, indent=2)
            print(f"[FEEDBACK] IC权重已更新({feedback_type}): {ic_file}")
        elif decay == 0:
            # decay=0 仍需写入文件（更新strategy_quality但不改变权重）
            ic_data['weights'] = weights
            ic_data['last_update'] = datetime.now().isoformat()
            ic_data['strategy_quality'] = {
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'keep_rate': keep_rate,
                'keep_avg_return': keep_avg_return,
                'n_keep': n_keep,
                'decay_applied': 0.0,
                'feedback_type': 'neutral',
                'source': 'autoresearch'
            }

            with open(ic_file, 'w') as f:
                _json.dump(ic_data, f, indent=2)
            print(f"[FEEDBACK] 策略质量正常，权重保持不变(neutral): {ic_file}")

    except Exception as e:
        print(f"[FEEDBACK] IC权重更新失败: {e}")


# 批量扫描: 固定参数网格
def grid_scan(
    coins: List[str] = None,
    timeframe: str = "1h",
    n_windows: int = 8,
):
    """
    网格扫描: 遍历参数空间的离散点
    比随机搜索更系统，但数量庞大
    """
    coins = coins or DEFAULT_COINS
    ensure_dirs()
    init_results_tsv()

    print(f"\n{'='*70}")
    print(f"GRID SCAN | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    # 加载数据
    print("[Loading data...]")
    data = load_all_data(timeframe, coins=coins)
    if not data:
        return

    # 定义网格
    grid_configs = []

    # RSI网格
    for rsi_oversold in [25, 30, 35]:
        for rsi_overbought in [65, 70, 75]:
            # ADX网格
            for adx in [12, 18, 25]:
                # SL ATR网格
                for sl_atr in [1.0, 1.5, 2.0]:
                    for tp1_atr in [2.5, 3.5, 5.0]:
                        c = StrategyConfig.baseline()
                        c.rsi_oversold = rsi_oversold
                        c.rsi_overbought = rsi_overbought
                        c.adx_threshold = adx
                        c.sl_atr_mult = sl_atr
                        c.tp1_atr_mult = tp1_atr
                        grid_configs.append(c)

    print(f"Total grid points: {len(grid_configs)}")
    print(f"Running... (this may take a while)\n")

    # 运行每个网格点
    best_sharpe = 0.0
    best_config = None

    for i, config in enumerate(grid_configs):
        results = run_experiment(config, data, coins, n_windows=n_windows)
        metrics = aggregate_results(results)
        sharpe = metrics["avg_sharpe"]

        desc = (f"RSI({config.rsi_oversold}/{config.rsi_overbought}) "
                f"ADX({config.adx_threshold}) "
                f"SL({config.sl_atr_mult}) TP1({config.tp1_atr_mult})")

        status = "keep" if sharpe >= best_sharpe else "discard"
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_config = config

        record = ExperimentRecord(
            commit=get_git_commit(),
            experiment_id=f"grid_{i+1}",
            coins=",".join(coins),
            val_return=metrics["avg_return"],
            sharpe=sharpe,
            max_dd=metrics["avg_dd"],
            win_rate=metrics["avg_win_rate"],
            wlr=metrics["avg_wlr"],
            total_trades=metrics["total_trades"],
            status=status,
            description=desc,
            timestamp=datetime.now().isoformat(),
            config=config.to_dict(),
            details={},
        )
        write_experiment(record)

        print(f"  [{i+1}/{len(grid_configs)}] {desc} | "
              f"Sharpe={sharpe:.3f} Return={metrics['avg_return']:.1f}% "
              f"WR={metrics['avg_win_rate']:.1f}% | {status.upper()}")

    print(f"\nGrid scan complete. Best Sharpe: {best_sharpe:.3f}")
    return best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos Autoresearch")
    parser.add_argument("--mode", choices=["autoresearch", "grid"], default="autoresearch")
    parser.add_argument("--experiments", type=int, default=50)
    parser.add_argument("--coins", type=str, default="BTC,ETH,ADA,DOGE,AVAX,DOT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--max-time", type=int, default=480)
    parser.add_argument("--wf-mode", type=str, default="expanding",
                        choices=["expanding", "rolling", "rolling_recent"],
                        help="Walk-Forward窗口模式: expanding(推荐)/rolling/rolling_recent")
    parser.add_argument("--no-bear-only", action="store_true",
                        help="使用完整历史数据（不用熊市过滤）")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据目录路径，默认使用 ~/kronos/")
    parser.add_argument("--oos", type=float, default=0.2,
                        help="OOS验证保留比例 (0~1)，默认0.2=保留最后20%做样本外验证")
    args = parser.parse_args()

    coins = args.coins.split(",")

    # 解析data_dir
    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / ".hermes/cron/output"
    print(f"[INFO] Data directory: {data_dir}")

    if args.mode == "grid":
        grid_scan(coins=coins, timeframe=args.timeframe, n_windows=args.windows)
    else:
        run_autoresearch(
            n_experiments=args.experiments,
            coins=coins,
            timeframe=args.timeframe,
            start_from_best=True,
            max_time_minutes=args.max_time,
            wf_mode=args.wf_mode,
            bear_only=not args.no_bear_only,
            data_dir=data_dir,
            oos_ratio=args.oos,
        )
