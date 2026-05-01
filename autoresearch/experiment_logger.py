"""
Experiment Logger - 结果记录与比较

results.tsv 格式:
commit\texperiment_id\tcoins\tval_return\tsharpe\tmax_dd\twin_rate\twlr\ttotal_trades\tstatus\tdescription
"""
import os
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_TSV = RESULTS_DIR / "results.tsv"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"


@dataclass
class ExperimentRecord:
    commit: str
    experiment_id: str
    coins: str
    val_return: float  # 平均Walk-Forward收益
    sharpe: float
    max_dd: float
    win_rate: float
    wlr: float
    total_trades: int
    status: str  # keep | discard | crash
    description: str
    timestamp: str
    config: Dict[str, Any]
    details: Dict[str, Any]  # 每个窗口的详细结果

    def to_tsv_row(self) -> str:
        return "\t".join([
            self.commit[:7],
            self.experiment_id,
            self.coins,
            f"{self.val_return:.4f}",
            f"{self.sharpe:.4f}",
            f"{self.max_dd:.4f}",
            f"{self.win_rate:.2f}",
            f"{self.wlr:.3f}",
            str(self.total_trades),
            self.status,
            self.description[:200],
        ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_git_commit() -> str:
    """获取当前git commit hash"""
    try:
        repo = Path(__file__).parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:7]
    except Exception:
        return "no-git"


def ensure_dirs():
    """确保结果目录存在"""
    RESULTS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)


def init_results_tsv():
    """初始化TSV文件"""
    ensure_dirs()
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "commit\texperiment_id\tcoins\tval_return\tsharpe\tmax_dd\t"
            "win_rate\twlr\ttotal_trades\tstatus\tdescription\n"
        )


def write_experiment(record: ExperimentRecord):
    """追加实验结果到TSV"""
    ensure_dirs()
    init_results_tsv()

    # TSV追加
    with open(RESULTS_TSV, "a") as f:
        f.write(record.to_tsv_row() + "\n")

    # JSON详细记录
    exp_file = EXPERIMENTS_DIR / f"{record.experiment_id}.json"
    with open(exp_file, "w") as f:
        json.dump(record.to_dict(), f, indent=2, default=str)

    print(f"[EXPERIMENT LOG] {record.status.upper()}: {record.experiment_id} | "
          f"Return={record.val_return:.2f}% Sharpe={record.sharpe:.3f} "
          f"DD={record.max_dd:.2f}% WR={record.win_rate:.1f}% WLR={record.wlr:.2f} "
          f"| {record.description[:80]}")


def read_results() -> List[ExperimentRecord]:
    """读取所有实验结果"""
    if not RESULTS_TSV.exists():
        return []

    records = []
    with open(RESULTS_TSV) as f:
        header = f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 11:
                record = ExperimentRecord(
                    commit=parts[0],
                    experiment_id=parts[1],
                    coins=parts[2],
                    val_return=float(parts[3]),
                    sharpe=float(parts[4]),
                    max_dd=float(parts[5]),
                    win_rate=float(parts[6]),
                    wlr=float(parts[7]),
                    total_trades=int(parts[8]),
                    status=parts[9],
                    description=parts[10],
                    timestamp="",
                    config={},
                    details={},
                )
                records.append(record)
    return records


def get_best_result() -> Optional[ExperimentRecord]:
    """获取最佳结果（按Sharpe排序）"""
    records = read_results()
    if not records:
        return None
    keep_records = [r for r in records if r.status == "keep"]
    if not keep_records:
        return None
    return max(keep_records, key=lambda r: r.sharpe)


def get_top_results(n: int = 5) -> List[ExperimentRecord]:
    """获取Top N结果"""
    records = read_results()
    keep_records = sorted(
        [r for r in records if r.status == "keep"],
        key=lambda r: r.sharpe,
        reverse=True,
    )
    return keep_records[:n]


def generate_experiment_id(config: Dict[str, Any], coin: str) -> str:
    """生成唯一实验ID"""
    import time
    config_str = json.dumps(config, sort_keys=True)
    hash_str = hashlib.md5(f"{config_str}{coin}{time.time()}".encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    return f"exp_{timestamp}_{hash_str}"


def print_summary(records: List[ExperimentRecord]):
    """打印实验汇总"""
    if not records:
        print("No experiments yet.")
        return

    keep = [r for r in records if r.status == "keep"]
    discard = [r for r in records if r.status == "discard"]
    crash = [r for r in records if r.status == "crash"]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY | Total: {len(records)} | Keep: {len(keep)} | Discard: {len(discard)} | Crash: {len(crash)}")
    print(f"{'='*70}")

    if keep:
        print(f"\nTop 5 by Sharpe (keep only):")
        for i, r in enumerate(get_top_results(5), 1):
            print(f"  {i}. [{r.experiment_id}] {r.coins} | "
                  f"Sharpe={r.sharpe:.3f} Return={r.val_return:.2f}% "
                  f"WR={r.win_rate:.1f}% WLR={r.wlr:.2f} | {r.description[:60]}")

    best = get_best_result()
    if best:
        print(f"\nBest Result: [{best.experiment_id}]")
        print(f"  Sharpe={best.sharpe:.3f} | Return={best.val_return:.2f}% | "
              f"DD={best.max_dd:.2f}% | WR={best.win_rate:.1f}% | WLR={best.wlr:.2f}")
        print(f"  Config: {json.dumps(best.config, sort_keys=True)}")


if __name__ == "__main__":
    records = read_results()
    print_summary(records)
