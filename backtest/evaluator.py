"""
Miracle 1.0.2 - Adaptive Evaluator
====================================
Factor and pattern evaluation, over拟合检测, PCA异常检测

Features:
1. Factor IC evaluation
2. Pattern performance evaluation (with whitelist/blacklist)
3. Overfitting detection
4. PCA anomaly detection and alerts
5. Performance reporting
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .learner import (
    WalkForwardValidator,
    calc_information_coefficient,
)

logger = logging.getLogger("miracle.adaptive_learner.evaluator")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# Factor Evaluator
# ============================================================

class FactorEvaluator:
    """
    因子评估器
    
    评估各因子的预测能力和表现
    """
    
    def __init__(self, min_sample_size: int = 20):
        self.min_sample_size = min_sample_size
        self.factor_performance = defaultdict(lambda: {
            "signals": [],
            "returns": [],
            "ic_history": []
        })
    
    def update(self, factor_name: str, signal: float, actual_return: float):
        """
        更新因子表现
        
        Args:
            factor_name: 因子名称
            signal: 因子信号值
            actual_return: 实际收益
        """
        perf = self.factor_performance[factor_name]
        perf["signals"].append(signal)
        perf["returns"].append(actual_return)
        
        # 保持最近100个样本
        if len(perf["signals"]) > 100:
            perf["signals"] = perf["signals"][-100:]
            perf["returns"] = perf["returns"][-100:]
    
    def get_ic(self, factor_name: str) -> Tuple[float, float]:
        """
        获取因子IC
        
        Args:
            factor_name: 因子名称
            
        Returns:
            (ic, p_value)
        """
        perf = self.factor_performance.get(factor_name)
        if not perf or len(perf["signals"]) < self.min_sample_size:
            return 0.0, 1.0
        
        return calc_information_coefficient(perf["signals"], perf["returns"])
    
    def get_report(self) -> Dict[str, Any]:
        """
        获取所有因子IC报告
        
        Returns:
            各因子的IC统计报告
        """
        report = {}
        for factor_name, perf in self.factor_performance.items():
            if len(perf["signals"]) >= self.min_sample_size:
                ic, p_value = calc_information_coefficient(perf["signals"], perf["returns"])
                report[factor_name] = {
                    "ic": ic,
                    "p_value": p_value,
                    "sample_size": len(perf["signals"]),
                    "ic_history_avg": float(np.mean(perf["ic_history"])) if perf["ic_history"] else 0.0
                }
        return report


# ============================================================
# Pattern Evaluator
# ============================================================

class PatternEvaluator:
    """
    模式评估器
    
    评估各交易模式的表现，支持白名单/黑名单过滤
    """
    
    def __init__(self, 
                 min_sample_size: int = 5,
                 pattern_whitelist: List[str] = None,
                 pattern_blacklist: List[str] = None):
        """
        Args:
            min_sample_size: 最少样本数
            pattern_whitelist: 允许的模式列表 (空列表或None表示不限制)
            pattern_blacklist: 禁止的模式列表
        """
        self.min_sample_size = min_sample_size
        self.pattern_performance = defaultdict(lambda: {
            "total": 0,
            "wins": 0,
            "total_rr": 0.0,
            "win_rate": 0.5
        })
        
        # 白名单/黑名单
        self.pattern_whitelist: Set[str] = set(pattern_whitelist or [])
        self.pattern_blacklist: Set[str] = set(pattern_blacklist or [])
    
    def update(self, pattern_key: str, won: bool, actual_rr: float):
        """
        更新模式表现
        
        Args:
            pattern_key: 模式键
            won: 是否盈利
            actual_rr: 实际盈亏比
        """
        perf = self.pattern_performance[pattern_key]
        perf["total"] += 1
        if won:
            perf["wins"] += 1
        perf["total_rr"] += actual_rr
        if perf["total"] > 0:
            perf["win_rate"] = perf["wins"] / perf["total"]
    
    def is_allowed(self, pattern_key: str) -> bool:
        """
        检查模式是否允许交易
        
        Args:
            pattern_key: 模式键
            
        Returns:
            是否允许交易
        """
        # 黑名单检查 - 直接禁止
        if pattern_key in self.pattern_blacklist:
            logger.warning(f"Pattern {pattern_key} is on blacklist, blocked")
            return False
        
        # 白名单检查 - 如果有白名单且不在白名单中，则禁止
        if self.pattern_whitelist and pattern_key not in self.pattern_whitelist:
            logger.info(f"Pattern {pattern_key} not in whitelist, blocked")
            return False
        
        perf = self.pattern_performance.get(pattern_key)
        
        # 样本不足，允许交易
        if not perf or perf["total"] < self.min_sample_size:
            return True
        
        # 胜率低于40%，禁止交易
        if perf["win_rate"] < 0.4:
            logger.warning(f"Pattern {pattern_key} blocked due to low win rate: {perf['win_rate']:.2%}")
            return False
        
        return True
    
    def add_to_whitelist(self, pattern_keys: List[str]):
        """添加模式到白名单"""
        self.pattern_whitelist.update(pattern_keys)
        logger.info(f"Added patterns to whitelist: {pattern_keys}")
    
    def remove_from_whitelist(self, pattern_keys: List[str]):
        """从白名单移除模式"""
        self.pattern_whitelist.difference_update(pattern_keys)
        logger.info(f"Removed patterns from whitelist: {pattern_keys}")
    
    def add_to_blacklist(self, pattern_keys: List[str]):
        """添加模式到黑名单"""
        self.pattern_blacklist.update(pattern_keys)
        logger.info(f"Added patterns to blacklist: {pattern_keys}")
    
    def remove_from_blacklist(self, pattern_keys: List[str]):
        """从黑名单移除模式"""
        self.pattern_blacklist.difference_update(pattern_keys)
        logger.info(f"Removed patterns from blacklist: {pattern_keys}")
    
    def get_whitelist(self) -> List[str]:
        """获取白名单"""
        return list(self.pattern_whitelist)
    
    def get_blacklist(self) -> List[str]:
        """获取黑名单"""
        return list(self.pattern_blacklist)
    
    def clear_whitelist(self):
        """清空白名单"""
        self.pattern_whitelist.clear()
        logger.info("Cleared pattern whitelist")
    
    def clear_blacklist(self):
        """清空黑名单"""
        self.pattern_blacklist.clear()
        logger.info("Cleared pattern blacklist")
    
    def get_stats(self, pattern_key: str) -> Dict[str, Any]:
        """
        获取模式统计
        
        Args:
            pattern_key: 模式键
            
        Returns:
            模式统计信息
        """
        perf = self.pattern_performance.get(pattern_key, {
            "total": 0, "wins": 0, "total_rr": 0.0, "win_rate": 0.5
        })
        avg_rr = perf["total_rr"] / perf["total"] if perf["total"] > 0 else 0.0
        return {
            "pattern": pattern_key,
            "total_trades": perf["total"],
            "wins": perf["wins"],
            "losses": perf["total"] - perf["wins"],
            "win_rate": perf["win_rate"],
            "avg_rr": avg_rr
        }
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """获取所有模式统计"""
        return [
            self.get_stats(pk)
            for pk in self.pattern_performance.keys()
        ]


# ============================================================
# Overfitting Detector
# ============================================================

class OverfittingDetector:
    """
    过拟合检测器
    
    使用Walk-Forward分析检测策略是否过拟合
    """
    
    def __init__(self, walk_forward_validator: WalkForwardValidator = None):
        self.walk_forward_validator = walk_forward_validator or WalkForwardValidator(
            train_window=50, test_window=20
        )
    
    def detect(self, factor_evaluator: FactorEvaluator) -> Dict[str, Any]:
        """
        检测过拟合
        
        Args:
            factor_evaluator: 因子评估器
            
        Returns:
            {
                "is_overfitting": bool,
                "train_ic_avg": float,
                "test_ic_avg": float,
                "ic_decay": float,
                "reason": str
            }
        """
        # 准备数据
        all_signals = []
        all_returns = []
        for perf in factor_evaluator.factor_performance.values():
            all_signals.extend(perf["signals"])
            all_returns.extend(perf["returns"])
        
        if len(all_signals) < 50:
            return {
                "is_overfitting": False,
                "reason": "样本不足，无法判断",
                "train_ic_avg": 0.0,
                "test_ic_avg": 0.0,
                "ic_decay": 0.0
            }
        
        # Walk-Forward验证
        data = [{"signal": s, "return": r} for s, r in zip(all_signals, all_returns)]
        
        def strategy_func(window_data):
            """Walk-forward验证：训练参数在train窗口，测试在test窗口"""
            n = len(window_data)
            if n < 10:
                return {"train_ic": 0.0, "test_ic": 0.0}
            
            split = n // 2
            train_window = window_data[:split]
            test_window = window_data[split:]
            
            # Train: compute IC on train window (in-sample)
            train_signals = [d["signal"] for d in train_window]
            train_returns = [d["return"] for d in train_window]
            train_ic, _ = calc_information_coefficient(train_signals, train_returns)
            
            # Test: use trained "params" (mean signal from train) on test window (out-of-sample)
            train_mean_signal = sum(train_signals) / len(train_signals)
            test_signals = [d["signal"] for d in test_window]
            test_returns = [d["return"] for d in test_window]
            test_ic, _ = calc_information_coefficient(test_signals, test_returns)
            
            return {
                "train_ic": train_ic,
                "test_ic": test_ic
            }
        
        wf_results = self.walk_forward_validator.validate(strategy_func, data, n_windows=5)
        
        ic_decay = wf_results.get("ic_decay", 0.0)
        is_overfitting = ic_decay > 0.3  # IC衰减超过30%认为过拟合
        
        return {
            "is_overfitting": is_overfitting,
            "train_ic_avg": wf_results.get("train_ic_avg", 0.0),
            "test_ic_avg": wf_results.get("test_ic_avg", 0.0),
            "ic_decay": ic_decay,
            "reason": "IC衰减超过30%" if is_overfitting else "未检测到过拟合"
        }


# ============================================================
# PCA Anomaly Detector
# ============================================================

@dataclass
class AnomalyAlert:
    """异常警报数据类"""
    timestamp: str
    alert_type: str  # "anomaly", "drift", "volatility"
    severity: str   # "low", "medium", "high", "critical"
    message: str
    anomaly_score: float  # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)


class PCAAnomalyDetector:
    """
    PCA异常检测器
    
    使用PCA分析因子空间的多维异常，检测：
    1. 异常样本点（因子空间中的离群点）
    2. 概念漂移（因子权重分布变化）
    3. 市场波动异常（收益分布变化）
    
    依赖: sklearn (scikit-learn)
    """
    
    def __init__(self,
                 n_components: int = 2,
                 contamination: float = 0.1,
                 drift_threshold: float = 0.05,
                 history_size: int = 100):
        """
        Args:
            n_components: PCA主成分数量
            contamination: 异常比例（用于T2统计量阈值）
            drift_threshold: 概念漂移检测阈值（基于重构误差变化）
            history_size: 保持的历史样本数
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, PCAAnomalyDetector disabled")
        
        self.n_components = n_components
        self.contamination = contamination
        self.drift_threshold = drift_threshold
        self.history_size = history_size
        
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.pca = PCA(n_components=n_components) if HAS_SKLEARN else None
        
        # 历史数据
        self.feature_history: List[List[float]] = []
        self.return_history: List[float] = []
        
        # 基线（参考分布）
        self.baseline_fitted = False
        self.baseline_components = None
        self.baseline_mean = None
        self.baseline_std = None
        
        # 警报记录
        self.alerts: List[AnomalyAlert] = []
        
        # 统计
        self.anomaly_scores: List[float] = []
    
    def _ensure_scikit_learn(self) -> bool:
        """检查scikit-learn是否可用"""
        if not HAS_SKLEARN:
            logger.warning("PCAAnomalyDetector: sklearn not available")
            return False
        return True
    
    def fit_baseline(self, features: List[List[float]], returns: List[float] = None):
        """
        拟合基线分布（正常市场状态）
        
        Args:
            features: 因子特征列表 [[f1, f2, ...], ...]
            returns: 可选的收益列表
        """
        if not self._ensure_scikit_learn():
            return
        
        if len(features) < 30:
            logger.warning(f"PCAAnomalyDetector: Need at least 30 samples for baseline, got {len(features)}")
            return
        
        # 标准化
        features_array = np.array(features)
        self.scaler.fit(features_array)
        scaled_features = self.scaler.transform(features_array)
        
        # PCA拟合
        self.pca.fit(scaled_features)
        
        # 保存基线
        self.baseline_components = self.pca.components_.copy()
        self.baseline_mean = self.scaler.mean_.copy()
        self.baseline_std = self.scaler.scale_.copy()
        self.baseline_fitted = True
        
        # 保存返回历史
        if returns:
            self.return_history = list(returns[-self.history_size:])
        
        logger.info(f"PCAAnomalyDetector: Baseline fitted with {len(features)} samples")
    
    def add_sample(self, features: List[float], return_value: float = None) -> float:
        """
        添加新样本并返回异常分数
        
        Args:
            features: 因子特征 [f1, f2, ...]
            return_value: 可选的收益值
            
        Returns:
            异常分数 (0.0-1.0, 越高越异常)
        """
        if not self._ensure_scikit_learn() or not self.baseline_fitted:
            return 0.0
        
        self.feature_history.append(features)
        if return_value is not None:
            self.return_history.append(return_value)
        
        # 保持历史大小
        if len(self.feature_history) > self.history_size:
            self.feature_history = self.feature_history[-self.history_size:]
        if len(self.return_history) > self.history_size:
            self.return_history = self.return_history[-self.history_size:]
        
        # 计算异常分数
        anomaly_score = self._calculate_anomaly_score(features)
        self.anomaly_scores.append(anomaly_score)
        
        # 保持分数历史
        if len(self.anomaly_scores) > self.history_size:
            self.anomaly_scores = self.anomaly_scores[-self.history_size:]
        
        return anomaly_score
    
    def _calculate_anomaly_score(self, features: List[float]) -> float:
        """
        计算样本的异常分数（基于T2统计量和重构误差）
        
        Returns:
            异常分数 0.0-1.0
        """
        if not HAS_SKLEARN or not self.baseline_fitted:
            return 0.0
        
        try:
            # 标准化
            scaled = (np.array(features) - self.baseline_mean) / self.baseline_std
            scaled = scaled.reshape(1, -1)
            
            # T2统计量（到PCA主成分空间的马氏距离）
            pca_transformed = self.pca.transform(scaled)
            t2_score = np.sum(pca_transformed[0] ** 2)
            
            # 重构误差
            reconstructed = self.pca.inverse_transform(pca_transformed)
            reconstruction_error = np.mean((scaled - reconstructed) ** 2)
            
            # 归一化分数
            # T2分数通常服从chi2分布，但这里用经验阈值
            anomaly_score = min(1.0, reconstruction_error * 10)
            
            return float(anomaly_score)
            
        except Exception as e:
            logger.warning(f"PCAAnomalyDetector: Error calculating anomaly score: {e}")
            return 0.0
    
    def detect_drift(self) -> Optional[Dict[str, Any]]:
        """
        检测概念漂移（因子权重分布变化）
        
        Returns:
            漂移报告或None
        """
        if not self._ensure_scikit_learn() or not self.baseline_fitted:
            return None
        
        if len(self.feature_history) < 30:
            return None
        
        # 用当前窗口重新拟合PCA
        current_features = np.array(self.feature_history[-self.history_size:])
        current_scaled = self.scaler.fit_transform(current_features)
        current_pca = PCA(n_components=self.n_components)
        current_pca.fit(current_scaled)
        
        # 比较主成分方向变化（余弦相似度）
        angle_diffs = []
        for i in range(min(len(self.baseline_components), len(current_pca.components_))):
            baseline_comp = self.baseline_components[i]
            current_comp = current_pca.components_[i]
            # 计算角度差异
            cos_sim = np.dot(baseline_comp, current_comp) / (
                np.linalg.norm(baseline_comp) * np.linalg.norm(current_comp) + 1e-10
            )
            angle_diff = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi
            angle_diffs.append(angle_diff)
        
        avg_angle_diff = np.mean(angle_diffs) if angle_diffs else 0.0
        
        # 检测重构误差变化
        recent_features = np.array(self.feature_history[-50:])
        recent_scaled = (recent_features - self.baseline_mean) / self.baseline_std
        recent_transformed = current_pca.transform(recent_scaled)
        recent_reconstructed = current_pca.inverse_transform(recent_transformed)
        recent_error = np.mean((recent_scaled - recent_reconstructed) ** 2)
        
        # 基线重构误差
        baseline_scaled = (current_features - self.baseline_mean) / self.baseline_std
        baseline_transformed = self.pca.transform(baseline_scaled)
        baseline_reconstructed = self.pca.inverse_transform(baseline_transformed)
        baseline_error = np.mean((baseline_scaled - baseline_reconstructed) ** 2)
        
        error_change = abs(recent_error - baseline_error) / (baseline_error + 1e-10)
        
        # 判断是否有漂移
        is_drift = avg_angle_diff > 15 or error_change > self.drift_threshold * 10
        
        return {
            "is_drift": is_drift,
            "avg_angle_diff": float(avg_angle_diff),
            "reconstruction_error_change": float(error_change),
            "current_reconstruction_error": float(recent_error),
            "baseline_reconstruction_error": float(baseline_error)
        }
    
    def check_anomaly(self) -> Optional[AnomalyAlert]:
        """
        检查是否应该触发警报
        
        Returns:
            AnomalyAlert或None
        """
        if not self.anomaly_scores:
            return None
        
        recent_scores = self.anomaly_scores[-10:] if len(self.anomaly_scores) >= 10 else self.anomaly_scores
        avg_score = np.mean(recent_scores)
        
        # 阈值判断
        if avg_score > 0.8:
            severity = "critical"
        elif avg_score > 0.6:
            severity = "high"
        elif avg_score > 0.4:
            severity = "medium"
        elif avg_score > 0.2:
            severity = "low"
        else:
            return None
        
        # 构建警报
        alert = AnomalyAlert(
            timestamp=datetime.now().isoformat(),
            alert_type="anomaly",
            severity=severity,
            message=f"PCA异常检测: 平均异常分数 {avg_score:.3f}",
            anomaly_score=avg_score,
            details={
                "recent_scores": [float(s) for s in recent_scores],
                "n_samples": len(self.feature_history)
            }
        )
        
        self.alerts.append(alert)
        
        # 保持最近100个警报
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        return alert
    
    def check_market_volatility(self, window: int = 20) -> Optional[AnomalyAlert]:
        """
        检测市场波动异常（基于收益分布）
        
        Args:
            window: 检测窗口
            
        Returns:
            波动异常警报或None
        """
        if len(self.return_history) < window * 2:
            return None
        
        recent_returns = self.return_history[-window:]
        historical_returns = self.return_history[-window*2:-window]
        
        # 计算波动率
        recent_vol = np.std(recent_returns) if recent_returns else 0
        historical_vol = np.std(historical_returns) if historical_returns else 0
        
        if historical_vol == 0:
            return None
        
        vol_ratio = recent_vol / historical_vol
        
        # 判断波动异常
        if vol_ratio > 3.0:
            severity = "critical"
        elif vol_ratio > 2.5:
            severity = "high"
        elif vol_ratio > 2.0:
            severity = "medium"
        elif vol_ratio > 1.5:
            severity = "low"
        else:
            return None
        
        alert = AnomalyAlert(
            timestamp=datetime.now().isoformat(),
            alert_type="volatility",
            severity=severity,
            message=f"市场波动异常: 当前波动率是历史的 {vol_ratio:.2f}倍",
            anomaly_score=min(1.0, (vol_ratio - 1.5) / 2.0),
            details={
                "recent_volatility": float(recent_vol),
                "historical_volatility": float(historical_vol),
                "volatility_ratio": float(vol_ratio)
            }
        )
        
        self.alerts.append(alert)
        return alert
    
    def run_full_check(self) -> Dict[str, Any]:
        """
        运行完整检查（异常+漂移+波动）
        
        Returns:
            完整检查报告
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": 0.0,
            "anomaly_alert": None,
            "drift_report": None,
            "volatility_alert": None,
            "has_alert": False,
            "alerts": []
        }
        
        # 异常检测
        anomaly_alert = self.check_anomaly()
        if anomaly_alert:
            report["anomaly_score"] = anomaly_alert.anomaly_score
            report["anomaly_alert"] = anomaly_alert
            report["has_alert"] = True
        
        # 漂移检测
        drift_report = self.detect_drift()
        if drift_report and drift_report["is_drift"]:
            report["drift_report"] = drift_report
            report["has_alert"] = True
            
            # 创建漂移警报
            drift_alert = AnomalyAlert(
                timestamp=datetime.now().isoformat(),
                alert_type="drift",
                severity="medium",
                message=f"概念漂移检测: 因子角度变化 {drift_report['avg_angle_diff']:.1f}°",
                anomaly_score=min(1.0, drift_report['avg_angle_diff'] / 45.0),
                details=drift_report
            )
            report["alerts"].append(drift_alert)
        
        # 波动检测
        vol_alert = self.check_market_volatility()
        if vol_alert:
            report["volatility_alert"] = vol_alert
            report["has_alert"] = True
        
        if anomaly_alert:
            report["alerts"].append(anomaly_alert)
        
        return report
    
    def get_recent_alerts(self, n: int = 10) -> List[AnomalyAlert]:
        """获取最近的n个警报"""
        return self.alerts[-n:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.anomaly_scores:
            return {
                "baseline_fitted": self.baseline_fitted,
                "n_samples": len(self.feature_history),
                "avg_anomaly_score": 0.0,
                "max_anomaly_score": 0.0
            }
        
        return {
            "baseline_fitted": self.baseline_fitted,
            "n_samples": len(self.feature_history),
            "avg_anomaly_score": float(np.mean(self.anomaly_scores)),
            "max_anomaly_score": float(np.max(self.anomaly_scores)),
            "recent_alerts_count": len([a for a in self.alerts if 
                (datetime.now() - datetime.fromisoformat(a.timestamp)).total_seconds() < 3600])
        }


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "FactorEvaluator",
    "PatternEvaluator",
    "OverfittingDetector",
    "PCAAnomalyDetector",
    "AnomalyAlert",
]
