#!/usr/bin/env python3
"""
Model Router - Fast/Slow Thinking Model Selection (P1.5)
=======================================================

根据场景自动选择快速/深度思考模型

功能:
    - should_use_deep() - 自动判断是否需要深度思考
    - 模型映射表可配置 (FAST/DEEP)
    - 成本统计日志

模型分配:
    | 场景 | 模型 | 触发条件 |
    |------|------|----------|
    | Bull/Bear分析 | quick_think | 标准扫描 |
    | Debate Judge | deep_think | 裁决阶段 |
    | IC权重更新 | deep_think | 置信度<0.4或冲突 |
    | 信号处理 | quick_think | 常规处理 |

Usage:
    from models.model_router import ModelRouter, should_use_deep

    router = ModelRouter()

    # 判断是否需要深度思考
    if router.should_use_deep(confidence=0.35, has_ic_update=True):
        model = router.get_model("ic_weight_update")
    else:
        model = router.get_model("signal_processing")

    # 获取成本统计
    stats = router.get_cost_stats()
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from datetime import datetime

logger = logging.getLogger(__name__)

# ==================== 常量 ====================

# 默认模型映射表
DEFAULT_MODEL_MAP = {
    # 快速思考模型 (标准技术分析/信号处理)
    "quick_think": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_tokens": 2048,
        "temperature": 0.7,
        "description": "快速分析模型 - 标准技术分析/信号处理"
    },
    # 深度思考模型 (复杂决策/IC更新/多因子冲突)
    "deep_think": {
        "provider": "openai",
        "model": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 0.5,
        "description": "深度决策模型 - 复杂决策/IC更新/多因子冲突"
    },
    # 辩论研究员 (快速)
    "bull_researcher": "quick_think",
    "bear_researcher": "quick_think",
    # 辩论裁决 (深度)
    "debate_judge": "deep_think",
    # IC权重更新 (深度)
    "ic_weight_update": "deep_think",
    # 信号处理 (快速)
    "signal_processing": "quick_think",
    # 趋势分析 (快速)
    "trend_analysis": "quick_think",
    # 置信度评估 (深度)
    "confidence_evaluation": "deep_think",
}

# 置信度阈值
DEFAULT_CONFIDENCE_THRESHOLD = 0.4

# 成本统计文件
COST_STATS_FILE = os.path.expanduser('~/.miracle_memory/model_cost_stats.json')


# ==================== 枚举 ====================

class ModelType(Enum):
    """模型类型枚举"""
    QUICK = "quick_think"
    DEEP = "deep_think"


class TaskType(Enum):
    """任务类型枚举"""
    BULL_RESEARCH = "bull_researcher"
    BEAR_RESEARCH = "bear_researcher"
    DEBATE_JUDGE = "debate_judge"
    IC_WEIGHT_UPDATE = "ic_weight_update"
    SIGNAL_PROCESSING = "signal_processing"
    TREND_ANALYSIS = "trend_analysis"
    CONFIDENCE_EVALUATION = "confidence_evaluation"
    STANDARD_SCAN = "standard_scan"


# ==================== 数据结构 ====================

@dataclass
class CostStats:
    """成本统计"""
    total_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0
    quick_think_count: int = 0
    deep_think_count: int = 0
    last_updated: Optional[str] = None


@dataclass
class ModelConfig:
    """模型配置"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 2048
    temperature: float = 0.7
    description: str = ""


@dataclass
class DeepThinkTriggers:
    """深度思考触发条件"""
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
    requires_ic_update: bool = False
    has_factor_conflict: bool = False
    is_critical_risk: bool = False
    survival_tier: str = "normal"


# ==================== 模型路由器 ====================

class ModelRouter:
    """
    快慢思考模型路由器

    根据场景自动选择合适的模型:
    - 快速模型: 标准技术分析、信号处理
    - 深度模型: IC权重更新、多因子冲突、低置信度(<0.4)

    使用单例模式确保全局唯一
    """

    _instance: Optional['ModelRouter'] = None
    _lock = Lock()

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型路由器

        Args:
            config_path: 可选的配置文件路径
        """
        self._model_map: Dict[str, Any] = dict(DEFAULT_MODEL_MAP)
        self._cost_stats = CostStats()
        self._load_config(config_path)
        self._load_cost_stats()

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'ModelRouter':
        """单例获取"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        return cls._instance

    def _load_config(self, config_path: Optional[str] = None) -> None:
        """从配置文件加载模型映射表"""
        if config_path is None:
            config_path = os.path.expanduser('~/.miracle_memory/model_router_config.json')

        if not os.path.exists(config_path):
            logger.info("[Router] 使用默认模型映射表")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._model_map = data.get('model_map', DEFAULT_MODEL_MAP)
            logger.info(f"[Router] 从 {config_path} 加载模型映射表")
        except Exception as e:
            logger.warning(f"[Router] 加载配置失败: {e}, 使用默认配置")

    def _load_cost_stats(self) -> None:
        """从磁盘加载成本统计"""
        if not os.path.exists(COST_STATS_FILE):
            return
        try:
            with open(COST_STATS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._cost_stats = CostStats(
                total_tokens=data.get('total_tokens', 0),
                total_cost=data.get('total_cost', 0.0),
                request_count=data.get('request_count', 0),
                quick_think_count=data.get('quick_think_count', 0),
                deep_think_count=data.get('deep_think_count', 0),
                last_updated=data.get('last_updated')
            )
            logger.info(f"[Router] 加载成本统计: 请求数={self._cost_stats.request_count}")
        except Exception as e:
            logger.warning(f"[Router] 加载成本统计失败: {e}")

    def _save_cost_stats(self) -> None:
        """持久化成本统计到磁盘"""
        os.makedirs(os.path.dirname(COST_STATS_FILE), exist_ok=True)
        try:
            self._cost_stats.last_updated = datetime.now().isoformat()
            with open(COST_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_tokens': self._cost_stats.total_tokens,
                    'total_cost': self._cost_stats.total_cost,
                    'request_count': self._cost_stats.request_count,
                    'quick_think_count': self._cost_stats.quick_think_count,
                    'deep_think_count': self._cost_stats.deep_think_count,
                    'last_updated': self._cost_stats.last_updated,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[Router] 保存成本统计失败: {e}")

    def should_use_deep(
        self,
        confidence: Optional[float] = None,
        requires_ic_update: bool = False,
        has_factor_conflict: bool = False,
        is_critical_risk: bool = False,
        survival_tier: str = "normal",
        task_type: Optional[TaskType] = None,
    ) -> bool:
        """
        判断是否需要使用深度思考模型

        深度思考触发条件:
        1. 置信度 < 0.4 (低置信度需要深度分析)
        2. IC权重更新场景
        3. 多因子冲突 (信号矛盾)
        4. 关键风险场景 (risk_level=critical)
        5. 生存层级为 critical/paused
        6. 辩论裁决任务

        快速思考场景:
        - 标准技术分析
        - 常规信号处理
        - Bull/Bear研究员分析

        Args:
            confidence: 置信度 0.0-1.0
            requires_ic_update: 是否需要IC权重更新
            has_factor_conflict: 是否存在多因子冲突
            is_critical_risk: 是否为关键风险场景
            survival_tier: 生存层级
            task_type: 任务类型

        Returns:
            True = 使用深度思考模型, False = 使用快速模型
        """
        # 任务类型明确指定的情况
        if task_type:
            deep_tasks = {
                TaskType.DEBATE_JUDGE,
                TaskType.IC_WEIGHT_UPDATE,
                TaskType.CONFIDENCE_EVALUATION,
            }
            if task_type in deep_tasks:
                logger.debug(f"[Router] 任务类型 {task_type.value} -> 使用深度模型")
                return True

            quick_tasks = {
                TaskType.BULL_RESEARCH,
                TaskType.BEAR_RESEARCH,
                TaskType.SIGNAL_PROCESSING,
                TaskType.TREND_ANALYSIS,
                TaskType.STANDARD_SCAN,
            }
            if task_type in quick_tasks:
                logger.debug(f"[Router] 任务类型 {task_type.value} -> 使用快速模型")
                return False

        # 深度思考触发条件检查
        triggers = DeepThinkTriggers(
            confidence=confidence or 1.0,
            requires_ic_update=requires_ic_update,
            has_factor_conflict=has_factor_conflict,
            is_critical_risk=is_critical_risk,
            survival_tier=survival_tier,
        )

        # 条件1: 低置信度
        if confidence is not None and confidence < DEFAULT_CONFIDENCE_THRESHOLD:
            logger.debug(f"[Router] 置信度 {confidence} < {DEFAULT_CONFIDENCE_THRESHOLD} -> 使用深度模型")
            return True

        # 条件2: IC权重更新
        if requires_ic_update:
            logger.debug(f"[Router] IC权重更新场景 -> 使用深度模型")
            return True

        # 条件3: 多因子冲突
        if has_factor_conflict:
            logger.debug(f"[Router] 多因子冲突 -> 使用深度模型")
            return True

        # 条件4: 关键风险场景
        if is_critical_risk:
            logger.debug(f"[Router] 关键风险场景 -> 使用深度模型")
            return True

        # 条件5: 危险生存层级
        if survival_tier in ("critical", "paused"):
            logger.debug(f"[Router] 生存层级 {survival_tier} -> 使用深度模型")
            return True

        # 默认使用快速模型
        return False

    def get_model_config(self, task_type: str) -> ModelConfig:
        """
        获取任务对应的模型配置

        Args:
            task_type: 任务类型 (如 "bull_researcher", "ic_weight_update")

        Returns:
            ModelConfig: 模型配置
        """
        model_key = self._model_map.get(task_type, "quick_think")

        # 如果映射到另一个任务类型,递归解析
        if isinstance(model_key, str) and model_key in self._model_map:
            return self.get_model_config(model_key)

        # 如果是完整配置字典
        if isinstance(model_key, dict):
            return ModelConfig(
                provider=model_key.get('provider', 'openai'),
                model=model_key.get('model', 'gpt-4o-mini'),
                max_tokens=model_key.get('max_tokens', 2048),
                temperature=model_key.get('temperature', 0.7),
                description=model_key.get('description', ''),
            )

        # 默认快速模型
        return ModelConfig(
            provider='openai',
            model='gpt-4o-mini',
            max_tokens=2048,
            temperature=0.7,
            description='快速分析模型',
        )

    def get_model(self, task_type: str) -> str:
        """
        获取任务对应的实际模型名称

        Args:
            task_type: 任务类型

        Returns:
            str: 模型名称 (如 "gpt-4o-mini" 或 "gpt-4o")
        """
        config = self.get_model_config(task_type)
        return config.model

    def get_provider(self, task_type: str) -> str:
        """
        获取任务对应的provider

        Args:
            task_type: 任务类型

        Returns:
            str: provider名称
        """
        config = self.get_model_config(task_type)
        return config.provider

    def record_usage(
        self,
        task_type: str,
        tokens_used: int,
        cost: float,
        model_type: Optional[ModelType] = None,
    ) -> None:
        """
        记录模型使用情况并更新成本统计

        Args:
            task_type: 任务类型
            tokens_used: 使用的token数
            cost: 成本 (美元)
            model_type: 模型类型 (快速/深度), 如果为None则自动判断
        """
        if model_type is None:
            # 根据任务类型自动判断
            if self.should_use_deep(task_type=self._task_type_to_enum(task_type)):
                model_type = ModelType.DEEP
            else:
                model_type = ModelType.QUICK

        self._cost_stats.total_tokens += tokens_used
        self._cost_stats.total_cost += cost
        self._cost_stats.request_count += 1

        if model_type == ModelType.QUICK:
            self._cost_stats.quick_think_count += 1
        else:
            self._cost_stats.deep_think_count += 1

        self._save_cost_stats()

        logger.info(
            f"[Router] 使用统计: task={task_type}, tokens={tokens_used}, "
            f"cost=${cost:.6f}, model={model_type.value}"
        )

    def _task_type_to_enum(self, task_type: str) -> Optional[TaskType]:
        """将字符串任务类型转换为枚举"""
        try:
            return TaskType(task_type)
        except ValueError:
            return None

    def get_cost_stats(self) -> Dict[str, Any]:
        """
        获取成本统计信息

        Returns:
            Dict: 成本统计字典
        """
        return {
            'total_tokens': self._cost_stats.total_tokens,
            'total_cost': self._cost_stats.total_cost,
            'request_count': self._cost_stats.request_count,
            'quick_think_count': self._cost_stats.quick_think_count,
            'deep_think_count': self._cost_stats.deep_think_count,
            'last_updated': self._cost_stats.last_updated,
            'avg_tokens_per_request': (
                self._cost_stats.total_tokens / self._cost_stats.request_count
                if self._cost_stats.request_count > 0 else 0
            ),
            'cost_ratio': (
                self._cost_stats.deep_think_count / self._cost_stats.request_count
                if self._cost_stats.request_count > 0 else 0
            ),
        }

    def reset_cost_stats(self) -> None:
        """重置成本统计"""
        self._cost_stats = CostStats()
        self._save_cost_stats()
        logger.info("[Router] 成本统计已重置")

    def update_model_map(self, model_map: Dict[str, Any]) -> None:
        """
        更新模型映射表

        Args:
            model_map: 新的模型映射表
        """
        self._model_map = model_map
        logger.info(f"[Router] 模型映射表已更新: {list(model_map.keys())}")

    def get_model_map(self) -> Dict[str, Any]:
        """获取当前模型映射表"""
        return dict(self._model_map)

    def get_recommended_model_for_decision(
        self,
        confidence: float,
        has_factor_conflict: bool = False,
        risk_level: str = "medium",
    ) -> Dict[str, Any]:
        """
        获取决策场景推荐模型

        综合考虑置信度、因子冲突、风险等级来决定使用哪个模型

        Args:
            confidence: 置信度 0.0-1.0
            has_factor_conflict: 是否存在多因子冲突
            risk_level: 风险等级

        Returns:
            Dict: 包含 model_type, model_name, provider, reasoning
        """
        if self.should_use_deep(
            confidence=confidence,
            has_factor_conflict=has_factor_conflict,
            is_critical_risk=(risk_level == "critical"),
        ):
            model_config = self.get_model_config("debate_judge")
            return {
                'model_type': 'deep_think',
                'model_name': model_config.model,
                'provider': model_config.provider,
                'reasoning': f"低置信度({confidence:.2f})或多因子冲突或高风险({risk_level})",
            }
        else:
            model_config = self.get_model_config("signal_processing")
            return {
                'model_type': 'quick_think',
                'model_name': model_config.model,
                'provider': model_config.provider,
                'reasoning': f"标准决策场景, 置信度={confidence:.2f}",
            }

    def export_config(self, path: Optional[str] = None) -> None:
        """导出当前配置到文件"""
        if path is None:
            path = os.path.expanduser('~/.miracle_memory/model_router_config.json')

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_map': self._model_map,
                'exported_at': datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"[Router] 配置已导出到 {path}")


# ==================== 便捷函数 ====================

_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """获取路由器单例"""
    global _router
    if _router is None:
        _router = ModelRouter.get_instance()
    return _router


def should_use_deep(
    confidence: Optional[float] = None,
    requires_ic_update: bool = False,
    has_factor_conflict: bool = False,
    is_critical_risk: bool = False,
    survival_tier: str = "normal",
    task_type: Optional[str] = None,
) -> bool:
    """
    判断是否需要深度思考 (快捷函数)

    See ModelRouter.should_use_deep() for details
    """
    router = get_router()
    task_type_enum = None
    if task_type:
        try:
            task_type_enum = TaskType(task_type)
        except ValueError:
            pass
    return router.should_use_deep(
        confidence=confidence,
        requires_ic_update=requires_ic_update,
        has_factor_conflict=has_factor_conflict,
        is_critical_risk=is_critical_risk,
        survival_tier=survival_tier,
        task_type=task_type_enum,
    )


def get_model(task_type: str) -> str:
    """获取任务对应的模型名称"""
    return get_router().get_model(task_type)


def get_cost_stats() -> Dict[str, Any]:
    """获取成本统计"""
    return get_router().get_cost_stats()


def record_usage(task_type: str, tokens_used: int, cost: float) -> None:
    """记录模型使用"""
    return get_router().record_usage(task_type, tokens_used, cost)


# ==================== 自检 ====================

if __name__ == '__main__':
    import pprint

    print("=== 模型路由器 (P1.5) 自检 ===\n")

    router = ModelRouter.get_instance()

    # 测试模型映射
    print("1. 模型映射表:")
    pprint.pprint(router.get_model_map())

    # 测试深度判断
    print("\n2. 深度判断测试:")
    test_cases = [
        {"confidence": 0.35, "requires_ic_update": False, "desc": "低置信度"},
        {"confidence": 0.6, "requires_ic_update": True, "desc": "IC权重更新"},
        {"confidence": 0.7, "has_factor_conflict": True, "desc": "多因子冲突"},
        {"confidence": 0.8, "is_critical_risk": True, "desc": "高风险场景"},
        {"confidence": 0.7, "survival_tier": "critical", "desc": "危险生存层级"},
        {"confidence": 0.8, "task_type": "debate_judge", "desc": "辩论裁决任务"},
        {"confidence": 0.8, "task_type": "bull_researcher", "desc": "多头研究任务"},
    ]

    for case in test_cases:
        task_type = case.get('task_type')
        result = router.should_use_deep(
            confidence=case.get('confidence'),
            requires_ic_update=case.get('requires_ic_update', False),
            has_factor_conflict=case.get('has_factor_conflict', False),
            is_critical_risk=case.get('is_critical_risk', False),
            survival_tier=case.get('survival_tier', 'normal'),
            task_type=TaskType(task_type) if task_type else None,
        )
        print(f"  - {case['desc']}: {'DEEP' if result else 'QUICK'}")

    # 测试模型推荐
    print("\n3. 决策场景模型推荐:")
    recommendations = [
        {"confidence": 0.35, "has_factor_conflict": False, "risk_level": "medium"},
        {"confidence": 0.7, "has_factor_conflict": True, "risk_level": "high"},
        {"confidence": 0.85, "has_factor_conflict": False, "risk_level": "low"},
    ]
    for rec in recommendations:
        result = router.get_recommended_model_for_decision(**rec)
        print(f"  - 置信度={rec['confidence']}, 冲突={rec['has_factor_conflict']}, 风险={rec['risk_level']}")
        print(f"    -> {result}")

    # 成本统计
    print("\n4. 成本统计:")
    pprint.pprint(router.get_cost_stats())

    print("\n=== 自检完成 ===")
