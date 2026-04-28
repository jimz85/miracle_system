from __future__ import annotations

"""
Multi-Agent Debate Layer for Miracle+TradingAgents Fusion

辩论层实现Bull/Bear Researcher + Debate Judge三Agent辩论机制。
支持并行多空分析，通过结构化辩论识别市场矛盾点。

模块:
    - BullResearcher: 多头论点研究
    - BearResearcher: 空头论点研究
    - DebateJudge: 辩论裁决
    - FusionDebateLayer: 辩论层编排器

Usage:
    from agents.debate import FusionDebateLayer

    layer = FusionDebateLayer(llm_manager)
    result = await layer.run_debate(debate_input)
"""

import logging
from typing import List

from .bear_researcher import BearResearcher
from .bull_researcher import BullResearcher
from .debate_judge import DebateJudge, DebateVerdict
from .fusion_debate import (
    DebateInput,
    DebateOutput,
    FusionDebateLayer,
    ResearchResult,
    VerdictResult,
)

__all__ = [
    "BullResearcher",
    "BearResearcher",
    "DebateJudge",
    "FusionDebateLayer",
    "DebateInput",
    "DebateOutput",
    "ResearchResult",
    "VerdictResult",
    "DebateVerdict",
]

logger = logging.getLogger(__name__)