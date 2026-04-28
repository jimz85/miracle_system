from __future__ import annotations

from .fusion_decision import FusionDecision


class FusionDecisionRenderer:
    """将FusionDecision渲染为Markdown格式"""

    @staticmethod
    def render(decision: FusionDecision) -> str:
        parts = [
            f"**Action**: {decision.action.value}",
            f"**Rating**: {decision.rating.value}",
            f"**Confidence**: {decision.confidence:.1%}",
            "",
            f"**Reasoning**: {decision.reasoning}",
        ]

        if decision.entry_price is not None:
            parts.extend(["", f"**Entry Price**: {decision.entry_price}"])
        if decision.stop_loss is not None:
            parts.extend(["", f"**Stop Loss**: {decision.stop_loss}"])
        if decision.take_profit is not None:
            parts.extend(["", f"**Take Profit**: {decision.take_profit}"])
        if decision.position_size is not None:
            parts.extend(["", f"**Position Size**: {decision.position_size}"])

        if decision.bull_evidence:
            parts.extend(["", "**Bull Evidence**:"])
            for evidence in decision.bull_evidence:
                parts.append(f"- {evidence}")

        if decision.bear_evidence:
            parts.append("**Bear Evidence**:")
            for evidence in decision.bear_evidence:
                parts.append(f"- {evidence}")

        if decision.debate_insights:
            parts.extend(["", "**Debate Insights**:"])
            for insight in decision.debate_insights:
                parts.append(f"- {insight}")

        if decision.ic_weights:
            parts.extend(["", "**IC Weights**:"])
            for factor, weight in decision.ic_weights.items():
                parts.append(f"- {factor}: {weight:.4f}")

        if decision.factor_signals:
            parts.extend(["", "**Factor Signals**:"])
            for factor, signal in decision.factor_signals.items():
                parts.append(f"- {factor}: {signal:.4f}")

        parts.extend(["", f"**Risk Level**: {decision.risk_level}"])
        parts.extend(["", f"**Survival Tier**: {decision.survival_tier}"])

        return "\n".join(parts)
