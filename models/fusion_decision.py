from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class TradeAction(str, Enum):
    """交易动作枚举"""
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    WAIT = "Wait"


class PositionRating(str, Enum):
    """仓位评级枚举"""
    STRONG_BUY = "StrongBuy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "StrongSell"


class FusionDecision(BaseModel):
    """融合系统统一决策格式"""

    action: TradeAction = Field(
        description="交易动作: Buy / Hold / Sell / Wait"
    )
    rating: PositionRating = Field(
        description="仓位评级: StrongBuy / Buy / Hold / Sell / StrongSell"
    )
    confidence: float = Field(
        description="置信度 0.0-1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="决策理由，2-4句话总结核心逻辑"
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="推荐入场价格"
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="止损价格"
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="止盈价格"
    )
    position_size: Optional[str] = Field(
        default=None,
        description="仓位大小，如 '5% of portfolio'"
    )
    bull_evidence: list[str] = Field(
        default_factory=list,
        description="多头证据列表"
    )
    bear_evidence: list[str] = Field(
        default_factory=list,
        description="空头证据列表"
    )
    debate_insights: list[str] = Field(
        default_factory=list,
        description="辩论关键洞察"
    )
    ic_weights: dict[str, float] = Field(
        default_factory=dict,
        description="因子IC权重"
    )
    factor_signals: dict[str, float] = Field(
        default_factory=dict,
        description="各因子信号值"
    )
    risk_level: str = Field(
        description="风险等级: low / medium / high / critical"
    )
    survival_tier: str = Field(
        description="生存层级: normal / caution / low / critical / paused"
    )
