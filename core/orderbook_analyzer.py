#!/usr/bin/env python3
"""
orderbook_analyzer.py - OKX Level2 订单簿分析器
===============================================

从订单簿数据提取微观结构信号:
  - 买卖盘压力比 (Bid/Ask Volume Ratio)
  - 加权价格偏离 (Weighted Price Pressure)
  - 大单墙检测 (Large Wall Detection)
  - 价差分析 (Spread Analysis)

这些信号是纯技术指标之外的独立信息源，
反映真实市场参与者的挂单意图和短期供需。

依赖: 无外部依赖（仅numpy）
数据源: OKX GET /api/v5/market/books (Level2, 200档)
        或 /api/v5/market/books-lite (轻量版, 5档)

用法:
    from core.orderbook_analyzer import OrderBookAnalyzer, compute_orderbook_signal
    
    # 从已有OrderBook对象
    signal = compute_orderbook_signal(bids, asks)
    
    # 或创建分析器实例
    analyzer = OrderBookAnalyzer(depth_levels=10)
    result = analyzer.analyze(bids, asks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OrderBookSignal:
    """订单簿信号结果"""
    # 核心信号 (-1~1, 负=空方主导, 正=多方主导)
    pressure_score: float = 0.0
    
    # 详细指标
    bid_ask_ratio: float = 1.0       # 买/卖量比 (>1=多方)
    weighted_pressure: float = 0.0    # 加权的买卖压力
    spread_pct: float = 0.0          # 价差百分比
    large_wall_bid: float = 0.0      # 买方大单墙强度
    large_wall_ask: float = 0.0      # 卖方大单墙强度
    mid_price: float = 0.0           # 中间价
    
    # 大单墙价格（支撑阻力位）
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # 元信息
    n_bids: int = 0
    n_asks: int = 0
    total_bid_size: float = 0.0
    total_ask_size: float = 0.0
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pressure_score": round(self.pressure_score, 4),
            "bid_ask_ratio": round(self.bid_ask_ratio, 4),
            "weighted_pressure": round(self.weighted_pressure, 4),
            "spread_pct": round(self.spread_pct, 4),
            "large_wall_bid": round(self.large_wall_bid, 4),
            "large_wall_ask": round(self.large_wall_ask, 4),
            "mid_price": round(self.mid_price, 2),
            "support_levels": [round(p, 2) for p in self.support_levels[:3]],
            "resistance_levels": [round(p, 2) for p in self.resistance_levels[:3]],
            "n_bids": self.n_bids,
            "n_asks": self.n_asks,
            "confidence": round(self.confidence, 4),
        }


class OrderBookAnalyzer:
    """订单簿微观结构分析器
    
    从Level2订单簿数据提取交易信号：
    1. 累计买卖量比（排除极值）
    2. 价格加权压力（靠近当前价的单子权重更高）
    3. 大单墙检测（超越均值2个标准差的挂单）
    4. 价差分析（宽度+深度）
    
    Args:
        depth_levels: 分析的深度档位数（默认20）
        wall_std_threshold: 大单墙标准差倍数（默认2.0）
        min_wall_size_ratio: 大单最小占总量的比例（默认0.05）
    """
    
    def __init__(
        self,
        depth_levels: int = 20,
        wall_std_threshold: float = 2.0,
        min_wall_size_ratio: float = 0.05,
    ):
        self.depth = max(5, min(200, depth_levels))
        self.wall_std = wall_std_threshold
        self.min_wall_ratio = min_wall_size_ratio
    
    def analyze(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
    ) -> OrderBookSignal:
        """分析订单簿
        
        Args:
            bids: 买单列表 [[price, size], ...]
            asks: 卖单列表 [[price, size], ...]
        
        Returns:
            OrderBookSignal: 信号结果
        """
        # 数据验证
        if not bids or not asks:
            return OrderBookSignal()
        
        # 限制深度
        bids = bids[:self.depth]
        asks = asks[:self.depth]
        
        # 解析价格和量
        bid_prices = np.array([b[0] for b in bids])
        bid_sizes = np.array([b[1] for b in bids])
        ask_prices = np.array([a[0] for a in asks])
        ask_sizes = np.array([a[1] for a in asks])
        
        # 中间价
        best_bid = float(bid_prices[0])
        best_ask = float(ask_prices[0])
        mid_price = (best_bid + best_ask) / 2.0
        
        # 价差百分比
        spread = best_ask - best_bid
        spread_pct = spread / mid_price if mid_price > 0 else 0.0
        
        # ---- 1. 累计买卖量比 ----
        total_bid = float(np.sum(bid_sizes))
        total_ask = float(np.sum(ask_sizes))
        bid_ask_ratio = total_bid / total_ask if total_ask > 0 else 1.0
        
        # 归一化到 -1~1（log尺度, ratio在0.5~2范围对应 -0.5~0.5）
        pressure_volume = np.clip((bid_ask_ratio - 1.0) / 1.5, -1.0, 1.0)
        
        # ---- 2. 加权价格压力 ----
        # 越靠近当前价的单子权重越高（指数衰减）
        bid_distance = np.abs(bid_prices - mid_price) / (spread + 1e-10)
        ask_distance = np.abs(ask_prices - mid_price) / (spread + 1e-10)
        bid_weight = np.exp(-bid_distance / 3.0)
        ask_weight = np.exp(-ask_distance / 3.0)
        
        weighted_bid = float(np.sum(bid_sizes * bid_weight))
        weighted_ask = float(np.sum(ask_sizes * ask_weight))
        weighted_pressure = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)
        
        # ---- 3. 大单墙检测 ----
        all_sizes = np.concatenate([bid_sizes, ask_sizes])
        mean_size = float(np.mean(all_sizes))
        std_size = float(np.std(all_sizes)) + 1e-10
        wall_threshold = max(mean_size + std_size * self.wall_std, 
                           total_bid * self.min_wall_ratio)
        
        # 买方大单墙
        bid_walls = bid_sizes > wall_threshold
        wall_bid_strength = float(np.sum(bid_sizes[bid_walls]) / (total_bid + 1e-10))
        
        # 卖方大单墙
        ask_walls = ask_sizes > wall_threshold
        wall_ask_strength = float(np.sum(ask_sizes[ask_walls]) / (total_ask + 1e-10))
        
        # 大单墙价格（支撑阻力位）
        support_levels = [float(p) for p, s in zip(bid_prices[bid_walls], bid_sizes[bid_walls])]
        resistance_levels = [float(p) for p, s in zip(ask_prices[ask_walls], ask_sizes[ask_walls])]
        
        # ---- 4. 综合信号 ----
        # 加权组合: 价格压力(0.5) + 量比(0.3) + 大单墙(0.2)
        pressure_score = (
            weighted_pressure * 0.5 +
            pressure_volume * 0.3 +
            (wall_bid_strength - wall_ask_strength) * 0.2
        )
        pressure_score = float(np.clip(pressure_score, -1.0, 1.0))
        
        # ---- 5. 置信度 ----
        # 深度越浅→置信度越低；价差越大→流动性差→置信度低
        depth_factor = min(1.0, self.depth / 10.0)
        spread_factor = max(0.2, 1.0 - spread_pct * 100)
        total_depth = total_bid + total_ask
        size_factor = min(1.0, total_depth / (mid_price * 10)) if mid_price > 0 else 0.5
        confidence = min(1.0, depth_factor * spread_factor * (0.5 + size_factor * 0.5))
        confidence = max(0.1, confidence)
        
        return OrderBookSignal(
            pressure_score=pressure_score,
            bid_ask_ratio=round(bid_ask_ratio, 4),
            weighted_pressure=round(weighted_pressure, 4),
            spread_pct=round(spread_pct, 6),
            large_wall_bid=round(wall_bid_strength, 4),
            large_wall_ask=round(wall_ask_strength, 4),
            mid_price=mid_price,
            support_levels=sorted(support_levels, reverse=True)[:5],
            resistance_levels=sorted(resistance_levels)[:5],
            n_bids=len(bids),
            n_asks=len(asks),
            total_bid_size=round(total_bid, 2),
            total_ask_size=round(total_ask, 2),
            confidence=confidence,
        )
    
    def signal_to_direction(self, signal: OrderBookSignal) -> Tuple[Optional[str], float]:
        """将信号转换为交易方向
        
        Returns:
            (direction, strength): ("long"/"short"/None, 0~1)
        """
        score = signal.pressure_score
        conf = signal.confidence
        
        if score > 0.3 and conf > 0.4:
            return "long", min(1.0, score * conf)
        elif score < -0.3 and conf > 0.4:
            return "short", min(1.0, abs(score) * conf)
        else:
            return None, 0.0
    
    def __repr__(self) -> str:
        return f"OrderBookAnalyzer(depth={self.depth}, wall_std={self.wall_std})"


def compute_orderbook_signal(
    bids: List[List[float]],
    asks: List[List[float]],
    depth: int = 20,
) -> OrderBookSignal:
    """便捷函数：一步计算订单簿信号"""
    analyzer = OrderBookAnalyzer(depth_levels=depth)
    return analyzer.analyze(bids, asks)
