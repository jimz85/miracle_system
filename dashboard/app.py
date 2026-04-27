"""
FastAPI Web Dashboard - 实时行情与交易信号监控
提供Web界面展示实时行情、持仓、信号

运行: uvicorn dashboard.app:app --reload --port 8000
访问: http://localhost:8000
"""
import asyncio
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# 尝试导入项目模块
try:
    from core.exchange_adapter import create_exchange_adapter, ExchangeType
    from core.risk_management import CrossCurrencyRiskMonitor, Position
    HAS_EXCHANGE = True
except ImportError:
    HAS_EXCHANGE = False


# ========================
# 数据模型
# ========================

@dataclass
class Quote:
    """实时行情"""
    symbol: str
    last_price: float
    bid: float
    ask: float
    high_24h: float
    low_24h: float
    volume_24h: float
    change_24h: float
    change_pct_24h: float
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class Signal:
    """交易信号"""
    id: str
    symbol: str
    direction: str  # "buy" or "sell"
    strength: float  # 0-1
    indicators: Dict[str, float]
    reason: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class TradeStats:
    """交易统计"""
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    equity: float


# ========================
# 数据模拟器
# ========================

class DataSimulator:
    """模拟实时数据"""
    
    def __init__(self):
        self.prices = {
            "BTC-USDT": 67500,
            "ETH-USDT": 3450,
            "SOL-USDT": 145,
            "DOGE-USDT": 0.125,
            "XRP-USDT": 0.52
        }
    
    def get_quotes(self) -> List[Quote]:
        """获取模拟行情"""
        quotes = []
        for symbol, base_price in self.prices.items():
            # 模拟价格波动
            change = random.uniform(-0.02, 0.025)
            price = base_price * (1 + change)
            self.prices[symbol] = price
            
            quotes.append(Quote(
                symbol=symbol,
                last_price=price,
                bid=price * 0.9998,
                ask=price * 1.0002,
                high_24h=price * 1.03,
                low_24h=price * 0.97,
                volume_24h=random.uniform(1000000, 10000000),
                change_24h=price - base_price,
                change_pct_24h=change * 100
            ))
        
        return quotes
    
    def get_signals(self) -> List[Signal]:
        """生成模拟信号"""
        signals = []
        
        # 随机生成信号
        symbols = list(self.prices.keys())
        for _ in range(random.randint(0, 3)):
            symbol = random.choice(symbols)
            direction = random.choice(["buy", "sell"])
            strength = random.uniform(0.5, 0.95)
            
            signals.append(Signal(
                id=f"SIG-{int(time.time() * 1000)}",
                symbol=symbol,
                direction=direction,
                strength=strength,
                indicators={
                    "RSI": random.uniform(20, 80),
                    "MACD": random.uniform(-100, 100),
                    "Bollinger": random.uniform(0.4, 0.6)
                },
                reason=f"{direction.upper()} signal from {random.choice(['RSI', 'MACD', 'Bollinger'])} indicator"
            ))
        
        return sorted(signals, key=lambda s: s.timestamp, reverse=True)
    
    def get_positions(self) -> List[Position]:
        """生成模拟持仓"""
        positions = []
        
        # 模拟几个持仓
        for symbol in random.sample(list(self.prices.keys()), min(2, len(self.prices))):
            price = self.prices[symbol]
            entry = price * random.uniform(0.95, 1.05)
            
            positions.append(Position(
                symbol=symbol,
                side=random.choice(["long", "short"]),
                size=random.uniform(0.1, 1.0),
                entry_price=entry,
                current_price=price,
                unrealized_pnl=(price - entry) * random.uniform(0.5, 2),
                unrealized_pnl_pct=((price - entry) / entry) * 100
            ))
        
        return positions
    
    def get_stats(self) -> TradeStats:
        """获取交易统计"""
        return TradeStats(
            total_trades=random.randint(50, 200),
            win_rate=random.uniform(0.45, 0.65),
            total_pnl=random.uniform(-500, 2000),
            sharpe_ratio=random.uniform(0.5, 2.5),
            max_drawdown=random.uniform(5, 20),
            equity=10000 + random.uniform(-1000, 3000)
        )


# ========================
# FastAPI应用
# ========================

app = FastAPI(title="Miracle 2.0 Dashboard")
simulator = DataSimulator()

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        """广播消息到所有连接"""
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


# ========================
# API路由
# ========================

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回Dashboard HTML"""
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    
    # 内联HTML
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Miracle 2.0 Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-gray-900 text-white p-6">
        <div class="max-w-7xl mx-auto">
            <h1 class="text-4xl font-bold mb-6 text-blue-400">🎯 Miracle 2.0 Dashboard</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div class="bg-gray-800 p-4 rounded-lg">
                    <div class="text-gray-400">Equity</div>
                    <div class="text-2xl font-bold text-green-400" id="equity">$0</div>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg">
                    <div class="text-gray-400">Win Rate</div>
                    <div class="text-2xl font-bold text-blue-400" id="winrate">0%</div>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg">
                    <div class="text-gray-400">Sharpe Ratio</div>
                    <div class="text-2xl font-bold text-purple-400" id="sharpe">0</div>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg">
                    <div class="text-gray-400">Max Drawdown</div>
                    <div class="text-2xl font-bold text-red-400" id="drawdown">0%</div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div class="bg-gray-800 p-4 rounded-lg">
                    <h2 class="text-xl font-bold mb-4">📊 Market Quotes</h2>
                    <div id="quotes" class="space-y-2"></div>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg">
                    <h2 class="text-xl font-bold mb-4">📈 Latest Signals</h2>
                    <div id="signals" class="space-y-2"></div>
                </div>
            </div>
            
            <div class="bg-gray-800 p-4 rounded-lg">
                <h2 class="text-xl font-bold mb-4">💼 Positions</h2>
                <div id="positions" class="space-y-2"></div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket(`ws://${location.host}/ws`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'stats') {
                    document.getElementById('equity').textContent = '$' + data.equity.toFixed(2);
                    document.getElementById('winrate').textContent = (data.win_rate * 100).toFixed(1) + '%';
                    document.getElementById('sharpe').textContent = data.sharpe_ratio.toFixed(2);
                    document.getElementById('drawdown').textContent = data.max_drawdown.toFixed(1) + '%';
                }
                
                if (data.type === 'quotes') {
                    const container = document.getElementById('quotes');
                    container.innerHTML = data.quotes.map(q => `
                        <div class="flex justify-between items-center p-2 bg-gray-700 rounded">
                            <span class="font-mono">${q.symbol}</span>
                            <span class="text-green-400">$${q.last_price.toFixed(2)}</span>
                            <span class="${q.change_pct_24h >= 0 ? 'text-green-400' : 'text-red-400'}">
                                ${q.change_pct_24h >= 0 ? '+' : ''}${q.change_pct_24h.toFixed(2)}%
                            </span>
                        </div>
                    `).join('');
                }
                
                if (data.type === 'signals') {
                    const container = document.getElementById('signals');
                    container.innerHTML = data.signals.slice(0, 5).map(s => `
                        <div class="p-2 bg-gray-700 rounded flex justify-between items-center">
                            <div>
                                <span class="${s.direction === 'buy' ? 'text-green-400' : 'text-red-400'} font-bold">
                                    ${s.direction.toUpperCase()}
                                </span>
                                <span class="ml-2">${s.symbol}</span>
                            </div>
                            <span class="text-yellow-400">${(s.strength * 100).toFixed(0)}%</span>
                        </div>
                    `).join('') || '<div class="text-gray-500">No signals</div>';
                }
                
                if (data.type === 'positions') {
                    const container = document.getElementById('positions');
                    container.innerHTML = data.positions.map(p => `
                        <div class="p-2 bg-gray-700 rounded flex justify-between items-center">
                            <div>
                                <span class="${p.side === 'long' ? 'text-green-400' : 'text-red-400'}">${p.side.toUpperCase()}</span>
                                <span class="ml-2">${p.symbol}</span>
                                <span class="text-gray-400">${p.size}</span>
                            </div>
                            <span class="${p.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}">
                                $${p.unrealized_pnl.toFixed(2)} (${p.unrealized_pnl_pct.toFixed(2)}%)
                            </span>
                        </div>
                    `).join('') || '<div class="text-gray-500">No positions</div>';
                }
            };
            
            ws.onclose = () => {
                setTimeout(() => location.reload(), 3000);
            };
        </script>
    </body>
    </html>
    """


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点 - 实时推送数据"""
    await manager.connect(websocket)
    try:
        while True:
            # 广播数据
            stats = simulator.get_stats()
            quotes = simulator.get_quotes()
            signals = simulator.get_signals()
            positions = simulator.get_positions()
            
            await manager.broadcast({
                "type": "stats",
                "equity": stats.equity,
                "win_rate": stats.win_rate,
                "sharpe_ratio": stats.sharpe_ratio,
                "max_drawdown": stats.max_drawdown,
                "total_trades": stats.total_trades
            })
            
            await manager.broadcast({
                "type": "quotes",
                "quotes": [
                    {
                        "symbol": q.symbol,
                        "last_price": q.last_price,
                        "bid": q.bid,
                        "ask": q.ask,
                        "high_24h": q.high_24h,
                        "low_24h": q.low_24h,
                        "volume_24h": q.volume_24h,
                        "change_24h": q.change_24h,
                        "change_pct_24h": q.change_pct_24h
                    }
                    for q in quotes
                ]
            })
            
            await manager.broadcast({
                "type": "signals",
                "signals": [
                    {
                        "id": s.id,
                        "symbol": s.symbol,
                        "direction": s.direction,
                        "strength": s.strength,
                        "indicators": s.indicators,
                        "reason": s.reason,
                        "timestamp": s.timestamp
                    }
                    for s in signals
                ]
            })
            
            await manager.broadcast({
                "type": "positions",
                "positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "size": p.size,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "unrealized_pnl_pct": p.unrealized_pnl_pct
                    }
                    for p in positions
                ]
            })
            
            await asyncio.sleep(2)  # 2秒更新一次
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_stats():
    """
    获取交易统计
    
    Returns:
        Stats: 包含总交易数、胜率、夏普比率等
    """
    stats = simulator.get_stats()
    return {
        "total_trades": stats.total_trades,
        "win_rate": stats.win_rate,
        "total_pnl": stats.total_pnl,
        "sharpe_ratio": stats.sharpe_ratio,
        "max_drawdown": stats.max_drawdown,
        "equity": stats.equity
    }


@app.get("/api/quotes")
async def get_quotes():
    """
    获取实时行情
    
    Returns:
        Quotes: 所有交易对的实时价格数据
    """
    quotes = simulator.get_quotes()
    return {
        "quotes": [
            {
                "symbol": q.symbol,
                "last_price": q.last_price,
                "bid": q.bid,
                "ask": q.ask,
                "high_24h": q.high_24h,
                "low_24h": q.low_24h,
                "volume_24h": q.volume_24h,
                "change_24h": q.change_24h,
                "change_pct_24h": q.change_pct_24h,
                "timestamp": q.timestamp
            }
            for q in quotes
        ]
    }


@app.get("/api/signals")
async def get_signals():
    """
    获取交易信号
    
    Returns:
        Signals: 当前生成的交易信号列表
    """
    signals = simulator.get_signals()
    return {"signals": signals}


@app.get("/api/positions")
async def get_positions():
    """
    获取当前持仓
    
    Returns:
        Positions: 当前开仓的持仓列表
    """
    positions = simulator.get_positions()
    return {"positions": positions}


# ========================
# 主函数
# ========================

def run():
    """运行Dashboard"""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
