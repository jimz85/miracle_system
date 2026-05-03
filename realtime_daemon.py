#!/usr/bin/env python3
"""
Miracle Realtime Daemon — 实时看盘+思考系统
=============================================

架构:
  OKX WebSocket ──→ Ticker数据流 ──→ 价格窗口滚动
                                          │
                    ┌─────────────────────┴────────────────────┐
                    ▼                                         ▼
           事件触发引擎                               RSI/ADX/布林带计算
                    │                                         │
                    ▼                                         ▼
           LLM异步分析队列                          信号生成器
                    │                                         │
                    └─────────────────┬───────────────────────┘
                                      ▼
                              交易决策执行器
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
               开仓+OCO            调整OCO           平仓

用法:
  python realtime_daemon.py                    # 前台运行
  python realtime_daemon.py --daemon           # 后台守护
  python realtime_daemon.py --status          # 查看状态
  python realtime_daemon.py --stop             # 停止守护进程
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import socket
import atexit
import logging
import threading
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

# WebSocket client
import websocket

# ============================================================
# 配置
# ============================================================

WORKSPACE = Path(__file__).parent
LOCK_FILE = WORKSPACE / "realtime_daemon.lock"
PID_FILE = WORKSPACE / "realtime_daemon.pid"
LOG_FILE = WORKSPACE / "logs" / "realtime_daemon.log"
STATE_DIR = WORKSPACE / "data"
STATE_DIR.mkdir(exist_ok=True)
(WORKSPACE / "logs").mkdir(exist_ok=True)

# OKX WebSocket
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
SIM_FLAG = os.getenv('OKX_FLAG', '1')
if SIM_FLAG == '1':
    OKX_WS_URL += '?x-simulated-trading=1'

# 订阅的交易对 (只监控有持仓的 + 候选币)
WATCHED_COINS = [
    "DOGE-USDT-SWAP",
    "BNB-USDT-SWAP",
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "FIL-USDT-SWAP",
]

# WebSocket 频道
# ticker: 每秒实时价格 (用于心跳+实时监控)
# candle5m: 5分钟K线 (用于RSI/ADX计算)
WS_TICKER_CHANNEL = "tickers"        # 每秒价格推送
WS_CANDLE_CHANNEL = "candle5m"     # 5分钟K线（每5分钟一条）

# 实时 RSI 参数
RSI_PERIOD = 14          # RSI周期
RSI_LONG_THRESHOLD = 35  # RSI低于35 → 超卖 → 做多信号
RSI_SHORT_THRESHOLD = 65  # RSI高于65 → 超买 → 做空信号
PRICE_WINDOW = 100        # 价格窗口 (ticker次数，约100秒)

# 信号冷却 (秒) — 防止信号过于频繁
SIGNAL_COOLDOWN = 60      # 同一币种同一方向，60秒内不重复触发

# LLM分析队列
LLM_QUEUE_SIZE = 10
LLM_TIMEOUT = 30          # LLM分析超时秒数

# 重连参数
RECONNECT_DELAY = 3        # 重连延迟秒
MAX_RECONNECT_ATTEMPTS = 20
HEARTBEAT_INTERVAL = 60   # ping保活秒数 (OKX服务器60s发一次ping)

# ============================================================
# 日志
# ============================================================

logger = logging.getLogger("RealtimeDaemon")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ============================================================
# 价格窗口 — 滚动窗口计算RSI
# ============================================================

class PriceWindow:
    """滚动价格窗口，支持RSI/ADX/布林带计算"""

    def __init__(self, symbol: str, max_size: int = 200):
        self.symbol = symbol
        self.max_size = max_size
        self.prices: list[float] = []   # 收盘价序列
        self.highs: list[float] = []     # 最高价序列
        self.lows: list[float] = []      # 最低价序列
        self.timestamps: list[str] = []
        self._lock = threading.Lock()

    def update(self, price: float, high: float = None, low: float = None, ts: str = None):
        with self._lock:
            self.prices.append(price)
            self.highs.append(high if high is not None else price)
            self.lows.append(low if low is not None else price)
            self.timestamps.append(ts or datetime.now().isoformat())
            if len(self.prices) > self.max_size:
                self.prices.pop(0)
                self.highs.pop(0)
                self.lows.pop(0)
                self.timestamps.pop(0)

    def update_last_price(self, price: float):
        """仅更新最新价格（用于ticker频道，不影响RSI计算）"""
        with self._lock:
            if self.prices:
                self.prices[-1] = price
            else:
                self.prices.append(price)
                self.highs.append(price)
                self.lows.append(price)
                self.timestamps.append(datetime.now().isoformat())

    def get_rsi(self, period: int = 14) -> Optional[float]:
        with self._lock:
            if len(self.prices) < period + 1:
                return None
            prices = self.prices[-period-1:]
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d for d in deltas if d > 0]
            losses = [-d for d in deltas if d < 0]
            avg_gain = statistics.mean(gains) if gains else 0
            avg_loss = statistics.mean(losses) if losses else 0
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

    def get_bollinger(self, period: int = 20, std_mult: float = 2.0) -> Optional[dict]:
        with self._lock:
            if len(self.prices) < period:
                return None
            window = self.prices[-period:]
            mean = statistics.mean(window)
            std = statistics.stdev(window) if len(window) > 1 else 0
            return {
                'middle': mean,
                'upper': mean + std_mult * std,
                'lower': mean - std_mult * std,
                'position': (self.prices[-1] - mean) / (std if std > 0 else 1),
            }

    def get_adx(self, period: int = 14) -> Optional[dict]:
        """简化ADX计算"""
        with self._lock:
            if len(self.prices) < period + 1:
                return None
            highs = self.highs[-period-1:]
            lows = self.lows[-period-1:]
            closes = self.prices[-period-1:]

            # +DM, -DM, TR
            tr_list = []
            plus_dm = []
            minus_dm = []
            for i in range(1, len(highs)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                up = highs[i] - highs[i-1]
                down = lows[i-1] - lows[i]
                tr_list.append(tr)
                plus_dm.append(up if up > down and up > 0 else 0)
                minus_dm.append(down if down > up and down > 0 else 0)

            if len(tr_list) < period:
                return None

            # Smooth
            def smooth(data):
                result = []
                for i in range(len(data)):
                    if i < period - 1:
                        result.append(None)
                    else:
                        val = sum(data[i-period+1:i+1]) / period
                        result.append(val)
                return result[period-1:]

            tr_smooth = sum(tr_list[-period:]) / period
            plus_smooth = sum(plus_dm[-period:]) / period
            minus_smooth = sum(minus_dm[-period:]) / period

            if tr_smooth == 0:
                return {'adx': 0, 'plus_di': 0, 'minus_di': 0}

            plus_di = 100 * plus_smooth / tr_smooth
            minus_di = 100 * minus_smooth / tr_smooth

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0

            return {
                'adx': dx,  # 简化: 直接返回DX而非平滑ADX
                'plus_di': plus_di,
                'minus_di': minus_di,
            }

    @property
    def last_price(self) -> Optional[float]:
        with self._lock:
            return self.prices[-1] if self.prices else None

    def __len__(self):
        with self._lock:
            return len(self.prices)


# ============================================================
# 信号管理器 — 防止信号风暴
# ============================================================

class SignalManager:
    """管理信号冷却，防止同一币种信号过于频繁"""

    def __init__(self):
        self._lock = threading.Lock()
        # {(symbol, direction): last_trigger_time}
        self._last_signal: dict[tuple, float] = {}

    def can_fire(self, symbol: str, direction: str) -> bool:
        """检查是否可以触发信号"""
        with self._lock:
            key = (symbol, direction)
            now = time.time()
            last = self._last_signal.get(key, 0)
            if now - last < SIGNAL_COOLDOWN:
                return False
            self._last_signal[key] = now
            return True

    def get_cooldown_remaining(self, symbol: str, direction: str) -> float:
        """获取剩余冷却时间"""
        with self._lock:
            key = (symbol, direction)
            last = self._last_signal.get(key, 0)
            remaining = SIGNAL_COOLDOWN - (time.time() - last)
            return max(0, remaining)


# ============================================================
# LLM 分析器 — 预热+异步队列
# ============================================================

class LLMAnalyzer:
    """
    LLM 异步分析器

    设计: 每次WebSocket tick不调LLM，只在信号触发时调。
    LLM平时预热在后台，有信号时立即分析。
    """

    def __init__(self):
        self._queue: list[dict] = []
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._results: dict[str, dict] = {}  # symbol → analysis result
        self._pending_events: dict[str, threading.Event] = {}

    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("LLM分析器已启动")

    def stop(self):
        self._running = False

    def submit(self, symbol: str, direction: str, price: float,
               rsi: float, adx: float, bb_pos: float,
               entry_price: float, pnl_pct: float, hold_hours: float,
               price_data: dict):
        """提交分析请求"""
        with self._lock:
            if len(self._queue) >= LLM_QUEUE_SIZE:
                logger.warning(f"LLM队列满({LLM_QUEUE_SIZE})，丢弃最旧请求")
                self._queue.pop(0)
            event = threading.Event()
            self._pending_events[symbol] = event
            self._queue.append({
                'symbol': symbol,
                'direction': direction,
                'price': price,
                'rsi': rsi,
                'adx': adx,
                'bb_pos': bb_pos,
                'entry_price': entry_price,
                'pnl_pct': pnl_pct,
                'hold_hours': hold_hours,
                'price_data': price_data,
                'event': event,
                'submitted_at': time.time(),
            })

    def get_result(self, symbol: str, timeout: float = LLM_TIMEOUT) -> Optional[dict]:
        """阻塞获取分析结果"""
        with self._lock:
            event = self._pending_events.get(symbol)
        if event is None:
            return self._results.get(symbol)
        if event.wait(timeout=timeout):
            return self._results.get(symbol)
        return None

    def _worker(self):
        while self._running:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)
            if item:
                try:
                    result = self._analyze(item)
                    with self._lock:
                        self._results[item['symbol']] = result
                    item['event'].set()
                except Exception as e:
                    logger.error(f"LLM分析失败: {e}")
                    item['event'].set()
            else:
                time.sleep(0.5)

    def _analyze(self, item: dict) -> dict:
        """实际调用LLM分析"""
        symbol = item['symbol']
        direction = item['direction']
        price = item['price']
        rsi = item['rsi']
        adx = item['adx']
        bb_pos = item['bb_pos']
        entry_price = item['entry_price']
        pnl_pct = item['pnl_pct']
        hold_hours = item['hold_hours']
        price_data = item['price_data']

        # 构建上下文
        context = (
            f"币种: {symbol.replace('-USDT-SWAP','')}\n"
            f"当前价格: {price:.6f}\n"
            f"入场价: {entry_price:.6f}\n"
            f"浮盈亏: {pnl_pct:+.2%}\n"
            f"持仓时间: {hold_hours:.1f}小时\n"
            f"RSI(14): {rsi:.1f}\n"
            f"ADX: {adx:.1f}\n"
            f"布林带位置: {bb_pos:+.2f}\n"
            f"24h高: {price_data.get('high24h','N/A')}\n"
            f"24h低: {price_data.get('low24h','N/A')}\n"
            f"24h开: {price_data.get('open24h','N/A')}\n"
        )

        prompt = (
            "你是一个有10年经验的加密货币交易员。基于以下市场数据给出持仓建议。\n"
            "只输出JSON，不要其他任何内容。\n\n"
            f"=== 市场数据 ===\n{context}\n\n"
            "=== 输出格式 ===\n"
            '{"judgment":"bullish/bearish/neutral","action":"hold/close/partial_tp",'
            '"sl_price":null或数字,"tp_price":null或数字,"reason":"一句话理由"}'
        )

        # 优先: NVIDIA Qwen3 Next 80B (免费，3s)
        nvidia_key = os.environ.get('NVIDIA_API_KEY', '')
        if nvidia_key:
            try:
                import requests as req
                resp = req.post(
                    'https://integrate.api.nvidia.com/v1/chat/completions',
                    headers={
                        "Authorization": f"Bearer {nvidia_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "qwen/qwen3-next-80b-a3b-instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 150,
                    },
                    timeout=LLM_TIMEOUT,
                )
                if resp.status_code == 200:
                    result_text = resp.json()['choices'][0]['message']['content']
                    import re
                    match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if match:
                        return json.loads(match.group())
            except Exception as e:
                logger.warning(f"NVIDIA API失败，回退到本地Gemma: {e}")

        # 后备: 本地Gemma (如果可用)
        try:
            return self._gemma_fallback(context, symbol)
        except Exception as e:
            logger.error(f"Gemma分析也失败: {e}")
            return {
                'judgment': 'neutral',
                'action': 'hold',
                'reason': f'LLM不可用: {e}',
                'model': 'fallback'
            }

    def _gemma_fallback(self, context: str, symbol: str) -> dict:
        """本地Gemma后备"""
        # 尝试用gemma4分析
        try:
            # 检查是否有gemma模型可用
            # 这里用简单规则作为fallback
            return {
                'judgment': 'neutral',
                'action': 'hold',
                'reason': 'gemma_fallback',
                'model': 'gemma_fallback'
            }
        except Exception:
            raise


# ============================================================
# OKX WebSocket 客户端
# ============================================================

class OKXWebSocketClient:
    """
    OKX WebSocket实时价格客户端

    功能:
    - 自动重连
    - 心跳保活
    - 价格窗口滚动更新
    - 信号触发检测
    """

    def __init__(self, on_signal: Callable):
        self.on_signal = on_signal  # (symbol, direction, price_data) → None
        self.url = OKX_WS_URL
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_count = 0
        self._last_ping = 0
        self._price_windows: dict[str, PriceWindow] = {}
        self._lock = threading.Lock()

        for coin in WATCHED_COINS:
            self._price_windows[coin] = PriceWindow(coin, max_size=PRICE_WINDOW)

        self._signal_manager = SignalManager()
        self._llm_analyzer = LLMAnalyzer()

    def start(self):
        self._running = True
        self._llm_analyzer.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket客户端已启动，监控: {WATCHED_COINS}")

    def stop(self):
        self._running = False
        self._llm_analyzer.stop()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        logger.info("WebSocket客户端已停止")

    def _run(self):
        while self._running and self._reconnect_count < MAX_RECONNECT_ATTEMPTS:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"WebSocket连接异常: {e}")
                self._reconnect_count += 1
                wait = RECONNECT_DELAY * min(self._reconnect_count, 10)
                logger.info(f"{wait}秒后重连 ({self._reconnect_count}/{MAX_RECONNECT_ATTEMPTS})")
                time.sleep(wait)

        if self._reconnect_count >= MAX_RECONNECT_ATTEMPTS:
            logger.error("达到最大重连次数，WebSocket客户端退出")

    def _connect(self):
        self._ws = websocket.WebSocketApp(
            self.url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._reconnect_count = 0  # 重置计数
        # 使用ping_timeout保持连接
        self._ws.run_forever()  # auto-reconnect on disconnect

    def _on_open(self, ws):
        # 订阅两个频道：ticker(实时价格) + candle5m(R SI计算)
        args = (
            [{"channel": WS_TICKER_CHANNEL, "instId": inst_id} for inst_id in WATCHED_COINS] +
            [{"channel": WS_CANDLE_CHANNEL, "instId": inst_id} for inst_id in WATCHED_COINS]
        )
        ws.send(json.dumps({"op": "subscribe", "args": args}))
        logger.info(f"已订阅 {len(WATCHED_COINS)} 币种 × 2频道 (ticker + candle5m)")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # 忽略subscribe确认消息
            if 'event' in data:
                return

            arg = data.get('arg', {})
            channel = arg.get('channel')
            inst_id = arg.get('instId')

            if not inst_id or inst_id not in self._price_windows:
                return

            if channel == WS_TICKER_CHANNEL:
                # ticker: [lastPx, askPx, bidPx, ...] — 实时价格
                ticker_list = data.get('data', [])
                if not ticker_list:
                    return
                ticker = ticker_list[0]
                last_price = float(ticker.get('last', 0))
                if last_price <= 0:
                    return
                # ticker 更新当前价格到窗口（用于心跳显示）
                window = self._price_windows[inst_id]
                window.update_last_price(last_price)

            elif channel == WS_CANDLE_CHANNEL:
                # candle5m: [ts, open, high, low, close, vol] — K线
                candle_list = data.get('data', [])
                if not candle_list:
                    return
                candle = candle_list[0]
                ts = candle[0]
                open_px = float(candle[1])
                high_px = float(candle[2])
                low_px = float(candle[3])
                close_px = float(candle[4])
                if close_px <= 0:
                    return
                window = self._price_windows[inst_id]
                window.update(close_px, high_px, low_px, ts)
                # 每收到K线就检测信号（5分钟一次）
                self._check_signals(inst_id, window)

        except Exception as e:
            logger.error(f"处理消息失败: {e}")

    def _check_signals(self, inst_id: str, window: PriceWindow):
        """检测交易信号"""
        if len(window) < 20:  # 数据不够，跳过
            return

        rsi = window.get_rsi(RSI_PERIOD)
        adx_data = window.get_adx(RSI_PERIOD)
        bb_data = window.get_bollinger(20)
        last_price = window.last_price

        if rsi is None or adx_data is None or bb_data is None:
            return

        adx = adx_data['adx']
        bb_pos = bb_data['position']

        price_data = {
            'rsi': rsi,
            'adx': adx,
            'bb_pos': bb_pos,
            'bb_upper': bb_data['upper'],
            'bb_lower': bb_data['lower'],
            'bb_middle': bb_data['middle'],
        }

        direction = None

        # 买入信号: RSI < 35 且 ADX > 20 (趋势明确)
        if rsi < RSI_LONG_THRESHOLD and adx > 20:
            if self._signal_manager.can_fire(inst_id, 'long'):
                direction = 'long'
                logger.info(
                    f"💚 BUY SIGNAL [{inst_id}] RSI={rsi:.1f}<{RSI_LONG_THRESHOLD} "
                    f"ADX={adx:.1f}>20 | 价格={last_price}"
                )
                self._trigger_signal(inst_id, direction, last_price, rsi, adx, bb_pos, price_data)

        # 卖出信号: RSI > 65 且 ADX > 20
        elif rsi > RSI_SHORT_THRESHOLD and adx > 20:
            if self._signal_manager.can_fire(inst_id, 'short'):
                direction = 'short'
                logger.info(
                    f"💔 SELL SIGNAL [{inst_id}] RSI={rsi:.1f}>{RSI_SHORT_THRESHOLD} "
                    f"ADX={adx:.1f}>20 | 价格={last_price}"
                )
                self._trigger_signal(inst_id, direction, last_price, rsi, adx, bb_pos, price_data)

    def _trigger_signal(self, inst_id: str, direction: str, price: float,
                        rsi: float, adx: float, bb_pos: float, price_data: dict):
        """触发信号处理"""
        # 读取持仓信息
        position_info = self._get_position_info(inst_id)

        if position_info:
            # 有持仓: 调用LLM分析持仓管理
            entry_price = position_info.get('entry_price', price)
            pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
            hold_hours = position_info.get('hold_hours', 0)
            self._llm_analyzer.submit(
                symbol=inst_id,
                direction=direction,
                price=price,
                rsi=rsi,
                adx=adx,
                bb_pos=bb_pos,
                entry_price=entry_price,
                pnl_pct=pnl_pct,
                hold_hours=hold_hours,
                price_data=price_data,
            )
            # 非阻塞获取结果（只在需要时）
            result = self._llm_analyzer.get_result(inst_id, timeout=5)
            if result:
                self._execute_management(inst_id, result, position_info)
                logger.info(f"📊 LLM分析 [{inst_id}]: {result}")
        else:
            # 无持仓: 触发开仓信号
            self.on_signal(inst_id, direction, price, {
                'rsi': rsi,
                'adx': adx,
                'bb_pos': bb_pos,
                'price_data': price_data,
            })

    def _get_position_info(self, inst_id: str) -> Optional[dict]:
        """从local_state获取持仓信息"""
        try:
            state_file = STATE_DIR / "local_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                if inst_id in state:
                    pos = state[inst_id]
                    return {
                        'direction': pos.get('direction'),
                        'entry_price': pos.get('entry_price', 0),
                        'contracts': pos.get('contracts', 0),
                        'hold_hours': 0,  # TODO: 计算持仓时间
                    }
        except Exception:
            pass
        return None

    def _execute_management(self, inst_id: str, result: dict, position_info: dict):
        """执行持仓管理决策"""
        action = result.get('action', 'hold')
        sl_price = result.get('sl_price')
        tp_price = result.get('tp_price')

        if action == 'hold':
            return  # 什么都不做

        logger.info(
            f"📋 持仓管理决策 [{inst_id}]: action={action} "
            f"sl={sl_price} tp={tp_price} reason={result.get('reason','')}"
        )

        try:
            from miracle_kronos import (
                close_position, adjust_trailing_oco,
                okx_req, set_coin_cooldown, clear_coin_cooldown
            )

            if action == 'close':
                # 平仓
                coin = inst_id.replace('-USDT-SWAP', '')
                res = close_position(coin, reason=f'llm_signal:{result.get("reason","")}', pos=position_info)
                if res.get('code') == '0':
                    logger.info(f"✅ 平仓成功 [{inst_id}]")
                    # 亏损后设置cooldown
                    pnl_pct = result.get('pnl_pct', 0)
                    if pnl_pct < 0:
                        set_coin_cooldown(coin, position_info.get('direction', 'long'))
                    # 更新local_state
                    self._remove_from_local_state(inst_id)
                else:
                    logger.warning(f"❌ 平仓失败 [{inst_id}]: {res.get('msg')}")

            elif action == 'partial_tp':
                # 部分止盈: 平50% + 调整OCO
                coin = inst_id.replace('-USDT-SWAP', '')
                res_close = close_position(coin, reason='partial_tp', pos=position_info, close_pct=0.5)
                if res_close.get('code') == '0':
                    logger.info(f"✅ 部分止盈50% [{inst_id}]")
                    # 调整OCO
                    entry = position_info.get('entry_price', 0)
                    current = position_info.get('last_price', 0)
                    if entry > 0 and current > 0:
                        res_adj = adjust_trailing_oco(inst_id, position_info.get('direction', 'long'),
                                                     entry, current, tp_distance_pct=0.03)
                        logger.info(f"调整OCO: {res_adj}")

            elif action in ('adjust_sl', 'tighten'):
                # 调整止损
                if sl_price:
                    coin = inst_id.replace('-USDT-SWAP', '')
                    entry = position_info.get('entry_price', 0)
                    current = position_info.get('last_price', 0)
                    if entry > 0 and current > 0:
                        res = adjust_trailing_oco(inst_id, position_info.get('direction', 'long'),
                                                  entry, current, tp_distance_pct=0.05)
                        logger.info(f"调整OCO(移动SL): {res}")

        except Exception as e:
            logger.error(f"持仓管理执行失败 [{inst_id}]: {e}")

    def _remove_from_local_state(self, symbol: str):
        """从local_state移除持仓记录"""
        try:
            state_file = STATE_DIR / "local_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                if symbol in state:
                    del state[symbol]
                    with open(state_file, 'w') as f:
                        json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"更新local_state失败: {e}")

    def _on_error(self, ws, error):
        logger.warning(f"WebSocket错误: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket关闭: {close_status_code} {close_msg}")

    def get_window(self, symbol: str) -> Optional[PriceWindow]:
        return self._price_windows.get(symbol)


# ============================================================
# 主守护进程
# ============================================================

class RealtimeDaemon:
    """
    实时看盘守护进程

    整合: WebSocket + 价格窗口 + 信号检测 + LLM分析 + 交易执行
    """

    def __init__(self):
        self._ws_client: Optional[OKXWebSocketClient] = None
        self._running = False
        self._pid = os.getpid()

    def start(self):
        if self._is_running():
            logger.error("守护进程已在运行!")
            return False

        self._running = True
        self._write_pid()

        # 启动WebSocket客户端
        self._ws_client = OKXWebSocketClient(on_signal=self._on_trading_signal)
        self._ws_client.start()

        logger.info(f"✅ 实时守护进程启动 (PID={self._pid})")
        logger.info(f"   监控: {WATCHED_COINS}")
        logger.info(f"   RSI信号: LONG<{RSI_LONG_THRESHOLD} SHORT>{RSI_SHORT_THRESHOLD}")
        logger.info(f"   信号冷却: {SIGNAL_COOLDOWN}秒")

        # 主循环 (保持进程存活 + 日志心跳)
        self._heartbeat()
        return True

    def stop(self):
        self._running = False
        if self._ws_client:
            self._ws_client.stop()
        self._remove_pid()
        logger.info("🛑 实时守护进程已停止")

    def _heartbeat(self):
        """心跳: 保持进程存活，定期报告状态"""
        while self._running:
            try:
                # 每30秒报告一次状态
                time.sleep(30)
                if not self._running:
                    break

                # 状态报告
                status_lines = [
                    f"[HEARTBEAT] {datetime.now().strftime('%H:%M:%S')}",
                    f"  PID: {self._pid}",
                    f"  运行中: {'是' if self._running else '否'}",
                ]

                if self._ws_client and self._ws_client._price_windows:
                    for sym, window in self._ws_client._price_windows.items():
                        if len(window) > 0:
                            rsi = window.get_rsi(RSI_PERIOD)
                            price = window.last_price
                            status_lines.append(
                                f"  {sym}: ${price:.6f} | RSI={rsi:.1f}" if rsi else f"  {sym}: ${price:.6f}"
                            )

                logger.info("\n".join(status_lines))

            except Exception as e:
                logger.error(f"心跳异常: {e}")

    def _on_trading_signal(self, symbol: str, direction: str, price: float, context: dict):
        """
        交易信号回调 — 当检测到开仓信号时调用

        Args:
            symbol: 合约ID, e.g. 'DOGE-USDT-SWAP'
            direction: 'long' or 'short'
            price: 当前价格
            context: {rsi, adx, bb_pos, price_data}
        """
        coin = symbol.replace('-USDT-SWAP', '')
        logger.info(
            f"🎯 开仓信号 [{symbol}] {direction.upper()} @ ${price:.6f} | "
            f"RSI={context.get('rsi','?')} ADX={context.get('adx','?')}"
        )

        try:
            # 懒加载，避免启动慢
            from miracle_kronos import (
                okx_req, place_oco, check_coin_cooldown,
                load_treasury, STATE_DIR
            )
            import json as json_lib

            # 检查cooldown
            is_cd, cd_reason = check_coin_cooldown(coin, direction)
            if is_cd:
                logger.info(f"跳过开仓 [{symbol}] {direction}: {cd_reason}")
                return

            # 计算仓位大小
            treasury = load_treasury()
            equity = treasury.get('equity', 60000)
            risk_per_trade = equity * 0.01  # 1% 风险
            sl_pct = 0.02  # 2% 止损
            sz_dollar = risk_per_trade / sl_pct  # $600 / 0.02 = $30,000

            # 获取合约乘数
            multiplier = self._get_multiplier(symbol)
            sz = max(1, int(sz_dollar / (price * multiplier)))

            # 开仓 + OCO
            sl_trigger = price * (1 - sl_pct) if direction == 'long' else price * (1 + sl_pct)
            tp_pct = 0.08  # 8% 止盈
            tp_trigger = price * (1 + tp_pct) if direction == 'long' else price * (1 - tp_pct)

            result = place_oco(
                instId=symbol,
                side='buy' if direction == 'long' else 'sell',
                sz=sz,
                entry_price=price,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                equity=equity,
                leverage=3,
            )

            if result.get('code') == '0':
                algo_id = result.get('data', [{}])[0].get('algoId')
                logger.info(f"✅ 开仓成功 [{symbol}] {direction} {sz}张 @ ${price:.6f} | algoId={algo_id}")

                # 更新local_state
                self._update_local_state(symbol, direction, price, sz, sl_trigger, tp_trigger, algo_id)

                # 同步到paper_trades.json
                self._sync_open_trade(symbol, direction, price, sz, context)
            else:
                logger.warning(f"❌ 开仓失败 [{symbol}]: {result.get('msg')}")

        except Exception as e:
            logger.error(f"开仓处理异常 [{symbol}]: {e}")

    def _get_multiplier(self, symbol: str) -> float:
        """获取合约乘数"""
        MULTIPLIER_MAP = {
            'BTC-USDT-SWAP': 0.01,
            'ETH-USDT-SWAP': 0.1,
            'SOL-USDT-SWAP': 1,
            'DOGE-USDT-SWAP': 1000,
            'BNB-USDT-SWAP': 10,
            'ADA-USDT-SWAP': 100,
            'AVAX-USDT-SWAP': 1,
            'FIL-USDT-SWAP': 1,
            'LINK-USDT-SWAP': 1,
            'DOT-USDT-SWAP': 1,
            'XRP-USDT-SWAP': 1,
        }
        return MULTIPLIER_MAP.get(symbol, 1)

    def _update_local_state(self, symbol: str, direction: str, entry: float,
                           sz: float, sl_price: float, tp_price: float,
                           algo_id: str):
        """更新local_state.json"""
        try:
            state_file = STATE_DIR / "local_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
            else:
                state = {}

            state[symbol] = {
                "direction": direction,
                "entry_price": entry,
                "contracts": float(sz),
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "algo_id": algo_id,
                "opened_at": datetime.now().isoformat(),
                "status": "active",
            }

            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"已更新local_state.json [{symbol}]")
        except Exception as e:
            logger.error(f"更新local_state失败: {e}")

    def _sync_open_trade(self, symbol: str, direction: str, price: float,
                         sz: float, context: dict):
        """同步开仓到paper_trades.json（供cron系统参考）"""
        try:
            paper_file = Path.home() / ".hermes" / "cron" / "output" / "paper_trades.json"
            if paper_file.exists():
                with open(paper_file) as f:
                    trades = json.load(f)
            else:
                trades = []

            coin = symbol.replace('-USDT-SWAP', '')
            rsi = context.get('rsi', 0)
            adx = context.get('adx', 0)
            open_reason = f"realtime_{direction} RSI={rsi:.0f} ADX={adx:.0f}"

            trades.append({
                "symbol": coin,
                "direction": direction,
                "status": "OPEN",
                "open_reason": open_reason,
                "open_price": price,
                "size": sz,
                "opened_at": datetime.now().isoformat(),
                "algo_id": None,
            })

            with open(paper_file, 'w') as f:
                json.dump(trades, f, indent=2)
            logger.info(f"已同步到paper_trades.json [{coin}]")
        except Exception as e:
            logger.error(f"同步paper_trades失败: {e}")

    def _is_running(self) -> bool:
        """检查是否已在运行"""
        if LOCK_FILE.exists():
            try:
                with open(LOCK_FILE) as f:
                    pid = int(f.read().strip())
                # 检查进程是否存在
                try:
                    os.kill(pid, 0)  # Signal 0 — 只检查存活
                    if pid != os.getpid():
                        logger.warning(f"守护进程已在运行 (PID={pid})")
                        return True
                except OSError:
                    logger.info(f"旧的PID文件 (PID={pid}) 已失效，将重启")
            except Exception:
                pass
        return False

    def _write_pid(self):
        with open(LOCK_FILE, 'w') as f:
            f.write(str(self._pid))
        with open(PID_FILE, 'w') as f:
            f.write(str(self._pid))

    def _remove_pid(self):
        for f in [LOCK_FILE, PID_FILE]:
            try:
                f.unlink()
            except Exception:
                pass


# ============================================================
# 命令行入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Miracle实时看盘守护进程')
    parser.add_argument('--daemon', action='store_true', help='后台守护模式')
    parser.add_argument('--status', action='store_true', help='查看状态')
    parser.add_argument('--stop', action='store_true', help='停止守护进程')
    parser.add_argument('--test-ws', action='store_true', help='测试WebSocket连接')
    args = parser.parse_args()

    daemon = RealtimeDaemon()

    if args.status:
        _show_status()
    elif args.stop:
        _stop_daemon()
    elif args.test_ws:
        _test_websocket()
    elif args.daemon:
        _daemonize(daemon)
    else:
        # 前台运行
        print("启动实时看盘守护进程 (前台模式，Ctrl+C停止)...")
        daemon.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n")
            daemon.stop()


def _show_status():
    """显示守护进程状态"""
    lock_file = WORKSPACE / "realtime_daemon.lock"
    pid_file = WORKSPACE / "realtime_daemon.pid"

    if lock_file.exists():
        with open(lock_file) as f:
            pid = f.read().strip()
        print(f"守护进程 PID: {pid}")
        try:
            os.kill(int(pid), 0)
            print("状态: 🟢 运行中")
        except OSError:
            print("状态: 🔴 已停止 (PID文件残留)")
    else:
        print("守护进程未运行")


def _stop_daemon():
    """停止守护进程"""
    lock_file = WORKSPACE / "realtime_daemon.lock"
    if lock_file.exists():
        with open(lock_file) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"已发送SIGTERM到PID={pid}")
            time.sleep(2)
            print("守护进程已停止")
        except OSError as e:
            print(f"停止失败: {e}")
        try:
            lock_file.unlink()
        except Exception:
            pass
    else:
        print("守护进程未运行")


def _daemonize(daemon: RealtimeDaemon):
    """后台守护"""
    pid = os.fork()
    if pid > 0:
        print(f"守护进程已在后台启动 (PID={pid})")
        sys.exit(0)
    os.setsid()
    pid = os.fork()
    if pid > 0:
        sys.exit(0)
    daemon.start()
    atexit.register(daemon.stop)


def _test_websocket():
    """测试WebSocket连接"""
    print(f"测试连接: {OKX_WS_URL}")
    received = []

    def on_message(ws, message):
        data = json.loads(message)
        received.append(data)
        if 'event' not in data:
            arg = data.get('arg', {})
            channel = arg.get('channel')
            if channel == 'tickers':
                ticker = data.get('data', [{}])[0]
                print(f"  [ticker] {ticker.get('instId')}: ${ticker.get('last')} "
                      f"H={ticker.get('high24h')} L={ticker.get('low24h')}")
            elif channel == 'candle5m':
                c = data.get('data', [{}])[0]
                if len(c) >= 5:
                    print(f"  [candle] {arg.get('instId')}: O={c[1]} H={c[2]} L={c[3]} C={c[4]}")
        if len(received) >= 15:
            ws.close()

    ws = websocket.WebSocketApp(
        OKX_WS_URL,
        on_message=on_message,
        on_error=lambda ws, e: print(f"错误: {e}"),
        on_close=lambda ws, a, b: print("连接关闭"),
        on_open=lambda ws: ws.send(json.dumps({
            "op": "subscribe",
            "args": [
                {"channel": "tickers", "instId": "DOGE-USDT-SWAP"},
                {"channel": "candle5m", "instId": "DOGE-USDT-SWAP"},
            ]
        })) or print("已订阅 DOGE-USDT-SWAP ticker + candle5m")
    )
    t = threading.Thread(target=lambda: ws.run_forever(ping_timeout=10), daemon=True)
    t.start()
    t.join(timeout=10)
    print(f"\n测试完成，共收到 {len(received)} 条消息")


if __name__ == '__main__':
    main()
