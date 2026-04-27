"""
Structured Memory using SQLite
===============================
结构化记忆模块 - 基于SQLite存储交易记录、因子表现、策略参数等
"""

import os
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)

# 全局连接
_db_connection = None
_db_path = None


def get_db_path() -> str:
    """获取数据库路径"""
    global _db_path
    if _db_path is None:
        db_dir = os.path.expanduser("~/.miracle_memory")
        os.makedirs(db_dir, exist_ok=True)
        _db_path = os.path.join(db_dir, "miracle_memory.db")
    return _db_path


@contextmanager
def get_db_connection():
    """获取数据库连接的上下文管理器"""
    global _db_connection
    
    if _db_connection is None:
        _db_connection = sqlite3.connect(get_db_path(), check_same_thread=False)
        _db_connection.row_factory = sqlite3.Row
        # 启用外键约束
        _db_connection.execute("PRAGMA foreign_keys = ON")
    
    try:
        yield _db_connection
    except Exception as e:
        _db_connection.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        pass  # 不关闭连接，保持复用


class MemoryType(Enum):
    """记忆类型枚举"""
    TRADE = "trade"
    SIGNAL = "signal"
    MARKET = "market"
    LESSON = "lesson"
    PATTERN = "pattern"
    STRATEGY = "strategy"
    CONFIG = "config"


@dataclass
class TradeRecord:
    """交易记录"""
    id: Optional[int] = None
    symbol: str = ""
    direction: str = ""  # LONG/SHORT
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    position_size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    strategy: str = ""
    signals: Dict[str, Any] = field(default_factory=dict)
    market_context: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    status: str = "open"  # open/closed/cancelled
    created_at: datetime = field(default_factory=datetime.now)


@dataclass  
class FactorPerformance:
    """因子表现记录"""
    id: Optional[int] = None
    factor_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    value: float = 0.0
    signal_direction: str = ""  # LONG/SHORT/NEUTRAL
    actual_outcome: Optional[str] = None  # WIN/LOSS
    pnl_contribution: float = 0.0
    market_regime: str = ""
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyParams:
    """策略参数"""
    id: Optional[int] = None
    strategy_name: str = ""
    symbol: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    performance: float = 0.0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    total_trades: int = 0
    market_regime: str = "default"
    notes: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Lesson:
    """学到的教训"""
    id: Optional[int] = None
    category: str = ""  # entry/exit/risk/management
    content: str = ""
    trigger_trade_id: Optional[int] = None
    trigger_symbol: str = ""
    trigger_direction: str = ""
    outcome: str = ""  # WIN/LOSS
    actionable: bool = True
    applied_count: int = 0
    success_rate: float = 0.0
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class StructuredMemory:
    """
    SQLite结构化记忆
    存储交易记录、因子表现、策略参数等结构化数据
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化结构化记忆
        
        Args:
            db_path: 可选指定数据库路径
        """
        global _db_path
        if db_path:
            _db_path = db_path
        
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        global _db_connection
        if _db_connection is None:
            _db_connection = sqlite3.connect(get_db_path(), check_same_thread=False)
            _db_connection.row_factory = sqlite3.Row
            _db_connection.execute("PRAGMA foreign_keys = ON")
        return _db_connection
    
    def _init_schema(self):
        """初始化数据库schema"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 交易记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL DEFAULT 0,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    position_size REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    strategy TEXT DEFAULT '',
                    signals TEXT DEFAULT '{}',
                    market_context TEXT DEFAULT '{}',
                    notes TEXT DEFAULT '',
                    status TEXT DEFAULT 'open',
                    created_at TEXT NOT NULL
                )
            """)
            
            # 因子表现表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS factor_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT DEFAULT '',
                    value REAL DEFAULT 0,
                    signal_direction TEXT DEFAULT '',
                    actual_outcome TEXT,
                    pnl_contribution REAL DEFAULT 0,
                    market_regime TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            
            # 策略参数表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    params TEXT DEFAULT '{}',
                    performance REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_rr REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    market_regime TEXT DEFAULT 'default',
                    notes TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(strategy_name, symbol, market_regime)
                )
            """)
            
            # 教训表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    trigger_trade_id INTEGER,
                    trigger_symbol TEXT DEFAULT '',
                    trigger_direction TEXT DEFAULT '',
                    outcome TEXT DEFAULT '',
                    actionable INTEGER DEFAULT 1,
                    applied_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0,
                    tags TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (trigger_trade_id) REFERENCES trades(id)
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_name ON factor_performance(factor_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lessons_category ON lessons(category)")
            
            conn.commit()
            logger.info("Database schema initialized")
    
    # ==================== Trade Operations ====================
    
    def add_trade(self, trade: TradeRecord) -> int:
        """添加交易记录"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    symbol, direction, entry_price, exit_price, entry_time, exit_time,
                    position_size, pnl, pnl_pct, commission, strategy, signals,
                    market_context, notes, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.direction, trade.entry_price, trade.exit_price,
                trade.entry_time.isoformat(), 
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.position_size, trade.pnl, trade.pnl_pct, trade.commission,
                trade.strategy, json.dumps(trade.signals),
                json.dumps(trade.market_context), trade.notes, trade.status,
                trade.created_at.isoformat()
            ))
            conn.commit()
            return cursor.lastrowid
    
    def update_trade(self, trade_id: int, **kwargs) -> bool:
        """更新交易记录"""
        # 定义允许更新的字段及其处理方式
        field_handlers = {
            "exit_price": lambda v: v,
            "exit_time": lambda v: v.isoformat() if isinstance(v, datetime) else v,
            "pnl": lambda v: v,
            "pnl_pct": lambda v: v,
            "commission": lambda v: v,
            "signals": lambda v: json.dumps(v) if isinstance(v, dict) else v,
            "market_context": lambda v: json.dumps(v) if isinstance(v, dict) else v,
            "notes": lambda v: v,
            "status": lambda v: v,
        }
        
        # 只处理白名单中的字段
        updates = {k: field_handlers[k](v) for k, v in kwargs.items() if k in field_handlers}
        if not updates:
            return False
        
        # 显式构建 UPDATE 语句，避免字符串拼接列名
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if "exit_price" in updates:
                cursor.execute("UPDATE trades SET exit_price = ? WHERE id = ?", (updates["exit_price"], trade_id))
            if "exit_time" in updates:
                cursor.execute("UPDATE trades SET exit_time = ? WHERE id = ?", (updates["exit_time"], trade_id))
            if "pnl" in updates:
                cursor.execute("UPDATE trades SET pnl = ? WHERE id = ?", (updates["pnl"], trade_id))
            if "pnl_pct" in updates:
                cursor.execute("UPDATE trades SET pnl_pct = ? WHERE id = ?", (updates["pnl_pct"], trade_id))
            if "commission" in updates:
                cursor.execute("UPDATE trades SET commission = ? WHERE id = ?", (updates["commission"], trade_id))
            if "signals" in updates:
                cursor.execute("UPDATE trades SET signals = ? WHERE id = ?", (updates["signals"], trade_id))
            if "market_context" in updates:
                cursor.execute("UPDATE trades SET market_context = ? WHERE id = ?", (updates["market_context"], trade_id))
            if "notes" in updates:
                cursor.execute("UPDATE trades SET notes = ? WHERE id = ?", (updates["notes"], trade_id))
            if "status" in updates:
                cursor.execute("UPDATE trades SET status = ? WHERE id = ?", (updates["status"], trade_id))
            conn.commit()
            return True
    
    def get_trade(self, trade_id: int) -> Optional[TradeRecord]:
        """获取交易记录"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            return self._row_to_trade(row) if row else None
    
    def get_trades(self, symbol: Optional[str] = None, status: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> List[TradeRecord]:
        """获取交易列表"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_trade(row) for row in cursor.fetchall()]
    
    def get_open_positions(self) -> List[TradeRecord]:
        """获取未平仓交易"""
        return self.get_trades(status="open")
    
    def get_trade_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取交易统计"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            base_query = "FROM trades WHERE status = 'closed'"
            params = []
            if symbol:
                base_query += " AND symbol = ?"
                params.append(symbol)
            
            cursor.execute(f"SELECT COUNT(*) {base_query}", params)
            total_trades = cursor.fetchone()[0]
            
            if total_trades == 0:
                return {"total": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
            
            cursor.execute(f"SELECT SUM(pnl) {base_query}", params)
            total_pnl = cursor.fetchone()[0] or 0
            
            # Fix: use WHERE instead of AND, and use parameterized query
            win_query = base_query.replace("status = 'closed'", "status = 'closed' AND pnl > 0")
            cursor.execute(f"SELECT COUNT(*) {win_query}", params)
            wins = cursor.fetchone()[0]
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            cursor.execute(f"SELECT AVG(pnl_pct) {base_query}", params)
            avg_pnl_pct = cursor.fetchone()[0] or 0
            
            return {
                "total": total_trades,
                "wins": wins,
                "losses": total_trades - wins,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl_pct": avg_pnl_pct
            }
    
    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        """将数据库行转换为TradeRecord"""
        return TradeRecord(
            id=row["id"],
            symbol=row["symbol"],
            direction=row["direction"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            entry_time=datetime.fromisoformat(row["entry_time"]),
            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
            position_size=row["position_size"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            commission=row["commission"],
            strategy=row["strategy"],
            signals=json.loads(row["signals"]) if row["signals"] else {},
            market_context=json.loads(row["market_context"]) if row["market_context"] else {},
            notes=row["notes"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"])
        )
    
    # ==================== Factor Performance Operations ====================
    
    def add_factor_performance(self, factor: FactorPerformance) -> int:
        """添加因子表现记录"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO factor_performance (
                    factor_name, symbol, timeframe, value, signal_direction,
                    actual_outcome, pnl_contribution, market_regime, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                factor.factor_name, factor.symbol, factor.timeframe, factor.value,
                factor.signal_direction, factor.actual_outcome, factor.pnl_contribution,
                factor.market_regime, factor.notes, factor.created_at.isoformat()
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_factor_performance(self, factor_name: str, symbol: str,
                               limit: int = 100) -> List[FactorPerformance]:
        """获取因子表现历史"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM factor_performance 
                WHERE factor_name = ? AND symbol = ?
                ORDER BY created_at DESC LIMIT ?
            """, (factor_name, symbol, limit))
            
            return [self._row_to_factor(row) for row in cursor.fetchall()]
    
    def get_factor_ic(self, factor_name: str, symbol: str) -> float:
        """计算因子的IC（信息系数）"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value, actual_outcome FROM factor_performance
                WHERE factor_name = ? AND symbol = ? AND actual_outcome IS NOT NULL
                ORDER BY created_at DESC LIMIT 100
            """, (factor_name, symbol))
            
            rows = cursor.fetchall()
            if len(rows) < 10:
                return 0.0
            
            # 简单IC计算：方向一致率
            correct = 0
            for row in rows:
                value = row["value"]
                outcome = row["actual_outcome"]
                
                # 假设value > 0 应该对应 LONG/WIN
                if (value > 0 and outcome in ["WIN", "LONG"]) or \
                   (value < 0 and outcome in ["LOSS", "SHORT"]):
                    correct += 1
            
            return correct / len(rows) if rows else 0.0
    
    def _row_to_factor(self, row: sqlite3.Row) -> FactorPerformance:
        """将数据库行转换为FactorPerformance"""
        return FactorPerformance(
            id=row["id"],
            factor_name=row["factor_name"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            value=row["value"],
            signal_direction=row["signal_direction"],
            actual_outcome=row["actual_outcome"],
            pnl_contribution=row["pnl_contribution"],
            market_regime=row["market_regime"],
            notes=row["notes"],
            created_at=datetime.fromisoformat(row["created_at"])
        )
    
    # ==================== Strategy Params Operations ====================
    
    def save_strategy_params(self, params: StrategyParams) -> int:
        """保存策略参数（存在则更新）"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO strategy_params (
                    strategy_name, symbol, params, performance, win_rate,
                    avg_rr, total_trades, market_regime, notes, is_active,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_name, symbol, market_regime) 
                DO UPDATE SET
                    params = excluded.params,
                    performance = excluded.performance,
                    win_rate = excluded.win_rate,
                    avg_rr = excluded.avg_rr,
                    total_trades = excluded.total_trades,
                    notes = excluded.notes,
                    is_active = excluded.is_active,
                    updated_at = excluded.updated_at
            """, (
                params.strategy_name, params.symbol, json.dumps(params.params),
                params.performance, params.win_rate, params.avg_rr,
                params.total_trades, params.market_regime, params.notes,
                1 if params.is_active else 0,
                params.created_at.isoformat(), datetime.now().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_strategy_params(self, strategy_name: str, symbol: str,
                           market_regime: str = "default") -> Optional[StrategyParams]:
        """获取策略参数"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM strategy_params
                WHERE strategy_name = ? AND symbol = ? AND market_regime = ?
            """, (strategy_name, symbol, market_regime))
            
            row = cursor.fetchone()
            return self._row_to_strategy(row) if row else None
    
    def get_active_strategies(self, symbol: Optional[str] = None) -> List[StrategyParams]:
        """获取活跃策略"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM strategy_params WHERE is_active = 1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY performance DESC"
            cursor.execute(query, params)
            
            return [self._row_to_strategy(row) for row in cursor.fetchall()]
    
    def _row_to_strategy(self, row: sqlite3.Row) -> StrategyParams:
        """将数据库行转换为StrategyParams"""
        return StrategyParams(
            id=row["id"],
            strategy_name=row["strategy_name"],
            symbol=row["symbol"],
            params=json.loads(row["params"]) if row["params"] else {},
            performance=row["performance"],
            win_rate=row["win_rate"],
            avg_rr=row["avg_rr"],
            total_trades=row["total_trades"],
            market_regime=row["market_regime"],
            notes=row["notes"],
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )
    
    # ==================== Lesson Operations ====================
    
    def add_lesson(self, lesson: Lesson) -> int:
        """添加教训记录"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO lessons (
                    category, content, trigger_trade_id, trigger_symbol,
                    trigger_direction, outcome, actionable, applied_count,
                    success_rate, tags, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lesson.category, lesson.content, lesson.trigger_trade_id,
                lesson.trigger_symbol, lesson.trigger_direction, lesson.outcome,
                1 if lesson.actionable else 0, lesson.applied_count,
                lesson.success_rate, json.dumps(lesson.tags),
                lesson.created_at.isoformat()
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_lessons(self, category: Optional[str] = None,
                    actionable_only: bool = True,
                    limit: int = 50) -> List[Lesson]:
        """获取教训列表"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM lessons WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
            if actionable_only:
                query += " AND actionable = 1"
            
            query += " ORDER BY success_rate DESC, applied_count DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [self._row_to_lesson(row) for row in cursor.fetchall()]
    
    def update_lesson_success(self, lesson_id: int, success: bool):
        """更新教训应用成功率"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 获取当前统计
            cursor.execute("SELECT applied_count, success_rate FROM lessons WHERE id = ?", (lesson_id,))
            row = cursor.fetchone()
            if not row:
                return
            
            current_count = row["applied_count"]
            current_rate = row["success_rate"]
            
            # 计算新成功率
            new_count = current_count + 1
            new_successes = current_rate * current_count + (1 if success else 0)
            new_rate = new_successes / new_count
            
            cursor.execute("""
                UPDATE lessons SET applied_count = ?, success_rate = ?
                WHERE id = ?
            """, (new_count, new_rate, lesson_id))
            conn.commit()
    
    def _row_to_lesson(self, row: sqlite3.Row) -> Lesson:
        """将数据库行转换为Lesson"""
        return Lesson(
            id=row["id"],
            category=row["category"],
            content=row["content"],
            trigger_trade_id=row["trigger_trade_id"],
            trigger_symbol=row["trigger_symbol"],
            trigger_direction=row["trigger_direction"],
            outcome=row["outcome"],
            actionable=bool(row["actionable"]),
            applied_count=row["applied_count"],
            success_rate=row["success_rate"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=datetime.fromisoformat(row["created_at"])
        )
    
    # ==================== Utility Operations ====================
    
    def reset(self):
        """重置数据库（危险操作）"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM lessons")
            cursor.execute("DELETE FROM strategy_params")
            cursor.execute("DELETE FROM factor_performance")
            cursor.execute("DELETE FROM trades")
            conn.commit()
            logger.warning("Structured memory reset - all data deleted")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("SELECT COUNT(*) FROM trades")
            stats["total_trades"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
            stats["closed_trades"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM lessons")
            stats["total_lessons"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM strategy_params WHERE is_active = 1")
            stats["active_strategies"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT factor_name) FROM factor_performance")
            stats["tracked_factors"] = cursor.fetchone()[0]
            
            return stats
    
    def close(self):
        """关闭数据库连接"""
        global _db_connection
        if _db_connection:
            _db_connection.close()
            _db_connection = None
            logger.info("Database connection closed")


# 全局实例
_structured_memory: Optional[StructuredMemory] = None

def get_structured_memory() -> StructuredMemory:
    """获取结构化记忆全局实例"""
    global _structured_memory
    if _structured_memory is None:
        _structured_memory = StructuredMemory()
    return _structured_memory
