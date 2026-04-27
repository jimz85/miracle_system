"""
Logging Configuration - 结构化日志配置
支持JSON格式输出、日志轮转、分类日志
"""
import os
import sys
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


# ========================
# JSON日志格式化
# ========================

class JSONFormatter(logging.Formatter):
    """
    JSON结构化日志格式化器
    
    输出格式:
    {
        "timestamp": "2026-04-27T19:00:00.000Z",
        "level": "INFO",
        "logger": "module.name",
        "message": "log message",
        "module": "module",
        "function": "function_name",
        "line": 123,
        ...extra_fields
    }
    """
    
    def __init__(
        self,
        include_extra: bool = True,
        include_location: bool = True,
        default_fields: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.include_extra = include_extra
        self.include_location = include_location
        self.default_fields = default_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加默认字段
        log_data.update(self.default_fields)
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # 添加额外字段
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in logging.LogRecord(
                    "", 0, "", 0, "", (), None
                ).__dict__ and not k.startswith("_")
            }
            log_data.update(extra_fields)
        
        return json.dumps(log_data, default=str)


class PlainFormatter(logging.Formatter):
    """普通格式日志"""
    
    def __init__(self, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        super().__init__(fmt)


# ========================
# 日志配置
# ========================

DEFAULT_LOG_DIR = Path.home() / ".miracle" / "logs"
DEFAULT_LOG_LEVEL = logging.INFO


def ensure_log_dir(log_dir: Path) -> Path:
    """确保日志目录存在"""
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_root_logger(
    log_level: int = DEFAULT_LOG_LEVEL,
    log_dir: Optional[Path] = None,
    json_logs: bool = False,
    log_file: Optional[str] = None,
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    配置根日志器
    
    Args:
        log_level: 日志级别
        log_dir: 日志目录
        json_logs: 是否使用JSON格式
        log_file: 日志文件名
        max_bytes: 单个日志文件最大字节
        backup_count: 保留的备份数
    """
    log_dir = ensure_log_dir(log_dir or DEFAULT_LOG_DIR)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有handler
    root_logger.handlers.clear()
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    if json_logs:
        console_handler.setFormatter(JSONFormatter(include_location=False))
    else:
        console_handler.setFormatter(PlainFormatter())
    root_logger.addHandler(console_handler)
    
    # 文件handler (如果指定)
    if log_file:
        file_path = log_dir / log_file
        
        # 使用RotatingFileHandler
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        
        if json_logs:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(PlainFormatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
            ))
        
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_json_logger(
    name: str,
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None,
    level: int = DEFAULT_LOG_LEVEL,
    default_fields: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    获取JSON日志器
    
    Args:
        name: 日志器名称
        log_file: 日志文件名
        log_dir: 日志目录
        level: 日志级别
        default_fields: 默认字段
    
    Returns:
        Logger实例
    """
    log_dir = ensure_log_dir(log_dir or DEFAULT_LOG_DIR)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # JSON文件handler
    if log_file:
        file_path = log_dir / log_file
        
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter(default_fields=default_fields))
        logger.addHandler(file_handler)
    
    return logger


def get_trade_logger(
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    获取交易专用日志器
    
    日志输出到: trades/{date}.json
    """
    log_dir = ensure_log_dir(log_dir or DEFAULT_LOG_DIR)
    trade_dir = log_dir / "trades"
    trade_dir.mkdir(exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = f"trades_{date_str}.json"
    
    logger = get_json_logger(
        name="miracle.trades",
        log_file=log_file,
        log_dir=trade_dir,
        default_fields={"type": "trade"}
    )
    
    return logger


def get_audit_logger(
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    获取审计专用日志器
    
    日志输出到: audit/{date}.json
    """
    log_dir = ensure_log_dir(log_dir or DEFAULT_LOG_DIR)
    audit_dir = log_dir / "audit"
    audit_dir.mkdir(exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = f"audit_{date_str}.json"
    
    logger = get_json_logger(
        name="miracle.audit",
        log_file=log_file,
        log_dir=audit_dir,
        default_fields={"type": "audit"}
    )
    
    return logger


# ========================
# 结构化日志辅助
# ========================

def log_trade_event(
    logger: logging.Logger,
    event: str,
    symbol: str,
    side: str,
    size: float,
    price: float,
    pnl: Optional[float] = None,
    **extra
):
    """记录交易事件"""
    log_data = {
        "event": event,
        "symbol": symbol,
        "side": side,
        "size": size,
        "price": price
    }
    if pnl is not None:
        log_data["pnl"] = pnl
    log_data.update(extra)
    
    logger.info(f"Trade {event}: {json.dumps(log_data)}")


def log_signal_event(
    logger: logging.Logger,
    symbol: str,
    direction: str,
    confidence: float,
    factors: Optional[Dict[str, Any]] = None
):
    """记录信号事件"""
    log_data = {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence
    }
    if factors:
        log_data["factors"] = factors
    
    logger.info(f"Signal: {json.dumps(log_data)}")


def log_risk_event(
    logger: logging.Logger,
    event: str,
    message: str,
    **extra
):
    """记录风险事件"""
    log_data = {
        "event": event,
        "message": message
    }
    log_data.update(extra)
    
    if event in ["warning", "danger", "emergency"]:
        logger.warning(f"Risk {event}: {json.dumps(log_data)}")
    else:
        logger.info(f"Risk {event}: {json.dumps(log_data)}")


# ========================
# 关闭所有日志器
# ========================

def close_all_loggers():
    """关闭所有日志处理器"""
    logging.shutdown()


if __name__ == "__main__":
    print("=== Logging Config Test ===\n")
    
    # 测试JSON日志
    logger = get_json_logger("test", "test.json", default_fields={"app": "miracle"})
    logger.info("Test info message")
    logger.warning("Test warning", extra={"code": 123})
    logger.error("Test error with exception", exc_info=Exception("test error"))
    
    # 测试交易日志
    trade_logger = get_trade_logger()
    log_trade_event(
        trade_logger, "OPEN", "BTC", "long", 0.1, 50000,
        stop_loss=49000, take_profit=52000
    )
    log_trade_event(
        trade_logger, "CLOSE", "BTC", "long", 0.1, 51000, pnl=100
    )
    
    # 测试审计日志
    audit_logger = get_audit_logger()
    log_risk_event(
        audit_logger, "position_warning", "High exposure",
        symbol="BTC", exposure_pct=85
    )
    
    print(f"\nLogs written to: {DEFAULT_LOG_DIR}")
    print("\n=== Test complete ===")
