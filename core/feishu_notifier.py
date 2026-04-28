#!/usr/bin/env python3
from __future__ import annotations

"""
Feishu Notifier - 分级告警系统 (P2.3)
=====================================

三级告警:
- P0 (CRITICAL): 资金安全/系统崩溃 - 立即推送
- P1 (WARNING): 异常交易/熔断触发 - 正常推送
- P2 (INFO): 正常心跳/性能报告 - 静默(本地日志)

配置:
- 环境变量: FEISHU_WEBHOOK_URL
- 交付方式: 本地静默 / 飞书推送

Usage:
    from core.feishu_notifier import FeishuNotifier, AlertLevel
    
    notifier = FeishuNotifier()
    notifier.critical('资金安全', 'BTC多头触发止损')
    notifier.warning('熔断触发', 'DOGE连续3亏')
    notifier.info('心跳', '系统正常运行')
"""

import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ==================== 配置 ====================

FEISHU_WEBHOOK_URL = os.getenv('FEISHU_WEBHOOK_URL', '')
DELIVERY_MODE = os.getenv('FEISHU_DELIVERY', 'local')  # 'local' | 'feishu'
OUTPUT_DIR = os.path.expanduser('~/.miracle_memory/feishu_alerts/')


def is_feishu_configured() -> bool:
    """检查飞书是否已配置"""
    return bool(FEISHU_WEBHOOK_URL)


class AlertLevel(Enum):
    """告警级别"""
    CRITICAL = 'P0'  # 资金安全/系统崩溃
    WARNING = 'P1'    # 异常交易/熔断
    INFO = 'P2'       # 心跳/性能报告


class AlertLevelConfig:
    """告警级别配置"""
    CONFIG = {
        AlertLevel.CRITICAL: {
            'name': 'CRITICAL',
            'emoji': '🚨',
            'deliver': 'feishu',  # 强制推送
            'file_level': 'ERROR',
        },
        AlertLevel.WARNING: {
            'name': 'WARNING', 
            'emoji': '⚠️',
            'deliver': 'feishu',  # 推送
            'file_level': 'WARNING',
        },
        AlertLevel.INFO: {
            'name': 'INFO',
            'emoji': '💤',
            'deliver': 'local',  # 静默
            'file_level': 'INFO',
        },
    }


class FeishuNotifier:
    """
    飞书分级告警系统
    
    设计原则:
    - P0/P1: 必须推送飞书
    - P2: 默认静默(本地日志)
    - 本地持久化: 所有告警都写入日志文件
    """

    def __init__(self, webhook_url: str = None, delivery_mode: str = None):
        self.webhook_url = webhook_url or FEISHU_WEBHOOK_URL
        self.delivery_mode = delivery_mode or DELIVERY_MODE
        
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 告警计数器
        self._counters = {
            AlertLevel.CRITICAL: 0,
            AlertLevel.WARNING: 0,
            AlertLevel.INFO: 0,
        }
        
        # 最近的告警 (内存缓存)
        self._recent_alerts = []
        self._max_recent = 100

    def _log_alert(self, level: AlertLevel, title: str, message: str, data: Dict = None):
        """写入本地日志"""
        config = AlertLevelConfig.CONFIG[level]
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'level': level.value,
            'title': title,
            'message': message,
            'data': data or {},
        }
        
        # 添加到最近告警缓存
        self._recent_alerts.append(log_entry)
        if len(self._recent_alerts) > self._max_recent:
            self._recent_alerts.pop(0)
        
        # 写入日志文件
        log_file = os.path.join(OUTPUT_DIR, f'alerts_{datetime.now().strftime("%Y%m%d")}.jsonl')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 日志输出
        log_func = getattr(logging, config['file_level'].lower())
        log_func(f"[{level.value}] {title}: {message}")

    def _send_to_feishu(self, level: AlertLevel, title: str, message: str, data: Dict = None):
        """发送飞书消息"""
        if not self.webhook_url:
            logger.debug(f"飞书Webhook未配置,跳过推送: {title}")
            return
        
        config = AlertLevelConfig.CONFIG[level]
        emoji = config['emoji']
        
        # 构建消息内容
        content = {
            'msg_type': 'text',
            'content': {
                'text': f"{emoji} **[{level.value}] {title}**\n\n{message}"
            }
        }
        
        # 如果有额外数据,添加详情
        if data:
            data_str = '\n'.join([f'- {k}: {v}' for k, v in data.items()])
            content['content']['text'] += f"\n\n📊 **详情:**\n{data_str}"
        
        try:
            import urllib.request
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(content).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info(f"飞书推送成功: {title}")
                else:
                    logger.warning(f"飞书推送失败: {resp.status}")
        except Exception as e:
            logger.error(f"飞书推送异常: {e}")

    def notify(self, level: AlertLevel, title: str, message: str, data: Dict = None):
        """
        发送告警
        
        Args:
            level: 告警级别
            title: 告警标题
            message: 告警消息
            data: 额外数据
        """
        self._counters[level] += 1
        
        # 1. 写入本地日志 (所有级别)
        self._log_alert(level, title, message, data)
        
        # 2. 判断是否推送飞书
        config = AlertLevelConfig.CONFIG[level]
        should_deliver_feishu = (
            config['deliver'] == 'feishu' or 
            self.delivery_mode == 'feishu' or
            level == AlertLevel.CRITICAL  # P0强制推送
        )
        
        if should_deliver_feishu:
            self._send_to_feishu(level, title, message, data)

    def critical(self, title: str, message: str, data: Dict = None):
        """P0级告警: 资金安全/系统崩溃"""
        self.notify(AlertLevel.CRITICAL, title, message, data)

    def warning(self, title: str, message: str, data: Dict = None):
        """P1级告警: 异常交易/熔断触发"""
        self.notify(AlertLevel.WARNING, title, message, data)

    def info(self, title: str, message: str, data: Dict = None):
        """P2级告警: 心跳/性能报告 (静默)"""
        self.notify(AlertLevel.INFO, title, message, data)

    def get_counters(self) -> Dict[str, int]:
        """获取告警计数"""
        return {level.value: count for level, count in self._counters.items()}

    def get_recent_alerts(self, level: AlertLevel = None, limit: int = 10) -> list:
        """获取最近告警"""
        alerts = self._recent_alerts
        if level:
            alerts = [a for a in alerts if a['level'] == level.value]
        return alerts[-limit:]


# ==================== 便捷函数 ====================

_notifier: FeishuNotifier | None = None


def get_notifier() -> FeishuNotifier:
    """获取全局notifier单例"""
    global _notifier
    if _notifier is None:
        _notifier = FeishuNotifier()
    return _notifier


def notify_critical(title: str, message: str, data: Dict = None):
    """发送P0告警"""
    get_notifier().critical(title, message, data)


def notify_warning(title: str, message: str, data: Dict = None):
    """发送P1告警"""
    get_notifier().warning(title, message, data)


def notify_info(title: str, message: str, data: Dict = None):
    """发送P2告警 (静默)"""
    get_notifier().info(title, message, data)


# ==================== 向后兼容函数 ====================

def push_feishu(message: str, level: str = 'INFO') -> bool:
    """
    推送飞书消息 (向后兼容)
    
    Args:
        message: 消息内容
        level: 级别 INFO/WARNING/CRITICAL
    """
    level_map = {
        'INFO': AlertLevel.INFO,
        'WARNING': AlertLevel.WARNING,
        'CRITICAL': AlertLevel.CRITICAL,
    }
    alert_level = level_map.get(level.upper(), AlertLevel.INFO)
    get_notifier().notify(alert_level, '通知', message)
    return True


def push_feishu_alert(title: str, message: str, alert_type: str = 'warning') -> bool:
    """
    推送飞书告警 (向后兼容)
    
    Args:
        title: 告警标题
        message: 告警消息
        alert_type: 告警类型 info/warning/error/critical
    """
    type_map = {
        'info': AlertLevel.INFO,
        'warning': AlertLevel.WARNING,
        'error': AlertLevel.WARNING,
        'critical': AlertLevel.CRITICAL,
    }
    alert_level = type_map.get(alert_type.lower(), AlertLevel.WARNING)
    get_notifier().notify(alert_level, title, message)
    return True


def push_feishu_report(title: str, report: str) -> bool:
    """
    推送飞书报告 (向后兼容)
    
    Args:
        title: 报告标题
        report: 报告内容
    """
    get_notifier().info(title, report)
    return True


# ==================== 自检 ====================

if __name__ == '__main__':
    import pprint
    
    print("=== 飞书分级告警系统 (P2.3) ===\n")
    
    notifier = FeishuNotifier()
    
    # 测试各级别告警
    print("发送测试告警...")
    notifier.critical('资金安全测试', '模拟P0告警', {'equity': 50000, 'position': 'LONG'})
    notifier.warning('熔断触发测试', '模拟P1告警', {'consecutive_losses': 3})
    notifier.info('心跳测试', '系统正常运行', {'uptime_hours': 24})
    
    print("\n告警计数:")
    pprint.pprint(notifier.get_counters())
    
    print("\n最近告警:")
    pprint.pprint(notifier.get_recent_alerts())
    
    print("\n=== 自检完成 ===")
