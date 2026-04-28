"""
ExecutorFeishuNotifier - 执行引擎专用飞书通知
==========================================

从 agents/agent_executor.py 提取的飞书通知类
与 core/feishu_notifier.py (分级告警系统) 不同：
- 本模块: 交易通知 (入场/出场/止损/错误)
- core/feishu_notifier.py: 系统级分级告警 (P0/P1/P2)

用法:
    from core.executor_feishu_notifier import ExecutorFeishuNotifier
    from agents.agent_executor import ExecutorFeishuNotifier  # 向后兼容导入
"""

import logging
from datetime import datetime
from typing import Dict

import requests

from core.executor_config import ExecutorConfig

logger = logging.getLogger(__name__)


class ExecutorFeishuNotifier:
    """飞书通知 - 执行引擎专用"""

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.enabled = config.feishu_enabled and bool(config.feishu_webhook)

    def notify(self, trade_record: Dict, notification_type: str):
        """
        发送飞书通知
        notification_type: "entry" / "exit" / "stop_loss" / "error"
        """
        if not self.enabled:
            return

        try:
            if notification_type == "entry":
                title = "🚨 新仓入场"
                color = "green"
            elif notification_type == "exit":
                title = "📤 平仓出场"
                color = "blue"
            elif notification_type == "stop_loss":
                title = "🔴 止损触发"
                color = "red"
            elif notification_type == "error":
                title = "⚠️ 执行错误"
                color = "red"
            else:
                title = f"📢 交易通知 [{notification_type}]"
                color = "grey"

            message = self._build_message(trade_record, title, notification_type)
            self._send(message, color)

        except Exception as e:
            logger.error(f"飞书通知发送失败: {e}")

    def _build_message(self, trade_record: Dict, title: str, notif_type: str) -> Dict:
        """构建消息内容"""
        elements = [
            {"tag": "markdown", "content": f"**{title}**"},
            {"tag": "hr"},
            {
                "tag": "element",
                "text": {
                    "tag": "lark_md",
                    "content": f"**交易标的:** {trade_record.get('symbol', 'N/A')}"
                }
            },
            {
                "tag": "element",
                "text": {
                    "tag": "lark_md",
                    "content": f"**交易方向:** {trade_record.get('side', 'N/A').upper()}"
                }
            }
        ]

        if notif_type in ["entry", "exit"]:
            elements.append({
                "tag": "element",
                "text": {
                    "tag": "lark_md",
                    "content": f"**入场价:** {trade_record.get('entry_price', 0):.4f}"
                }
            })

            if trade_record.get("exit_price"):
                elements.append({
                    "tag": "element",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**出场价:** {trade_record.get('exit_price'):.4f}"
                    }
                })

        if trade_record.get("pnl") is not None:
            pnl = trade_record["pnl"]
            emoji = "🟢" if pnl >= 0 else "🔴"
            elements.append({
                "tag": "element",
                "text": {
                    "tag": "lark_md",
                    "content": f"**盈亏:** {emoji} {pnl:.2f} USDT"
                }
            })

        if trade_record.get("exit_reason"):
            elements.append({
                "tag": "element",
                "text": {
                    "tag": "lark_md",
                    "content": f"**出场原因:** {trade_record.get('exit_reason')}"
                }
            })

        if trade_record.get("trade_id"):
            elements.append({
                "tag": "element",
                "text": {
                    "tag": "lark_md",
                    "content": f"**Trade ID:** `{trade_record.get('trade_id')[:8]}...`"
                }
            })

        elements.append({
            "tag": "element",
            "text": {
                "tag": "lark_md",
                "content": f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        })

        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": "Miracle 1.0.1 交易通知"},
                    "template": "purple" if notif_type == "entry" else "orange" if notif_type == "exit" else "red"
                },
                "elements": elements
            }
        }

    def _send(self, message: Dict, color: str = "blue"):
        """发送消息"""
        try:
            response = requests.post(
                self.config.feishu_webhook,
                json=message,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            result = response.json()

            if result.get("code") != 0:
                logger.warning(f"飞书API返回错误: {result}")
            else:
                logger.info(f"飞书通知发送成功")

        except Exception as e:
            logger.error(f"飞书通知发送失败: {e}")

    def send_alert(self, title: str, message: str, level: str = "warning"):
        """发送告警消息"""
        if not self.enabled:
            return

        color_map = {
            "info": "blue",
            "warning": "orange",
            "error": "red"
        }

        self._send({
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"⚠️ {title}"},
                    "template": color_map.get(level, "grey")
                },
                "elements": [
                    {"tag": "markdown", "content": message},
                    {"tag": "hr"},
                    {"tag": "element", "text": {"tag": "lark_md", "content": f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}}
                ]
            }
        }, color_map.get(level, "grey"))


# 向后兼容别名
FeishuNotifier = ExecutorFeishuNotifier
