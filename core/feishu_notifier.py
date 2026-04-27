#!/usr/bin/env python3
"""
feishu_notifier.py - 飞书通知模块
=================================

Miracle System 的飞书通知功能，基于 Kronos 的 feishu_notifier.py 实现。

主要功能:
    - push_feishu(): 发送文本消息到飞书群

配置项 (环境变量):
    - FEISHU_APP_ID: 飞书应用 App ID
    - FEISHU_APP_SECRET: 飞书应用 App Secret  
    - FEISHU_CHAT_ID: 飞书群 Chat ID (可选，默认使用配置的值)

版本: 1.0.1
"""

import os
import json
import logging
from typing import Optional

import requests

logger = logging.getLogger("miracle.feishu")

# 飞书 API 配置
FEISHU_API_BASE = "https://open.feishu.cn/open-apis"

# 默认 Chat ID (如果环境变量未设置)
DEFAULT_CHAT_ID = "oc_bfd8a7cc1a606f190b53e3fd0167f5a0"


class FeishuNotifier:
    """
    飞书通知器类
    
    提供飞书消息推送功能，支持:
    - 文本消息发送
    - 错误告警
    - 系统状态通知
    """
    
    _instance: Optional['FeishuNotifier'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._app_id: str = ''
        self._app_secret: str = ''
        self._chat_id: str = ''
        self._token: Optional[str] = None
        self._token_expires_at: float = 0
        self._initialized = True
        self._load_config()
    
    def _load_config(self):
        """从环境变量加载飞书配置"""
        self._app_id = os.environ.get('FEISHU_APP_ID', '')
        self._app_secret = os.environ.get('FEISHU_APP_SECRET', '')
        self._chat_id = os.environ.get('FEISHU_CHAT_ID', DEFAULT_CHAT_ID)
        
        if self._app_id and self._app_secret:
            logger.info("飞书配置已加载")
        else:
            logger.warning("飞书配置未完成 (APP_ID/APP_SECRET 未设置)")
    
    @property
    def is_configured(self) -> bool:
        """检查飞书是否已配置"""
        return bool(self._app_id and self._app_secret)
    
    def _get_token(self) -> Optional[str]:
        """
        获取飞书 Tenant Access Token
        
        使用缓存机制，避免频繁请求
        """
        import time
        
        # 检查缓存的 token 是否有效 (提前5分钟过期)
        if self._token and time.time() < (self._token_expires_at - 300):
            return self._token
        
        if not self.is_configured:
            return None
        
        try:
            url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
            payload = {
                'app_id': self._app_id,
                'app_secret': self._app_secret
            }
            
            response = requests.post(url, json=payload, timeout=10)
            data = response.json()
            
            if data.get('code') == 0:
                self._token = data.get('tenant_access_token')
                # 飞书 token 有效期约2小时，这里设置为1.5小时
                self._token_expires_at = time.time() + 5400
                logger.debug("获取飞书 Token 成功")
                return self._token
            else:
                logger.error(f"获取飞书 Token 失败: code={data.get('code')}, msg={data.get('msg')}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"获取飞书 Token 网络错误: {e}")
            return None
        except Exception as e:
            logger.error(f"获取飞书 Token 异常: {e}")
            return None
    
    def send_message(self, message: str, chat_id: Optional[str] = None) -> bool:
        """
        发送文本消息到飞书
        
        Args:
            message: 要发送的消息内容 (最多4000字符)
            chat_id: 可选的群 ID，默认使用配置的 CHAT_ID
        
        Returns:
            bool: 发送是否成功
        """
        if not self.is_configured:
            logger.warning("飞书未配置，跳过消息发送")
            return False
        
        token = self._get_token()
        if not token:
            logger.error("无法获取飞书 Token，发送失败")
            return False
        
        try:
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            # 截断消息到4000字符
            truncated_message = message[:4000]
            
            payload = {
                'receive_id': chat_id or self._chat_id,
                'msg_type': 'text',
                'content': json.dumps({'text': truncated_message})
            }
            
            params = {'receive_id_type': 'chat_id'}
            
            url = f"{FEISHU_API_BASE}/im/v1/messages"
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                params=params, 
                timeout=10
            )
            
            result = response.json()
            
            if result.get('code') == 0:
                logger.info("飞书消息发送成功")
                return True
            else:
                logger.error(f"飞书消息发送失败: code={result.get('code')}, msg={result.get('msg')}")
                # Token 过期时清除缓存，下次重试
                if result.get('code') == 99991663:
                    self._token = None
                return False
                
        except requests.RequestException as e:
            logger.error(f"飞书消息发送网络错误: {e}")
            return False
        except Exception as e:
            logger.error(f"飞书消息发送异常: {e}")
            return False
    
    def send_alert(self, title: str, message: str) -> bool:
        """
        发送告警消息 (带标题)
        
        Args:
            title: 告警标题
            message: 告警详情
        
        Returns:
            bool: 发送是否成功
        """
        alert_message = f"🚨 {title}\n\n{message}"
        return self.send_message(alert_message)
    
    def send_report(self, title: str, content: str) -> bool:
        """
        发送报告消息 (带标题)
        
        Args:
            title: 报告标题
            content: 报告内容
        
        Returns:
            bool: 发送是否成功
        """
        report_message = f"📊 {title}\n\n{content}"
        return self.send_message(report_message)
    
    def reload_config(self):
        """重新加载配置"""
        self._token = None
        self._token_expires_at = 0
        self._load_config()


# 全局实例
_notifier_instance: Optional[FeishuNotifier] = None


def get_notifier() -> FeishuNotifier:
    """获取全局飞书通知器实例"""
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = FeishuNotifier()
    return _notifier_instance


def push_feishu(message: str, chat_id: Optional[str] = None) -> bool:
    """
    发送消息到飞书 (便捷函数)
    
    这是从 Kronos 移植的便捷接口，保持向后兼容。
    
    Args:
        message: 要发送的消息内容
        chat_id: 可选的群 ID
    
    Returns:
        bool: 发送是否成功
    
    Example:
        >>> push_feishu("Kronos 交易信号: BTC 多头")
        True
    """
    notifier = get_notifier()
    return notifier.send_message(message, chat_id)


def push_feishu_alert(title: str, message: str) -> bool:
    """
    发送告警到飞书 (便捷函数)
    
    Args:
        title: 告警标题
        message: 告警详情
    
    Returns:
        bool: 发送是否成功
    
    Example:
        >>> push_feishu_alert("止损触发", "BTC 多单止损 @ $50000")
        True
    """
    notifier = get_notifier()
    return notifier.send_alert(title, message)


def push_feishu_report(title: str, content: str) -> bool:
    """
    发送报告到飞书 (便捷函数)
    
    Args:
        title: 报告标题
        content: 报告内容
    
    Returns:
        bool: 发送是否成功
    
    Example:
        >>> push_feishu_report("每日交易报告", "今日收益: +2.5%")
        True
    """
    notifier = get_notifier()
    return notifier.send_report(title, content)


def is_feishu_configured() -> bool:
    """检查飞书是否已配置"""
    return get_notifier().is_configured


# 向后兼容: 直接使用模块级函数
if __name__ == "__main__":
    # 测试代码
    import sys
    
    # 检查配置
    if not is_feishu_configured():
        print("⚠️ 飞书未配置，请设置环境变量:")
        print("   FEISHU_APP_ID")
        print("   FEISHU_APP_SECRET")
        print("   FEISHU_CHAT_ID (可选)")
        sys.exit(1)
    
    # 发送测试消息
    test_msg = "Miracle System 飞书通知测试\n时间: 测试"
    success = push_feishu(test_msg)
    print(f"测试消息发送{'成功' if success else '失败'}")
