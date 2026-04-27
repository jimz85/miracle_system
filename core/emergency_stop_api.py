#!/usr/bin/env python3
"""
Miracle System - Emergency Stop API
====================================
全局紧急停止API，用于在紧急情况下立即停止所有交易活动

Endpoints:
- GET  /health          - 健康检查
- POST /emergency/stop  - 触发紧急停止（需要认证）
- POST /emergency/resume - 恢复交易（需要认证）
- GET  /status          - 获取当前交易状态

Usage:
    python emergency_stop_api.py --port 8080 --token YOUR_TOKEN
"""

import argparse
import logging
import os
import sys
import json
import threading
from datetime import datetime
from typing import Optional
from flask import Flask, request, jsonify

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('emergency_stop_api')


# ============================================================
# 全局停止状态管理
# ============================================================

class EmergencyStopManager:
    """全局紧急停止管理器（单例）"""
    
    _instance = None
    _lock = threading.Lock()
    
    # 类级别的状态变量
    _emergency_stopped = False
    _stop_reason: Optional[str] = None
    _stop_time: Optional[str] = None
    _stopped_by: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped
    
    @property
    def stop_reason(self) -> Optional[str]:
        return self._stop_reason
    
    @property
    def stop_time(self) -> Optional[str]:
        return self._stop_time
    
    @property
    def stopped_by(self) -> Optional[str]:
        return self._stopped_by
    
    def trigger_emergency_stop(self, reason: str, triggered_by: str = "unknown") -> dict:
        """触发紧急停止"""
        with self._lock:
            self._emergency_stopped = True
            self._stop_reason = reason
            self._stop_time = datetime.now().isoformat()
            self._stopped_by = triggered_by
            
            logger.critical(f"🚨 EMERGENCY STOP TRIGGERED!")
            logger.critical(f"   Reason: {reason}")
            logger.critical(f"   Triggered by: {triggered_by}")
            logger.critical(f"   Time: {self._stop_time}")
            
            # 执行清理操作
            self._execute_cleanup()
            
            return {
                "status": "stopped",
                "reason": reason,
                "stop_time": self._stop_time,
                "stopped_by": triggered_by
            }
    
    def resume_trading(self, resumed_by: str = "unknown") -> dict:
        """恢复交易"""
        with self._lock:
            if not self._emergency_stopped:
                return {
                    "status": "not_stopped",
                    "message": "Trading was not stopped"
                }
            
            previous_stop = {
                "reason": self._stop_reason,
                "stop_time": self._stop_time,
                "stopped_by": self._stopped_by
            }
            
            self._emergency_stopped = False
            self._stop_reason = None
            self._stop_time = None
            self._stopped_by = None
            
            logger.info(f"✅ Trading resumed by {resumed_by}")
            logger.info(f"   Previous stop: {previous_stop}")
            
            return {
                "status": "resumed",
                "previous_stop": previous_stop,
                "resumed_by": resumed_by,
                "resumed_at": datetime.now().isoformat()
            }
    
    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "emergency_stopped": self._emergency_stopped,
            "reason": self._stop_reason,
            "stop_time": self._stop_time,
            "stopped_by": self._stopped_by,
            "server_time": datetime.now().isoformat()
        }
    
    def _execute_cleanup(self):
        """执行清理操作"""
        try:
            # 取消所有活跃订单
            self._cancel_all_orders()
            
            # 通知相关系统
            self._notify_systems()
            
            logger.info("Cleanup operations completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _cancel_all_orders(self):
        """取消所有活跃订单"""
        logger.info("Cancelling all active orders...")
        # 这里可以实现实际的取消订单逻辑
        # 取决于具体的交易平台接口
        pass
    
    def _notify_systems(self):
        """通知相关系统"""
        logger.info("Notifying other systems...")
        # 这里可以实现通知逻辑（如Feishu、邮件等）
        pass


# ============================================================
# Flask API 应用
# ============================================================

app = Flask(__name__)
stop_manager = EmergencyStopManager()

# API Token (should be set via environment variable or passed as argument)
API_TOKEN = os.getenv("EMERGENCY_STOP_TOKEN", "default_token_change_me")


def verify_token() -> bool:
    """验证API Token"""
    token = request.headers.get('Authorization', '')
    if token.startswith('Bearer '):
        token = token[7:]
    return token == API_TOKEN


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "server_time": datetime.now().isoformat()
    })


@app.route('/status', methods=['GET'])
def get_status():
    """获取当前交易状态"""
    status = stop_manager.get_status()
    return jsonify(status)


@app.route('/emergency/stop', methods=['POST'])
def emergency_stop():
    """
    触发紧急停止
    
    Headers:
        Authorization: Bearer <API_TOKEN>
    
    Body (JSON):
        reason: 停止原因 (optional, default: "Manual emergency stop")
    
    Returns:
        JSON: {"status": "stopped", "reason": "...", "stop_time": "...", "stopped_by": "..."}
    """
    # 验证Token
    if not verify_token():
        logger.warning(f"Unauthorized emergency stop attempt from {request.remote_addr}")
        return jsonify({
            "error": "Unauthorized",
            "message": "Invalid or missing API token"
        }), 401
    
    # 获取停止原因
    data = request.get_json() or {}
    reason = data.get('reason', 'Manual emergency stop')
    triggered_by = data.get('triggered_by', request.remote_addr)
    
    # 触发紧急停止
    result = stop_manager.trigger_emergency_stop(reason, triggered_by)
    
    logger.critical(f"Emergency stop API called by {triggered_by}: {reason}")
    
    return jsonify(result)


@app.route('/emergency/resume', methods=['POST'])
def resume_trading():
    """
    恢复交易
    
    Headers:
        Authorization: Bearer <API_TOKEN>
    
    Body (JSON):
        resumed_by: 恢复操作人 (optional, default: request remote address)
    
    Returns:
        JSON: {"status": "resumed", "previous_stop": {...}, "resumed_by": "...", "resumed_at": "..."}
    """
    # 验证Token
    if not verify_token():
        logger.warning(f"Unauthorized resume attempt from {request.remote_addr}")
        return jsonify({
            "error": "Unauthorized",
            "message": "Invalid or missing API token"
        }), 401
    
    # 获取恢复操作人
    data = request.get_json() or {}
    resumed_by = data.get('resumed_by', request.remote_addr)
    
    # 恢复交易
    result = stop_manager.resume_trading(resumed_by)
    
    logger.info(f"Resume trading API called by {resumed_by}")
    
    return jsonify(result)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================
# 主程序入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Miracle Emergency Stop API Server')
    parser.add_argument('--host', '-H', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Port to bind to (default: 8080)')
    parser.add_argument('--token', '-t', default=None, help='API token (default: from EMERGENCY_STOP_TOKEN env var)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置API Token
    global API_TOKEN
    if args.token:
        API_TOKEN = args.token
    else:
        API_TOKEN = os.getenv("EMERGENCY_STOP_TOKEN", "default_token_change_me")
    
    logger.info("=" * 60)
    logger.info("Miracle Emergency Stop API Server")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"API Token: {'*' * len(API_TOKEN) if API_TOKEN != 'default_token_change_me' else 'NOT SET (using default)'}")
    logger.info("=" * 60)
    logger.info("Endpoints:")
    logger.info("  GET  /health          - Health check")
    logger.info("  GET  /status          - Get trading status")
    logger.info("  POST /emergency/stop   - Trigger emergency stop")
    logger.info("  POST /emergency/resume - Resume trading")
    logger.info("=" * 60)
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
