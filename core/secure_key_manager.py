from __future__ import annotations

"""
SecureKeyManager - 安全密钥管理器
================================

支持:
1. 环境变量读取（优先）
2. 加密文件存储
3. 运行时解密

从 agents/agent_executor.py 提取到 core/ 模块
以减少 agent_executor.py 的体积 (1956行 → 拆分)

用法:
    from core.secure_key_manager import SecureKeyManager, get_key_manager
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SecureKeyManager:
    """
    安全密钥管理器

    支持:
    1. 环境变量读取（优先）
    2. 加密文件存储
    3. 运行时解密
    """

    def __init__(self, encryption_key_path: str | None = None):
        self.encryption_key = self._load_or_create_key(encryption_key_path)
        self.cipher = None
        if self.encryption_key:
            try:
                from cryptography.fernet import Fernet
                self.cipher = Fernet(self.encryption_key)
            except ImportError:
                logger.warning("cryptography库未安装，加密功能不可用")
        self._keys_cache = {}

    def _load_or_create_key(self, key_path: str | None) -> bytes | None:
        """加载或创建加密密钥"""
        if key_path and Path(key_path).exists():
            with open(key_path, 'rb') as f:
                return f.read()
        # 尝试从环境变量获取主密钥
        master_key = os.getenv("MIRACLE_MASTER_KEY")
        if master_key:
            return base64.urlsafe_b64decode(master_key)
        return None

    def get_key(self, exchange: str, key_type: str) -> str | None:
        """
        获取API密钥

        优先级:
        1. 环境变量
        2. 加密配置文件
        3. 返回None
        """
        cache_key = f"{exchange}_{key_type}"
        if cache_key in self._keys_cache:
            return self._keys_cache[cache_key]

        # 1. 尝试环境变量
        exchange_upper = exchange.upper()
        key_type_upper = key_type.upper()
        # OKX的特殊映射：secret_key → OKX_SECRET（不是OKX_SECRET_KEY）
        env_key_map = {
            ('OKX', 'SECRET_KEY'): 'OKX_SECRET',
            ('OKX', 'PASSPHRASE'): 'OKX_PASSPHRASE',
        }
        env_key = env_key_map.get((exchange_upper, key_type_upper),
                                  f"{exchange_upper}_{key_type_upper}")
        value = os.getenv(env_key)
        if value:
            self._keys_cache[cache_key] = value
            return value

        # 2. 尝试加密配置文件
        encrypted_file = Path(__file__).parent.parent / "data" / f".keys_{exchange}.enc"
        if encrypted_file.exists() and self.cipher:
            try:
                with open(encrypted_file, 'rb') as f:
                    encrypted_data = f.read()
                value = self.cipher.decrypt(encrypted_data).decode()
                self._keys_cache[cache_key] = value
                return value
            except Exception as e:
                logger.warning(f"解密密钥文件失败: {e}")

        return None

    def set_key(self, exchange: str, key_type: str, value: str):
        """存储API密钥（加密）"""
        if not self.cipher:
            raise RuntimeError("加密不可用，请设置MIRACLE_MASTER_KEY环境变量")

        encrypted_file = Path(__file__).parent.parent / "data" / f".keys_{exchange}.enc"
        encrypted_file.parent.mkdir(parents=True, exist_ok=True)
        encrypted = self.cipher.encrypt(value.encode())
        with open(encrypted_file, 'wb') as f:
            f.write(encrypted)

        self._keys_cache[f"{exchange}_{key_type}"] = value

    def clear_cache(self):
        """清除密钥缓存"""
        self._keys_cache = {}


# 全局密钥管理器实例
_key_manager: SecureKeyManager | None = None


def get_key_manager() -> SecureKeyManager:
    """获取全局密钥管理器"""
    global _key_manager
    if _key_manager is None:
        _key_manager = SecureKeyManager()
    return _key_manager
