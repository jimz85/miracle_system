"""
Miracle 1.0.1 - 配置管理器
统一配置管理系统

职责:
1. 集中管理所有配置项
2. 支持配置热重载
3. 配置验证
4. 嵌套键访问
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Dict

logger = logging.getLogger("miracle")


class ConfigError(Exception):
    """配置错误异常"""
    pass


class ConfigManager:
    """
    统一配置管理器

    使用单例模式，确保全局配置一致
    """

    _instance: Optional['ConfigManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None
        self._initialized = True
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        config_path = Path(__file__).parent.parent / "miracle_config.json"
        self._config_path = config_path

        if not config_path.exists():
            raise ConfigError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            logger.info(f"配置加载成功: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"配置文件JSON格式错误: {e}")
        except Exception as e:
            raise ConfigError(f"加载配置文件失败: {e}")

        # 验证必需的配置项
        self._validate_config()

    def _validate_config(self):
        """验证配置完整性"""
        required_sections = ["trading", "position", "leverage", "risk", "factors"]
        missing = [s for s in required_sections if s not in self._config]
        if missing:
            raise ConfigError(f"配置缺少必需section: {missing}")

    def get(self, *keys, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套访问

        Examples:
            config.get("risk", "max_loss_per_trade_pct")
            config.get("factors", "price_momentum", "weight")
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, *keys, value: Any):
        """
        设置配置值（仅在内存中，不持久化）

        Examples:
            config.set("risk", "max_loss_per_trade_pct", 0.02)
        """
        if len(keys) < 2:
            raise ConfigError("set方法至少需要2个参数: key和value")

        target = self._config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        target[keys[-1]] = value
        logger.debug(f"配置已更新: {'.'.join(keys)} = {value}")

    def reload(self):
        """重新加载配置"""
        logger.info("重新加载配置...")
        self._load_config()

    @property
    def trading(self) -> Dict[str, Any]:
        """获取trading配置"""
        return self._config.get("trading", {})

    @property
    def position(self) -> Dict[str, Any]:
        """获取position配置"""
        return self._config.get("position", {})

    @property
    def leverage(self) -> Dict[str, Any]:
        """获取leverage配置"""
        return self._config.get("leverage", {})

    @property
    def risk(self) -> Dict[str, Any]:
        """获取risk配置"""
        return self._config.get("risk", {})

    @property
    def factors(self) -> Dict[str, Any]:
        """获取factors配置"""
        return self._config.get("factors", {})

    def to_dict(self) -> Dict[str, Any]:
        """返回配置副本"""
        return self._config.copy()


# 全局配置实例
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def reload_config():
    """重新加载全局配置"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()
