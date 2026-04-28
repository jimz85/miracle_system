from __future__ import annotations

"""
ExecutorConfig - 执行器配置
===========================

从 agents/agent_executor.py 提取到 core/ 模块

用法:
    from core.executor_config import ExecutorConfig
    from agents.agent_executor import ExecutorConfig  # 向后兼容导入
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from core.secure_key_manager import get_key_manager


@dataclass
class ExecutorConfig:
    """执行器配置"""
    # 交易所配置
    default_exchange: str = "okx"           # 默认交易所
    use_backup_on_fail: bool = True          # 失败时切换备用交易所

    # API配置 (OKX) - 现在通过SecureKeyManager管理，不再直接存储
    okx_testnet: bool = False  # OKX永续无testnet URL，用x-simulated-trading:1头切换模拟盘

    def __post_init__(self):
        # 如果未显式传入okx_testnet，则从环境变量推断
        # 让ExchangeClient与okx_req保持一致的模拟盘头
        pass  # 占位，后续需要时启用

    # API配置 (Binance) - 现在通过SecureKeyManager管理
    binance_testnet: bool = True

    # 交易配置
    max_retry: int = 3                       # 最大重试次数
    retry_interval: float = 1.0              # 重试间隔(秒)
    order_timeout: float = 10.0              # 订单超时(秒)
    max_loss_per_trade_pct: float = 1.0      # 每笔交易最大损失百分比(1%)

    # 滑点配置
    slippage_warning_threshold: float = 0.01  # 滑点警告阈值(1%)

    # 飞书通知
    feishu_webhook: str = ""
    feishu_enabled: bool = False

    # 监控配置
    monitor_interval: float = 1.0            # 监控间隔(秒)
    max_hold_hours: float = 24.0             # 最大持仓时间(小时)

    # 日志配置
    log_dir: str = "logs"
    trade_log_file: str = "trades.json"
    slippage_log_file: str = "slippage.json"

    def get_okx_keys(self) -> Tuple[str | None, str | None, str | None]:
        """获取OKX API密钥（从安全密钥管理器）"""
        km = get_key_manager()
        return (
            km.get_key("okx", "api_key"),
            km.get_key("okx", "secret_key"),
            km.get_key("okx", "passphrase")
        )

    def get_binance_keys(self) -> Tuple[str | None, str | None]:
        """获取Binance API密钥（从安全密钥管理器）"""
        km = get_key_manager()
        return (
            km.get_key("binance", "api_key"),
            km.get_key("binance", "secret_key")
        )
