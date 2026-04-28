# 开发指南

本文档包含 Miracle 2.0 的开发相关说明。

## 添加新的Agent

```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    """自定义Agent"""

    async def execute(self, context):
        """执行逻辑

        Args:
            context: 执行上下文

        Returns:
            dict: 执行结果
        """
        # 执行逻辑
        return {"result": "..."}

    async def validate(self, context) -> bool:
        """验证上下文是否有效"""
        return True
```

## 添加新的工具

```python
from core.tools import register_tool

@register_tool
async def my_tool(param1, param2):
    """工具描述

    Args:
        param1: 参数1描述
        param2: 参数2描述

    Returns:
        工具执行结果
    """
    return await do_something(param1, param2)
```

## 添加新的LLM Provider

```python
from core.llm_provider import LLMProvider, register_provider

@register_provider("my_provider")
class MyProvider(LLMProvider):
    """自定义LLM Provider"""

    async def complete(self, prompt: str, **kwargs) -> str:
        """生成补全"""
        # 调用自定义API
        return result

    async def embed(self, text: str) -> list:
        """生成嵌入向量"""
        return embedding
```

## 测试指南

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_circuit_breaker.py -v

# 运行带覆盖率的测试
python -m pytest tests/ --cov=. --cov-report=term-missing

# 跳过网络测试
python -m pytest tests/ -m "not network"
```

## 添加新的技术指标

在 `agents/agent_signal.py` 中添加新指标：

```python
def calc_custom_indicator(prices: List[float], param: float) -> float:
    """自定义指标计算

    Args:
        prices: 价格序列
        param: 指标参数

    Returns:
        指标值
    """
    # 实现指标计算逻辑
    return value
```

## 代码规范

- 使用 `ruff` 进行代码格式化 (`ruff check` + `ruff format`)
- 类型注解必须完整
- 所有公开函数必须有docstring
- 单元测试覆盖率目标: >70%

## 提交规范

```
<type>(<scope>): <subject>

Types:
- feat: 新功能
- fix: Bug修复
- refactor: 重构
- test: 测试相关
- docs: 文档相关
- chore: 构建/工具变更
```

## 环境变量配置

```bash
# .env 文件模板
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
OKX_API_KEY=...
OKX_SECRET_KEY=...
OKX_PASSPHRASE=...
FEISHU_WEBHOOK_URL=https://open.feishu.cn/...
MIRACLE_MODE=simulation  # simulation | paper | live
```
