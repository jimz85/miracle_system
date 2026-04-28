#!/usr/bin/env python3
from __future__ import annotations

"""
Miracle 2.0 - LLM Provider
===========================
统一的大语言模型接口，支持多后端切换（Claude/GPT/Gemini/DeepSeek/Ollama）
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==================== 枚举和配置 ====================

class LLMProviderType(Enum):
    """支持的LLM提供商"""
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


@dataclass
class Message:
    """对话消息"""
    role: str  # system/user/assistant
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    provider: LLMProviderType
    model: str
    tokens_used: int | None = None
    cost: float | None = None
    latency_ms: float | None = None
    raw_response: Dict[str, Any] | None = None
    error: str | None = None


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProviderType = LLMProviderType.CLAUDE
    model: str = "claude-3-5-sonnet-20241022"
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120
    max_retries: int = 3
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, provider: str | None = None) -> LLMConfig:
        """从环境变量加载配置"""
        if provider:
            provider_type = LLMProviderType(provider.lower())
        else:
            if os.getenv("ANTHROPIC_API_KEY"):
                provider_type = LLMProviderType.CLAUDE
            elif os.getenv("OPENAI_API_KEY"):
                provider_type = LLMProviderType.GPT
            elif os.getenv("GOOGLE_API_KEY"):
                provider_type = LLMProviderType.GEMINI
            elif os.getenv("DEEPSEEK_API_KEY"):
                provider_type = LLMProviderType.DEEPSEEK
            else:
                provider_type = LLMProviderType.OLLAMA
        
        config = cls(provider=provider_type)
        
        if provider_type == LLMProviderType.CLAUDE:
            config.api_key = os.getenv("ANTHROPIC_API_KEY")
            config.model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        elif provider_type == LLMProviderType.GPT:
            config.api_key = os.getenv("OPENAI_API_KEY")
            config.model = os.getenv("OPENAI_MODEL", "gpt-4o")
            config.base_url = os.getenv("OPENAI_BASE_URL")
        elif provider_type == LLMProviderType.GEMINI:
            config.api_key = os.getenv("GOOGLE_API_KEY")
            config.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        elif provider_type == LLMProviderType.DEEPSEEK:
            config.api_key = os.getenv("DEEPSEEK_API_KEY")
            config.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            config.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        elif provider_type == LLMProviderType.OLLAMA:
            config.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            config.model = os.getenv("OLLAMA_MODEL", "llama3")
        
        return config


# ==================== 抽象接口 ====================

class BaseLLMProvider(ABC):
    """LLM Provider 基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @abstractmethod
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """发送对话请求"""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """获取文本嵌入向量"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        pass
    
    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """准备消息格式"""
        return [msg.to_dict() for msg in messages]
    
    async def chat_simple(self, prompt: str, system: str | None = None) -> LLMResponse:
        """简单的对话接口"""
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return await self.chat(messages)


# ==================== Claude Provider ====================

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude Provider"""
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        import anthropic
        
        if not self._client:
            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=120
            )
        
        start_time = time.time()
        try:
            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                messages=self._prepare_messages(messages)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.content[0].text,
                provider=LLMProviderType.CLAUDE,
                model=self.config.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                cost=response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.015,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProviderType.CLAUDE,
                model=self.config.model,
                error=str(e)
            )
    
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts).tolist()
            return embeddings
        except ImportError:
            logger.warning("sentence-transformers not installed, using mock embeddings")
            import random
            dim = self.get_embedding_dimension()
            return [[random.random() for _ in range(dim)] for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        return 384


# ==================== GPT Provider ====================

class GPTProvider(BaseLLMProvider):
    """OpenAI GPT Provider"""
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        import openai
        
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=120
            )
        
        start_time = time.time()
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                messages=self._prepare_messages(messages)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            choice = response.choices[0]
            
            return LLMResponse(
                content=choice.message.content or "",
                provider=LLMProviderType.GPT,
                model=self.config.model,
                tokens_used=response.usage.total_tokens,
                cost=response.usage.total_tokens * 0.000015,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProviderType.GPT,
                model=self.config.model,
                error=str(e)
            )
    
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        import openai
        
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=120
            )
        
        try:
            response = await self._client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI Embedding error: {e}")
            import random
            dim = self.get_embedding_dimension()
            return [[random.random() for _ in range(dim)] for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        return 1536


# ==================== Gemini Provider ====================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini Provider"""
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        import google.genai as genai
        
        if not self._client:
            genai.configure(api_key=self.config.api_key)
            self._client = genai
        
        start_time = time.time()
        try:
            contents = []
            for msg in messages:
                if msg.role == "user":
                    contents.append({"role": "user", "parts": [msg.content]})
                elif msg.role == "assistant":
                    contents.append({"role": "model", "parts": [msg.content]})
                elif msg.role == "system":
                    contents.append({"role": "user", "parts": [msg.content]})
            
            model = self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
                generation_config={
                    "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature)
                }
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=model.text,
                provider=LLMProviderType.GEMINI,
                model=self.config.model,
                tokens_used=model.usage_metadata.total_token_count if hasattr(model, 'usage_metadata') else None,
                latency_ms=latency_ms,
                raw_response=model.to_dict() if hasattr(model, 'to_dict') else None
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProviderType.GEMINI,
                model=self.config.model,
                error=str(e)
            )
    
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        import google.genai as genai
        
        if not self._client:
            genai.configure(api_key=self.config.api_key)
            self._client = genai
        
        try:
            embeddings = []
            for text in texts:
                result = self._client.models.embed_content(
                    model="models/text-embedding-004",
                    contents=text
                )
                embeddings.append(result.embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Gemini Embedding error: {e}")
            import random
            dim = self.get_embedding_dimension()
            return [[random.random() for _ in range(dim)] for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        return 768


# ==================== DeepSeek Provider ====================

class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek Provider"""
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        import openai
        
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.deepseek.com",
                timeout=120
            )
        
        start_time = time.time()
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                messages=self._prepare_messages(messages)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            choice = response.choices[0]
            
            return LLMResponse(
                content=choice.message.content or "",
                provider=LLMProviderType.DEEPSEEK,
                model=self.config.model,
                tokens_used=response.usage.total_tokens,
                cost=response.usage.total_tokens * 0.000001,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProviderType.DEEPSEEK,
                model=self.config.model,
                error=str(e)
            )
    
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        import openai
        
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.deepseek.com",
                timeout=120
            )
        
        try:
            response = await self._client.embeddings.create(
                model="deepseek-embed",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"DeepSeek Embedding error: {e}")
            import random
            dim = self.get_embedding_dimension()
            return [[random.random() for _ in range(dim)] for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        return 1536


# ==================== Ollama Provider ====================

class OllamaProvider(BaseLLMProvider):
    """Ollama 本地模型 Provider"""
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        import aiohttp
        
        start_time = time.time()
        try:
            url = f"{self.config.base_url}/api/chat"
            
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            payload = {
                "model": self.config.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature)
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"Ollama error: {error_text}")
                    
                    response = await resp.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.get("message", {}).get("content", ""),
                provider=LLMProviderType.OLLAMA,
                model=self.config.model,
                tokens_used=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
                latency_ms=latency_ms,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProviderType.OLLAMA,
                model=self.config.model,
                error=str(e)
            )
    
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        import aiohttp
        
        try:
            url = f"{self.config.base_url}/api/embeddings"
            embeddings = []
            
            async with aiohttp.ClientSession() as session:
                for text in texts:
                    payload = {
                        "model": self.config.model,
                        "prompt": text
                    }
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            response = await resp.json()
                            embeddings.append(response.get("embedding", []))
                        else:
                            embeddings.append([0.0] * self.get_embedding_dimension())
            return embeddings
        except Exception as e:
            logger.error(f"Ollama Embedding error: {e}")
            import random
            dim = self.get_embedding_dimension()
            return [[random.random() for _ in range(dim)] for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        return 4096


# ==================== Provider 管理器 ====================

class LLMProviderManager:
    """LLM Provider 管理器 - 支持多Provider切换和自动故障转移"""
    
    _instance = None
    _providers: Dict[LLMProviderType, BaseLLMProvider] = {}
    _current_provider: LLMProviderType | None = None
    _fallback_provider: LLMProviderType | None = None
    _failure_count: Dict[LLMProviderType, int] = {}
    _max_retries: int = 3
    _auto_fallback_enabled: bool = True
    
    # Provider优先级列表（用于故障转移时的顺序）
    _provider_priority = [
        LLMProviderType.CLAUDE,
        LLMProviderType.GPT,
        LLMProviderType.DEEPSEEK,
        LLMProviderType.GEMINI,
        LLMProviderType.OLLAMA
    ]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._providers:
            self._init_providers()
            self._load_fallback_config()
    
    def _load_fallback_config(self):
        """从环境变量加载故障转移配置"""
        import os
        self._auto_fallback_enabled = os.getenv("LLM_AUTO_FALLBACK", "true").lower() == "true"
        self._max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        
        # 设置备用provider
        fallback_name = os.getenv("LLM_FALLBACK_PROVIDER", "deepseek")
        try:
            self._fallback_provider = LLMProviderType(fallback_name.lower())
        except ValueError:
            self._fallback_provider = LLMProviderType.DEEPSEEK
        
        # 设置主provider
        primary_name = os.getenv("LLM_PRIMARY_PROVIDER", "claude")
        try:
            primary = LLMProviderType(primary_name.lower())
            if primary in self._providers:
                self._current_provider = primary
        except ValueError:
            pass
        
        logger.info(f"LLM Fallback config: auto_fallback={self._auto_fallback_enabled}, "
                   f"max_retries={self._max_retries}, primary={self._current_provider}, "
                   f"fallback={self._fallback_provider}")
    
    def _init_providers(self):
        """初始化所有可用的Provider"""
        for provider_type in LLMProviderType:
            try:
                config = LLMConfig.from_env(provider_type.value)
                if config.api_key or provider_type == LLMProviderType.OLLAMA:
                    provider = self._create_provider(provider_type, config)
                    if provider:
                        self._providers[provider_type] = provider
                        self._failure_count[provider_type] = 0
                        logger.info(f"Initialized {provider_type.value} provider")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_type.value}: {e}")
        
        if self._providers:
            for priority in self._provider_priority:
                if priority in self._providers:
                    self._current_provider = priority
                    break
    
    def _create_provider(self, provider_type: LLMProviderType, config: LLMConfig) -> BaseLLMProvider | None:
        """创建指定类型的Provider"""
        if provider_type == LLMProviderType.CLAUDE:
            return ClaudeProvider(config)
        elif provider_type == LLMProviderType.GPT:
            return GPTProvider(config)
        elif provider_type == LLMProviderType.GEMINI:
            return GeminiProvider(config)
        elif provider_type == LLMProviderType.DEEPSEEK:
            return DeepSeekProvider(config)
        elif provider_type == LLMProviderType.OLLAMA:
            return OllamaProvider(config)
        return None
    
    def get_provider(self, provider_type: LLMProviderType | None = None) -> BaseLLMProvider | None:
        """获取指定Provider"""
        if provider_type is None:
            provider_type = self._current_provider
        return self._providers.get(provider_type)
    
    def set_provider(self, provider_type: LLMProviderType) -> bool:
        """切换当前Provider"""
        if provider_type in self._providers:
            self._current_provider = provider_type
            logger.info(f"Switched to {provider_type.value} provider")
            return True
        return False
    
    def _record_failure(self, provider_type: LLMProviderType):
        """记录provider失败次数"""
        if provider_type not in self._failure_count:
            self._failure_count[provider_type] = 0
        self._failure_count[provider_type] += 1
        logger.warning(f"{provider_type.value} failure count: {self._failure_count[provider_type]}/{self._max_retries}")
        
        if self._failure_count[provider_type] >= self._max_retries:
            self._trigger_fallback(provider_type)
    
    def _record_success(self, provider_type: LLMProviderType):
        """记录provider成功，重置失败计数"""
        if provider_type in self._failure_count:
            self._failure_count[provider_type] = 0
    
    def _trigger_fallback(self, failed_provider: LLMProviderType):
        """触发故障转移"""
        logger.warning(f"Triggering fallback from {failed_provider.value}")
        
        # 首先尝试配置的fallback provider
        if self._fallback_provider and self._fallback_provider in self._providers:
            if self._fallback_provider != failed_provider:
                self.set_provider(self._fallback_provider)
                logger.info(f"Fallback activated: using {self._fallback_provider.value}")
                return
        
        # 否则按优先级找一个可用的provider
        for provider in self._provider_priority:
            if provider in self._providers and provider != failed_provider:
                self.set_provider(provider)
                logger.info(f"Auto-fallback activated: using {provider.value}")
                return
        
        logger.error("No available fallback provider!")
    
    def reset_failure_count(self, provider_type: LLMProviderType | None = None):
        """重置失败计数"""
        if provider_type:
            self._failure_count[provider_type] = 0
        else:
            for pt in self._failure_count:
                self._failure_count[pt] = 0
    
    @property
    def current_provider(self) -> LLMProviderType | None:
        return self._current_provider
    
    @property
    def available_providers(self) -> List[LLMProviderType]:
        return list(self._providers.keys())
    
    async def chat(self, messages: List[Message], provider: LLMProviderType | None = None, **kwargs) -> LLMResponse:
        """使用指定或当前Provider发送对话（带自动降级）"""
        prov = self.get_provider(provider)
        if not prov:
            return LLMResponse(
                content="",
                provider=provider or LLMProviderType.CLAUDE,
                model="",
                error="No provider available"
            )
        
        # 如果启用自动降级，使用带降级的chat_with_fallback
        if self._auto_fallback_enabled and provider is None:
            return await self.chat_with_fallback(messages, **kwargs)
        
        return await prov.chat(messages, **kwargs)
    
    async def chat_with_fallback(self, messages: List[Message], **kwargs) -> LLMResponse:
        """带自动降级的chat方法 - 主力失败自动切换备用"""
        attempted_providers = []
        
        # 首先尝试当前配置的provider
        current = self._current_provider
        if current:
            attempted_providers.append(current)
            prov = self.get_provider(current)
            if prov:
                try:
                    response = await prov.chat(messages, **kwargs)
                    if not response.error:
                        self._record_success(current)
                        return response
                    else:
                        logger.warning(f"{current.value} returned error: {response.error}")
                        self._record_failure(current)
                except Exception as e:
                    logger.error(f"{current.value} exception: {e}")
                    self._record_failure(current)
        
        # 如果失败了，尝试fallback provider
        if self._fallback_provider and self._fallback_provider not in attempted_providers:
            attempted_providers.append(self._fallback_provider)
            prov = self.get_provider(self._fallback_provider)
            if prov:
                try:
                    logger.info(f"Trying fallback provider: {self._fallback_provider.value}")
                    response = await prov.chat(messages, **kwargs)
                    if not response.error:
                        self._record_success(self._fallback_provider)
                        # 切换到fallback作为新的主provider
                        self.set_provider(self._fallback_provider)
                        return response
                    else:
                        logger.warning(f"{self._fallback_provider.value} returned error: {response.error}")
                        self._record_failure(self._fallback_provider)
                except Exception as e:
                    logger.error(f"{self._fallback_provider.value} exception: {e}")
                    self._record_failure(self._fallback_provider)
        
        # 最后尝试所有其他可用的provider
        for provider_type in self._provider_priority:
            if provider_type in self._providers and provider_type not in attempted_providers:
                attempted_providers.append(provider_type)
                prov = self._providers[provider_type]
                try:
                    logger.info(f"Trying emergency fallback provider: {provider_type.value}")
                    response = await prov.chat(messages, **kwargs)
                    if not response.error:
                        self._record_success(provider_type)
                        self.set_provider(provider_type)
                        return response
                except Exception as e:
                    logger.error(f"{provider_type.value} exception: {e}")
        
        # 所有provider都失败了
        logger.error("All LLM providers failed!")
        return LLMResponse(
            content="",
            provider=current or LLMProviderType.CLAUDE,
            model="",
            error="All LLM providers failed"
        )
    
    async def embed(self, texts: List[str], provider: LLMProviderType | None = None, **kwargs) -> List[List[float]]:
        """获取文本嵌入"""
        prov = self.get_provider(provider)
        if not prov:
            import random
            return [[random.random() for _ in range(384)] for _ in texts]
        return await prov.embed(texts, **kwargs)
    
    async def chat_simple(self, prompt: str, system: str | None = None, 
                          provider: LLMProviderType | None = None) -> LLMResponse:
        """简单对话接口"""
        prov = self.get_provider(provider)
        if not prov:
            return LLMResponse(
                content="",
                provider=provider or LLMProviderType.CLAUDE,
                model="",
                error="No provider available"
            )
        return await prov.chat_simple(prompt, system)


# ==================== 全局单例 ====================

_llm_manager: LLMProviderManager | None = None

def get_llm_manager() -> LLMProviderManager:
    """获取LLM管理器单例"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMProviderManager()
    return _llm_manager

def get_llm_provider(provider: str | None = None) -> BaseLLMProvider | None:
    """获取LLM Provider快捷函数"""
    if provider is not None and isinstance(provider, str):
        try:
            provider = LLMProviderType(provider.lower())
        except ValueError:
            logger.warning(f"Unknown LLM provider: {provider}, using default")
            provider = None
    return get_llm_manager().get_provider(provider)
