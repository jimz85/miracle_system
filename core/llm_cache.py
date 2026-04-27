"""
LLM Cache - LLM请求缓存层
使用Redis减少重复API调用，降低成本提高响应速度
"""
import hashlib
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 尝试导入Redis
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("redis not installed, LLM cache disabled")


# ========================
# 缓存配置
# ========================

DEFAULT_TTL = {
    "ollama": 600,      # 10分钟 本地模型
    "minimax": 300,     # 5分钟 远程模型
    "claude": 600,      # 10分钟
    "gpt4": 600,        # 10分钟
    "gemini": 300,      # 5分钟
    "deepseek": 300,    # 5分钟
}

DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ========================
# LLM缓存类
# ========================

class LLMCache:
    """
    LLM响应缓存
    
    功能:
    - 基于prompt哈希的缓存键
    - 按provider分开缓存
    - TTL自动过期
    - 缓存统计
    """
    
    def __init__(
        self,
        redis_url: str = DEFAULT_REDIS_URL,
        enabled: bool = True
    ):
        self.enabled = enabled and HAS_REDIS
        self.redis_url = redis_url
        self._client = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0
        }
        
        if self.enabled:
            try:
                self._client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                # 测试连接
                self._client.ping()
                logger.info("LLM cache connected to Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, cache disabled")
                self.enabled = False
    
    @property
    def client(self):
        return self._client
    
    @property
    def stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "total": total,
            "hit_rate": hit_rate,
            "enabled": self.enabled
        }
    
    def _make_key(self, prompt: str, provider: str, model: Optional[str] = None) -> str:
        """生成缓存键"""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:32]
        parts = ["llm", provider]
        if model:
            parts.append(model)
        parts.append(prompt_hash)
        return ":".join(parts)
    
    def get(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存的响应
        
        Args:
            prompt: 输入提示
            provider: LLM提供商 (ollama, minimax, claude等)
            model: 模型名称 (可选)
        
        Returns:
            缓存的响应数据，或None
        """
        if not self.enabled:
            return None
        
        try:
            key = self._make_key(prompt, provider, model)
            data = self._client.get(key)
            
            if data:
                self._stats["hits"] += 1
                logger.debug(f"Cache hit: {key[:50]}...")
                return json.loads(data)
            else:
                self._stats["misses"] += 1
                return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        prompt: str,
        response: Any,
        provider: str,
        model: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        缓存响应
        
        Args:
            prompt: 输入提示
            response: LLM响应内容
            provider: LLM提供商
            model: 模型名称 (可选)
            ttl: 过期秒数 (默认按provider)
        
        Returns:
            是否成功
        """
        if not self.enabled:
            return False
        
        try:
            key = self._make_key(prompt, provider, model)
            ttl = ttl or DEFAULT_TTL.get(provider, 300)
            
            # 构建缓存数据
            cache_data = {
                "response": response,
                "provider": provider,
                "model": model,
                "cached_at": datetime.now().isoformat(),
                "prompt_hash": key.split(":")[-1]
            }
            
            self._client.setex(key, ttl, json.dumps(cache_data))
            logger.debug(f"Cache set: {key[:50]}... (TTL={ttl}s)")
            return True
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None
    ) -> bool:
        """删除缓存"""
        if not self.enabled:
            return False
        
        try:
            key = self._make_key(prompt, provider, model)
            self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_provider(self, provider: str) -> int:
        """清除provider的所有缓存"""
        if not self.enabled:
            return 0
        
        try:
            pattern = f"llm:{provider}:*"
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """清除所有LLM缓存"""
        if not self.enabled:
            return False
        
        try:
            pattern = "llm:*"
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Cache clear all error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.stats
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {"hits": 0, "misses": 0, "errors": 0}


# ========================
# 全局单例
# ========================

_global_cache: Optional[LLMCache] = None


def get_llm_cache() -> LLMCache:
    """获取LLM缓存单例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache()
    return _global_cache


# ========================
# 便捷函数
# ========================

def cache_get(prompt: str, provider: str, model: Optional[str] = None) -> Optional[Any]:
    """获取缓存的LLM响应"""
    cache = get_llm_cache()
    data = cache.get(prompt, provider, model)
    if data:
        return data.get("response")
    return None


def cache_set(
    prompt: str,
    response: Any,
    provider: str,
    model: Optional[str] = None,
    ttl: Optional[int] = None
) -> bool:
    """缓存LLM响应"""
    cache = get_llm_cache()
    return cache.set(prompt, response, provider, model, ttl)


def cache_stats() -> Dict[str, Any]:
    """获取缓存统计"""
    return get_llm_cache().get_stats()


# ========================
# LLM调用包装器
# ========================

class CachedLLMCaller:
    """
    带缓存的LLM调用器
    
    用法:
    ```python
    caller = CachedLLMCaller(provider="ollama", model="gemma3:4b")
    
    # 第一次调用 (miss)
    response = caller.chat("Hello")
    
    # 第二次调用 (hit)
    response = caller.chat("Hello")  # 直接从缓存返回
    ```
    """
    
    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        ttl: Optional[int] = None,
        cache: Optional[LLMCache] = None
    ):
        self.provider = provider
        self.model = model
        self.ttl = ttl
        self._cache = cache or get_llm_cache()
        self._llm_provider = None
    
    def _get_llm_provider(self):
        """获取LLM provider (延迟导入)"""
        if self._llm_provider is None:
            try:
                from core.llm_provider import get_llm_provider
                self._llm_provider = get_llm_provider()
            except ImportError:
                return None
        return self._llm_provider
    
    def chat(self, prompt: str, **kwargs) -> str:
        """
        发送聊天请求 (带缓存)
        
        Args:
            prompt: 输入提示
            **kwargs: 传递给LLM的其他参数
        
        Returns:
            LLM响应字符串
        """
        # 尝试从缓存获取
        cached = self._cache.get(prompt, self.provider, self.model)
        if cached:
            return cached.get("response")
        
        # 调用LLM
        llm = self._get_llm_provider()
        if llm is None:
            raise RuntimeError(f"LLM provider {self.provider} not available")
        
        response = llm.chat(prompt, **kwargs)
        
        # 缓存响应
        self._cache.set(
            prompt=prompt,
            response=response,
            provider=self.provider,
            model=self.model,
            ttl=self.ttl
        )
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self._cache.get_stats()


if __name__ == "__main__":
    # 测试
    print("=== LLM Cache Test ===\n")
    
    cache = get_llm_cache()
    print(f"Cache enabled: {cache.enabled}")
    print(f"Stats: {cache.stats}")
    
    if cache.enabled:
        # 测试缓存
        test_prompt = "What is the meaning of life?"
        
        # 第一次 (miss)
        result = cache.get(test_prompt, "test")
        print(f"First get: {result}")
        
        # 设置缓存
        success = cache.set(test_prompt, "42", "test", ttl=60)
        print(f"Set cache: {success}")
        
        # 第二次 (hit)
        result = cache.get(test_prompt, "test")
        print(f"Second get: {result}")
        
        # 统计
        print(f"Stats: {cache.get_stats()}")
    else:
        print("Redis not available, cache disabled")
    
    print("\n=== Test complete ===")
