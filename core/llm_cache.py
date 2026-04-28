from __future__ import annotations

"""
LLM Cache - LLM请求缓存层
使用Redis减少重复API调用，降低成本提高响应速度
"""
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

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
    
    def _make_key(self, prompt: str, provider: str, model: str | None = None) -> str:
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
        model: str | None = None
    ) -> Dict[str, Any] | None:
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
        model: str | None = None,
        ttl: int | None = None
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
        model: str | None = None
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

_global_cache: LLMCache | None = None


def get_llm_cache() -> LLMCache:
    """获取LLM缓存单例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache()
    return _global_cache


# ========================
# 便捷函数
# ========================

def cache_get(prompt: str, provider: str, model: str | None = None) -> Any | None:
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
    model: str | None = None,
    ttl: int | None = None
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
        model: str | None = None,
        ttl: int | None = None,
        cache: LLMCache | None = None
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


# ============================================================
# SQLite-based LLM Response Cache (P3-3)
# ============================================================

import sqlite3
import threading
import time
from pathlib import Path


_LLM_CACHE_TTL_SECONDS = 3600  # 1 hour


class LLMResponseCache:
    """
    SQLite-based LLM响应缓存
    
    特性:
    - key = SHA256(prompt + state_json_sorted) 防止状态差异缓存污染
    - TTL = 1小时（可配置）
    - JSON Schema校验（可选）
    - 线程安全
    
    用法:
        cache = LLMResponseCache()
        cache.set(prompt="Hello", state={"mode":"trade"}, response={"text":"Hi"})
        result = cache.get(prompt="Hello", state={"mode":"trade"})
    """
    
    def __init__(
        self,
        db_path: str | None = None,
        ttl_seconds: int = _LLM_CACHE_TTL_SECONDS,
        schema: Dict[str, Any] | None = None,  # JSON Schema for validation
        verbose: bool = False
    ):
        """
        Args:
            db_path: SQLite数据库路径，默认 ~/.miracle_memory/llm_cache.db
            ttl_seconds: 缓存过期时间（秒），默认3600（1小时）
            schema: JSON Schema dict，用于校验response结构
            verbose: 打印详细日志
        """
        if db_path is None:
            db_path = os.path.expanduser("~/.miracle_memory/llm_cache.db")
        
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.schema = schema
        self.verbose = verbose
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        
        self._init_db()
        self._stats = {"hits": 0, "misses": 0, "errors": 0, "schema_rejects": 0}
    
    def _init_db(self):
        """初始化缓存表"""
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                state_json TEXT,  -- JSON serialized state
                response TEXT NOT NULL,  -- JSON serialized response
                provider TEXT,
                model TEXT,
                created_at REAL NOT NULL,  -- Unix timestamp
                expires_at REAL NOT NULL   -- Unix timestamp
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires ON llm_cache(expires_at)")
        self._conn.commit()
    
    def _make_key(self, prompt: str, state: Dict[str, Any] | None = None) -> str:
        """生成缓存key: SHA256(prompt + sorted_state_json)"""
        state_str = json.dumps(state or {}, sort_keys=True, ensure_ascii=False)
        combined = f"{prompt}|{state_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _validate_response(self, response: Any) -> bool:
        """使用JSON Schema校验response结构"""
        if self.schema is None:
            return True
        
        try:
            import jsonschema
            jsonschema.validate(instance=response, schema=self.schema)
            return True
        except ImportError:
            # Fallback: manual schema check
            return self._manual_schema_validate(response, self.schema)
        except Exception as e:
            self._log(f"Schema validation failed: {e}")
            return False
    
    def _manual_schema_validate(self, instance: Any, schema: Dict[str, Any]) -> bool:
        """Minimal JSON Schema validation (type + required fields)"""
        # Only check 'type' and 'required' for simplicity
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(instance, dict):
            return False
        if expected_type == "string" and not isinstance(instance, str):
            return False
        if expected_type == "array" and not isinstance(instance, list):
            return False
        
        required = schema.get("required", [])
        if isinstance(instance, dict):
            for field in required:
                if field not in instance:
                    return False
        return True
    
    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)
    
    def get(
        self,
        prompt: str,
        state: Dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None
    ) -> Optional[Any]:
        """
        获取缓存的LLM响应
        
        Args:
            prompt: 输入提示
            state: 状态字典（key的一部分）
            provider: LLM提供商
            model: 模型名称
        
        Returns:
            缓存的响应，或None（未命中/过期/校验失败）
        """
        key = self._make_key(prompt, state)
        
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM llm_cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
        
        if row is None:
            self._stats["misses"] += 1
            self._log(f"Cache miss: {key[:16]}...")
            return None
        
        # 检查TTL
        now = time.time()
        if now > row["expires_at"]:
            self._delete_key(key)
            self._stats["misses"] += 1
            self._log(f"Cache expired: {key[:16]}...")
            return None
        
        # 解析response
        try:
            response = json.loads(row["response"])
        except Exception as e:
            self._delete_key(key)
            self._stats["errors"] += 1
            self._log(f"Cache corrupt (delete): {e}")
            return None
        
        # JSON Schema校验
        if not self._validate_response(response):
            self._delete_key(key)
            self._stats["schema_rejects"] += 1
            self._log(f"Schema validation failed, deleted: {key[:16]}...")
            return None
        
        self._stats["hits"] += 1
        self._log(f"Cache hit: {key[:16]}...")
        return response
    
    def set(
        self,
        prompt: str,
        response: Any,
        state: Dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        ttl: int | None = None
    ) -> bool:
        """
        缓存LLM响应
        
        Args:
            prompt: 输入提示
            response: LLM响应（会被JSON序列化）
            state: 状态字典
            provider: LLM提供商
            model: 模型名称
            ttl: 过期秒数（默认1小时）
        
        Returns:
            是否成功
        """
        key = self._make_key(prompt, state)
        ttl = ttl or self.ttl_seconds
        
        # JSON Schema校验（存储前）
        if not self._validate_response(response):
            self._stats["schema_rejects"] += 1
            self._log(f"Response failed schema validation, not caching")
            return False
        
        now = time.time()
        expires_at = now + ttl
        
        try:
            response_json = json.dumps(response, ensure_ascii=False)
            state_json = json.dumps(state or {}, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            self._stats["errors"] += 1
            self._log(f"JSON serialization failed: {e}")
            return False
        
        with self._write_lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO llm_cache 
                (key, prompt, state_json, response, provider, model, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (key, prompt, state_json, response_json, provider, model, now, expires_at))
            self._conn.commit()
        
        self._log(f"Cache set: {key[:16]}... TTL={ttl}s")
        return True
    
    def _delete_key(self, key: str):
        """删除指定key"""
        with self._write_lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
            self._conn.commit()
    
    def delete(
        self,
        prompt: str,
        state: Dict[str, Any] | None = None
    ) -> bool:
        """删除缓存"""
        key = self._make_key(prompt, state)
        self._delete_key(key)
        return True
    
    def clear_expired(self) -> int:
        """清理过期缓存，返回删除数量"""
        now = time.time()
        with self._write_lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM llm_cache WHERE expires_at < ?", (now,))
            deleted = cursor.rowcount
            self._conn.commit()
        if deleted > 0:
            self._log(f"Cleared {deleted} expired cache entries")
        return deleted
    
    def clear_all(self) -> bool:
        """清空所有缓存"""
        with self._write_lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM llm_cache")
            self._conn.commit()
        self._log("Cleared all cache entries")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            **self._stats,
            "total_queries": total,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
            "db_path": self.db_path
        }
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {"hits": 0, "misses": 0, "errors": 0, "schema_rejects": 0}
    
    def close(self):
        """关闭数据库连接"""
        self._conn.close()


# ============================================================
# 全局单例
# ============================================================

_llm_response_cache: LLMResponseCache | None = None


def get_llm_response_cache(
    db_path: str | None = None,
    ttl_seconds: int = _LLM_CACHE_TTL_SECONDS,
    schema: Dict[str, Any] | None = None
) -> LLMResponseCache:
    """获取LLMResponseCache全局单例"""
    global _llm_response_cache
    if _llm_response_cache is None:
        _llm_response_cache = LLMResponseCache(
            db_path=db_path,
            ttl_seconds=ttl_seconds,
            schema=schema
        )
    return _llm_response_cache


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
