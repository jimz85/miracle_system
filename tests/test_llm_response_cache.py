"""
Tests for core/llm_cache.py - LLMResponseCache (SQLite-based)
==============================================================
"""
from __future__ import annotations

import json
import tempfile
import time
import pytest
from pathlib import Path


class TestLLMResponseCache:
    """Tests for SQLite-based LLMResponseCache"""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Create a test cache instance with temp DB"""
        from core.llm_cache import LLMResponseCache
        db_path = str(tmp_path / "test_llm_cache.db")
        cache = LLMResponseCache(
            db_path=db_path,
            ttl_seconds=60,
            verbose=False
        )
        yield cache
        cache.close()
    
    def test_set_and_get(self, cache):
        """Basic set/get roundtrip"""
        prompt = "What is Bitcoin?"
        state = {"mode": "trade", "coin": "BTC"}
        response = {"text": "Bitcoin is a cryptocurrency.", "price": 72000}
        
        # First get should miss
        result = cache.get(prompt, state)
        assert result is None
        
        # Set cache
        success = cache.set(prompt, response, state)
        assert success is True
        
        # Second get should hit
        result = cache.get(prompt, state)
        assert result == response
    
    def test_different_state_different_cache(self, cache):
        """Different state should produce different cache entry"""
        prompt = "What is Bitcoin?"
        response1 = {"text": "Mode: trade"}
        response2 = {"text": "Mode: research"}
        
        cache.set(prompt, response1, state={"mode": "trade"})
        cache.set(prompt, response2, state={"mode": "research"})
        
        result1 = cache.get(prompt, state={"mode": "trade"})
        result2 = cache.get(prompt, state={"mode": "research"})
        
        assert result1["text"] == "Mode: trade"
        assert result2["text"] == "Mode: research"
    
    def test_hash_key_deterministic(self, cache):
        """Same prompt+state should always produce same key"""
        prompt = "Hello"
        state = {"a": 1, "b": 2}
        
        key1 = cache._make_key(prompt, state)
        key2 = cache._make_key(prompt, state)
        
        assert key1 == key2
    
    def test_hash_key_order_independent(self, cache):
        """State key order should not affect hash"""
        prompt = "Hello"
        
        key1 = cache._make_key(prompt, {"a": 1, "b": 2})
        key2 = cache._make_key(prompt, {"b": 2, "a": 1})
        
        assert key1 == key2  # sort_keys=True ensures this
    
    def test_ttl_expired(self, tmp_path):
        """Cache entry should expire after TTL"""
        from core.llm_cache import LLMResponseCache
        
        db_path = str(tmp_path / "test_ttl.db")
        cache = LLMResponseCache(db_path=db_path, ttl_seconds=1, verbose=False)
        
        cache.set("test", {"result": "data"})
        
        # Immediate get should work
        result = cache.get("test")
        assert result == {"result": "data"}
        
        # Wait for TTL to expire
        time.sleep(1.5)
        
        # Should now be None (expired)
        result = cache.get("test")
        assert result is None
        
        cache.close()
    
    def test_delete(self, cache):
        """Delete should remove cache entry"""
        cache.set("test", {"data": "value"})
        assert cache.get("test") == {"data": "value"}
        
        cache.delete("test")
        assert cache.get("test") is None
    
    def test_clear_expired(self, tmp_path):
        """clear_expired should remove only expired entries"""
        from core.llm_cache import LLMResponseCache
        
        db_path = str(tmp_path / "test_clear.db")
        
        # Entry with 10 second TTL
        cache1 = LLMResponseCache(db_path=db_path, ttl_seconds=10, verbose=False)
        cache1.set("keep", {"data": "keep"}, ttl=10)
        cache1.close()
        
        # Entry with 1 second TTL
        cache2 = LLMResponseCache(db_path=db_path, ttl_seconds=10, verbose=False)
        cache2.set("expire", {"data": "expire"}, ttl=1)
        time.sleep(1.5)
        
        deleted = cache2.clear_expired()
        assert deleted == 1
        
        # "keep" should still exist
        assert cache2.get("keep") == {"data": "keep"}
        
        # "expire" should be gone
        assert cache2.get("expire") is None
        cache2.close()
    
    def test_clear_all(self, cache):
        """clear_all should remove all entries"""
        cache.set("a", {"1": "one"})
        cache.set("b", {"2": "two"})
        cache.set("c", {"3": "three"})
        
        assert cache.get("a") is not None
        cache.clear_all()
        assert cache.get("a") is None
        assert cache.get("b") is None
    
    def test_stats(self, cache):
        """Stats should track hits/misses"""
        cache.set("test", {"data": "value"})
        cache.get("test")  # hit
        cache.get("missing")  # miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_queries"] == 2
        assert stats["hit_rate"] == 0.5
    
    def test_schema_validation_object(self, cache):
        """Schema validation with object type"""
        from core.llm_cache import LLMResponseCache
        
        schema = {
            "type": "object",
            "required": ["text", "price"]
        }
        
        db_path = cache.db_path
        cache.close()
        
        # Recreate with schema
        cache = LLMResponseCache(db_path=db_path, schema=schema, verbose=False)
        
        # Valid response
        valid_response = {"text": "Hello", "price": 100}
        cache.set("valid", valid_response)
        assert cache.get("valid") == valid_response
        
        # Invalid response (missing 'price')
        invalid_response = {"text": "Hello"}  # missing 'price'
        cache.set("invalid", invalid_response)
        # Should not be cached due to schema validation
        assert cache.get("invalid") is None
        
        # Stats should show schema_rejects
        stats = cache.get_stats()
        assert stats["schema_rejects"] >= 1
        
        cache.close()
    def test_schema_validation_string(self, cache):
        """Schema validation with string type"""
        schema = {"type": "string"}
        
        db_path = cache.db_path
        cache.close()
        
        # Import inside to avoid referencing closed fixture object
        from core.llm_cache import LLMResponseCache
        cache2 = LLMResponseCache(db_path=db_path, schema=schema, verbose=False)
        
        # Valid string response
        cache2.set("string", "hello world")
        assert cache2.get("string") == "hello world"
        
        # Invalid: object instead of string
        cache2.set("wrong_type", {"text": "hello"})
        assert cache2.get("wrong_type") is None
        
        cache2.close()
    
    def test_schema_validation_array(self, cache):
        """Schema validation with array type"""
        schema = {"type": "array"}
        
        db_path = cache.db_path
        cache.close()
        
        from core.llm_cache import LLMResponseCache
        cache3 = LLMResponseCache(db_path=db_path, schema=schema, verbose=False)
        
        # Valid array response
        cache3.set("array", [1, 2, 3])
        assert cache3.get("array") == [1, 2, 3]
        
        cache3.close()
    
    def test_schema_not_required_when_none(self, cache):
        """When no schema provided, all responses should be cached"""
        cache.set("test", {"any": "thing"})
        assert cache.get("test") == {"any": "thing"}
    
    def test_reset_stats(self, cache):
        """reset_stats should zero out counters"""
        cache.set("a", {"1": 1})
        cache.get("a")
        cache.get("b")
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        
        cache.reset_stats()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
    
    def test_corrupt_json_handling(self, tmp_path):
        """Corrupt JSON in cache should be handled gracefully"""
        from core.llm_cache import LLMResponseCache
        
        db_path = str(tmp_path / "test_corrupt.db")
        cache = LLMResponseCache(db_path=db_path, verbose=False)
        
        # Insert corrupt data directly via SQL
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO llm_cache (key, prompt, state_json, response, provider, model, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("corrupt_key", "test", "{}", "NOT_JSON", None, None, time.time(), time.time() + 3600))
        conn.commit()
        conn.close()
        
        # Should return None for corrupt entry
        result = cache.get("test")
        assert result is None
        
        cache.close()


class TestLLMResponseCacheSingleton:
    """Test global singleton pattern"""
    
    def test_get_llm_response_cache(self):
        from core.llm_cache import get_llm_response_cache, _llm_response_cache
        
        # Clear global
        import core.llm_cache as m
        m._llm_response_cache = None
        
        cache1 = get_llm_response_cache()
        cache2 = get_llm_response_cache()
        
        # Should be the same instance
        assert cache1 is cache2
        
        # Clean up
        m._llm_response_cache = None
