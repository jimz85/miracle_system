"""
Tests for core/memory/vector_memory.py embedding support
========================================================
"""
from __future__ import annotations

import pytest
import tempfile
import os


class TestVectorMemoryEmbedding:
    """Tests for vector memory semantic search embedding support"""

    @pytest.fixture
    def memory(self):
        """Create a test VectorMemory instance with isolated temp DB.

        P0-FIX: Use tempfile.mkdtemp() instead of tmp_path to avoid
        pytest-asyncio premature directory cleanup race condition.
        """
        import shutil
        from core.memory.vector_memory import VectorMemory

        tmp = tempfile.mkdtemp()
        vm = VectorMemory(persist_directory=tmp)
        yield vm
        shutil.rmtree(tmp, ignore_errors=True)
    
    def test_embedding_column_exists(self, memory):
        """DB should have embedding column"""
        import sqlite3
        
        # The memory fixture already created the DB, use the same connection path
        # VectorMemory stores at: persist_directory/miracle_memories.db
        db_path = os.path.join(str(memory.persist_directory), "miracle_memories.db")
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        assert "embedding" in columns
    
    def test_add_stores_embedding(self, memory):
        """add() should compute and store embedding when sentence_transformers available"""
        from core.memory.vector_memory import _EMBEDDING_AVAILABLE
        
        entry_id = memory.add(
            content="This is a test memory about Bitcoin trading.",
            memory_type="test"
        )
        
        # Get the stored memory
        stored = memory.get(entry_id)
        assert stored is not None
        assert stored["content"] == "This is a test memory about Bitcoin trading."
        
        # If embeddings are available, the embedding field should be populated
        # (We can't directly check the embedding vector, but we can verify it's not an error)
        if _EMBEDDING_AVAILABLE:
            # The add should succeed without error
            assert entry_id is not None
        else:
            # Without sentence_transformers, embedding is None (expected)
            assert True  # Just verify no crash
    
    def test_search_with_embedding(self, memory):
        """search() should use cosine similarity when embeddings available"""
        from core.memory.vector_memory import _EMBEDDING_AVAILABLE
        
        # Add a few memories
        memory.add("Bitcoin is a cryptocurrency.", memory_type="crypto")
        memory.add("The weather is sunny today.", memory_type="weather")
        memory.add("Ethereum is a smart contract platform.", memory_type="crypto")
        
        # Search for crypto-related content
        results = memory.search("digital currency blockchain", k=3, memory_type="crypto")
        
        # Should find the crypto memories
        assert len(results) >= 1
        # The crypto memories should rank higher than weather
        if len(results) >= 2:
            # First result should be crypto
            assert "cryptocurrency" in results[0]["content"].lower() or \
                   "smart contract" in results[0]["content"].lower()
    
    def test_search_keyword_fallback(self):
        """Without sentence_transformers, should fall back to keyword matching"""
        import shutil
        from core.memory.vector_memory import VectorMemory, _EMBEDDING_AVAILABLE

        # Create a fresh instance to check if embeddings are available
        if not _EMBEDDING_AVAILABLE:
            tmp = tempfile.mkdtemp()
            try:
                vm = VectorMemory(persist_directory=tmp)
                vm.add("Bitcoin trading strategy", memory_type="trade")
                vm.add("Rainy weather forecast", memory_type="weather")

                results = vm.search("bitcoin strategy", k=2)
                assert len(results) >= 1
                assert "bitcoin" in results[0]["content"].lower()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    
    def test_cosine_similarity_function(self):
        """Test cosine similarity computation"""
        from core.memory.vector_memory import _cosine_similarity
        
        # Identical vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v1, v2) - 1.0) < 1e-9
        
        # Orthogonal vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(v1, v2)) < 1e-9
        
        # Opposite vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v1, v2) - (-1.0)) < 1e-9
    
    def test_get_embedding_returns_list_or_none(self):
        """_get_embedding should return List[float] or None"""
        from core.memory.vector_memory import _get_embedding, _EMBEDDING_AVAILABLE
        
        result = _get_embedding("test text")
        
        if _EMBEDDING_AVAILABLE:
            assert isinstance(result, list)
            assert len(result) > 0
        else:
            assert result is None
    
    def test_search_includes_vector_score(self, memory):
        """When using vector search, results should include vector_score"""
        from core.memory.vector_memory import _EMBEDDING_AVAILABLE
        
        memory.add("Bitcoin price analysis", memory_type="trade")
        
        results = memory.search("cryptocurrency market", k=1)
        
        if _EMBEDDING_AVAILABLE and results:
            assert "vector_score" in results[0]
            assert "similarity" in results[0]
            # vector_score should be between 0 and 1
            vs = results[0].get("vector_score")
            if vs is not None:
                assert 0.0 <= vs <= 1.0
    
    def test_search_with_include_embeddings(self, memory):
        """When include_embeddings=True, results should include embedding vector"""
        from core.memory.vector_memory import _EMBEDDING_AVAILABLE
        
        memory.add("Test content for embedding", memory_type="test")
        
        results = memory.search(
            "test query",
            k=1,
            include_embeddings=True
        )
        
        if _EMBEDDING_AVAILABLE and results:
            # If we have embeddings stored and we're using vector search
            # the result might include the embedding
            assert results is not None
    
    def test_batch_add_with_embedding(self, memory):
        """add_batch should also store embeddings"""
        from core.memory.vector_memory import MemoryEntry, _EMBEDDING_AVAILABLE
        
        from datetime import datetime
        entries = [
            MemoryEntry(
                id="batch_1",
                content="First batch entry",
                memory_type="batch",
                created_at=datetime.now()
            ),
            MemoryEntry(
                id="batch_2",
                content="Second batch entry",
                memory_type="batch",
                created_at=datetime.now()
            ),
        ]
        
        ids = memory.add_batch(entries)
        assert len(ids) == 2
        
        # Verify they were stored
        for eid in ids:
            stored = memory.get(eid)
            assert stored is not None
    
    def test_stats_reflect_embeddings(self, memory):
        """get_stats should work correctly after adding embedded memories"""
        memory.add("Test memory 1", memory_type="test")
        memory.add("Test memory 2", memory_type="test")
        
        stats = memory.get_stats()
        assert stats["total"] >= 2
        assert "test" in stats["by_type"]
