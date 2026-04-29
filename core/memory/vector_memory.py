from __future__ import annotations

"""
Vector Memory using SQLite (Downgraded from ChromaDB)
====================================================
向量记忆模块 - 基于SQLite实现存储和检索
支持语义向量搜索（sentence_transformers）和关键词匹配双模式
"""

import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# Embedding Support (sentence_transformers with keyword fallback)
# ============================================================
_EMBEDDING_MODEL = None
_EMBEDDING_DIM = 384  # default for all-MiniLM-L6-v2
_EMBEDDING_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    _EMBEDDING_AVAILABLE = True
    logger.info("sentence_transformers loaded - semantic search ENABLED (local_files_only)")
except ImportError:
    logger.warning(
        "sentence_transformers not available - falling back to keyword matching. "
        "Install with: pip install sentence-transformers"
    )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_embedding(text: str) -> List[float] | None:
    """Get embedding vector for text, or None if sentence_transformers unavailable."""
    if not _EMBEDDING_AVAILABLE or _EMBEDDING_MODEL is None:
        return None
    try:
        # Returns numpy array, convert to list
        embedding = _EMBEDDING_MODEL.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.warning(f"Embedding computation failed: {e}")
        return None

# 全局SQLite连接
_sqlite_conn: sqlite3.Connection | None = None
_sqlite_path: str | None = None
_write_lock = threading.Lock()  # 保护并发写操作


def _get_db_path(persist_directory: str | None = None) -> str:
    """获取SQLite数据库路径"""
    if persist_directory is None:
        persist_directory = os.path.expanduser("~/.miracle_memory/sqlite")
    os.makedirs(persist_directory, exist_ok=True)
    return os.path.join(persist_directory, "miracle_memories.db")


def _get_connection(persist_directory: str | None = None) -> sqlite3.Connection:
    """获取SQLite连接"""
    global _sqlite_conn, _sqlite_path
    
    db_path = _get_db_path(persist_directory)
    
    if _sqlite_conn is None or _sqlite_path != db_path:
        if _sqlite_conn is not None:
            _sqlite_conn.close()
        
        _sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        _sqlite_conn.row_factory = sqlite3.Row
        _sqlite_path = db_path
        
        # 启用WAL模式，提升并发读写性能
        _sqlite_conn.execute("PRAGMA journal_mode=WAL")
        
        # 初始化表结构
        _init_db(_sqlite_conn)
        logger.info(f"Initialized SQLite at {db_path}")
    
    return _sqlite_conn


def _init_db(conn: sqlite3.Connection) -> None:
    """初始化数据库表"""
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT,  -- JSON字符串
            memory_type TEXT DEFAULT 'general',
            created_at TEXT NOT NULL,
            expires_at TEXT,
            keywords TEXT,  -- 逗号分隔的关键词
            embedding TEXT   -- JSON序列化的向量 (sentence_transformers)
        )
    """)
    
    # Migration: add embedding column if it doesn't exist
    cursor.execute("PRAGMA table_info(memories)")
    columns = [row[1] for row in cursor.fetchall()]
    if "embedding" not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        logger.info("Added embedding column to memories table")
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_expires_at ON memories(expires_at)
    """)
    
    conn.commit()


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    memory_type: str = "general"  # general, trade, market, lesson, pattern
    expires_at: datetime | None = None  # 过期时间，None表示永不过期
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "memory_type": self.memory_type
        }
        if self.expires_at:
            result["expires_at"] = self.expires_at.isoformat()
        return result
    
    def is_expired(self) -> bool:
        """检查记忆是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


def _extract_keywords(text: str) -> str:
    """从文本中提取关键词用于搜索"""
    import re
    # 移除特殊字符，转小写
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # 提取长度>=3的单词
    words = [w for w in text.split() if len(w) >= 3]
    # 去重并取前20个
    unique_words = list(dict.fromkeys(words))[:20]
    return ','.join(unique_words)


class VectorMemory:
    """
    SQLite向量记忆
    支持关键词搜索、过滤、元数据存储
    
    注意: 由于移除了ChromaDB，不再支持语义向量搜索。
    搜索改为基于关键词匹配+时间衰减排序。
    """
    
    COLLECTION_NAME = "miracle_memories"
    DB_NAME = "miracle_memories.db"
    
    def __init__(self, persist_directory: str | None = None, 
                 collection_name: str = COLLECTION_NAME,
                 llm_provider=None):
        """
        初始化向量记忆
        
        Args:
            persist_directory: SQLite数据持久化目录
            collection_name: 集合名称（兼容ChromaDB接口，但实际使用DB文件）
            llm_provider: LLM Provider（已弃用，保留接口兼容）
        """
        self.persist_directory = persist_directory or os.path.expanduser("~/.miracle_memory/sqlite")
        self.collection_name = collection_name
        self.llm_provider = llm_provider  # 保留但不使用
        
        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 初始化数据库连接
        self._conn = _get_connection(self.persist_directory)
    
    def _row_to_memory(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转换为内存字典"""
        import json
        metadata = {}
        if row['metadata']:
            try:
                metadata = json.loads(row['metadata'])
            except Exception as e:
                logger.debug(f"JSON解析metadata失败: {e}")
        
        return {
            "id": row['id'],
            "content": row['content'],
            "metadata": metadata,
            "created_at": row['created_at'],
            "memory_type": row['memory_type'],
            "expires_at": row['expires_at'],
            "distance": 0.0,  # 兼容ChromaDB接口
            "similarity": 1.0  # 兼容ChromaDB接口
        }
    
    def add(self, content: str, metadata: Dict[str, Any] | None = None,
            memory_type: str = "general", id: str | None = None,
            expires_at: datetime | None = None) -> str:
        """
        添加记忆条目
        
        Args:
            content: 记忆内容
            metadata: 元数据
            memory_type: 记忆类型
            id: 可选指定ID
            expires_at: 过期时间，None表示永不过期
            
        Returns:
            str: 记忆ID
        """
        import json
        
        entry_id = id or str(uuid.uuid4())
        metadata = metadata or {}
        metadata["memory_type"] = memory_type
        
        created_at = datetime.now()
        expires_at_str = expires_at.isoformat() if expires_at else None
        
        # 提取关键词
        keywords = _extract_keywords(content)
        
        # 计算语义向量（sentence_transformers）
        embedding = _get_embedding(content)
        embedding_json = json.dumps(embedding) if embedding else None
        
        with _write_lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO memories (id, content, metadata, memory_type, created_at, expires_at, keywords, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry_id, content, json.dumps(metadata, ensure_ascii=False), memory_type, 
                  created_at.isoformat(), expires_at_str, keywords, embedding_json))
            
            self._conn.commit()
        logger.debug(f"Added memory: {entry_id}, type={memory_type}, expires={expires_at}, embedding={embedding is not None}")
        return entry_id
    
    def add_batch(self, entries: List[MemoryEntry]) -> List[str]:
        """
        批量添加记忆
        
        Args:
            entries: MemoryEntry列表
            
        Returns:
            List[str]: 记忆ID列表
        """
        import json
        
        if not entries:
            return []
        
        with _write_lock:
            cursor = self._conn.cursor()
            ids = []
            
            for entry in entries:
                ids.append(entry.id)
                keywords = _extract_keywords(entry.content)
                expires_at_str = entry.expires_at.isoformat() if entry.expires_at else None
                
                cursor.execute("""
                    INSERT INTO memories (id, content, metadata, memory_type, created_at, expires_at, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (entry.id, entry.content, json.dumps(entry.metadata, ensure_ascii=False),
                      entry.memory_type, entry.created_at.isoformat(), expires_at_str, keywords))
            
            self._conn.commit()
        logger.info(f"Added {len(entries)} memories in batch")
        return ids
    
    def search(self, query: str, k: int = 5, 
               memory_type: str | None = None,
               filter_metadata: Dict[str, Any] | None = None,
               include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        搜索记忆（语义向量相似度 + 关键词匹配 + 时间衰减）
        
        当sentence_transformers可用时：
            - 计算查询文本的embedding
            - 与所有记忆的embedding做余弦相似度
            - 综合分数 = 向量相似度 * 0.6 + 关键词匹配 * 0.25 + 时间衰减 * 0.15
        
        当sentence_transformers不可用时：
            - 纯关键词匹配 + 时间衰减（原有逻辑）
        
        Args:
            query: 查询文本
            k: 返回数量
            memory_type: 按类型过滤
            filter_metadata: 按元数据过滤
            include_embeddings: 是否返回嵌入向量
            
        Returns:
            List[Dict]: 检索结果列表
        """
        import json
        from datetime import datetime as dt
        
        # 提取查询关键词
        query_keywords = _extract_keywords(query).split(',')
        
        # 计算查询文本的语义向量
        query_embedding = _get_embedding(query)
        use_vector_search = query_embedding is not None
        
        if use_vector_search:
            logger.debug("Using semantic vector search")
        else:
            logger.debug("sentence_transformers unavailable, using keyword search")
        
        cursor = self._conn.cursor()
        
        # 构建SQL查询
        sql = "SELECT * FROM memories WHERE 1=1"
        params = []
        
        # 过滤已过期
        sql += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(dt.now().isoformat())
        
        # 类型过滤
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        
        # 元数据过滤（简单实现：JSON包含关键key-value）
        if filter_metadata:
            for key, value in filter_metadata.items():
                sql += " AND metadata LIKE ?"
                params.append(f'%"{key}": "{value}"%')
        
        # 获取所有候选记录
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        # 计算匹配分数并排序
        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            
            # 计算关键词匹配分数
            row_keywords = row['keywords'].split(',') if row['keywords'] else []
            match_count = sum(1 for kw in query_keywords if kw in row_keywords)
            keyword_score = match_count / max(len(query_keywords), 1) if query_keywords else 0
            
            # 时间衰减分数（越新的记忆分数越高）
            try:
                created = dt.fromisoformat(row['created_at'])
                hours_age = (dt.now() - created).total_seconds() / 3600
                time_score = max(0, 1 - hours_age / (24 * 30))  # 30天后衰减为0
            except Exception as e:
                logger.debug(f"datetime解析失败: {e}")
                time_score = 0.5
            
            # 计算向量相似度（如果可用）
            vector_score = 0.0
            if use_vector_search and row['embedding']:
                try:
                    stored_embedding = json.loads(row['embedding'])
                    vector_score = _cosine_similarity(query_embedding, stored_embedding)
                except Exception as e:
                    logger.debug(f"向量相似度计算失败: {e}")
                    vector_score = 0.0
            
            # 综合分数
            if use_vector_search and vector_score > 0:
                # 优先使用向量相似度
                similarity = vector_score * 0.6 + keyword_score * 0.25 + time_score * 0.15
            else:
                # 降级：纯关键词 + 时间
                similarity = keyword_score * 0.7 + time_score * 0.3
            
            if similarity > 0:
                memory['similarity'] = similarity
                memory['distance'] = 1 - similarity
                memory['match_score'] = match_count
                memory['vector_score'] = vector_score if use_vector_search else None
                if include_embeddings and row['embedding']:
                    try:
                        memory['embedding'] = json.loads(row['embedding'])
                    except Exception:
                        pass
                results.append(memory)
        
        # 按相似度排序
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return results[:k]
    
    def get(self, memory_id: str) -> Dict[str, Any] | None:
        """获取指定记忆"""
        import json
        
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_memory(row)
        return None
    
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        with _write_lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
            
            if cursor.rowcount > 0:
                logger.debug(f"Deleted memory: {memory_id}")
                return True
        return False
    
    def update(self, memory_id: str, content: str | None = None,
               metadata: Dict[str, Any] | None = None) -> bool:
        """更新记忆"""
        import json
        
        with _write_lock:
            cursor = self._conn.cursor()
            
            # 获取现有记录
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            update_fields = []
            params = []
            
            if content:
                update_fields.append("content = ?")
                params.append(content)
                update_fields.append("keywords = ?")
                params.append(_extract_keywords(content))
            
            if metadata:
                update_fields.append("metadata = ?")
                params.append(json.dumps(metadata, ensure_ascii=False))
            
            if not update_fields:
                return True
            
            params.append(memory_id)
            sql = f"UPDATE memories SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(sql, params)
            self._conn.commit()
            
            logger.debug(f"Updated memory: {memory_id}")
            return True
    
    def get_by_type(self, memory_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取指定类型的所有记忆"""
        import json
        from datetime import datetime as dt
        
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM memories 
            WHERE memory_type = ? AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY created_at DESC
            LIMIT ?
        """, (memory_type, dt.now().isoformat(), limit))
        
        rows = cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]
    
    def count(self, memory_type: str | None = None) -> int:
        """统计记忆数量"""
        from datetime import datetime as dt
        
        cursor = self._conn.cursor()
        
        if memory_type:
            cursor.execute("""
                SELECT COUNT(*) FROM memories 
                WHERE memory_type = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (memory_type, dt.now().isoformat()))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM memories 
                WHERE expires_at IS NULL OR expires_at > ?
            """, (dt.now().isoformat(),))
        
        return cursor.fetchone()[0]
    
    def reset(self):
        """清空所有记忆（危险操作）"""
        with _write_lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM memories")
            self._conn.commit()
        logger.warning("Vector memory reset - all memories deleted")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        from datetime import datetime as dt
        
        cursor = self._conn.cursor()
        
        # 总数
        cursor.execute("""
            SELECT COUNT(*) FROM memories 
            WHERE expires_at IS NULL OR expires_at > ?
        """, (dt.now().isoformat(),))
        total = cursor.fetchone()[0]
        
        # 按类型统计
        cursor.execute("""
            SELECT memory_type, COUNT(*) as cnt FROM memories 
            WHERE expires_at IS NULL OR expires_at > ?
            GROUP BY memory_type
        """, (dt.now().isoformat(),))
        
        by_type = {}
        for row in cursor.fetchall():
            by_type[row[0]] = row[1]
        
        return {"total": total, "by_type": by_type}
    
    def search_with_context(self, query: str, k: int = 5,
                            memory_types: List[str] | None = None) -> str:
        """
        检索记忆并格式化为上下文字符串
        
        Args:
            query: 查询文本
            k: 返回数量
            memory_types: 记忆类型列表
            
        Returns:
            str: 格式化的上下文字符串
        """
        if memory_types:
            memories = []
            for mt in memory_types:
                results = self.search(query, k=k // len(memory_types), memory_type=mt)
                memories.extend(results)
            # 按相似度排序
            memories.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            memories = memories[:k]
        else:
            memories = self.search(query, k=k)
        
        if not memories:
            return ""
        
        context_parts = []
        context_parts.append("=== 相关历史记忆 ===")
        for i, mem in enumerate(memories, 1):
            meta = mem.get('metadata', {})
            mem_type = mem.get('memory_type', 'general')
            created = meta.get('created_at', 'unknown')
            context_parts.append(
                f"\n[{i}] [{mem_type}] ({created})\n"
                f"{mem['content']}\n"
                f"相似度: {mem.get('similarity', 0):.2%}"
            )
        
        return "\n".join(context_parts)
    
    def cleanup_expired(self, dry_run: bool = False) -> Dict[str, int]:
        """
        清理过期的记忆条目
        
        Args:
            dry_run: 若为True，只统计不删除
            
        Returns:
            Dict: 清理统计 {"deleted": N, "by_type": {...}}
        """
        import json
        from datetime import datetime as dt
        
        cursor = self._conn.cursor()
        
        # 获取所有过期记忆
        cursor.execute("""
            SELECT id, memory_type FROM memories 
            WHERE expires_at IS NOT NULL AND expires_at <= ?
        """, (dt.now().isoformat(),))
        
        expired_rows = cursor.fetchall()
        
        expired_ids = [row[0] for row in expired_rows]
        by_type = {}
        
        for row in expired_rows:
            mem_type = row[1]
            by_type[mem_type] = by_type.get(mem_type, 0) + 1
        
        if not dry_run and expired_ids:
            with _write_lock:
                placeholders = ','.join('?' * len(expired_ids))
                cursor.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", expired_ids)
                self._conn.commit()
            logger.info(f"VectorMemory: Cleaned up {len(expired_ids)} expired memories")
        
        return {
            "deleted": len(expired_ids) if not dry_run else 0,
            "would_delete": len(expired_ids) if dry_run else 0,
            "by_type": by_type,
            "total_checked": self.count()
        }
    
    def cleanup_by_age(self, max_age_days: int = 30, 
                      memory_types: List[str] | None = None,
                      dry_run: bool = False) -> Dict[str, int]:
        """
        按年龄清理记忆
        
        Args:
            max_age_days: 最大保留天数
            memory_types: 只清理这些类型，为None则清理所有
            dry_run: 若为True，只统计不删除
            
        Returns:
            Dict: 清理统计
        """
        from datetime import datetime as dt
        
        cutoff_time = dt.now() - timedelta(days=max_age_days)
        
        cursor = self._conn.cursor()
        
        # 构建查询
        sql = "SELECT id, memory_type FROM memories WHERE created_at <= ?"
        params = [cutoff_time.isoformat()]
        
        if memory_types:
            placeholders = ','.join('?' * len(memory_types))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
        
        cursor.execute(sql, params)
        old_rows = cursor.fetchall()
        
        old_ids = [row[0] for row in old_rows]
        by_type = {}
        
        for row in old_rows:
            mem_type = row[1]
            by_type[mem_type] = by_type.get(mem_type, 0) + 1
        
        if not dry_run and old_ids:
            with _write_lock:
                placeholders = ','.join('?' * len(old_ids))
                cursor.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", old_ids)
                self._conn.commit()
            logger.info(f"VectorMemory: Cleaned up {len(old_ids)} old memories (>{max_age_days} days)")
        
        return {
            "deleted": len(old_ids) if not dry_run else 0,
            "would_delete": len(old_ids) if dry_run else 0,
            "by_type": by_type,
            "total_checked": len(old_ids),
            "max_age_days": max_age_days
        }
    
    def get_memory_stats_with_expiry(self) -> Dict[str, Any]:
        """
        获取包含过期信息的记忆统计
        
        Returns:
            Dict: 统计信息
        """
        from datetime import datetime as dt
        
        cursor = self._conn.cursor()
        
        cursor.execute("""
            SELECT memory_type, expires_at FROM memories
        """)
        
        rows = cursor.fetchall()
        
        total = len(rows)
        by_type = {}
        expired = 0
        expiring_soon = 0  # 7天内过期
        
        soon_cutoff = dt.now() + timedelta(days=7)
        
        for row in rows:
            mem_type = row[0]
            by_type[mem_type] = by_type.get(mem_type, 0) + 1
            
            if row[1]:
                try:
                    expires_at = dt.fromisoformat(row[1])
                    if dt.now() > expires_at:
                        expired += 1
                    elif expires_at < soon_cutoff:
                        expiring_soon += 1
                except (ValueError, TypeError):
                    pass
        
        return {
            "total": total,
            "by_type": by_type,
            "expired": expired,
            "expiring_soon": expiring_soon
        }


# 全局实例
_vector_memory: VectorMemory | None = None

def get_vector_memory(llm_provider=None) -> VectorMemory:
    """获取向量记忆全局实例"""
    global _vector_memory
    if _vector_memory is None:
        _vector_memory = VectorMemory(llm_provider=llm_provider)
    return _vector_memory
