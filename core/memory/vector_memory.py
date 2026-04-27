"""
Vector Memory using ChromaDB
=============================
向量记忆模块 - 基于ChromaDB实现语义检索
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# 全局ChromaDB客户端
_chroma_client = None
_default_collection = None


def get_chroma_client(persist_directory: Optional[str] = None):
    """获取ChromaDB客户端"""
    global _chroma_client
    
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        
        if persist_directory is None:
            persist_directory = os.path.expanduser("~/.miracle_memory/chroma")
        
        os.makedirs(persist_directory, exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logger.info(f"Initialized ChromaDB at {persist_directory}")
    
    return _chroma_client


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    memory_type: str = "general"  # general, trade, market, lesson, pattern
    expires_at: Optional[datetime] = None  # 过期时间，None表示永不过期
    
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


class VectorMemory:
    """
    ChromaDB向量记忆
    支持语义检索、过滤、元数据存储
    """
    
    COLLECTION_NAME = "miracle_memories"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 本地嵌入模型
    
    def __init__(self, persist_directory: Optional[str] = None, 
                 collection_name: str = COLLECTION_NAME,
                 llm_provider=None):
        """
        初始化向量记忆
        
        Args:
            persist_directory: ChromaDB数据持久化目录
            collection_name: 集合名称
            llm_provider: LLM Provider用于生成嵌入向量
        """
        self.persist_directory = persist_directory or os.path.expanduser("~/.miracle_memory/chroma")
        self.collection_name = collection_name
        self.llm_provider = llm_provider
        
        # 初始化ChromaDB
        self.client = get_chroma_client(self.persist_directory)
        self._collection = None
        
        # 初始化嵌入模型
        self._embedding_model = None
    
    @property
    def collection(self):
        """延迟加载collection"""
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Miracle 2.0 Vector Memory"}
                )
            except Exception as e:
                logger.error(f"Failed to get/create collection: {e}")
                # 尝试重置
                self.client.reset()
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Miracle 2.0 Vector Memory"}
                )
        return self._collection
    
    def _get_embedding_model(self):
        """获取嵌入模型"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
                logger.info(f"Loaded embedding model: {self.EMBEDDING_MODEL}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using LLM provider for embeddings")
        return self._embedding_model
    
    def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        # 优先使用本地模型
        model = self._get_embedding_model()
        if model:
            embedding = model.encode(text).tolist()
            return embedding
        
        # fallback到LLM provider
        if self.llm_provider:
            import asyncio
            try:
                embeddings = asyncio.run(self.llm_provider.embed([text]))
                return embeddings[0] if embeddings else [0.0] * 384
            except Exception as e:
                logger.error(f"Failed to generate embedding via LLM: {e}")
        
        # 最后fallback到零向量
        return [0.0] * 384
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        model = self._get_embedding_model()
        if model:
            embeddings = model.encode(texts).tolist()
            return embeddings
        
        # fallback到LLM provider
        if self.llm_provider:
            import asyncio
            try:
                return asyncio.run(self.llm_provider.embed(texts))
            except Exception as e:
                logger.error(f"Failed to generate embeddings via LLM: {e}")
        
        return [[0.0] * 384 for _ in texts]
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None,
            memory_type: str = "general", id: Optional[str] = None,
            expires_at: Optional[datetime] = None) -> str:
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
        import uuid
        
        entry_id = id or str(uuid.uuid4())
        metadata = metadata or {}
        metadata["memory_type"] = memory_type
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
            memory_type=memory_type,
            expires_at=expires_at
        )
        
        # 生成嵌入
        embedding = self._generate_embedding(content)
        
        try:
            entry_metadata = {
                **metadata,
                "created_at": entry.created_at.isoformat(),
                "memory_type": memory_type
            }
            if expires_at:
                entry_metadata["expires_at"] = expires_at.isoformat()
            
            self.collection.add(
                ids=[entry_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[entry_metadata]
            )
            logger.debug(f"Added memory: {entry_id}, type={memory_type}, expires={expires_at}")
            return entry_id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
    
    def add_batch(self, entries: List[MemoryEntry]) -> List[str]:
        """
        批量添加记忆
        
        Args:
            entries: MemoryEntry列表
            
        Returns:
            List[str]: 记忆ID列表
        """
        if not entries:
            return []
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for entry in entries:
            ids.append(entry.id)
            documents.append(entry.content)
            embeddings.append(self._generate_embedding(entry.content))
            
            entry_metadata = {
                **entry.metadata,
                "created_at": entry.created_at.isoformat(),
                "memory_type": entry.memory_type
            }
            if entry.expires_at:
                entry_metadata["expires_at"] = entry.expires_at.isoformat()
            metadatas.append(entry_metadata)
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Added {len(entries)} memories in batch")
            return ids
        except Exception as e:
            logger.error(f"Failed to add memories batch: {e}")
            raise
    
    def search(self, query: str, k: int = 5, 
               memory_type: Optional[str] = None,
               filter_metadata: Optional[Dict[str, Any]] = None,
               include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        语义检索记忆
        
        Args:
            query: 查询文本
            k: 返回数量
            memory_type: 按类型过滤
            filter_metadata: 按元数据过滤
            include_embeddings: 是否返回嵌入向量
            
        Returns:
            List[Dict]: 检索结果列表
        """
        # 生成查询向量
        query_embedding = self._generate_embedding(query)
        
        # 构建where过滤条件
        where = {}
        if memory_type:
            where["memory_type"] = memory_type
        if filter_metadata:
            where.update(filter_metadata)
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where if where else None,
                include=["documents", "metadatas", "distances"] + 
                        (["embeddings"] if include_embeddings else [])
            )
            
            # 格式化结果
            memories = []
            if results["ids"] and len(results["ids"]) > 0:
                for i, mem_id in enumerate(results["ids"][0]):
                    memory = {
                        "id": mem_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity": 1 - results["distances"][0][i]  # 转换距离为相似度
                    }
                    if include_embeddings and "embeddings" in results:
                        memory["embedding"] = results["embeddings"][0][i]
                    memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取指定记忆"""
        try:
            results = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def update(self, memory_id: str, content: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新记忆"""
        try:
            update_data = {}
            if content:
                update_data["documents"] = [content]
                # 重新生成嵌入
                embedding = self._generate_embedding(content)
                update_data["embeddings"] = [embedding]
            
            if metadata:
                # 获取现有metadata并更新
                existing = self.get(memory_id)
                if existing:
                    new_metadata = {**existing["metadata"], **metadata}
                    update_data["metadatas"] = [new_metadata]
            
            if update_data:
                self.collection.update(ids=[memory_id], **update_data)
                logger.debug(f"Updated memory: {memory_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    def get_by_type(self, memory_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取指定类型的所有记忆"""
        try:
            results = self.collection.get(
                where={"memory_type": memory_type},
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            memories = []
            if results["ids"]:
                for i, mem_id in enumerate(results["ids"]):
                    memories.append({
                        "id": mem_id,
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    })
            return memories
        except Exception as e:
            logger.error(f"Failed to get memories by type {memory_type}: {e}")
            return []
    
    def count(self, memory_type: Optional[str] = None) -> int:
        """统计记忆数量"""
        try:
            where = {"memory_type": memory_type} if memory_type else None
            return self.collection.count(where=where)
        except Exception as e:
            logger.error(f"Failed to count memories: {e}")
            return 0
    
    def reset(self):
        """清空所有记忆（危险操作）"""
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = None
            logger.warning("Vector memory reset - all memories deleted")
        except Exception as e:
            logger.error(f"Failed to reset memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        try:
            all_data = self.collection.get(include=["metadatas"])
            stats = {
                "total": len(all_data["ids"]) if all_data["ids"] else 0,
                "by_type": {}
            }
            
            if all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    mem_type = meta.get("memory_type", "unknown")
                    stats["by_type"][mem_type] = stats["by_type"].get(mem_type, 0) + 1
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total": 0, "by_type": {}}
    
    def search_with_context(self, query: str, k: int = 5,
                            memory_types: Optional[List[str]] = None) -> str:
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
            memories.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            memories = memories[:k]
        else:
            memories = self.search(query, k=k)
        
        if not memories:
            return ""
        
        context_parts = []
        context_parts.append("=== 相关历史记忆 ===")
        for i, mem in enumerate(memories, 1):
            meta = mem.get("metadata", {})
            mem_type = meta.get("memory_type", "general")
            created = meta.get("created_at", "unknown")
            context_parts.append(
                f"\n[{i}] [{mem_type}] ({created})\n"
                f"{mem['content']}\n"
                f"相似度: {mem.get('similarity', 0):.2%}"
            )
        
        return "\n".join(context_parts)

    # ==================== Expiration Cleanup ====================
    
    def cleanup_expired(self, dry_run: bool = False) -> Dict[str, int]:
        """
        清理过期的记忆条目
        
        Args:
            dry_run: 若为True，只统计不删除
            
        Returns:
            Dict: 清理统计 {"deleted": N, "by_type": {...}}
        """
        from datetime import datetime as dt
        
        try:
            # 获取所有记忆
            all_data = self.collection.get(include=["metadatas", "documents", "ids"])
            
            if not all_data["ids"]:
                return {"deleted": 0, "by_type": {}, "total_checked": 0}
            
            expired_ids = []
            by_type = {}
            total_checked = len(all_data["ids"])
            
            for i, mem_id in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i]
                expires_at_str = meta.get("expires_at")
                
                if expires_at_str:
                    try:
                        expires_at = dt.fromisoformat(expires_at_str)
                        if dt.now() > expires_at:
                            expired_ids.append(mem_id)
                            mem_type = meta.get("memory_type", "unknown")
                            by_type[mem_type] = by_type.get(mem_type, 0) + 1
                    except (ValueError, TypeError):
                        pass
            
            if not dry_run and expired_ids:
                self.collection.delete(ids=expired_ids)
                logger.info(f"VectorMemory: Cleaned up {len(expired_ids)} expired memories")
            
            return {
                "deleted": len(expired_ids) if not dry_run else 0,
                "would_delete": len(expired_ids) if dry_run else 0,
                "by_type": by_type,
                "total_checked": total_checked
            }
        except Exception as e:
            logger.error(f"Failed to cleanup expired memories: {e}")
            return {"deleted": 0, "by_type": {}, "error": str(e)}
    
    def cleanup_by_age(self, max_age_days: int = 30, 
                      memory_types: Optional[List[str]] = None,
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
        from datetime import timedelta
        
        cutoff_time = dt.now() - timedelta(days=max_age_days)
        
        try:
            all_data = self.collection.get(include=["metadatas", "ids"])
            
            if not all_data["ids"]:
                return {"deleted": 0, "by_type": {}, "total_checked": 0}
            
            old_ids = []
            by_type = {}
            total_checked = 0
            
            for i, mem_id in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i]
                mem_type = meta.get("memory_type", "unknown")
                
                # 按类型过滤
                if memory_types and mem_type not in memory_types:
                    continue
                
                total_checked += 1
                created_at_str = meta.get("created_at")
                
                if created_at_str:
                    try:
                        created_at = dt.fromisoformat(created_at_str)
                        if created_at < cutoff_time:
                            old_ids.append(mem_id)
                            by_type[mem_type] = by_type.get(mem_type, 0) + 1
                    except (ValueError, TypeError):
                        pass
            
            if not dry_run and old_ids:
                self.collection.delete(ids=old_ids)
                logger.info(f"VectorMemory: Cleaned up {len(old_ids)} old memories (>{max_age_days} days)")
            
            return {
                "deleted": len(old_ids) if not dry_run else 0,
                "would_delete": len(old_ids) if dry_run else 0,
                "by_type": by_type,
                "total_checked": total_checked,
                "max_age_days": max_age_days
            }
        except Exception as e:
            logger.error(f"Failed to cleanup old memories: {e}")
            return {"deleted": 0, "by_type": {}, "error": str(e)}
    
    def get_memory_stats_with_expiry(self) -> Dict[str, Any]:
        """
        获取包含过期信息的记忆统计
        
        Returns:
            Dict: 统计信息
        """
        from datetime import datetime as dt, timedelta
        
        try:
            all_data = self.collection.get(include=["metadatas"])
            
            if not all_data["ids"]:
                return {"total": 0, "by_type": {}, "expired": 0, "expiring_soon": 0}
            
            total = len(all_data["ids"])
            by_type = {}
            expired = 0
            expiring_soon = 0  # 7天内过期
            
            soon_cutoff = dt.now() + timedelta(days=7)
            
            for meta in all_data["metadatas"]:
                mem_type = meta.get("memory_type", "unknown")
                by_type[mem_type] = by_type.get(mem_type, 0) + 1
                
                expires_at_str = meta.get("expires_at")
                if expires_at_str:
                    try:
                        expires_at = dt.fromisoformat(expires_at_str)
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
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}


# 全局实例
_vector_memory: Optional[VectorMemory] = None

def get_vector_memory(llm_provider=None) -> VectorMemory:
    """获取向量记忆全局实例"""
    global _vector_memory
    if _vector_memory is None:
        _vector_memory = VectorMemory(llm_provider=llm_provider)
    return _vector_memory
