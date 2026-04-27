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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "memory_type": self.memory_type
        }


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
            memory_type: str = "general", id: Optional[str] = None) -> str:
        """
        添加记忆条目
        
        Args:
            content: 记忆内容
            metadata: 元数据
            memory_type: 记忆类型
            id: 可选指定ID
            
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
            memory_type=memory_type
        )
        
        # 生成嵌入
        embedding = self._generate_embedding(content)
        
        try:
            self.collection.add(
                ids=[entry_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{
                    **metadata,
                    "created_at": entry.created_at.isoformat(),
                    "memory_type": memory_type
                }]
            )
            logger.debug(f"Added memory: {entry_id}, type={memory_type}")
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
            metadatas.append({
                **entry.metadata,
                "created_at": entry.created_at.isoformat(),
                "memory_type": entry.memory_type
            })
        
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


# 全局实例
_vector_memory: Optional[VectorMemory] = None

def get_vector_memory(llm_provider=None) -> VectorMemory:
    """获取向量记忆全局实例"""
    global _vector_memory
    if _vector_memory is None:
        _vector_memory = VectorMemory(llm_provider=llm_provider)
    return _vector_memory
