from __future__ import annotations

"""
Strategy Version Control - 策略版本控制与一键回滚
防止错误策略导致重大损失，支持版本历史追溯和快速回滚
"""
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock


class StrategyVersion:
    """策略版本数据类"""
    
    def __init__(self, version_id: str, params: Dict[str, Any], description: str,
                 created_at: str, created_by: str = "system", parent_version: str | None = None):
        self.version_id = version_id
        self.params = params
        self.description = description
        self.created_at = created_at
        self.created_by = created_by
        self.parent_version = parent_version
        self.is_stable = False
        self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "params": self.params,
            "description": self.description,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "parent_version": self.parent_version,
            "is_stable": self.is_stable,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyVersion:
        v = cls(
            version_id=data["version_id"],
            params=data["params"],
            description=data["description"],
            created_at=data["created_at"],
            created_by=data.get("created_by", "system"),
            parent_version=data.get("parent_version")
        )
        v.is_stable = data.get("is_stable", False)
        v.metrics = data.get("metrics", {})
        return v


class StrategyVersionControl:
    """
    策略版本控制系统
    
    功能:
    - 版本快照: 保存策略配置的完整历史
    - 一键回滚: 快速回滚到任意版本
    - 紧急回滚: 立即回滚到上一个稳定版本
    - 版本对比: 对比不同版本的参数差异
    """
    
    VERSION_DIR = Path.home() / ".kronos" / "data" / "strategy_versions"
    MAX_VERSIONS = 50
    
    def __init__(self, strategy_name: str = "default"):
        self.strategy_name = strategy_name
        self.version_dir = self.VERSION_DIR / strategy_name
        self.index_file = self.version_dir / "version_index.json"
        self.active_file = self.version_dir / "active_version.json"
        self.lock = FileLock(str(self.version_dir / ".lock"), timeout=10)
        
        self._ensure_dirs()
        self._load_index()
    
    def _ensure_dirs(self):
        """确保目录存在"""
        self.version_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["backups", "archives"]:
            (self.version_dir / subdir).mkdir(exist_ok=True)
    
    def _load_index(self):
        """加载版本索引"""
        if self.index_file.exists():
            with open(self.index_file) as f:
                data = json.load(f)
                self.versions = {v["version_id"]: StrategyVersion.from_dict(v) 
                               for v in data.get("versions", [])}
                self.deleted_versions = set(data.get("deleted_versions", []))
        else:
            self.versions = {}
            self.deleted_versions = set()
    
    def _save_index(self):
        """保存版本索引"""
        with self.lock:
            data = {
                "versions": [v.to_dict() for v in self.versions.values()],
                "deleted_versions": list(self.deleted_versions)
            }
            tmp_file = self.index_file.with_suffix('.tmp')
            with open(tmp_file, 'w') as f:
                json.dump(data, f, indent=2)
            tmp_file.replace(self.index_file)
    
    def _generate_version_id(self, params: Dict[str, Any]) -> str:
        """生成版本ID"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        return f"v{timestamp}-{param_hash}"
    
    def _save_version_file(self, version: StrategyVersion) -> Path:
        """保存版本文件"""
        version_file = self.version_dir / f"{version.version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        return version_file
    
    def _backup_active(self):
        """备份当前活跃版本"""
        if self.active_file.exists():
            backup_dir = self.version_dir / "backups"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_file = backup_dir / f"active_{timestamp}.json"
            shutil.copy2(self.active_file, backup_file)
            return backup_file
        return None
    
    def create_version(self, params: Dict[str, Any], description: str = "",
                      created_by: str = "system", parent_version: str | None = None,
                      metrics: Dict[str, Any] | None = None) -> StrategyVersion:
        """
        创建新版本
        
        Args:
            params: 策略参数
            description: 版本描述
            created_by: 创建者
            parent_version: 父版本ID
            metrics: 性能指标
        
        Returns:
            StrategyVersion: 新创建的版本
        """
        # 获取父版本
        active = self.get_active_version()
        parent = parent_version or (active.version_id if active else None)
        
        # 创建版本
        version_id = self._generate_version_id(params)
        version = StrategyVersion(
            version_id=version_id,
            params=params,
            description=description,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            parent_version=parent
        )
        
        if metrics:
            version.metrics = metrics
        
        # 保存
        self._backup_active()
        self._save_version_file(version)
        self.versions[version_id] = version
        self._save_index()
        
        # 更新活跃版本
        self._set_active(version)
        
        # 清理旧版本
        self._cleanup_old_versions()
        
        return version
    
    def _set_active(self, version: StrategyVersion):
        """设置活跃版本"""
        with open(self.active_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def get_active_version(self) -> StrategyVersion | None:
        """获取当前活跃版本"""
        if self.active_file.exists():
            with open(self.active_file) as f:
                data = json.load(f)
                return StrategyVersion.from_dict(data)
        return None
    
    def get_version(self, version_id: str) -> StrategyVersion | None:
        """获取指定版本"""
        return self.versions.get(version_id)
    
    def get_version_history(self, limit: int = 20) -> List[StrategyVersion]:
        """获取版本历史"""
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.created_at,
            reverse=True
        )
        return sorted_versions[:limit]
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        对比两个版本的差异
        
        Returns:
            Dict with 'only_in_v1', 'only_in_v2', 'different' keys
        """
        v1 = self.versions.get(version_id1)
        v2 = self.versions.get(version_id2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        params1 = v1.params
        params2 = v2.params
        
        all_keys = set(params1.keys()) | set(params2.keys())
        
        return {
            "version1": version_id1,
            "version2": version_id2,
            "only_in_v1": {k: params1[k] for k in params1 if k not in params2},
            "only_in_v2": {k: params2[k] for k in params2 if k not in params1},
            "different": {
                k: {"v1": params1[k], "v2": params2[k]}
                for k in all_keys
                if k in params1 and k in params2 and params1[k] != params2[k]
            },
            "same": {k: params1[k] for k in params1 if k in params2 and params1[k] == params2[k]}
        }
    
    def rollback_to(self, version_id: str, force: bool = False) -> Dict[str, Any]:
        """
        回滚到指定版本
        
        Args:
            version_id: 目标版本ID
            force: 强制回滚（跳过确认）
        
        Returns:
            Dict with status and message
        """
        version = self.versions.get(version_id)
        if not version:
            return {"success": False, "error": f"Version {version_id} not found"}
        
        active = self.get_active_version()
        
        # 备份当前活跃版本
        self._backup_active()
        
        # 设置新活跃版本
        self._set_active(version)
        
        return {
            "success": True,
            "rolled_back_to": version_id,
            "previous_version": active.version_id if active else None,
            "message": f"Successfully rolled back to {version_id}"
        }
    
    def emergency_rollback(self) -> Dict[str, Any]:
        """
        紧急回滚到上一个稳定版本
        
        Returns:
            Dict with status and message
        """
        # 查找上一个稳定版本
        stable_versions = [
            v for v in self.versions.values()
            if v.is_stable
        ]
        
        if not stable_versions:
            return {"success": False, "error": "No stable version found for emergency rollback"}
        
        # 按时间排序，取最新的稳定版本
        latest_stable = max(stable_versions, key=lambda v: v.created_at)
        
        result = self.rollback_to(latest_stable.version_id)
        result["emergency"] = True
        return result
    
    def mark_as_stable(self, version_id: str) -> bool:
        """标记版本为稳定"""
        version = self.versions.get(version_id)
        if not version:
            return False
        
        version.is_stable = True
        self._save_version_file(version)
        self._save_index()
        return True
    
    def mark_as_unstable(self, version_id: str) -> bool:
        """标记版本为不稳定"""
        version = self.versions.get(version_id)
        if not version:
            return False
        
        version.is_stable = False
        self._save_version_file(version)
        self._save_index()
        return True
    
    def _cleanup_old_versions(self):
        """清理旧版本"""
        if len(self.versions) <= self.MAX_VERSIONS:
            return
        
        # 按时间排序，删除最旧的非稳定版本
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.created_at
        )
        
        while len(self.versions) > self.MAX_VERSIONS:
            oldest = sorted_versions.pop(0)
            if not oldest.is_stable:
                # 移动到归档
                archive_dir = self.version_dir / "archives"
                src = self.version_dir / f"{oldest.version_id}.json"
                dst = archive_dir / f"{oldest.version_id}.json"
                if src.exists():
                    shutil.move(str(src), str(dst))
                
                # 从索引移除
                del self.versions[oldest.version_id]
                self.deleted_versions.add(oldest.version_id)
    
    def delete_version(self, version_id: str, force: bool = False) -> Dict[str, Any]:
        """删除版本（只能删除非稳定版本）"""
        version = self.versions.get(version_id)
        if not version:
            return {"success": False, "error": "Version not found"}
        
        if version.is_stable and not force:
            return {"success": False, "error": "Cannot delete stable version without force=True"}
        
        # 检查是否是活跃版本
        active = self.get_active_version()
        if active and active.version_id == version_id:
            return {"success": False, "error": "Cannot delete active version"}
        
        # 移动到归档
        archive_dir = self.version_dir / "archives"
        src = self.version_dir / f"{version_id}.json"
        dst = archive_dir / f"{version_id}.json"
        if src.exists():
            shutil.move(str(src), str(dst))
        
        del self.versions[version_id]
        self.deleted_versions.add(version_id)
        self._save_index()
        
        return {"success": True, "deleted": version_id}


# 全局单例
_global_vc: StrategyVersionControl | None = None


def get_version_control(strategy_name: str = "default") -> StrategyVersionControl:
    """获取版本控制器单例"""
    global _global_vc
    if _global_vc is None or _global_vc.strategy_name != strategy_name:
        _global_vc = StrategyVersionControl(strategy_name)
    return _global_vc


def quick_rollback(target: str = "last_stable") -> Dict[str, Any]:
    """
    快速回滚函数
    
    Args:
        target: "last_stable" 或版本ID
    """
    vc = get_version_control()
    
    if target == "last_stable":
        return vc.emergency_rollback()
    else:
        return vc.rollback_to(target)


# 便捷函数
def create_strategy_version(params: Dict[str, Any], description: str = "", **kwargs) -> StrategyVersion:
    """创建策略版本的便捷函数"""
    return get_version_control().create_version(params, description, **kwargs)


def rollback_to_version(version_id: str) -> Dict[str, Any]:
    """回滚到指定版本的便捷函数"""
    return get_version_control().rollback_to(version_id)


if __name__ == "__main__":
    # 测试
    vc = get_version_control("test_strategy")
    
    # 创建版本
    v1 = vc.create_version(
        params={"rsi_period": 14, "rsi_threshold": 30},
        description="Initial RSI strategy"
    )
    print(f"Created version: {v1.version_id}")
    
    # 创建新版本
    v2 = vc.create_version(
        params={"rsi_period": 14, "rsi_threshold": 25},
        description="Lower RSI threshold",
        metrics={"sharpe_ratio": 1.5}
    )
    print(f"Created version: {v2.version_id}")
    
    # 标记为稳定
    vc.mark_as_stable(v1.version_id)
    
    # 对比版本
    diff = vc.compare_versions(v1.version_id, v2.version_id)
    print(f"Version difference: {diff}")
    
    # 回滚
    result = vc.rollback_to(v1.version_id)
    print(f"Rollback result: {result}")
    
    print("Version control test completed!")
