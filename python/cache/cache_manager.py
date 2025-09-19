"""
缓存管理器 - 统一管理曲线和分析结果的缓存
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import threading

from core.curve import Curve, Peak


class CacheManager:
    """缓存管理器 - 提供曲线和分析结果的持久化缓存"""
    
    def __init__(self, cache_dir: str = "cache_data"):
        """
        初始化缓存管理器
        
        参数:
        - cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.curves_dir = self.cache_dir / "curves"
        self.analyses_dir = self.cache_dir / "analyses"
        self.metadata_dir = self.cache_dir / "metadata"
        
        for dir_path in [self.curves_dir, self.analyses_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 内存缓存
        self._memory_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        
        # 缓存元数据
        self.metadata_file = self.metadata_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """加载缓存元数据"""
        self.metadata = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"加载缓存元数据失败: {e}")
                self.metadata = {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"保存缓存元数据失败: {e}")
    
    def _generate_cache_key(self, file_path: str, curve_type: str, 
                          mz_range: tuple, rt_range: Optional[tuple] = None,
                          ms_level: Optional[int] = None, 
                          processing_params: Optional[Dict] = None) -> str:
        """
        生成缓存键
        
        参数包括文件路径、曲线类型、m/z范围、RT范围、MS级别和处理参数
        """
        # 获取文件的修改时间和大小作为文件标识
        try:
            stat = os.stat(file_path)
            file_signature = f"{stat.st_mtime}_{stat.st_size}"
        except:
            file_signature = "unknown"
        
        # 创建参数字符串
        params_str = f"{file_path}_{file_signature}_{curve_type}_{mz_range}"
        
        if rt_range:
            params_str += f"_rt{rt_range}"
        if ms_level is not None:
            params_str += f"_ms{ms_level}"
        if processing_params:
            # 将处理参数排序后加入
            sorted_params = json.dumps(processing_params, sort_keys=True)
            params_str += f"_proc{sorted_params}"
        
        # 生成MD5哈希
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def cache_curve(self, curve: Curve, file_path: str, 
                   extraction_params: Dict[str, Any],
                   processing_params: Optional[Dict[str, Any]] = None) -> str:
        """
        缓存曲线
        
        参数:
        - curve: 要缓存的曲线
        - file_path: 原始文件路径
        - extraction_params: 提取参数
        - processing_params: 处理参数
        
        返回:
        - 缓存键
        """
        cache_key = self._generate_cache_key(
            file_path=file_path,
            curve_type=extraction_params.get('curve_type', 'unknown'),
            mz_range=extraction_params.get('mz_range', (0, 0)),
            rt_range=extraction_params.get('rt_range'),
            ms_level=extraction_params.get('ms_level'),
            processing_params=processing_params
        )
        
        with self._cache_lock:
            # 保存到磁盘
            curve_file = self.curves_dir / f"{cache_key}.pkl"
            try:
                with open(curve_file, 'wb') as f:
                    pickle.dump(curve, f)
                
                # 更新元数据
                self.metadata[cache_key] = {
                    'type': 'curve',
                    'curve_id': curve.curve_id,
                    'curve_type': curve.curve_type,
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'extraction_params': extraction_params,
                    'processing_params': processing_params,
                    'cached_at': datetime.now().isoformat(),
                    'data_points': len(curve.x_values),
                    'peaks_count': len(curve.peaks)
                }
                
                # 也保存到内存缓存
                self._memory_cache[cache_key] = curve
                
                self._save_metadata()
                
                return cache_key
                
            except Exception as e:
                print(f"缓存曲线失败: {e}")
                return ""
    
    def get_cached_curve(self, cache_key: str) -> Optional[Curve]:
        """
        获取缓存的曲线
        
        参数:
        - cache_key: 缓存键
        
        返回:
        - 缓存的曲线，如果不存在则返回None
        """
        with self._cache_lock:
            # 先检查内存缓存
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]
            
            # 检查磁盘缓存
            curve_file = self.curves_dir / f"{cache_key}.pkl"
            if curve_file.exists():
                try:
                    with open(curve_file, 'rb') as f:
                        curve = pickle.load(f)
                    
                    # 加载到内存缓存
                    self._memory_cache[cache_key] = curve
                    return curve
                    
                except Exception as e:
                    print(f"加载缓存曲线失败: {e}")
                    return None
            
            return None
    
    def find_cached_curve(self, file_path: str, curve_type: str,
                         mz_range: tuple, rt_range: Optional[tuple] = None,
                         ms_level: Optional[int] = None,
                         processing_params: Optional[Dict] = None) -> Optional[str]:
        """
        查找匹配的缓存曲线键
        
        返回:
        - 缓存键，如果找到匹配的缓存
        """
        cache_key = self._generate_cache_key(
            file_path=file_path,
            curve_type=curve_type,
            mz_range=mz_range,
            rt_range=rt_range,
            ms_level=ms_level,
            processing_params=processing_params
        )
        
        if cache_key in self.metadata:
            return cache_key
        
        return None
    
    def list_cached_curves(self, file_path: Optional[str] = None,
                          curve_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出缓存的曲线
        
        参数:
        - file_path: 可选，过滤特定文件
        - curve_type: 可选，过滤特定曲线类型
        
        返回:
        - 匹配的缓存条目列表
        """
        results = []
        
        for cache_key, metadata in self.metadata.items():
            if metadata.get('type') != 'curve':
                continue
            
            # 应用过滤条件
            if file_path and metadata.get('file_path') != file_path:
                continue
            
            if curve_type and metadata.get('curve_type') != curve_type:
                continue
            
            results.append({
                'cache_key': cache_key,
                **metadata
            })
        
        # 按创建时间排序
        results.sort(key=lambda x: x.get('cached_at', ''), reverse=True)
        
        return results
    
    def delete_cached_curve(self, cache_key: str) -> bool:
        """
        删除缓存的曲线
        
        参数:
        - cache_key: 缓存键
        
        返回:
        - 是否删除成功
        """
        with self._cache_lock:
            try:
                # 从内存缓存删除
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                
                # 从磁盘删除
                curve_file = self.curves_dir / f"{cache_key}.pkl"
                if curve_file.exists():
                    curve_file.unlink()
                
                # 从元数据删除
                if cache_key in self.metadata:
                    del self.metadata[cache_key]
                    self._save_metadata()
                
                return True
                
            except Exception as e:
                print(f"删除缓存曲线失败: {e}")
                return False
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        清理缓存
        
        参数:
        - older_than_days: 可选，删除多少天前的缓存
        """
        with self._cache_lock:
            keys_to_delete = []
            
            if older_than_days is not None:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                
                for cache_key, metadata in self.metadata.items():
                    cached_at_str = metadata.get('cached_at', '')
                    try:
                        cached_at = datetime.fromisoformat(cached_at_str)
                        if cached_at < cutoff_date:
                            keys_to_delete.append(cache_key)
                    except:
                        # 如果日期解析失败，也删除
                        keys_to_delete.append(cache_key)
            else:
                # 删除所有缓存
                keys_to_delete = list(self.metadata.keys())
            
            # 删除选中的缓存
            for cache_key in keys_to_delete:
                self.delete_cached_curve(cache_key)
            
            print(f"清理了 {len(keys_to_delete)} 个缓存条目")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_curves = len([k for k, v in self.metadata.items() 
                           if v.get('type') == 'curve'])
        
        total_size = 0
        for file_path in self.curves_dir.glob("*.pkl"):
            try:
                total_size += file_path.stat().st_size
            except:
                pass
        
        # 按文件分组统计
        files_stats = {}
        for metadata in self.metadata.values():
            if metadata.get('type') == 'curve':
                file_name = metadata.get('file_name', 'unknown')
                if file_name not in files_stats:
                    files_stats[file_name] = {'count': 0, 'curve_types': set()}
                files_stats[file_name]['count'] += 1
                files_stats[file_name]['curve_types'].add(metadata.get('curve_type', 'unknown'))
        
        # 转换set为list以便JSON序列化
        for stats in files_stats.values():
            stats['curve_types'] = list(stats['curve_types'])
        
        return {
            'total_curves': total_curves,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir),
            'memory_cache_size': len(self._memory_cache),
            'files_stats': files_stats
        }


# 全局缓存管理器实例
cache_manager = CacheManager()
