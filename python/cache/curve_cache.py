"""
曲线缓存 - 专门用于曲线对象的缓存管理
"""

from typing import Dict, List, Optional, Any
import threading
import time
from collections import OrderedDict

from core.curve import Curve


class CurveCache:
    """曲线内存缓存 - 提供快速的内存访问"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        初始化曲线缓存
        
        参数:
        - max_size: 最大缓存曲线数量
        - ttl_seconds: 缓存生存时间（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # 使用OrderedDict实现LRU缓存
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        
        # 启动清理线程
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
    
    def put(self, curve_id: str, curve: Curve) -> bool:
        """
        添加曲线到缓存
        
        参数:
        - curve_id: 曲线ID
        - curve: 曲线对象
        
        返回:
        - 是否成功添加
        """
        with self._lock:
            current_time = time.time()
            
            # 如果已存在，更新位置（LRU）
            if curve_id in self._cache:
                self._cache.move_to_end(curve_id)
                self._cache[curve_id]['curve'] = curve
                self._cache[curve_id]['timestamp'] = current_time
                self._cache[curve_id]['access_count'] += 1
                return True
            
            # 如果缓存已满，移除最久未使用的
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # 移除最旧的项
            
            # 添加新项
            self._cache[curve_id] = {
                'curve': curve,
                'timestamp': current_time,
                'access_count': 1
            }
            
            return True
    
    def get(self, curve_id: str) -> Optional[Curve]:
        """
        从缓存获取曲线
        
        参数:
        - curve_id: 曲线ID
        
        返回:
        - 曲线对象，如果不存在或已过期则返回None
        """
        with self._lock:
            if curve_id not in self._cache:
                return None
            
            current_time = time.time()
            cache_item = self._cache[curve_id]
            
            # 检查是否过期
            if current_time - cache_item['timestamp'] > self.ttl_seconds:
                del self._cache[curve_id]
                return None
            
            # 更新访问信息（LRU）
            self._cache.move_to_end(curve_id)
            cache_item['access_count'] += 1
            
            return cache_item['curve']
    
    def remove(self, curve_id: str) -> bool:
        """
        从缓存移除曲线
        
        参数:
        - curve_id: 曲线ID
        
        返回:
        - 是否成功移除
        """
        with self._lock:
            if curve_id in self._cache:
                del self._cache[curve_id]
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            current_time = time.time()
            
            # 统计信息
            total_curves = len(self._cache)
            expired_count = 0
            total_access_count = 0
            
            for cache_item in self._cache.values():
                if current_time - cache_item['timestamp'] > self.ttl_seconds:
                    expired_count += 1
                total_access_count += cache_item['access_count']
            
            return {
                'total_curves': total_curves,
                'expired_curves': expired_count,
                'active_curves': total_curves - expired_count,
                'max_size': self.max_size,
                'usage_percentage': (total_curves / self.max_size) * 100,
                'total_access_count': total_access_count,
                'average_access_count': total_access_count / total_curves if total_curves > 0 else 0,
                'ttl_seconds': self.ttl_seconds
            }
    
    def list_cached_curves(self) -> List[Dict[str, Any]]:
        """列出缓存中的曲线信息"""
        with self._lock:
            current_time = time.time()
            curves_info = []
            
            for curve_id, cache_item in self._cache.items():
                curve = cache_item['curve']
                age_seconds = current_time - cache_item['timestamp']
                is_expired = age_seconds > self.ttl_seconds
                
                curves_info.append({
                    'curve_id': curve_id,
                    'curve_type': curve.curve_type,
                    'data_points': len(curve.x_values),
                    'peaks_count': len(curve.peaks),
                    'age_seconds': age_seconds,
                    'is_expired': is_expired,
                    'access_count': cache_item['access_count'],
                    'file_name': curve.metadata.get('original_filename', '未知')
                })
            
            # 按访问次数排序
            curves_info.sort(key=lambda x: x['access_count'], reverse=True)
            
            return curves_info
    
    def get_most_accessed_curves(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取访问次数最多的曲线"""
        curves_info = self.list_cached_curves()
        return curves_info[:limit]
    
    def cleanup_expired(self) -> int:
        """手动清理过期缓存"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for curve_id, cache_item in self._cache.items():
                if current_time - cache_item['timestamp'] > self.ttl_seconds:
                    expired_keys.append(curve_id)
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def _cleanup_expired(self):
        """后台清理过期缓存的线程函数"""
        while True:
            try:
                self.cleanup_expired()
                time.sleep(60)  # 每分钟清理一次
            except Exception:
                # 忽略清理过程中的错误，继续运行
                time.sleep(60)
    
    def extend_ttl(self, curve_id: str) -> bool:
        """
        延长特定曲线的生存时间
        
        参数:
        - curve_id: 曲线ID
        
        返回:
        - 是否成功延长
        """
        with self._lock:
            if curve_id in self._cache:
                self._cache[curve_id]['timestamp'] = time.time()
                return True
            return False
    
    def get_cache_memory_usage(self) -> Dict[str, Any]:
        """估算缓存的内存使用情况"""
        import sys
        
        with self._lock:
            total_size = 0
            curve_sizes = []
            
            for curve_id, cache_item in self._cache.items():
                curve = cache_item['curve']
                
                # 估算单个曲线的内存使用
                curve_size = (
                    sys.getsizeof(curve.x_values) +
                    sys.getsizeof(curve.y_values) +
                    sys.getsizeof(curve.metadata) +
                    sys.getsizeof(curve.peaks) +
                    sys.getsizeof(curve_id) +
                    1024  # 其他属性的估算
                )
                
                curve_sizes.append({
                    'curve_id': curve_id,
                    'size_bytes': curve_size,
                    'size_mb': curve_size / (1024 * 1024)
                })
                
                total_size += curve_size
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'average_curve_size_mb': (total_size / len(self._cache)) / (1024 * 1024) if self._cache else 0,
                'curve_sizes': sorted(curve_sizes, key=lambda x: x['size_bytes'], reverse=True)
            }


# 全局曲线缓存实例
curve_cache = CurveCache()
