"""
曲线和峰数据结构定义
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import json
from datetime import datetime


@dataclass
class Peak:
    """峰数据结构 - 作为曲线分析的结果"""
    
    # 基本信息
    peak_id: str
    curve_id: str  # 所属曲线ID
    
    # 峰位置信息
    rt: float  # 保留时间 (分钟)
    rt_start: float  # 峰起始时间
    rt_end: float  # 峰结束时间
    intensity: float  # 峰强度
    
    # 峰形状参数
    area: float  # 峰面积
    height: float  # 峰高
    fwhm: float  # 半峰宽 (Full Width at Half Maximum)
    asymmetry: float = 0.0  # 峰不对称因子
    tailing_factor: float = 0.0  # 拖尾因子
    
    # 质量相关
    mz_range: Optional[tuple] = None  # m/z范围 (min, max)
    target_mz: Optional[float] = None  # 目标m/z值
    
    # 质量参数
    signal_to_noise: float = 0.0  # 信噪比
    resolution: float = 0.0  # 分辨率
    purity: float = 0.0  # 峰纯度 (0-1)
    
    # 分析结果
    is_valid: bool = True  # 是否为有效峰
    confidence: float = 1.0  # 置信度 (0-1)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = {
            'peak_id': self.peak_id,
            'curve_id': self.curve_id,
            'rt': self.rt,
            'rt_start': self.rt_start,
            'rt_end': self.rt_end,
            'intensity': self.intensity,
            'area': self.area,
            'height': self.height,
            'fwhm': self.fwhm,
            'asymmetry': self.asymmetry,
            'tailing_factor': self.tailing_factor,
            'signal_to_noise': self.signal_to_noise,
            'resolution': self.resolution,
            'purity': self.purity,
            'is_valid': self.is_valid,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
        
        if self.mz_range:
            data['mz_range'] = self.mz_range
        if self.target_mz:
            data['target_mz'] = self.target_mz
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Peak':
        """从字典创建Peak对象"""
        # 处理datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@dataclass
class Curve:
    """曲线数据结构 - Python端的完整曲线表示"""
    
    # 基本信息
    curve_id: str
    curve_type: str  # TIC, EIC, BPC等
    
    # 数据点
    x_values: np.ndarray  # X轴数据 (通常是时间)
    y_values: np.ndarray  # Y轴数据 (通常是强度)
    
    # 轴标签
    x_label: str = "Retention Time"
    y_label: str = "Intensity"
    x_unit: str = "min"
    y_unit: str = "counts"
    
    # 处理状态
    is_baseline_corrected: bool = False
    is_smoothed: bool = False
    is_normalized: bool = False
    
    # 原始数据备份（用于重置）
    original_x: Optional[np.ndarray] = None
    original_y: Optional[np.ndarray] = None
    
    # 分析结果
    peaks: List[Peak] = field(default_factory=list)
    
    # 处理参数记录
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保数据是numpy数组
        if not isinstance(self.x_values, np.ndarray):
            self.x_values = np.array(self.x_values)
        if not isinstance(self.y_values, np.ndarray):
            self.y_values = np.array(self.y_values)
            
        # 备份原始数据
        if self.original_x is None:
            self.original_x = self.x_values.copy()
        if self.original_y is None:
            self.original_y = self.y_values.copy()
    
    @property
    def length(self) -> int:
        """数据点数量"""
        return len(self.x_values)
    
    @property
    def x_range(self) -> tuple:
        """X轴范围"""
        return (float(np.min(self.x_values)), float(np.max(self.x_values)))
    
    @property
    def y_range(self) -> tuple:
        """Y轴范围"""
        return (float(np.min(self.y_values)), float(np.max(self.y_values)))
    
    @property
    def max_intensity(self) -> float:
        """最大强度"""
        return float(np.max(self.y_values))
    
    @property
    def min_intensity(self) -> float:
        """最小强度"""
        return float(np.min(self.y_values))
    
    @property
    def total_area(self) -> float:
        """总面积（梯形积分）"""
        return float(np.trapz(self.y_values, self.x_values))
    
    def add_peak(self, peak: Peak):
        """添加峰"""
        self.peaks.append(peak)
        self.updated_at = datetime.now()
    
    def remove_peak(self, peak_id: str):
        """移除峰"""
        self.peaks = [p for p in self.peaks if p.peak_id != peak_id]
        self.updated_at = datetime.now()
    
    def get_peak(self, peak_id: str) -> Optional[Peak]:
        """获取特定峰"""
        for peak in self.peaks:
            if peak.peak_id == peak_id:
                return peak
        return None
    
    def clear_peaks(self):
        """清除所有峰"""
        self.peaks.clear()
        self.updated_at = datetime.now()
    
    def add_processing_step(self, step_name: str, parameters: Dict[str, Any]):
        """记录处理步骤"""
        step = {
            'step': step_name,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat()
        }
        self.processing_history.append(step)
        self.updated_at = datetime.now()
    
    def reset_to_original(self):
        """重置到原始数据"""
        if self.original_x is not None and self.original_y is not None:
            self.x_values = self.original_x.copy()
            self.y_values = self.original_y.copy()
            self.is_baseline_corrected = False
            self.is_smoothed = False
            self.is_normalized = False
            self.processing_history.clear()
            self.clear_peaks()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        return {
            'curve_id': self.curve_id,
            'curve_type': self.curve_type,
            'x_values': self.x_values.tolist(),
            'y_values': self.y_values.tolist(),
            'x_label': self.x_label,
            'y_label': self.y_label,
            'x_unit': self.x_unit,
            'y_unit': self.y_unit,
            'is_baseline_corrected': self.is_baseline_corrected,
            'is_smoothed': self.is_smoothed,
            'is_normalized': self.is_normalized,
            'original_x': self.original_x.tolist() if self.original_x is not None else None,
            'original_y': self.original_y.tolist() if self.original_y is not None else None,
            'peaks': [peak.to_dict() for peak in self.peaks],
            'processing_history': self.processing_history,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Curve':
        """从字典创建Curve对象"""
        # 处理numpy数组
        data['x_values'] = np.array(data['x_values'])
        data['y_values'] = np.array(data['y_values'])
        
        if data.get('original_x'):
            data['original_x'] = np.array(data['original_x'])
        if data.get('original_y'):
            data['original_y'] = np.array(data['original_y'])
        
        # 处理峰数据
        if 'peaks' in data:
            data['peaks'] = [Peak.from_dict(peak_data) for peak_data in data['peaks']]
        
        # 处理datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """保存到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Curve':
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
