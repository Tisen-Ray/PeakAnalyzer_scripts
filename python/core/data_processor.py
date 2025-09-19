"""
数据处理调度器 - 调度peak_analysis模块的功能
"""

import numpy as np
from typing import Dict, Any
from .curve import Curve


class DataProcessor:
    """数据处理调度器 - 只负责调度，不包含具体实现"""
    
    def __init__(self):
        """初始化调度器"""
        pass
    
    def process_curve(self, curve: Curve, operations: Dict[str, Any]) -> Curve:
        """
        处理曲线数据 - 调度给具体的处理模块
        
        参数:
        - curve: 输入曲线
        - operations: 操作参数字典
        
        返回:
        - 处理后的曲线
        """
        # 创建曲线副本
        processed_curve = self._create_curve_copy(curve)
        
        # 调度基线校正
        if 'baseline_correction' in operations:
            processed_curve = self._dispatch_baseline_correction(
                processed_curve, operations['baseline_correction']
            )
        
        # 调度平滑处理
        if 'smoothing' in operations:
            processed_curve = self._dispatch_smoothing(
                processed_curve, operations['smoothing']
            )
        
        # 调度归一化
        if 'normalization' in operations:
            processed_curve = self._dispatch_normalization(
                processed_curve, operations['normalization']
            )
        
        return processed_curve
    
    def _create_curve_copy(self, curve: Curve) -> Curve:
        """创建曲线副本"""
        return Curve(
            curve_id=curve.curve_id,
            curve_type=curve.curve_type,
            x_values=curve.x_values.copy(),
            y_values=curve.y_values.copy(),
            x_label=curve.x_label,
            y_label=curve.y_label,
            x_unit=curve.x_unit,
            y_unit=curve.y_unit,
            original_x=curve.original_x,
            original_y=curve.original_y,
            processing_history=curve.processing_history.copy(),
            metadata=curve.metadata.copy()
        )
        
    def _dispatch_baseline_correction(self, curve: Curve, config: Dict[str, Any]) -> Curve:
        """调度基线校正处理"""
        # 这里应该调用peak_analysis模块的基线校正功能
        # 目前提供简单实现
        method = config.get('method', 'simple')
        
        if method == 'simple':
            # 简单的基线校正：减去最小值
            baseline = np.min(curve.y_values)
            curve.y_values = curve.y_values - baseline
        
        curve.is_baseline_corrected = True
        curve.add_processing_step('baseline_correction', config)
        
        return curve
    
    def _dispatch_smoothing(self, curve: Curve, config: Dict[str, Any]) -> Curve:
        """调度平滑处理"""
        # 这里应该调用peak_analysis模块的平滑功能
        # 目前提供简单实现
        method = config.get('method', 'simple')
        
        if method == 'simple':
            # 简单的移动平均
            window_size = config.get('params', {}).get('window_size', 3)
            if window_size > 1:
                kernel = np.ones(window_size) / window_size
                curve.y_values = np.convolve(curve.y_values, kernel, mode='same')
        
        curve.is_smoothed = True
        curve.add_processing_step('smoothing', config)
        
        return curve
    
    def _dispatch_normalization(self, curve: Curve, config: Dict[str, Any]) -> Curve:
        """调度归一化处理"""
        method = config.get('method', 'minmax')
        
        if method == 'minmax':
            y_min, y_max = np.min(curve.y_values), np.max(curve.y_values)
            if y_max > y_min:
                curve.y_values = (curve.y_values - y_min) / (y_max - y_min)
        
        curve.is_normalized = True
        curve.add_processing_step('normalization', config)
        
        return curve