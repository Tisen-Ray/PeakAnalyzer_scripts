"""
曲线适配器 - 处理Rust PyCurve和Python Curve之间的转换
"""

import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from .curve import Curve, Peak


class CurveAdapter:
    """曲线适配器 - 负责不同曲线对象之间的转换"""
    
    @staticmethod
    def pycurve_to_curve(py_curve) -> Curve:
        """
        将Rust的PyCurve转换为Python的Curve对象
        
        参数:
        - py_curve: Rust返回的PyCurve对象
        
        返回:
        - Python的Curve对象
        """
        # 创建Python Curve对象
        curve = Curve(
            curve_id=py_curve.curve_id,
            curve_type=py_curve.curve_type,
            x_values=np.array(py_curve.x_values),
            y_values=np.array(py_curve.y_values),
            x_label=py_curve.x_label,
            y_label=py_curve.y_label,
            x_unit=py_curve.x_unit,
            y_unit=py_curve.y_unit
        )
        
        # 转换元数据
        for key, value in py_curve.metadata.items():
            curve.metadata[key] = value
        
        # 转换峰信息（如果有）
        if hasattr(py_curve, 'peaks') and py_curve.peaks:
            for peak_str in py_curve.peaks:
                # 这里可以解析峰字符串并创建Peak对象
                # 目前先简单处理
                pass
        
        return curve
    
    @staticmethod
    def curve_to_pycurve(curve: Curve):
        """
        将Python的Curve转换为类似PyCurve的字典结构
        （用于在页面间传递）
        """
        return {
            'curve_id': curve.curve_id,
            'curve_type': curve.curve_type,
            'x_values': curve.x_values.tolist() if hasattr(curve.x_values, 'tolist') else list(curve.x_values),
            'y_values': curve.y_values.tolist() if hasattr(curve.y_values, 'tolist') else list(curve.y_values),
            'x_label': curve.x_label,
            'y_label': curve.y_label,
            'x_unit': curve.x_unit,
            'y_unit': curve.y_unit,
            'metadata': curve.metadata,
            'peaks': curve.peaks,
            'processing_history': curve.processing_history,
            # 添加计算属性
            'x_range': curve.x_range,
            'y_range': curve.y_range,
            'max_intensity': curve.max_intensity,
            'min_intensity': curve.min_intensity,
            'total_area': curve.total_area
        }
    
    @staticmethod
    def create_mock_curve_dict(curve_id: str, curve_type: str, file_name: str = "mock.mzML"):
        """创建模拟曲线字典（当Rust不可用时）"""
        # 生成模拟数据
        x_values = np.linspace(0, 30, 300)
        
        # 创建模拟峰
        y_values = np.random.normal(50, 10, len(x_values))
        peaks_params = [(5.0, 1000, 0.5), (12.0, 2500, 0.8), (18.5, 1500, 0.6)]
        
        for pos, intensity, width in peaks_params:
            peak = intensity * np.exp(-0.5 * ((x_values - pos) / width) ** 2)
            y_values += peak
        
        y_values = np.maximum(y_values, 0)
        
        return {
            'curve_id': curve_id,
            'curve_type': curve_type,
            'x_values': x_values.tolist(),
            'y_values': y_values.tolist(),
            'x_label': "Retention Time",
            'y_label': "Intensity", 
            'x_unit': "min",
            'y_unit': "counts",
            'metadata': {'file_name': file_name, 'is_mock': 'true'},
            'peaks': [],
            'processing_history': [],
            'x_range': (float(np.min(x_values)), float(np.max(x_values))),
            'y_range': (float(np.min(y_values)), float(np.max(y_values))),
            'max_intensity': float(np.max(y_values)),
            'min_intensity': float(np.min(y_values)),
            'total_area': float(np.trapz(y_values, x_values))
        }


# 全局适配器实例
curve_adapter = CurveAdapter()
