"""
Rust桥接模块 - 调用Rust后端进行曲线提取
"""

import sys
import os
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import uuid

# 添加Rust编译的模块路径
rust_module_path = Path(__file__).parent.parent.parent / "target" / "release"
if rust_module_path.exists():
    sys.path.insert(0, str(rust_module_path))

from .curve import Curve


class RustBridge:
    """Rust后端桥接类"""
    
    def __init__(self):
        self._rust_module = None
        self._initialize_rust_module()
    
    def _initialize_rust_module(self):
        """初始化Rust模块"""
        try:
            # 尝试导入编译好的Rust模块
            import peakanalyzer_scripts
            self._rust_module = peakanalyzer_scripts
            print("✅ Rust模块加载成功")
        except ImportError as e:
            print(f"⚠️ Rust模块未找到: {e}")
            print("请确保已经编译了Rust模块: maturin develop")
            self._rust_module = None
    
    def is_available(self) -> bool:
        """检查Rust模块是否可用"""
        return self._rust_module is not None
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取MS文件信息"""
        if not self.is_available():
            raise RuntimeError("Rust模块不可用")
        
        try:
            info = self._rust_module.get_file_info(file_path)
            return dict(info)
        except Exception as e:
            raise RuntimeError(f"获取文件信息失败: {e}")
    
    def extract_tic_curve(
        self,
        file_path: str,
        mz_min: float,
        mz_max: float,
        rt_min: Optional[float] = None,
        rt_max: Optional[float] = None,
        ms_level: Optional[int] = None
    ) -> Curve:
        """提取TIC曲线"""
        if not self.is_available():
            return self._create_mock_curve("TIC", file_path)
        
        try:
            rust_curve = self._rust_module.extract_tic_curve(
                file_path, mz_min, mz_max, rt_min, rt_max, ms_level
            )
            return self._convert_rust_curve_to_python(rust_curve, file_path)
        except Exception as e:
            raise RuntimeError(f"TIC曲线提取失败: {e}")
    
    def extract_eic_curve(
        self,
        file_path: str,
        target_mz: float,
        mz_tolerance: float,
        rt_min: Optional[float] = None,
        rt_max: Optional[float] = None,
        ms_level: Optional[int] = None
    ) -> Curve:
        """提取EIC曲线"""
        if not self.is_available():
            return self._create_mock_curve("EIC", file_path)
        
        try:
            rust_curve = self._rust_module.extract_eic_curve(
                file_path, target_mz, mz_tolerance, rt_min, rt_max, ms_level
            )
            curve = self._convert_rust_curve_to_python(rust_curve, file_path)
            curve.metadata['target_mz'] = target_mz
            curve.metadata['mz_tolerance'] = mz_tolerance
            return curve
        except Exception as e:
            raise RuntimeError(f"EIC曲线提取失败: {e}")
    
    def extract_bpc_curve(
        self,
        file_path: str,
        mz_min: float,
        mz_max: float,
        rt_min: Optional[float] = None,
        rt_max: Optional[float] = None,
        ms_level: Optional[int] = None
    ) -> Curve:
        """提取BPC曲线"""
        if not self.is_available():
            return self._create_mock_curve("BPC", file_path)
        
        try:
            rust_curve = self._rust_module.extract_bpc_curve(
                file_path, mz_min, mz_max, rt_min, rt_max, ms_level
            )
            return self._convert_rust_curve_to_python(rust_curve, file_path)
        except Exception as e:
            raise RuntimeError(f"BPC曲线提取失败: {e}")
    
    def batch_extract_curves(
        self,
        file_paths: List[str],
        curve_type: str,
        mz_min: float,
        mz_max: float,
        rt_min: Optional[float] = None,
        rt_max: Optional[float] = None,
        ms_level: Optional[int] = None
    ) -> List[Curve]:
        """批量提取曲线"""
        if not self.is_available():
            return [self._create_mock_curve(curve_type, path) for path in file_paths]
        
        try:
            rust_curves = self._rust_module.batch_extract_curves(
                file_paths, curve_type, mz_min, mz_max, rt_min, rt_max, ms_level
            )
            
            curves = []
            for i, rust_curve in enumerate(rust_curves):
                file_path = file_paths[i] if i < len(file_paths) else "unknown"
                curve = self._convert_rust_curve_to_python(rust_curve, file_path)
                curves.append(curve)
            
            return curves
        except Exception as e:
            raise RuntimeError(f"批量曲线提取失败: {e}")
    
    def _convert_rust_curve_to_python(self, rust_curve, file_path: str) -> Curve:
        """将Rust曲线对象转换为Python曲线对象"""
        curve_id = f"{rust_curve.curve_type.lower()}_{uuid.uuid4().hex[:8]}"
        
        # 创建Python Curve对象
        curve = Curve(
            curve_id=curve_id,
            curve_type=rust_curve.curve_type,
            x_values=np.array(rust_curve.x_values),
            y_values=np.array(rust_curve.y_values),
            x_label=rust_curve.x_label,
            y_label=rust_curve.y_label,
            x_unit=rust_curve.x_unit,
            y_unit=rust_curve.y_unit
        )
        
        # 添加元数据
        curve.metadata.update({
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'extraction_method': 'rust_backend'
        })
        
        # 添加Rust传来的元数据
        if hasattr(rust_curve, 'metadata'):
            for key, value in rust_curve.metadata.items():
                curve.metadata[f'rust_{key}'] = value
        
        return curve
    
    def _create_mock_curve(self, curve_type: str, file_path: str) -> Curve:
        """创建模拟曲线（当Rust模块不可用时）"""
        # 生成模拟数据
        x_values = np.linspace(0, 30, 300)  # 30分钟，300个点
        
        # 创建一些模拟峰
        peaks_params = [
            (5.0, 1000, 0.5),   # (位置, 强度, 宽度)
            (12.0, 2500, 0.8),
            (18.5, 1500, 0.6),
            (25.0, 800, 0.4)
        ]
        
        y_values = np.random.normal(50, 10, len(x_values))  # 基线噪声
        
        for pos, intensity, width in peaks_params:
            peak = intensity * np.exp(-0.5 * ((x_values - pos) / width) ** 2)
            y_values += peak
        
        y_values = np.maximum(y_values, 0)  # 确保非负
        
        curve_id = f"{curve_type.lower()}_mock_{uuid.uuid4().hex[:8]}"
        
        curve = Curve(
            curve_id=curve_id,
            curve_type=curve_type,
            x_values=x_values,
            y_values=y_values,
            x_label="Retention Time",
            y_label="Intensity",
            x_unit="min",
            y_unit="counts"
        )
        
        curve.metadata.update({
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'extraction_method': 'mock_data',
            'is_mock': True
        })
        
        return curve


# 全局实例
rust_bridge = RustBridge()
