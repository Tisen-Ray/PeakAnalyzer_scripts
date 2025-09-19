"""
峰检测模块 - 使用scipy进行峰检测
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Any, Optional
import uuid

from core.curve import Curve, Peak


class PeakDetector:
    """峰检测器 - 提供多种峰检测算法"""
    
    def __init__(self):
        self.detection_methods = {
            'scipy_find_peaks': self._detect_peaks_scipy,
            'cwt': self._detect_peaks_cwt,
            'derivative': self._detect_peaks_derivative,
            'threshold': self._detect_peaks_threshold
        }
    
    def detect_peaks(self, curve: Curve, method: str = 'scipy_find_peaks', 
                    **kwargs) -> List[Peak]:
        """
        检测峰
        
        参数:
        - curve: 输入曲线
        - method: 检测方法
        - **kwargs: 方法特定参数
        
        返回:
        - 检测到的峰列表
        """
        if method not in self.detection_methods:
            raise ValueError(f"未知的峰检测方法: {method}")
        
        return self.detection_methods[method](curve, **kwargs)
    
    def _detect_peaks_scipy(self, curve: Curve, 
                           height: Optional[float] = None,
                           prominence: Optional[float] = None,
                           distance: Optional[int] = None,
                           width: Optional[float] = None,
                           **kwargs) -> List[Peak]:
        """
        使用scipy.signal.find_peaks检测峰
        
        参数:
        - height: 最小峰高度
        - prominence: 最小突出度
        - distance: 峰之间的最小距离（索引）
        - width: 最小峰宽度
        """
        # 自动设置参数
        if height is None:
            height = np.mean(curve.y_values) + 2 * np.std(curve.y_values)
        
        if prominence is None:
            prominence = (np.max(curve.y_values) - np.min(curve.y_values)) * 0.1
        
        if distance is None:
            # 基于数据点密度设置最小距离
            avg_spacing = (curve.x_values[-1] - curve.x_values[0]) / len(curve.x_values)
            min_peak_width_time = 0.1  # 最小峰宽0.1分钟
            distance = max(1, int(min_peak_width_time / avg_spacing))
        
        # 检测峰
        peaks_idx, properties = signal.find_peaks(
            curve.y_values,
            height=height,
            prominence=prominence,
            distance=distance,
            width=width,
            **kwargs
        )
        
        return self._create_peaks_from_indices(curve, peaks_idx, properties)
    
    def _detect_peaks_cwt(self, curve: Curve, 
                         widths: Optional[np.ndarray] = None,
                         min_snr: float = 1.0,
                         noise_perc: float = 10.0,
                         **kwargs) -> List[Peak]:
        """
        使用连续小波变换检测峰
        
        参数:
        - widths: 小波宽度范围
        - min_snr: 最小信噪比
        - noise_perc: 噪声百分位数
        """
        if widths is None:
            # 自动设置宽度范围
            widths = np.arange(1, min(50, len(curve.y_values)//10))
        
        peaks_idx = signal.find_peaks_cwt(
            curve.y_values,
            widths,
            min_snr=min_snr,
            noise_perc=noise_perc,
            **kwargs
        )
        
        # 获取峰的属性
        properties = {}
        if len(peaks_idx) > 0:
            properties['peak_heights'] = curve.y_values[peaks_idx]
        
        return self._create_peaks_from_indices(curve, peaks_idx, properties)
    
    def _detect_peaks_derivative(self, curve: Curve,
                                threshold: Optional[float] = None,
                                min_distance: Optional[int] = None,
                                **kwargs) -> List[Peak]:
        """
        使用导数方法检测峰
        
        参数:
        - threshold: 导数阈值
        - min_distance: 峰之间最小距离
        """
        # 计算一阶导数
        dy = np.gradient(curve.y_values, curve.x_values)
        
        # 找到导数从正变负的点（峰顶）
        sign_changes = np.diff(np.sign(dy))
        peaks_idx = np.where(sign_changes < 0)[0] + 1
        
        # 应用阈值过滤
        if threshold is None:
            threshold = np.mean(curve.y_values) + np.std(curve.y_values)
        
        peaks_idx = peaks_idx[curve.y_values[peaks_idx] > threshold]
        
        # 应用最小距离过滤
        if min_distance is not None and len(peaks_idx) > 1:
            # 简单的距离过滤
            filtered_peaks = [peaks_idx[0]]
            for peak in peaks_idx[1:]:
                if peak - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peak)
            peaks_idx = np.array(filtered_peaks)
        
        # 获取峰的属性
        properties = {}
        if len(peaks_idx) > 0:
            properties['peak_heights'] = curve.y_values[peaks_idx]
        
        return self._create_peaks_from_indices(curve, peaks_idx, properties)
    
    def _detect_peaks_threshold(self, curve: Curve,
                               threshold: Optional[float] = None,
                               min_distance: Optional[int] = None,
                               **kwargs) -> List[Peak]:
        """
        使用简单阈值方法检测峰
        
        参数:
        - threshold: 强度阈值
        - min_distance: 峰之间最小距离
        """
        if threshold is None:
            threshold = np.mean(curve.y_values) + 2 * np.std(curve.y_values)
        
        # 找到超过阈值的点
        above_threshold = curve.y_values > threshold
        
        # 找到连续区域的峰值点
        peaks_idx = []
        i = 0
        while i < len(above_threshold):
            if above_threshold[i]:
                # 找到连续区域的结束
                j = i
                while j < len(above_threshold) and above_threshold[j]:
                    j += 1
                
                # 在这个区域内找到最高点
                if j > i:
                    region_max_idx = i + np.argmax(curve.y_values[i:j])
                    peaks_idx.append(region_max_idx)
                
                i = j
            else:
                i += 1
        
        peaks_idx = np.array(peaks_idx)
        
        # 应用最小距离过滤
        if min_distance is not None and len(peaks_idx) > 1:
            filtered_peaks = [peaks_idx[0]]
            for peak in peaks_idx[1:]:
                if peak - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peak)
            peaks_idx = np.array(filtered_peaks)
        
        # 获取峰的属性
        properties = {}
        if len(peaks_idx) > 0:
            properties['peak_heights'] = curve.y_values[peaks_idx]
        
        return self._create_peaks_from_indices(curve, peaks_idx, properties)
    
    def _create_peaks_from_indices(self, curve: Curve, peaks_idx: np.ndarray, 
                                  properties: Dict[str, Any]) -> List[Peak]:
        """从峰索引创建Peak对象"""
        peaks = []
        
        for i, idx in enumerate(peaks_idx):
            if idx >= len(curve.x_values) or idx >= len(curve.y_values):
                continue
            
            peak_id = f"peak_{uuid.uuid4().hex[:8]}"
            rt = float(curve.x_values[idx])
            intensity = float(curve.y_values[idx])
            
            # 估算峰的起始和结束位置
            rt_start, rt_end = self._estimate_peak_boundaries(curve, idx)
            
            # 计算峰面积（简单的三角形近似）
            width_indices = int((rt_end - rt_start) / 
                              (curve.x_values[-1] - curve.x_values[0]) * len(curve.x_values))
            start_idx = max(0, idx - width_indices//2)
            end_idx = min(len(curve.y_values), idx + width_indices//2)
            
            if end_idx > start_idx:
                area = float(np.trapz(curve.y_values[start_idx:end_idx], 
                                    curve.x_values[start_idx:end_idx]))
            else:
                area = 0.0
            
            # 估算FWHM
            fwhm = self._estimate_fwhm(curve, idx)
            
            # 计算信噪比
            noise_level = np.std(curve.y_values[:min(50, len(curve.y_values))])
            signal_to_noise = intensity / noise_level if noise_level > 0 else float('inf')
            
            peak = Peak(
                peak_id=peak_id,
                curve_id=curve.curve_id,
                rt=rt,
                rt_start=rt_start,
                rt_end=rt_end,
                intensity=intensity,
                area=area,
                height=intensity,
                fwhm=fwhm,
                signal_to_noise=signal_to_noise
            )
            
            # 添加检测方法相关的元数据
            if 'peak_heights' in properties and i < len(properties['peak_heights']):
                peak.metadata['detected_height'] = float(properties['peak_heights'][i])
            
            if 'prominences' in properties and i < len(properties['prominences']):
                peak.metadata['prominence'] = float(properties['prominences'][i])
            
            if 'widths' in properties and i < len(properties['widths']):
                peak.metadata['detected_width'] = float(properties['widths'][i])
            
            peaks.append(peak)
        
        return peaks
    
    def _estimate_peak_boundaries(self, curve: Curve, peak_idx: int) -> Tuple[float, float]:
        """估算峰的边界"""
        peak_height = curve.y_values[peak_idx]
        half_height = peak_height / 2
        
        # 向左找边界
        left_idx = peak_idx
        while left_idx > 0 and curve.y_values[left_idx] > half_height:
            left_idx -= 1
        
        # 向右找边界
        right_idx = peak_idx
        while right_idx < len(curve.y_values) - 1 and curve.y_values[right_idx] > half_height:
            right_idx += 1
        
        rt_start = float(curve.x_values[left_idx])
        rt_end = float(curve.x_values[right_idx])
        
        return rt_start, rt_end
    
    def _estimate_fwhm(self, curve: Curve, peak_idx: int) -> float:
        """
        正确估算半峰宽（FWHM）
        在半高水平线上找到与曲线的交点
        """
        peak_height = curve.y_values[peak_idx]
        half_height = peak_height / 2
        
        # 找到左侧交点
        left_intersection = None
        for i in range(peak_idx, -1, -1):
            if curve.y_values[i] <= half_height:
                if i < len(curve.y_values) - 1:
                    # 线性插值
                    x1, y1 = curve.x_values[i], curve.y_values[i]
                    x2, y2 = curve.x_values[i + 1], curve.y_values[i + 1]
                    if abs(y2 - y1) > 1e-10:
                        left_intersection = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
                    else:
                        left_intersection = x1
                else:
                    left_intersection = curve.x_values[i]
                break
        
        # 找到右侧交点
        right_intersection = None
        for i in range(peak_idx, len(curve.y_values)):
            if curve.y_values[i] <= half_height:
                if i > 0:
                    # 线性插值
                    x1, y1 = curve.x_values[i - 1], curve.y_values[i - 1]
                    x2, y2 = curve.x_values[i], curve.y_values[i]
                    if abs(y2 - y1) > 1e-10:
                        right_intersection = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
                    else:
                        right_intersection = x2
                else:
                    right_intersection = curve.x_values[i]
                break
        
        # 计算FWHM
        if left_intersection is not None and right_intersection is not None:
            fwhm = right_intersection - left_intersection
            return float(max(fwhm, 0.001))
        
        # 备用方法：使用边界估算
        rt_start, rt_end = self._estimate_peak_boundaries(curve, peak_idx)
        return max(rt_end - rt_start, 0.001)
    
    def get_available_methods(self) -> List[str]:
        """获取可用的检测方法"""
        return list(self.detection_methods.keys())
    
    def get_method_parameters(self, method: str) -> Dict[str, Any]:
        """获取方法的参数说明"""
        param_info = {
            'scipy_find_peaks': {
                'height': 'float, 最小峰高度',
                'prominence': 'float, 最小突出度', 
                'distance': 'int, 峰之间最小距离（索引）',
                'width': 'float, 最小峰宽度'
            },
            'cwt': {
                'widths': 'array, 小波宽度范围',
                'min_snr': 'float, 最小信噪比',
                'noise_perc': 'float, 噪声百分位数'
            },
            'derivative': {
                'threshold': 'float, 导数阈值',
                'min_distance': 'int, 峰之间最小距离'
            },
            'threshold': {
                'threshold': 'float, 强度阈值',
                'min_distance': 'int, 峰之间最小距离'
            }
        }
        
        return param_info.get(method, {})
