"""
峰分析器 - 对检测到的峰进行详细分析
"""

import numpy as np
from scipy import optimize, ndimage
from typing import List, Dict, Any, Optional, Tuple
import uuid

from core.curve import Curve, Peak


class PeakAnalyzer:
    """峰分析器 - 提供峰的详细分析功能"""
    
    def __init__(self):
        pass
    
    def analyze_peak(self, curve: Curve, peak: Peak, 
                    extend_range: float = 2.0, 
                    integration_method: str = '垂直分割法',
                    baseline_method: str = '自动基线',
                    boundary_method: str = '自动选择（基于灵敏度）',
                    peak_sensitivity: int = 5,
                    noise_tolerance: int = 5,
                    boundary_smoothing: bool = True,
                    calc_theoretical_plates: bool = True,
                    calc_tailing_factor: bool = True,
                    calc_asymmetry_factor: bool = True,
                    calc_resolution: bool = True,
                    calc_capacity_factor: bool = False,
                    calc_selectivity: bool = False) -> Peak:
        """
        使用色谱分析标准方法分析单个峰
        
        参数:
        - curve: 所属曲线
        - peak: 要分析的峰
        - extend_range: 扩展分析范围的倍数（相对于FWHM）
        - integration_method: 积分方法
        - baseline_method: 基线处理方法
        - peak_sensitivity: 峰检测灵敏度
        - calc_*: 各种参数计算开关
        
        返回:
        - 更新后的峰对象
        """
        import copy
        updated_peak = copy.deepcopy(peak)
        
        # 获取峰周围的数据
        try:
            peak_data = self._extract_peak_region(curve, peak, extend_range)
            if not peak_data:
                return peak
            
            x_data, y_data, peak_idx = peak_data
            
            # 验证数据有效性
            if len(x_data) < 3 or len(y_data) < 3:
                return peak
            if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
                return peak
            if np.any(np.isinf(x_data)) or np.any(np.isinf(y_data)):
                return peak
        except Exception as e:
            print(f"Error extracting peak region: {e}")
            return peak
        
        try:
            # 1. 基线处理（色谱分析标准）
            baseline_y = self._process_baseline(x_data, y_data, baseline_method)
            corrected_y = y_data - baseline_y
            
            # 确保校正后的数据有效
            if np.any(np.isnan(corrected_y)) or np.any(np.isinf(corrected_y)):
                corrected_y = y_data.copy()
            
            # 2. 峰边界检测（使用色谱标准方法）
            boundaries = self._detect_chromatographic_boundaries_with_method(
                x_data, corrected_y, peak_idx, boundary_method,
                peak_sensitivity, noise_tolerance, boundary_smoothing
            )
            
            # 3. 峰积分（使用色谱标准方法）
            area = self._chromatographic_integration(
                x_data, corrected_y, boundaries, integration_method
            )
            
            # 4. 计算色谱峰参数
            fwhm = self._calculate_chromatographic_fwhm(x_data, corrected_y, peak_idx, boundaries)
        except Exception as e:
            print(f"Error in peak analysis calculations: {e}")
            # 使用基础值
            area = 0.0
            fwhm = 0.1
            boundaries = (peak.rt - 0.1, peak.rt + 0.1)
        
        # 更新基本峰参数
        updated_peak.area = area
        updated_peak.fwhm = fwhm
        updated_peak.rt_start = boundaries[0]
        updated_peak.rt_end = boundaries[1]
        
        # 5. 计算色谱质量参数（带异常处理）
        try:
            if calc_theoretical_plates:
                updated_peak.theoretical_plates = self._calculate_theoretical_plates(peak.rt, fwhm)
        except Exception as e:
            print(f"Error calculating theoretical plates: {e}")
            updated_peak.theoretical_plates = 0.0
        
        try:
            if calc_tailing_factor:
                updated_peak.tailing_factor = self._calculate_tailing_factor_usp(x_data, corrected_y, peak_idx)
        except Exception as e:
            print(f"Error calculating tailing factor: {e}")
            updated_peak.tailing_factor = 1.0
        
        try:
            if calc_asymmetry_factor:
                updated_peak.asymmetry_factor = self._calculate_asymmetry_factor_usp(x_data, corrected_y, peak_idx)
        except Exception as e:
            print(f"Error calculating asymmetry factor: {e}")
            updated_peak.asymmetry_factor = 1.0
        
        try:
            if calc_resolution:
                updated_peak.resolution = self._calculate_resolution_usp(curve, peak)
        except Exception as e:
            print(f"Error calculating resolution: {e}")
            updated_peak.resolution = 0.0
        
        try:
            if calc_capacity_factor:
                updated_peak.capacity_factor = self._calculate_capacity_factor(peak.rt, curve)
        except Exception as e:
            print(f"Error calculating capacity factor: {e}")
            updated_peak.capacity_factor = 0.0
        
        try:
            if calc_selectivity:
                updated_peak.selectivity = self._calculate_selectivity_factor(curve, peak)
        except Exception as e:
            print(f"Error calculating selectivity: {e}")
            updated_peak.selectivity = 1.0
        
        # 计算信噪比（色谱标准方法）
        try:
            updated_peak.signal_to_noise = self._calculate_chromatographic_snr(
                curve, peak, x_data, corrected_y, peak_idx
            )
        except Exception as e:
            print(f"Error calculating SNR: {e}")
            updated_peak.signal_to_noise = 0.0
        
        # 保存可视化数据到峰的元数据中（使用完整曲线坐标系）
        try:
            # 将局部基线数据映射回完整曲线坐标系
            full_baseline_y = self._map_baseline_to_full_curve(
                curve, x_data, baseline_y, boundaries
            )
            
            updated_peak.metadata['visualization_data'] = {
                'peak_region_x': x_data.tolist(),  # 峰区域x坐标
                'peak_region_y': y_data.tolist(),  # 峰区域y坐标
                'peak_region_baseline': baseline_y.tolist(),  # 峰区域基线
                'full_curve_baseline': full_baseline_y.tolist(),  # 完整曲线基线
                'corrected_y': corrected_y.tolist(),  # 校正后的峰区域数据
                'boundaries': boundaries,
                'integration_method': integration_method,
                'baseline_method': baseline_method
            }
        except Exception as e:
            print(f"Error saving visualization data: {e}")
            # 提供默认的可视化数据
            updated_peak.metadata['visualization_data'] = {
                'peak_region_x': [peak.rt - 0.1, peak.rt, peak.rt + 0.1],
                'peak_region_y': [peak.intensity * 0.1, peak.intensity, peak.intensity * 0.1],
                'peak_region_baseline': [peak.intensity * 0.05, peak.intensity * 0.05, peak.intensity * 0.05],
                'full_curve_baseline': [0] * len(curve.x_values),
                'corrected_y': [0, peak.intensity * 0.95, 0],
                'boundaries': (peak.rt - 0.1, peak.rt + 0.1),
                'integration_method': integration_method,
                'baseline_method': baseline_method
            }
        
        return updated_peak
    
    def analyze_all_peaks(self, curve: Curve) -> List[Peak]:
        """分析曲线中的所有峰"""
        analyzed_peaks = []
        
        for peak in curve.peaks:
            analyzed_peak = self.analyze_peak(curve, peak)
            analyzed_peaks.append(analyzed_peak)
        
        # 更新曲线中的峰
        curve.peaks = analyzed_peaks
        
        return analyzed_peaks
    
    def calculate_peak_statistics(self, peaks: List[Peak]) -> Dict[str, Any]:
        """计算峰的统计信息"""
        if not peaks:
            return {}
        
        areas = [peak.area for peak in peaks]
        heights = [peak.height for peak in peaks]
        fwhms = [peak.fwhm for peak in peaks]
        snrs = [peak.signal_to_noise for peak in peaks if peak.signal_to_noise > 0]
        
        stats = {
            'peak_count': len(peaks),
            'total_area': sum(areas),
            'area_stats': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas),
                'median': np.median(areas)
            },
            'height_stats': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': np.min(heights),
                'max': np.max(heights),
                'median': np.median(heights)
            },
            'fwhm_stats': {
                'mean': np.mean(fwhms),
                'std': np.std(fwhms),
                'min': np.min(fwhms),
                'max': np.max(fwhms),
                'median': np.median(fwhms)
            }
        }
        
        if snrs:
            stats['snr_stats'] = {
                'mean': np.mean(snrs),
                'std': np.std(snrs),
                'min': np.min(snrs),
                'max': np.max(snrs),
                'median': np.median(snrs)
            }
        
        return stats
    
    def _extract_peak_data(self, curve: Curve, peak: Peak, 
                          extend_range: float) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
        """提取峰周围的数据"""
        # 找到峰位置对应的索引
        peak_idx = self._find_nearest_index(curve.x_values, peak.rt)
        if peak_idx is None:
            return None
        
        # 计算扩展范围
        extend_width = peak.fwhm * extend_range
        start_rt = peak.rt - extend_width
        end_rt = peak.rt + extend_width
        
        # 找到范围内的索引
        start_idx = self._find_nearest_index(curve.x_values, start_rt)
        end_idx = self._find_nearest_index(curve.x_values, end_rt)
        
        if start_idx is None or end_idx is None:
            return None
        
        # 确保范围有效
        start_idx = max(0, start_idx)
        end_idx = min(len(curve.x_values) - 1, end_idx)
        
        if end_idx <= start_idx:
            return None
        
        x_data = curve.x_values[start_idx:end_idx+1]
        y_data = curve.y_values[start_idx:end_idx+1]
        
        return x_data, y_data, start_idx, end_idx
    
    def _find_nearest_index(self, array: np.ndarray, value: float) -> Optional[int]:
        """找到数组中最接近给定值的索引"""
        if len(array) == 0:
            return None
        
        idx = np.argmin(np.abs(array - value))
        return int(idx)
    
    def _calculate_fwhm(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """计算半峰宽"""
        if len(y_data) < 3:
            return 0.0
        
        # 找到峰顶
        peak_idx = np.argmax(y_data)
        peak_height = y_data[peak_idx]
        half_height = peak_height / 2
        
        # 向左找半高点
        left_idx = peak_idx
        while left_idx > 0 and y_data[left_idx] > half_height:
            left_idx -= 1
        
        # 向右找半高点
        right_idx = peak_idx
        while right_idx < len(y_data) - 1 and y_data[right_idx] > half_height:
            right_idx += 1
        
        # 线性插值找到精确的半高点
        if left_idx < peak_idx:
            # 左侧插值
            x1, y1 = x_data[left_idx], y_data[left_idx]
            x2, y2 = x_data[left_idx + 1], y_data[left_idx + 1]
            if y2 != y1:
                left_x = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
            else:
                left_x = x1
        else:
            left_x = x_data[left_idx]
        
        if right_idx > peak_idx:
            # 右侧插值
            x1, y1 = x_data[right_idx - 1], y_data[right_idx - 1]
            x2, y2 = x_data[right_idx], y_data[right_idx]
            if y1 != y2:
                right_x = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
            else:
                right_x = x2
        else:
            right_x = x_data[right_idx]
        
        return float(right_x - left_x)
    
    def _calculate_asymmetry(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """计算峰的不对称因子"""
        if len(y_data) < 3:
            return 1.0
        
        peak_idx = np.argmax(y_data)
        peak_x = x_data[peak_idx]
        peak_height = y_data[peak_idx]
        
        # 在10%峰高处测量不对称性
        target_height = peak_height * 0.1
        
        # 找到左右两侧的10%峰高点
        left_x = None
        right_x = None
        
        # 向左搜索
        for i in range(peak_idx, -1, -1):
            if y_data[i] <= target_height:
                if i < peak_idx:
                    # 线性插值
                    x1, y1 = x_data[i], y_data[i]
                    x2, y2 = x_data[i + 1], y_data[i + 1]
                    if y2 != y1:
                        left_x = x1 + (target_height - y1) * (x2 - x1) / (y2 - y1)
                    else:
                        left_x = x1
                break
        
        # 向右搜索
        for i in range(peak_idx, len(y_data)):
            if y_data[i] <= target_height:
                if i > peak_idx:
                    # 线性插值
                    x1, y1 = x_data[i - 1], y_data[i - 1]
                    x2, y2 = x_data[i], y_data[i]
                    if y1 != y2:
                        right_x = x1 + (target_height - y1) * (x2 - x1) / (y2 - y1)
                    else:
                        right_x = x2
                break
        
        if left_x is not None and right_x is not None:
            # 不对称因子 = b/a，其中a是峰顶到左边的距离，b是峰顶到右边的距离
            a = peak_x - left_x
            b = right_x - peak_x
            if a > 0:
                return float(b / a)
        
        return 1.0  # 对称峰
    
    def _calculate_tailing_factor(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """计算拖尾因子"""
        if len(y_data) < 3:
            return 1.0
        
        peak_idx = np.argmax(y_data)
        peak_height = y_data[peak_idx]
        
        # 在5%峰高处测量拖尾
        target_height = peak_height * 0.05
        
        # 找到前沿和拖尾的5%峰高点
        leading_width = None
        tailing_width = None
        
        # 计算前沿宽度（峰顶左侧）
        for i in range(peak_idx, -1, -1):
            if y_data[i] <= target_height:
                leading_width = x_data[peak_idx] - x_data[i]
                break
        
        # 计算拖尾宽度（峰顶右侧）
        for i in range(peak_idx, len(y_data)):
            if y_data[i] <= target_height:
                tailing_width = x_data[i] - x_data[peak_idx]
                break
        
        if leading_width and tailing_width and leading_width > 0:
            return float(tailing_width / leading_width)
        
        return 1.0
    
    def _calculate_resolution(self, curve: Curve, current_peak: Peak) -> float:
        """计算与相邻峰的分辨率"""
        if len(curve.peaks) < 2:
            return 0.0
        
        # 找到最近的相邻峰
        closest_peak = None
        min_distance = float('inf')
        
        for peak in curve.peaks:
            if peak.peak_id != current_peak.peak_id:
                distance = abs(peak.rt - current_peak.rt)
                if distance < min_distance:
                    min_distance = distance
                    closest_peak = peak
        
        if closest_peak is None:
            return 0.0
        
        # 计算分辨率 Rs = 2 * (rt2 - rt1) / (w1 + w2)
        # 其中w是峰宽（这里使用FWHM）
        rt_diff = abs(closest_peak.rt - current_peak.rt)
        width_sum = current_peak.fwhm + closest_peak.fwhm
        
        if width_sum > 0:
            resolution = 2 * rt_diff / width_sum
            return float(resolution)
        
        return 0.0
    
    def _calculate_purity(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """计算峰纯度（简化版本）"""
        if len(y_data) < 3:
            return 1.0
        
        # 简单的纯度估算：基于峰形的对称性
        peak_idx = np.argmax(y_data)
        
        # 计算峰的理论高斯形状
        peak_x = x_data[peak_idx]
        peak_height = y_data[peak_idx]
        
        # 估算标准差
        fwhm = self._calculate_fwhm(x_data, y_data)
        if fwhm <= 0:
            return 0.5
        
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma conversion
        
        if sigma <= 1e-10:  # 避免极小的sigma值
            return 0.5
        
        # 生成理论高斯曲线
        theoretical_y = peak_height * np.exp(-0.5 * ((x_data - peak_x) / sigma) ** 2)
        
        # 计算相关系数作为纯度指标
        correlation = np.corrcoef(y_data, theoretical_y)[0, 1]
        
        # 将相关系数转换为0-1的纯度值
        purity = max(0.0, min(1.0, correlation))
        
        return float(purity)
    
    def _calculate_signal_to_noise(self, curve: Curve, peak: Peak, 
                                 start_idx: int, end_idx: int) -> float:
        """计算信噪比"""
        # 使用峰周围的基线区域估算噪声
        noise_regions = []
        
        # 左侧噪声区域
        left_start = max(0, start_idx - 100)
        left_end = max(0, start_idx - 10)
        if left_end > left_start:
            noise_regions.extend(curve.y_values[left_start:left_end])
        
        # 右侧噪声区域
        right_start = min(len(curve.y_values), end_idx + 10)
        right_end = min(len(curve.y_values), end_idx + 100)
        if right_end > right_start:
            noise_regions.extend(curve.y_values[right_start:right_end])
        
        if noise_regions:
            noise_std = np.std(noise_regions)
            if noise_std > 0:
                return float(peak.height / noise_std)
        
        return 0.0
    
    # === 色谱分析标准方法实现 ===
    
    def _extract_peak_region(self, curve: Curve, peak, extend_range: float):
        """提取峰周围的数据区域"""
        # 找到峰位置对应的索引
        peak_idx_in_curve = np.argmin(np.abs(curve.x_values - peak.rt))
        
        # 估算扩展范围
        if hasattr(peak, 'fwhm') and peak.fwhm > 0:
            extend_width = peak.fwhm * extend_range
        else:
            # 如果没有FWHM，使用默认范围
            extend_width = max(0.5 * extend_range, 0.1)  # 最小0.1分钟
        
        start_rt = peak.rt - extend_width
        end_rt = peak.rt + extend_width
        
        # 提取数据
        mask = (curve.x_values >= start_rt) & (curve.x_values <= end_rt)
        x_data = curve.x_values[mask]
        y_data = curve.y_values[mask]
        
        if len(x_data) < 5:
            return None
        
        # 找到峰在提取数据中的索引
        peak_idx = np.argmin(np.abs(x_data - peak.rt))
        
        return x_data, y_data, peak_idx
    
    def _process_baseline(self, x_data, y_data, method: str):
        """基线处理（色谱分析标准）"""
        if method == "自动基线":
            return self._auto_baseline_chromatographic(x_data, y_data)
        elif method == "线性基线":
            return np.linspace(y_data[0], y_data[-1], len(y_data))
        elif method == "多项式基线":
            return self._polynomial_baseline(x_data, y_data)
        elif method == "指数基线":
            return self._exponential_baseline(x_data, y_data)
        else:
            return np.zeros_like(y_data)  # 水平基线
    
    def _auto_baseline_chromatographic(self, x_data, y_data):
        """自动基线检测（色谱分析标准）"""
        window_size = max(5, len(y_data) // 20)
        baseline = ndimage.minimum_filter1d(y_data, size=window_size)
        return baseline
    
    def _polynomial_baseline(self, x_data, y_data):
        """多项式基线拟合"""
        coeffs = np.polyfit(x_data, y_data, 2)  # 二次多项式
        return np.polyval(coeffs, x_data)
    
    def _exponential_baseline(self, x_data, y_data):
        """指数基线（适合梯度洗脱）"""
        try:
            from scipy.optimize import curve_fit
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            
            popt, _ = curve_fit(exp_func, x_data, y_data, 
                              p0=[y_data[0], 0.01, np.min(y_data)],
                              maxfev=1000)
            return exp_func(x_data, *popt)
        except:
            return np.linspace(y_data[0], y_data[-1], len(y_data))
    
    def _detect_chromatographic_boundaries_with_method(self, x_data, y_data, peak_idx: int,
                                                     boundary_method: str, sensitivity: int, 
                                                     noise_tolerance: int = 5, 
                                                     boundary_smoothing: bool = True):
        """
        根据用户选择的方法进行边界检测
        """
        try:
            if boundary_method == "切线撇取法 (Tangent Skim)":
                return self._tangent_skim_boundaries(x_data, y_data, peak_idx, noise_tolerance)
            elif boundary_method == "指数撇取法 (Exponential Skim)":
                return self._exponential_skim_boundaries(x_data, y_data, peak_idx, noise_tolerance, boundary_smoothing)
            elif boundary_method == "谷到谷法 (Valley-to-Valley)":
                return self._valley_to_valley_boundaries(x_data, y_data, peak_idx, noise_tolerance, boundary_smoothing)
            elif boundary_method == "垂直分割法 (Perpendicular Drop)":
                return self._perpendicular_drop_boundaries(x_data, y_data, peak_idx, noise_tolerance, boundary_smoothing)
            else:
                # 自动选择（基于灵敏度）
                return self._detect_chromatographic_boundaries(x_data, y_data, peak_idx, sensitivity, noise_tolerance, boundary_smoothing)
        except Exception as e:
            # 出错时使用备用方法
            return self._fallback_boundary_detection(x_data, y_data, peak_idx, noise_tolerance)
    
    def _detect_chromatographic_boundaries(self, x_data, y_data, peak_idx: int, 
                                         sensitivity: int, noise_tolerance: int = 5, 
                                         boundary_smoothing: bool = True):
        """
        色谱峰边界检测 - 使用色谱分析领域的标准方法
        
        实现多种经典的色谱峰边界检测算法：
        1. 切线撇取法 (Tangent Skim)
        2. 指数撇取法 (Exponential Skim) 
        3. 谷到谷法 (Valley-to-Valley)
        4. 垂直分割法 (Perpendicular Drop)
        5. 基线到基线法 (Baseline-to-Baseline)
        """
        # 根据灵敏度选择最佳方法
        if sensitivity <= 3:
            # 低敏感度：使用最保守的方法
            return self._tangent_skim_boundaries(x_data, y_data, peak_idx, noise_tolerance)
        elif sensitivity <= 6:
            # 中等敏感度：使用指数撇取法
            return self._exponential_skim_boundaries(x_data, y_data, peak_idx, noise_tolerance, boundary_smoothing)
        elif sensitivity <= 8:
            # 高敏感度：使用谷到谷法
            return self._valley_to_valley_boundaries(x_data, y_data, peak_idx, noise_tolerance, boundary_smoothing)
        else:
            # 最高敏感度：使用垂直分割法
            return self._perpendicular_drop_boundaries(x_data, y_data, peak_idx, noise_tolerance, boundary_smoothing)
    
    def _tangent_skim_boundaries(self, x_data, y_data, peak_idx: int, noise_tolerance: int):
        """
        切线撇取法 (Tangent Skim Method)
        
        色谱分析中最保守和准确的方法之一：
        1. 在峰的拐点处绘制切线
        2. 切线与基线的交点作为积分边界
        3. 适用于基线漂移和重叠峰的情况
        """
        try:
            # 寻找峰的拐点（二阶导数的极值点）
            left_inflection, right_inflection = self._find_inflection_points(x_data, y_data, peak_idx)
            
            # 在拐点处计算切线
            left_boundary = self._calculate_tangent_intersection(
                x_data, y_data, left_inflection, direction='left', noise_tolerance=noise_tolerance
            )
            right_boundary = self._calculate_tangent_intersection(
                x_data, y_data, right_inflection, direction='right', noise_tolerance=noise_tolerance
            )
            
            return left_boundary, right_boundary
            
        except Exception as e:
            # 备用：使用简化的切线方法
            return self._simplified_tangent_boundaries(x_data, y_data, peak_idx, noise_tolerance)
    
    def _exponential_skim_boundaries(self, x_data, y_data, peak_idx: int, noise_tolerance: int, smoothing: bool):
        """
        指数撇取法 (Exponential Skim Method)
        
        适用于拖尾峰的边界检测：
        1. 在峰的前沿和拖尾部分拟合指数函数
        2. 指数函数与基线的交点作为边界
        3. 特别适合处理不对称峰形
        """
        try:
            if smoothing:
                smoothed_y = self._apply_local_smoothing(y_data, window_size=3)
            else:
                smoothed_y = y_data
            
            # 估计基线水平
            baseline_level = self._estimate_baseline_level(smoothed_y, peak_idx, noise_tolerance)
            
            # 在峰的前沿拟合指数上升
            left_boundary = self._fit_exponential_rise(
                x_data, smoothed_y, peak_idx, baseline_level, noise_tolerance
            )
            
            # 在峰的拖尾部分拟合指数衰减
            right_boundary = self._fit_exponential_decay(
                x_data, smoothed_y, peak_idx, baseline_level, noise_tolerance
            )
            
            return left_boundary, right_boundary
            
        except Exception as e:
            # 备用方法
            return self._fallback_boundary_detection(x_data, y_data, peak_idx, noise_tolerance)
    
    def _valley_to_valley_boundaries(self, x_data, y_data, peak_idx: int, noise_tolerance: int, smoothing: bool):
        """
        谷到谷法 (Valley-to-Valley Method)
        
        经典的峰边界检测方法：
        1. 寻找峰两侧的最低点（谷点）
        2. 谷点作为积分边界
        3. 适用于基线相对平稳的情况
        """
        try:
            if smoothing:
                smoothed_y = self._apply_local_smoothing(y_data, window_size=5)
            else:
                smoothed_y = y_data
            
            # 寻找左侧谷点
            left_valley_idx = self._find_valley_point(
                smoothed_y, peak_idx, direction='left', noise_tolerance=noise_tolerance
            )
            
            # 寻找右侧谷点  
            right_valley_idx = self._find_valley_point(
                smoothed_y, peak_idx, direction='right', noise_tolerance=noise_tolerance
            )
            
            return x_data[left_valley_idx], x_data[right_valley_idx]
            
        except Exception as e:
            return self._fallback_boundary_detection(x_data, y_data, peak_idx, noise_tolerance)
    
    def _perpendicular_drop_boundaries(self, x_data, y_data, peak_idx: int, noise_tolerance: int, smoothing: bool):
        """
        垂直分割法 (Perpendicular Drop Method)
        
        最简单但有效的方法：
        1. 从峰顶向下画垂直线到基线
        2. 在指定高度比例处确定边界
        3. 适用于对称峰形
        """
        try:
            if smoothing:
                smoothed_y = self._apply_local_smoothing(y_data, window_size=3)
            else:
                smoothed_y = y_data
            
            peak_height = smoothed_y[peak_idx]
            baseline_level = self._estimate_baseline_level(smoothed_y, peak_idx, noise_tolerance)
            
            # 根据噪声容忍度调整阈值比例
            threshold_ratio = 0.01 + (10 - noise_tolerance) * 0.01  # 1%-10%
            threshold = baseline_level + (peak_height - baseline_level) * threshold_ratio
            
            # 寻找阈值交点
            left_idx = self._find_threshold_crossing(
                smoothed_y, peak_idx, threshold, direction='left'
            )
            right_idx = self._find_threshold_crossing(
                smoothed_y, peak_idx, threshold, direction='right'
            )
            
            return x_data[left_idx], x_data[right_idx]
            
        except Exception as e:
            return self._fallback_boundary_detection(x_data, y_data, peak_idx, noise_tolerance)
    
    # ====== 辅助函数：实现各种边界检测方法的核心算法 ======
    
    def _find_inflection_points(self, x_data, y_data, peak_idx):
        """寻找峰的拐点（二阶导数极值点）"""
        try:
            # 计算二阶导数
            second_derivative = np.gradient(np.gradient(y_data))
            
            # 在峰的左侧寻找拐点
            left_inflection = peak_idx // 2
            for i in range(peak_idx, max(0, peak_idx - len(y_data)//4), -1):
                if i > 1 and i < len(second_derivative) - 1:
                    if (second_derivative[i-1] < second_derivative[i] > second_derivative[i+1]):
                        left_inflection = i
                        break
            
            # 在峰的右侧寻找拐点
            right_inflection = min(len(y_data) - 1, peak_idx + (len(y_data) - peak_idx) // 2)
            for i in range(peak_idx, min(len(y_data) - 1, peak_idx + len(y_data)//4)):
                if i > 1 and i < len(second_derivative) - 1:
                    if (second_derivative[i-1] > second_derivative[i] < second_derivative[i+1]):
                        right_inflection = i
                        break
            
            return left_inflection, right_inflection
        except:
            # 备用：使用经验位置
            quarter_width = (len(y_data) - peak_idx) // 4
            return max(0, peak_idx - quarter_width), min(len(y_data) - 1, peak_idx + quarter_width)
    
    def _estimate_baseline_level(self, y_data, peak_idx, noise_tolerance):
        """估计基线水平"""
        try:
            # 取峰两端的数据点估计基线
            edge_points = []
            n_points = max(5, len(y_data) // 20)  # 取5%的数据点
            
            # 左端点
            edge_points.extend(y_data[:n_points])
            # 右端点
            edge_points.extend(y_data[-n_points:])
            
            # 使用中位数作为基线估计，更鲁棒
            baseline = np.median(edge_points)
            return float(baseline)
        except:
            return float(np.min(y_data))
    
    def _find_valley_point(self, y_data, peak_idx, direction, noise_tolerance):
        """寻找谷点"""
        try:
            if direction == 'left':
                search_range = range(peak_idx, max(0, peak_idx - len(y_data)//3), -1)
            else:
                search_range = range(peak_idx, min(len(y_data), peak_idx + len(y_data)//3))
            
            min_val = float('inf')
            valley_idx = peak_idx
            
            # 寻找局部最小值
            for i in search_range:
                if i > 2 and i < len(y_data) - 2:
                    # 检查是否为局部最小值
                    if (y_data[i] <= y_data[i-1] and y_data[i] <= y_data[i+1] and 
                        y_data[i] <= y_data[i-2] and y_data[i] <= y_data[i+2]):
                        if y_data[i] < min_val:
                            min_val = y_data[i]
                            valley_idx = i
            
            return valley_idx
        except:
            # 备用：返回边界
            return 0 if direction == 'left' else len(y_data) - 1
    
    def _find_threshold_crossing(self, y_data, peak_idx, threshold, direction):
        """寻找阈值交叉点"""
        try:
            if direction == 'left':
                for i in range(peak_idx, -1, -1):
                    if y_data[i] <= threshold:
                        return i
                return 0
            else:
                for i in range(peak_idx, len(y_data)):
                    if y_data[i] <= threshold:
                        return i
                return len(y_data) - 1
        except:
            return 0 if direction == 'left' else len(y_data) - 1
    
    def _fallback_boundary_detection(self, x_data, y_data, peak_idx, noise_tolerance):
        """备用边界检测方法"""
        try:
            peak_height = y_data[peak_idx]
            baseline = np.median([np.min(y_data[:len(y_data)//4]), np.min(y_data[-len(y_data)//4:])])
            
            # 使用5%峰高作为阈值
            threshold = baseline + (peak_height - baseline) * 0.05
            
            left_idx = self._find_threshold_crossing(y_data, peak_idx, threshold, 'left')
            right_idx = self._find_threshold_crossing(y_data, peak_idx, threshold, 'right')
            
            return x_data[left_idx], x_data[right_idx]
        except:
            # 最后的备用方法
            left_boundary = x_data[max(0, peak_idx - len(x_data)//10)]
            right_boundary = x_data[min(len(x_data) - 1, peak_idx + len(x_data)//10)]
            return left_boundary, right_boundary
    
    def _simplified_tangent_boundaries(self, x_data, y_data, peak_idx, noise_tolerance):
        """简化的切线边界检测"""
        try:
            # 使用一阶导数找到斜率变化最大的点
            gradient = np.gradient(y_data)
            
            # 左侧：找到斜率最大的点
            left_max_slope_idx = peak_idx
            max_slope = 0
            for i in range(peak_idx, max(0, peak_idx - len(y_data)//4), -1):
                if gradient[i] > max_slope:
                    max_slope = gradient[i]
                    left_max_slope_idx = i
            
            # 右侧：找到斜率最小的点（最负）
            right_max_slope_idx = peak_idx
            min_slope = 0
            for i in range(peak_idx, min(len(y_data), peak_idx + len(y_data)//4)):
                if gradient[i] < min_slope:
                    min_slope = gradient[i]
                    right_max_slope_idx = i
            
            # 从这些点向基线延伸
            baseline = self._estimate_baseline_level(y_data, peak_idx, noise_tolerance)
            
            # 简化：直接使用这些点作为边界
            left_boundary = x_data[max(0, left_max_slope_idx - 2)]
            right_boundary = x_data[min(len(x_data) - 1, right_max_slope_idx + 2)]
            
            return left_boundary, right_boundary
        except:
            return self._fallback_boundary_detection(x_data, y_data, peak_idx, noise_tolerance)
    
    def _calculate_tangent_intersection(self, x_data, y_data, inflection_idx, direction, noise_tolerance):
        """计算切线与基线的交点（简化版本）"""
        # 这是一个简化实现，实际的切线计算会更复杂
        baseline = self._estimate_baseline_level(y_data, len(y_data)//2, noise_tolerance)
        
        if direction == 'left':
            return x_data[max(0, inflection_idx - 3)]
        else:
            return x_data[min(len(x_data) - 1, inflection_idx + 3)]
    
    def _fit_exponential_rise(self, x_data, y_data, peak_idx, baseline_level, noise_tolerance):
        """拟合指数上升（简化版本）"""
        try:
            # 寻找上升部分的起点
            for i in range(peak_idx, -1, -1):
                if y_data[i] <= baseline_level * 1.1:  # 基线上方10%
                    return x_data[i]
            return x_data[0]
        except:
            return x_data[0]
    
    def _fit_exponential_decay(self, x_data, y_data, peak_idx, baseline_level, noise_tolerance):
        """拟合指数衰减（简化版本）"""
        try:
            # 寻找衰减部分的终点
            for i in range(peak_idx, len(y_data)):
                if y_data[i] <= baseline_level * 1.1:  # 基线上方10%
                    return x_data[i]
            return x_data[-1]
        except:
            return x_data[-1]
    
    def _estimate_local_noise(self, y_data: np.ndarray, peak_idx: int, window_size: int = 20) -> float:
        """估计峰附近的噪声水平"""
        try:
            # 选择峰两侧的区域估计噪声（避开峰本身）
            left_start = max(0, peak_idx - window_size * 3)
            left_end = max(0, peak_idx - window_size)
            right_start = min(len(y_data), peak_idx + window_size)
            right_end = min(len(y_data), peak_idx + window_size * 3)
            
            noise_regions = []
            if left_end > left_start:
                noise_regions.extend(y_data[left_start:left_end])
            if right_end > right_start:
                noise_regions.extend(y_data[right_start:right_end])
            
            if len(noise_regions) > 5:
                return float(np.std(noise_regions))
            else:
                # 备用方法：使用全局噪声估计
                return float(np.std(y_data) * 0.1)
        except:
            return float(np.std(y_data) * 0.1)
    
    def _apply_local_smoothing(self, y_data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """应用局部平滑以减少小波动"""
        try:
            from scipy.ndimage import uniform_filter1d
            # 使用均匀滤波器进行平滑
            smoothed = uniform_filter1d(y_data.astype(float), size=window_size, mode='nearest')
            return smoothed
        except:
            # 备用方法：简单移动平均
            if len(y_data) < window_size:
                return y_data
            
            smoothed = np.copy(y_data).astype(float)
            half_window = window_size // 2
            
            for i in range(half_window, len(y_data) - half_window):
                smoothed[i] = np.mean(y_data[i - half_window:i + half_window + 1])
            
            return smoothed
    
    def _find_robust_left_boundary(self, y_data: np.ndarray, peak_idx: int, 
                                 threshold: float, sensitivity: int) -> int:
        """鲁棒的左边界搜索"""
        left_idx = peak_idx
        consecutive_below_count = 0
        required_consecutive = max(2, (10 - sensitivity) // 2)  # 需要连续低于阈值的点数
        
        while left_idx > 0:
            if y_data[left_idx] <= threshold:
                consecutive_below_count += 1
                if consecutive_below_count >= required_consecutive:
                    # 找到稳定的边界，回退到第一个低于阈值的点
                    return left_idx + required_consecutive - 1
            else:
                consecutive_below_count = 0
            left_idx -= 1
        
        return left_idx
    
    def _find_robust_right_boundary(self, y_data: np.ndarray, peak_idx: int, 
                                  threshold: float, sensitivity: int) -> int:
        """鲁棒的右边界搜索"""
        right_idx = peak_idx
        consecutive_below_count = 0
        required_consecutive = max(2, (10 - sensitivity) // 2)  # 需要连续低于阈值的点数
        
        while right_idx < len(y_data) - 1:
            if y_data[right_idx] <= threshold:
                consecutive_below_count += 1
                if consecutive_below_count >= required_consecutive:
                    # 找到稳定的边界，回退到第一个低于阈值的点
                    return right_idx - required_consecutive + 1
            else:
                consecutive_below_count = 0
            right_idx += 1
        
        return right_idx
    
    def _validate_and_adjust_boundaries(self, x_data: np.ndarray, y_data: np.ndarray, 
                                      peak_idx: int, left_idx: int, right_idx: int, 
                                      threshold: float) -> Tuple[int, int]:
        """验证和调整边界"""
        # 1. 确保边界合理
        left_idx = max(0, min(left_idx, peak_idx - 1))
        right_idx = min(len(y_data) - 1, max(right_idx, peak_idx + 1))
        
        # 2. 检查峰宽是否合理
        peak_width_time = x_data[right_idx] - x_data[left_idx]
        min_width = 0.01  # 最小峰宽0.01分钟
        max_width = (x_data[-1] - x_data[0]) * 0.5  # 最大峰宽不超过总时间的一半
        
        if peak_width_time < min_width:
            # 峰太窄，适当扩展
            expand_points = max(1, int(min_width / np.mean(np.diff(x_data))))
            left_idx = max(0, left_idx - expand_points)
            right_idx = min(len(y_data) - 1, right_idx + expand_points)
        elif peak_width_time > max_width:
            # 峰太宽，可能检测错误，使用更严格的阈值重新搜索
            stricter_threshold = threshold * 2
            
            # 重新搜索左边界
            temp_left = peak_idx
            while temp_left > left_idx and y_data[temp_left] > stricter_threshold:
                temp_left -= 1
            left_idx = temp_left
            
            # 重新搜索右边界
            temp_right = peak_idx
            while temp_right < right_idx and y_data[temp_right] > stricter_threshold:
                temp_right += 1
            right_idx = temp_right
        
        # 3. 最终边界检查
        left_idx = max(0, left_idx)
        right_idx = min(len(y_data) - 1, right_idx)
        
        return left_idx, right_idx
    
    def _chromatographic_integration(self, x_data, y_data, boundaries, method: str):
        """色谱峰积分（标准方法）"""
        start_rt, end_rt = boundaries
        mask = (x_data >= start_rt) & (x_data <= end_rt)
        x_region = x_data[mask]
        y_region = y_data[mask]
        
        if len(x_region) < 3:
            return 0.0
        
        if method == "垂直分割法":
            return float(np.trapz(np.maximum(y_region, 0), x_region))
        elif method == "谷到谷积分":
            return self._valley_to_valley_integration(x_region, y_region)
        elif method == "切线基线法":
            return self._tangent_baseline_integration(x_region, y_region)
        elif method == "指数衰减基线":
            return self._exponential_skim_integration(x_region, y_region)
        else:  # 水平基线法
            baseline = np.min(y_region)
            corrected_y = y_region - baseline
            return float(np.trapz(np.maximum(corrected_y, 0), x_region))
    
    def _valley_to_valley_integration(self, x_data, y_data):
        """谷到谷积分"""
        valley_baseline = np.linspace(y_data[0], y_data[-1], len(y_data))
        corrected_y = y_data - valley_baseline
        return float(np.trapz(np.maximum(corrected_y, 0), x_data))
    
    def _tangent_baseline_integration(self, x_data, y_data):
        """切线基线法积分"""
        n = len(y_data)
        if n < 5:
            return float(np.trapz(y_data, x_data))
        
        # 起始点切线斜率
        dx_start = x_data[2] - x_data[0]
        start_slope = (y_data[2] - y_data[0]) / dx_start if abs(dx_start) > 1e-10 else 0.0
        # 结束点切线斜率  
        dx_end = x_data[-1] - x_data[-3]
        end_slope = (y_data[-1] - y_data[-3]) / dx_end if abs(dx_end) > 1e-10 else 0.0
        
        # 构建切线基线
        baseline = np.zeros_like(y_data)
        for i, x in enumerate(x_data):
            if i < n // 2:
                baseline[i] = y_data[0] + start_slope * (x - x_data[0])
            else:
                baseline[i] = y_data[-1] + end_slope * (x - x_data[-1])
        
        corrected_y = y_data - baseline
        return float(np.trapz(np.maximum(corrected_y, 0), x_data))
    
    def _exponential_skim_integration(self, x_data, y_data):
        """指数衰减基线积分"""
        try:
            from scipy.optimize import curve_fit
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * (x - x_data[0])) + c
            
            edge_indices = [0, 1, -2, -1]
            x_edge = x_data[edge_indices]
            y_edge = y_data[edge_indices]
            
            popt, _ = curve_fit(exp_decay, x_edge, y_edge, 
                              p0=[y_data[0] - y_data[-1], 0.1, y_data[-1]],
                              maxfev=1000)
            baseline = exp_decay(x_data, *popt)
        except:
            baseline = np.linspace(y_data[0], y_data[-1], len(y_data))
        
        corrected_y = y_data - baseline
        return float(np.trapz(np.maximum(corrected_y, 0), x_data))
    
    def _calculate_chromatographic_fwhm(self, x_data, y_data, peak_idx: int, boundaries):
        """
        正确计算色谱峰半峰宽（FWHM）
        
        FWHM定义：在峰高一半的水平线上，找到与曲线的两个交点，
        这两点之间的距离就是半高全宽
        """
        peak_height = y_data[peak_idx]
        half_height = peak_height / 2
        
        start_rt, end_rt = boundaries
        mask = (x_data >= start_rt) & (x_data <= end_rt)
        x_region = x_data[mask]
        y_region = y_data[mask]
        
        if len(x_region) < 3:
            return 0.001
        
        peak_idx_region = np.argmax(y_region)
        
        # 找到半高线与曲线的交点
        left_intersection = self._find_half_height_intersection(
            x_region, y_region, half_height, peak_idx_region, direction='left'
        )
        
        right_intersection = self._find_half_height_intersection(
            x_region, y_region, half_height, peak_idx_region, direction='right'
        )
        
        if left_intersection is not None and right_intersection is not None:
            fwhm = right_intersection - left_intersection
            return float(max(fwhm, 0.001))  # 确保FWHM不为零
        
        # 备用方法：如果找不到交点，使用简化估算
        return self._estimate_fwhm_fallback(x_region, y_region, peak_idx_region, half_height)
    
    def _find_half_height_intersection(self, x_data, y_data, half_height, peak_idx, direction='left'):
        """
        找到半高线与曲线的精确交点
        
        正确的FWHM计算方法：
        1. 确定峰高 (peak_height)
        2. 计算半高 (half_height = peak_height / 2)
        3. 画一条水平线在半高位置
        4. 找到这条水平线与曲线的两个交点
        5. 两个交点之间的距离就是FWHM
        
        参数:
        - x_data, y_data: 曲线数据
        - half_height: 半高值
        - peak_idx: 峰顶索引
        - direction: 'left' 或 'right'
        
        返回:
        - 交点的x坐标，如果找不到则返回None
        """
        try:
            if direction == 'left':
                # 从峰顶向左搜索，找到第一个低于半高的点
                for i in range(peak_idx, -1, -1):
                    if y_data[i] <= half_height:
                        # 找到交叉点，进行线性插值
                        if i < len(y_data) - 1:
                            # 在点i和i+1之间进行插值
                            x1, y1 = x_data[i], y_data[i]
                            x2, y2 = x_data[i + 1], y_data[i + 1]
                            
                            # 线性插值找到精确交点
                            if abs(y2 - y1) > 1e-10:
                                intersection_x = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
                                return float(intersection_x)
                        return float(x_data[i])
                
                # 如果没找到交点，返回边界点
                return float(x_data[0])
                
            else:  # direction == 'right'
                # 从峰顶向右搜索，找到第一个低于半高的点
                for i in range(peak_idx, len(y_data)):
                    if y_data[i] <= half_height:
                        # 找到交叉点，进行线性插值
                        if i > 0:
                            # 在点i-1和i之间进行插值
                            x1, y1 = x_data[i - 1], y_data[i - 1]
                            x2, y2 = x_data[i], y_data[i]
                            
                            # 线性插值找到精确交点
                            if abs(y2 - y1) > 1e-10:
                                intersection_x = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
                                return float(intersection_x)
                        return float(x_data[i])
                
                # 如果没找到交点，返回边界点
                return float(x_data[-1])
                
        except Exception as e:
            # 发生错误时返回None
            return None
    
    def _estimate_fwhm_fallback(self, x_data, y_data, peak_idx, half_height):
        """
        备用FWHM估算方法
        当无法找到精确交点时使用
        """
        try:
            # 使用简单的边界估算
            left_boundary = x_data[0]
            right_boundary = x_data[-1]
            
            # 尝试找到大致的半高边界
            for i in range(peak_idx):
                if y_data[i] <= half_height * 1.2:  # 稍微放宽条件
                    left_boundary = x_data[i]
                    break
            
            for i in range(peak_idx, len(y_data)):
                if y_data[i] <= half_height * 1.2:  # 稍微放宽条件
                    right_boundary = x_data[i]
                    break
            
            fwhm = right_boundary - left_boundary
            return float(max(fwhm, 0.001))
            
        except:
            # 最后的备用方法：基于峰宽的经验估算
            peak_width = x_data[-1] - x_data[0]
            return float(max(peak_width * 0.5, 0.001))
    
    def _calculate_theoretical_plates(self, rt, fwhm):
        """计算理论塔板数 N = 5.54 * (tR / W1/2)²"""
        if fwhm <= 0:
            return 0.0
        return 5.54 * (rt / fwhm) ** 2
    
    def _calculate_tailing_factor_usp(self, x_data, y_data, peak_idx):
        """计算USP拖尾因子 Tf = W0.05 / 2f"""
        peak_height = y_data[peak_idx]
        height_5_percent = peak_height * 0.05
        
        # 找到5%峰高的点
        left_idx = peak_idx
        while left_idx > 0 and y_data[left_idx] > height_5_percent:
            left_idx -= 1
        
        right_idx = peak_idx
        while right_idx < len(y_data) - 1 and y_data[right_idx] > height_5_percent:
            right_idx += 1
        
        if left_idx >= peak_idx or right_idx <= peak_idx:
            return 1.0
        
        w_005 = x_data[right_idx] - x_data[left_idx]  # 5%峰高处的峰宽
        f = x_data[peak_idx] - x_data[left_idx]  # 峰顶到前沿的距离
        
        if f <= 0:
            return 1.0
        
        return w_005 / (2 * f) if f != 0 else 1.0
    
    def _calculate_asymmetry_factor_usp(self, x_data, y_data, peak_idx):
        """计算USP不对称因子 As = b / a (在10%峰高处)"""
        peak_height = y_data[peak_idx]
        height_10_percent = peak_height * 0.1
        
        # 找到10%峰高的点
        left_idx = peak_idx
        while left_idx > 0 and y_data[left_idx] > height_10_percent:
            left_idx -= 1
        
        right_idx = peak_idx
        while right_idx < len(y_data) - 1 and y_data[right_idx] > height_10_percent:
            right_idx += 1
        
        if left_idx >= peak_idx or right_idx <= peak_idx:
            return 1.0
        
        a = x_data[peak_idx] - x_data[left_idx]   # 前沿距离
        b = x_data[right_idx] - x_data[peak_idx]  # 后沿距离
        
        if a <= 0:
            return 1.0
        
        return b / a if a != 0 else 1.0
    
    def _calculate_resolution_usp(self, curve, current_peak):
        """计算USP分离度 Rs = 2(tR2 - tR1) / (W1 + W2)"""
        if len(curve.peaks) < 2:
            return 0.0
        
        # 找到最近的相邻峰
        closest_peak = None
        min_distance = float('inf')
        
        for peak in curve.peaks:
            if peak.peak_id != current_peak.peak_id:
                distance = abs(peak.rt - current_peak.rt)
                if distance < min_distance:
                    min_distance = distance
                    closest_peak = peak
        
        if closest_peak is None:
            return 0.0
        
        rt_diff = abs(closest_peak.rt - current_peak.rt)
        width_sum = current_peak.fwhm + closest_peak.fwhm
        
        if width_sum <= 0:
            return 0.0
        
        return 2 * rt_diff / width_sum if width_sum != 0 else 0.0
    
    def _calculate_capacity_factor(self, rt, curve):
        """计算容量因子 k' = (tR - t0) / t0"""
        if len(curve.peaks) > 0:
            t0 = min(peak.rt for peak in curve.peaks)
            if t0 <= 0:
                t0 = 1.0
        else:
            t0 = 1.0
        
        return (rt - t0) / t0 if t0 != 0 else 0.0
    
    def _calculate_selectivity_factor(self, curve, current_peak):
        """计算选择性因子 α = k'2 / k'1"""
        if len(curve.peaks) < 2:
            return 1.0
        
        # 找到相邻峰
        adjacent_peak = None
        for peak in curve.peaks:
            if peak.peak_id != current_peak.peak_id:
                if adjacent_peak is None or abs(peak.rt - current_peak.rt) < abs(adjacent_peak.rt - current_peak.rt):
                    adjacent_peak = peak
        
        if adjacent_peak is None:
            return 1.0
        
        k1 = self._calculate_capacity_factor(current_peak.rt, curve)
        k2 = self._calculate_capacity_factor(adjacent_peak.rt, curve)
        
        if k1 <= 0:
            return 1.0
        
        min_k = min(k2, k1)
        max_k = max(k2, k1)
        return max_k / min_k if min_k != 0 else 1.0  # 确保α >= 1
    
    def _calculate_chromatographic_snr(self, curve, peak, x_data, y_data, peak_idx):
        """计算色谱信噪比"""
        peak_height = y_data[peak_idx]
        
        try:
            # 获取峰前后的噪声区域
            full_x = curve.x_values
            full_y = curve.y_values
            
            # 峰前区域
            before_mask = full_x < (peak.rt - 2 * peak.fwhm)
            before_noise = full_y[before_mask][-50:] if np.any(before_mask) else []
            
            # 峰后区域  
            after_mask = full_x > (peak.rt + 2 * peak.fwhm)
            after_noise = full_y[after_mask][:50] if np.any(after_mask) else []
            
            # 合并噪声数据
            noise_data = np.concatenate([before_noise, after_noise]) if len(before_noise) > 0 or len(after_noise) > 0 else full_y[:50]
            
            if len(noise_data) > 0:
                noise_level = np.std(noise_data)
                return peak_height / noise_level if noise_level > 0 else 0.0
            else:
                return 0.0
        except:
            return 0.0
    
    def _map_baseline_to_full_curve(self, curve, peak_x_data, peak_baseline_y, boundaries):
        """将峰区域的基线数据映射到完整曲线坐标系"""
        # 创建与完整曲线长度相同的基线数组
        full_baseline = np.zeros_like(curve.y_values)
        
        try:
            # 找到峰区域在完整曲线中的索引范围
            start_rt, end_rt = boundaries
            
            # 在完整曲线中找到对应的索引
            start_idx = np.argmin(np.abs(curve.x_values - start_rt))
            end_idx = np.argmin(np.abs(curve.x_values - end_rt))
            
            # 确保索引有效
            start_idx = max(0, min(start_idx, len(curve.x_values) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(curve.x_values) - 1))
            
            # 将峰区域的基线插值到完整曲线的对应区域
            if len(peak_x_data) > 1 and len(peak_baseline_y) > 1:
                # 使用插值将峰基线映射到完整曲线的x坐标
                full_x_region = curve.x_values[start_idx:end_idx+1]
                interpolated_baseline = np.interp(
                    full_x_region, 
                    peak_x_data, 
                    peak_baseline_y
                )
                full_baseline[start_idx:end_idx+1] = interpolated_baseline
            else:
                # 如果数据不足，使用简单的线性基线
                baseline_value = np.mean(peak_baseline_y) if len(peak_baseline_y) > 0 else 0
                full_baseline[start_idx:end_idx+1] = baseline_value
                
        except Exception as e:
            print(f"Error mapping baseline to full curve: {e}")
            # 使用零基线作为后备
            pass
        
        return full_baseline
