"""
峰分析器 - 重构版本，清晰的数据流和逻辑
"""

import numpy as np
from scipy import optimize, ndimage
from typing import List, Dict, Any, Optional, Tuple
import copy

from core.curve import Curve, Peak


class PeakAnalyzer:
    """峰分析器 - 提供峰的详细分析功能"""
    
    def __init__(self):
        pass
    
    def analyze_peak(self, curve: Curve, peak: Peak, 
                    extend_range: float = 2.0, 
                    baseline_method: str = '线性基线',
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
        
        清晰的数据流：
        1. 获取平滑参数 → 应用到整个曲线
        2. 从平滑曲线提取峰区域
        3. 边界检测（平滑数据） + 积分（原始数据） + FWHM（原始数据）
        4. 计算色谱质量参数
        """
        updated_peak = copy.deepcopy(peak)
        
        try:
            # === 第1步：获取和应用平滑参数 ===
            smoothed_curve_y = self._get_smoothed_curve_data(curve, boundary_smoothing)
            
            # === 第2步：从平滑曲线提取峰区域 ===
            region_data = self._extract_peak_region_clean(curve, peak, extend_range, smoothed_curve_y)
            if not region_data:
                return updated_peak
            
            x_data, y_original, y_smoothed, peak_idx = region_data
            
            # === 第3步：峰分析计算 ===
            # 边界检测（使用平滑数据）
            boundaries = self._detect_boundaries_clean(
                x_data, y_smoothed, peak_idx, boundary_method, 
                peak_sensitivity, noise_tolerance
            )
            
            # 峰积分（使用原始数据，基线校正）
            area = self._calculate_peak_area_clean(
                x_data, y_original, boundaries, baseline_method
            )
            
            # FWHM计算（使用原始数据，数值精确）
            fwhm = self._calculate_fwhm_clean(x_data, y_original, peak_idx)
            
            # === 第4步：更新峰参数 ===
            updated_peak.area = area
            updated_peak.fwhm = fwhm
            updated_peak.rt_start = boundaries[0]
            updated_peak.rt_end = boundaries[1]
            
            # === 第5步：计算色谱质量参数 ===
            self._calculate_chromatographic_parameters(
                updated_peak, x_data, y_original, peak_idx, curve,
                calc_theoretical_plates, calc_tailing_factor, calc_asymmetry_factor,
                calc_resolution, calc_capacity_factor, calc_selectivity
            )
            
            # === 第6步：保存可视化数据 ===
            self._save_visualization_data(
                updated_peak, curve, x_data, y_original, boundaries, 
                baseline_method, area, fwhm
            )
            
        except Exception as e:
            print(f"❌ 峰分析失败: {e}")
            # 返回基本更新的峰
            updated_peak.area = getattr(peak, 'area', 0.0)
            updated_peak.fwhm = max(getattr(peak, 'fwhm', 0.1), 0.001)
        
        return updated_peak
    
    # ===== 第1步：平滑数据获取 =====
    def _get_smoothed_curve_data(self, curve: Curve, boundary_smoothing: bool) -> np.ndarray:
        """获取平滑后的曲线数据"""
        if not boundary_smoothing:
            return curve.y_values.copy()
        
        # 优先级1：session状态中的平滑参数
        try:
            import streamlit as st
            working_key = f"working_curve_{curve.curve_id}"
            if working_key in st.session_state:
                working_data = st.session_state[working_key]
                if "smoothing_method" in working_data and "smoothing_params" in working_data:
                    from ui.pages.processing.smoothing import SmoothingProcessor
                    processor = SmoothingProcessor()
                    smoothed_y = processor.methods[working_data["smoothing_method"]](curve.y_values, working_data["smoothing_params"])
                    print(f"🔧 使用session平滑参数: {working_data['smoothing_method']}")
                    return smoothed_y
        except Exception as e:
            print(f"⚠️ session平滑参数获取失败: {e}")
        
        # 优先级2：curve对象中的平滑参数
        try:
            if hasattr(curve, 'smoothing_method') and hasattr(curve, 'smoothing_params'):
                from ui.pages.processing.smoothing import SmoothingProcessor
                processor = SmoothingProcessor()
                smoothed_y = processor.methods[curve.smoothing_method](curve.y_values, curve.smoothing_params)
                print(f"🔧 使用curve平滑参数: {curve.smoothing_method}")
                return smoothed_y
        except Exception as e:
            print(f"⚠️ curve平滑参数获取失败: {e}")
        
        # 优先级3：默认轻度平滑
        try:
            smoothed_y = ndimage.uniform_filter1d(curve.y_values, size=3)
            print(f"🔧 使用默认平滑（移动平均，窗口=3）")
            return smoothed_y
        except:
            print(f"🔧 平滑失败，使用原始数据")
            return curve.y_values.copy()
    
    # ===== 第2步：峰区域提取 =====
    def _extract_peak_region_clean(self, curve: Curve, peak: Peak, extend_range: float, 
                                 smoothed_curve_y: np.ndarray) -> Optional[Tuple]:
        """从平滑曲线中提取峰区域"""
        try:
            # 计算扩展范围
            if hasattr(peak, 'fwhm') and peak.fwhm > 0:
                extend_width = peak.fwhm * extend_range
            else:
                extend_width = (curve.x_values[-1] - curve.x_values[0]) * 0.05
            
            # 确定提取范围
            start_rt = peak.rt - extend_width
            end_rt = peak.rt + extend_width
            
            # 提取数据
            mask = (curve.x_values >= start_rt) & (curve.x_values <= end_rt)
            x_data = curve.x_values[mask]
            y_original = curve.y_values[mask]  # 原始数据
            y_smoothed = smoothed_curve_y[mask]  # 平滑数据
            
            if len(x_data) < 5:
                return None
            
            # 找到峰在区域中的索引
            peak_idx = np.argmin(np.abs(x_data - peak.rt))
            
            return x_data, y_original, y_smoothed, peak_idx
            
        except Exception as e:
            print(f"❌ 峰区域提取失败: {e}")
            return None
    
    # ===== 第3步：峰分析计算 =====
    def _detect_boundaries_clean(self, x_data: np.ndarray, y_smoothed: np.ndarray, 
                               peak_idx: int, boundary_method: str, 
                               sensitivity: int, noise_tolerance: int) -> Tuple[float, float]:
        """清晰的边界检测逻辑"""
        try:
            if boundary_method == "切线撇取法 (Tangent Skim)":
                return self._tangent_skim_method(x_data, y_smoothed, peak_idx, noise_tolerance)
            elif boundary_method == "指数撇取法 (Exponential Skim)":
                return self._exponential_skim_method(x_data, y_smoothed, peak_idx, noise_tolerance)
            elif boundary_method == "谷到谷法 (Valley-to-Valley)":
                return self._valley_to_valley_method(x_data, y_smoothed, peak_idx, noise_tolerance)
            elif boundary_method == "垂直分割法 (Perpendicular Drop)":
                return self._perpendicular_drop_method(x_data, y_smoothed, peak_idx, noise_tolerance)
            else:
                # 自动选择
                return self._auto_select_boundary_method(x_data, y_smoothed, peak_idx, sensitivity, noise_tolerance)
        except Exception as e:
            print(f"❌ 边界检测失败: {e}")
            # 备用方法：使用5%峰高
            return self._fallback_boundary_method(x_data, y_smoothed, peak_idx)
    
    def _calculate_peak_area_clean(self, x_data: np.ndarray, y_original: np.ndarray, 
                                 boundaries: Tuple[float, float], baseline_method: str) -> float:
        """清晰的峰面积计算"""
        try:
            # 基线校正（只影响积分）
            baseline_y = self._calculate_baseline_clean(x_data, y_original, baseline_method)
            corrected_y = y_original - baseline_y
            
            # 确定积分范围
            left_rt, right_rt = boundaries
            left_idx = np.argmin(np.abs(x_data - left_rt))
            right_idx = np.argmin(np.abs(x_data - right_rt))
            
            if left_idx >= right_idx:
                return 0.0
            
            # 梯形积分
            area = np.trapz(
                np.maximum(corrected_y[left_idx:right_idx+1], 0),
                x_data[left_idx:right_idx+1]
            )
            
            return float(max(area, 0.0))
            
        except Exception as e:
            print(f"❌ 面积计算失败: {e}")
            return 0.0
    
    def _calculate_fwhm_clean(self, x_data: np.ndarray, y_original: np.ndarray, peak_idx: int) -> float:
        """清晰的FWHM计算 - 数值精确方法"""
        try:
            peak_height = y_original[peak_idx]
            
            # 估算基线
            edge_size = max(2, len(y_original) // 10)
            baseline = (np.min(y_original[:edge_size]) + np.min(y_original[-edge_size:])) / 2
            
            # 真实峰高和半高
            true_height = peak_height - baseline
            half_height = baseline + true_height / 2
            
            # 数值方法找到精确交点
            left_rt = self._find_intersection_numerical(x_data, y_original, half_height, peak_idx, 'left')
            right_rt = self._find_intersection_numerical(x_data, y_original, half_height, peak_idx, 'right')
            
            if left_rt is not None and right_rt is not None:
                fwhm = right_rt - left_rt
                print(f"✅ FWHM计算: {fwhm:.4f} (左={left_rt:.4f}, 右={right_rt:.4f})")
                return float(max(fwhm, 0.001))
            
            # 备用方法
            return self._estimate_fwhm_backup(x_data, y_original, peak_idx)
            
        except Exception as e:
            print(f"❌ FWHM计算失败: {e}")
            return 0.001
    
    # ===== 辅助方法 =====
    def _calculate_baseline_clean(self, x_data: np.ndarray, y_data: np.ndarray, method: str) -> np.ndarray:
        """基线计算"""
        if method == "线性基线":
            return np.linspace(y_data[0], y_data[-1], len(y_data))
        elif method == "多项式基线":
            try:
                coeffs = np.polyfit(x_data, y_data, 2)
                return np.polyval(coeffs, x_data)
            except:
                return np.linspace(y_data[0], y_data[-1], len(y_data))
        elif method == "指数基线":
            try:
                from scipy.optimize import curve_fit
                def exp_func(x, a, b, c):
                    return a * np.exp(b * x) + c
                popt, _ = curve_fit(exp_func, x_data, y_data, 
                                  p0=[y_data[0], 0.01, np.min(y_data)], maxfev=1000)
                return exp_func(x_data, *popt)
            except:
                return np.linspace(y_data[0], y_data[-1], len(y_data))
        else:
            return np.linspace(y_data[0], y_data[-1], len(y_data))
    
    def _find_intersection_numerical(self, x_data: np.ndarray, y_data: np.ndarray, 
                                   target_height: float, peak_idx: int, direction: str) -> Optional[float]:
        """数值方法找到精确交点"""
        try:
            if direction == 'left':
                search_indices = range(peak_idx, -1, -1)
            else:
                search_indices = range(peak_idx, len(y_data))
            
            for i in search_indices[:-1]:
                next_i = i - 1 if direction == 'left' else i + 1
                
                if next_i < 0 or next_i >= len(y_data):
                    continue
                
                y1, y2 = y_data[i], y_data[next_i]
                
                # 检查是否跨越目标高度
                if (y1 >= target_height >= y2) or (y1 <= target_height <= y2):
                    # 线性插值
                    if abs(y2 - y1) > 1e-10:
                        t = (target_height - y1) / (y2 - y1)
                        intersection = x_data[i] + t * (x_data[next_i] - x_data[i])
                        
                        # 确保在有效范围内
                        if x_data[0] <= intersection <= x_data[-1]:
                            return float(intersection)
            
            # 未找到交点，返回边界
            return float(x_data[0] if direction == 'left' else x_data[-1])
            
        except Exception as e:
            print(f"❌ 交点计算错误: {e}")
            return None
    
    def _auto_select_boundary_method(self, x_data: np.ndarray, y_data: np.ndarray, 
                                   peak_idx: int, sensitivity: int, noise_tolerance: int) -> Tuple[float, float]:
        """根据灵敏度自动选择边界检测方法"""
        if sensitivity <= 3:
            return self._tangent_skim_method(x_data, y_data, peak_idx, noise_tolerance)
        elif sensitivity <= 6:
            return self._exponential_skim_method(x_data, y_data, peak_idx, noise_tolerance)
        elif sensitivity <= 8:
            return self._valley_to_valley_method(x_data, y_data, peak_idx, noise_tolerance)
        else:
            return self._perpendicular_drop_method(x_data, y_data, peak_idx, noise_tolerance)
    
    def _tangent_skim_method(self, x_data: np.ndarray, y_data: np.ndarray, 
                           peak_idx: int, noise_tolerance: int) -> Tuple[float, float]:
        """切线撇取法"""
        try:
            # 简化实现：找到斜率变化最大的点
            gradient = np.gradient(y_data)
            
            # 左侧最大正斜率点
            left_idx = peak_idx
            max_slope = 0
            for i in range(peak_idx, max(0, peak_idx - len(y_data)//3), -1):
                if gradient[i] > max_slope:
                    max_slope = gradient[i]
                    left_idx = i
            
            # 右侧最大负斜率点
            right_idx = peak_idx
            min_slope = 0
            for i in range(peak_idx, min(len(y_data), peak_idx + len(y_data)//3)):
                if gradient[i] < min_slope:
                    min_slope = gradient[i]
                    right_idx = i
            
            return float(x_data[left_idx]), float(x_data[right_idx])
        except:
            return self._fallback_boundary_method(x_data, y_data, peak_idx)
    
    def _exponential_skim_method(self, x_data: np.ndarray, y_data: np.ndarray, 
                               peak_idx: int, noise_tolerance: int) -> Tuple[float, float]:
        """指数撇取法"""
        try:
            baseline = (y_data[0] + y_data[-1]) / 2
            threshold = baseline + (y_data[peak_idx] - baseline) * 0.1
            
            # 找到阈值交点
            left_idx = self._find_threshold_crossing_simple(y_data, peak_idx, threshold, 'left')
            right_idx = self._find_threshold_crossing_simple(y_data, peak_idx, threshold, 'right')
            
            return float(x_data[left_idx]), float(x_data[right_idx])
        except:
            return self._fallback_boundary_method(x_data, y_data, peak_idx)
    
    def _valley_to_valley_method(self, x_data: np.ndarray, y_data: np.ndarray, 
                               peak_idx: int, noise_tolerance: int) -> Tuple[float, float]:
        """谷到谷法"""
        try:
            # 寻找左谷
            left_valley_idx = 0
            min_left = float('inf')
            for i in range(peak_idx, max(0, peak_idx - len(y_data)//3), -1):
                if y_data[i] < min_left:
                    min_left = y_data[i]
                    left_valley_idx = i
            
            # 寻找右谷
            right_valley_idx = len(y_data) - 1
            min_right = float('inf')
            for i in range(peak_idx, min(len(y_data), peak_idx + len(y_data)//3)):
                if y_data[i] < min_right:
                    min_right = y_data[i]
                    right_valley_idx = i
            
            return float(x_data[left_valley_idx]), float(x_data[right_valley_idx])
        except:
            return self._fallback_boundary_method(x_data, y_data, peak_idx)
    
    def _perpendicular_drop_method(self, x_data: np.ndarray, y_data: np.ndarray, 
                                 peak_idx: int, noise_tolerance: int) -> Tuple[float, float]:
        """垂直分割法"""
        try:
            peak_height = y_data[peak_idx]
            baseline = (y_data[0] + y_data[-1]) / 2
            threshold_ratio = 0.01 + (10 - noise_tolerance) * 0.01
            threshold = baseline + (peak_height - baseline) * threshold_ratio
            
            left_idx = self._find_threshold_crossing_simple(y_data, peak_idx, threshold, 'left')
            right_idx = self._find_threshold_crossing_simple(y_data, peak_idx, threshold, 'right')
            
            return float(x_data[left_idx]), float(x_data[right_idx])
        except:
            return self._fallback_boundary_method(x_data, y_data, peak_idx)
    
    def _find_threshold_crossing_simple(self, y_data: np.ndarray, peak_idx: int, 
                                      threshold: float, direction: str) -> int:
        """简单的阈值交叉查找"""
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
    
    def _fallback_boundary_method(self, x_data: np.ndarray, y_data: np.ndarray, peak_idx: int) -> Tuple[float, float]:
        """备用边界检测"""
        try:
            # 使用5%峰高作为简单阈值
            peak_height = y_data[peak_idx]
            baseline = (y_data[0] + y_data[-1]) / 2
            threshold = baseline + (peak_height - baseline) * 0.05
            
            left_idx = self._find_threshold_crossing_simple(y_data, peak_idx, threshold, 'left')
            right_idx = self._find_threshold_crossing_simple(y_data, peak_idx, threshold, 'right')
            
            return float(x_data[left_idx]), float(x_data[right_idx])
        except:
            # 最后备用：使用峰周围10%范围
            width = len(x_data) // 10
            left_idx = max(0, peak_idx - width)
            right_idx = min(len(x_data) - 1, peak_idx + width)
            return float(x_data[left_idx]), float(x_data[right_idx])
    
    def _estimate_fwhm_backup(self, x_data: np.ndarray, y_data: np.ndarray, peak_idx: int) -> float:
        """FWHM备用估算"""
        try:
            # 使用90%高度宽度估算
            peak_height = y_data[peak_idx]
            threshold = peak_height * 0.9
            
            left_idx = self._find_threshold_crossing_simple(y_data, peak_idx, threshold, 'left')
            right_idx = self._find_threshold_crossing_simple(y_data, peak_idx, threshold, 'right')
            
            width_90 = x_data[right_idx] - x_data[left_idx]
            return float(max(width_90 * 0.6, 0.001))  # FWHM ≈ 90%宽度 × 0.6
        except:
            return 0.001
    
    # ===== 第5步：色谱质量参数计算 =====
    def _calculate_chromatographic_parameters(self, peak: Peak, x_data: np.ndarray, y_data: np.ndarray, 
                                            peak_idx: int, curve: Curve, *calc_flags):
        """计算色谱质量参数"""
        (calc_theoretical_plates, calc_tailing_factor, calc_asymmetry_factor,
         calc_resolution, calc_capacity_factor, calc_selectivity) = calc_flags
        
        try:
            if calc_theoretical_plates:
                peak.theoretical_plates = self._calc_theoretical_plates(peak.rt, peak.fwhm)
        except: peak.theoretical_plates = 0.0
        
        try:
            if calc_tailing_factor:
                peak.tailing_factor = self._calc_tailing_factor(x_data, y_data, peak_idx)
        except: peak.tailing_factor = 1.0
        
        try:
            if calc_asymmetry_factor:
                peak.asymmetry_factor = self._calc_asymmetry_factor(x_data, y_data, peak_idx)
        except: peak.asymmetry_factor = 1.0
        
        try:
            if calc_resolution:
                peak.resolution = self._calc_resolution(curve, peak)
        except: peak.resolution = 0.0
        
        try:
            if calc_capacity_factor:
                peak.capacity_factor = self._calc_capacity_factor(peak.rt, curve)
        except: peak.capacity_factor = 0.0
        
        try:
            if calc_selectivity:
                peak.selectivity = self._calc_selectivity_factor(curve, peak)
        except: peak.selectivity = 1.0
    
    def _calc_theoretical_plates(self, rt: float, fwhm: float) -> float:
        """理论塔板数 N = 5.54 * (tR / W1/2)²"""
        return 5.54 * (rt / fwhm) ** 2 if fwhm > 0 else 0.0
    
    def _calc_tailing_factor(self, x_data: np.ndarray, y_data: np.ndarray, peak_idx: int) -> float:
        """USP拖尾因子"""
        try:
            peak_height = y_data[peak_idx]
            h_5 = peak_height * 0.05
            
            left_idx = self._find_threshold_crossing_simple(y_data, peak_idx, h_5, 'left')
            right_idx = self._find_threshold_crossing_simple(y_data, peak_idx, h_5, 'right')
            
            w = x_data[right_idx] - x_data[left_idx]
            f = x_data[peak_idx] - x_data[left_idx]
            
            return w / (2 * f) if f > 0 else 1.0
        except:
            return 1.0
    
    def _calc_asymmetry_factor(self, x_data: np.ndarray, y_data: np.ndarray, peak_idx: int) -> float:
        """USP不对称因子"""
        try:
            peak_height = y_data[peak_idx]
            h_10 = peak_height * 0.1
            
            left_idx = self._find_threshold_crossing_simple(y_data, peak_idx, h_10, 'left')
            right_idx = self._find_threshold_crossing_simple(y_data, peak_idx, h_10, 'right')
            
            a = x_data[peak_idx] - x_data[left_idx]
            b = x_data[right_idx] - x_data[peak_idx]
            
            return b / a if a > 0 else 1.0
        except:
            return 1.0
    
    def _calc_resolution(self, curve: Curve, current_peak: Peak) -> float:
        """USP分离度"""
        try:
            if len(curve.peaks) < 2:
                return 0.0
            
            # 找到最近的峰
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
            
            return 2 * rt_diff / width_sum if width_sum > 0 else 0.0
        except:
            return 0.0
    
    def _calc_capacity_factor(self, rt: float, curve: Curve) -> float:
        """容量因子"""
        try:
            t0 = min(peak.rt for peak in curve.peaks) if curve.peaks else 1.0
            t0 = max(t0, 0.1)  # 避免过小的t0
            return (rt - t0) / t0
        except:
            return 0.0
    
    def _calc_selectivity_factor(self, curve: Curve, current_peak: Peak) -> float:
        """选择性因子"""
        try:
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
            
            k1 = self._calc_capacity_factor(current_peak.rt, curve)
            k2 = self._calc_capacity_factor(adjacent_peak.rt, curve)
            
            return max(k2, k1) / min(k2, k1) if min(k2, k1) > 0 else 1.0
        except:
            return 1.0
    
    # ===== 第6步：可视化数据保存 =====
    def _save_visualization_data(self, peak: Peak, curve: Curve, x_data: np.ndarray, 
                               y_data: np.ndarray, boundaries: Tuple[float, float],
                               baseline_method: str, area: float, fwhm: float):
        """保存可视化数据"""
        try:
            # 计算基线用于可视化
            baseline_y = self._calculate_baseline_clean(x_data, y_data, baseline_method)
            corrected_y = y_data - baseline_y
            
            # 映射到完整曲线
            full_baseline = self._map_baseline_to_full_curve(curve, x_data, baseline_y, boundaries)
            
            peak.metadata['visualization_data'] = {
                'peak_region_x': x_data.tolist(),
                'peak_region_y': y_data.tolist(),
                'peak_region_baseline': baseline_y.tolist(),
                'full_curve_baseline': full_baseline.tolist(),
                'corrected_y': corrected_y.tolist(),
                'boundaries': boundaries,
                'integration_method': 'simple_trapezoid',
                'baseline_method': baseline_method
            }
        except Exception as e:
            print(f"❌ 可视化数据保存失败: {e}")
            # 提供默认数据
            peak.metadata['visualization_data'] = {
                'peak_region_x': [peak.rt - 0.1, peak.rt, peak.rt + 0.1],
                'peak_region_y': [peak.intensity * 0.1, peak.intensity, peak.intensity * 0.1],
                'peak_region_baseline': [0, 0, 0],
                'full_curve_baseline': [0] * len(curve.x_values),
                'corrected_y': [peak.intensity * 0.1, peak.intensity, peak.intensity * 0.1],
                'boundaries': (peak.rt - 0.1, peak.rt + 0.1),
                'integration_method': 'simple_trapezoid',
                'baseline_method': baseline_method
            }
    
    def _map_baseline_to_full_curve(self, curve: Curve, peak_x_data: np.ndarray, 
                                  peak_baseline_y: np.ndarray, boundaries: Tuple[float, float]) -> np.ndarray:
        """映射基线到完整曲线"""
        full_baseline = np.zeros_like(curve.y_values)
        
        try:
            start_rt, end_rt = boundaries
            start_idx = np.argmin(np.abs(curve.x_values - start_rt))
            end_idx = np.argmin(np.abs(curve.x_values - end_rt))
            
            start_idx = max(0, min(start_idx, len(curve.x_values) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(curve.x_values) - 1))
            
            if len(peak_x_data) > 1 and len(peak_baseline_y) > 1:
                full_x_region = curve.x_values[start_idx:end_idx+1]
                interpolated_baseline = np.interp(full_x_region, peak_x_data, peak_baseline_y)
                full_baseline[start_idx:end_idx+1] = interpolated_baseline
        except:
            pass
        
        return full_baseline
