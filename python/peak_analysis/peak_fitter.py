"""
峰拟合器 - 使用数学模型拟合峰形
"""

import numpy as np
from scipy import optimize
from typing import List, Dict, Any, Optional, Tuple, Callable
import warnings

from core.curve import Curve, Peak


class PeakFitter:
    """峰拟合器 - 提供多种峰形拟合功能"""
    
    def __init__(self):
        self.models = {
            'gaussian': self._gaussian_model,
            'lorentzian': self._lorentzian_model,
            'voigt': self._voigt_model,
            'exponential_gaussian': self._exp_gaussian_model,
            'skewed_gaussian': self._skewed_gaussian_model
        }
    
    def fit_peak(self, curve: Curve, peak: Peak, 
                model: str = 'gaussian',
                extend_range: float = 3.0) -> Dict[str, Any]:
        """
        拟合单个峰
        
        参数:
        - curve: 所属曲线
        - peak: 要拟合的峰
        - model: 拟合模型类型
        - extend_range: 扩展拟合范围的倍数
        
        返回:
        - 拟合结果字典
        """
        if model not in self.models:
            raise ValueError(f"未知的拟合模型: {model}")
        
        # 提取峰数据
        peak_data = self._extract_peak_data(curve, peak, extend_range)
        if not peak_data:
            return {'success': False, 'error': 'Cannot extract peak data'}
        
        x_data, y_data = peak_data
        
        # 执行拟合
        try:
            result = self._fit_model(x_data, y_data, model, peak)
            result['model'] = model
            result['success'] = True
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def fit_multiple_peaks(self, curve: Curve, peaks: List[Peak],
                          model: str = 'gaussian') -> List[Dict[str, Any]]:
        """拟合多个峰"""
        results = []
        
        for peak in peaks:
            result = self.fit_peak(curve, peak, model)
            results.append(result)
        
        return results
    
    def deconvolve_overlapping_peaks(self, curve: Curve, peak_positions: List[float],
                                   model: str = 'gaussian',
                                   rt_window: float = 2.0) -> Dict[str, Any]:
        """
        反卷积重叠峰
        
        参数:
        - curve: 曲线数据
        - peak_positions: 峰位置列表
        - model: 拟合模型
        - rt_window: RT窗口范围
        
        返回:
        - 反卷积结果
        """
        if not peak_positions:
            return {'success': False, 'error': 'No peak positions provided'}
        
        # 确定拟合范围
        rt_min = min(peak_positions) - rt_window
        rt_max = max(peak_positions) + rt_window
        
        # 提取数据
        mask = (curve.x_values >= rt_min) & (curve.x_values <= rt_max)
        x_data = curve.x_values[mask]
        y_data = curve.y_values[mask]
        
        if len(x_data) < 10:
            return {'success': False, 'error': 'Insufficient data points'}
        
        try:
            # 多峰拟合
            result = self._fit_multiple_peaks_model(x_data, y_data, peak_positions, model)
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_peak_data(self, curve: Curve, peak: Peak, 
                          extend_range: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """提取峰周围的数据"""
        # 计算扩展范围
        extend_width = max(peak.fwhm * extend_range, 0.1)  # 最小0.1分钟
        start_rt = peak.rt - extend_width
        end_rt = peak.rt + extend_width
        
        # 提取数据
        mask = (curve.x_values >= start_rt) & (curve.x_values <= end_rt)
        x_data = curve.x_values[mask]
        y_data = curve.y_values[mask]
        
        if len(x_data) < 5:  # 至少需要5个数据点
            return None
        
        return x_data, y_data
    
    def _fit_model(self, x_data: np.ndarray, y_data: np.ndarray, 
                  model: str, peak: Peak) -> Dict[str, Any]:
        """拟合单峰模型"""
        model_func = self.models[model]
        
        # 初始参数估计
        initial_params = self._estimate_initial_params(x_data, y_data, model, peak)
        
        # 参数边界
        bounds = self._get_parameter_bounds(x_data, y_data, model, initial_params)
        
        # 执行拟合
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                popt, pcov = optimize.curve_fit(
                    model_func, x_data, y_data,
                    p0=initial_params,
                    bounds=bounds,
                    maxfev=5000
                )
            except Exception as e:
                # 如果拟合失败，尝试更宽松的边界
                try:
                    popt, pcov = optimize.curve_fit(
                        model_func, x_data, y_data,
                        p0=initial_params,
                        maxfev=10000
                    )
                except:
                    raise e
        
        # 计算拟合质量
        y_fitted = model_func(x_data, *popt)
        r_squared = self._calculate_r_squared(y_data, y_fitted)
        rmse = np.sqrt(np.mean((y_data - y_fitted) ** 2))
        
        # 参数标准误差
        param_errors = np.sqrt(np.diag(pcov)) if pcov.size > 0 else np.zeros_like(popt)
        
        # 组织结果
        result = {
            'parameters': popt.tolist(),
            'parameter_errors': param_errors.tolist(),
            'parameter_names': self._get_parameter_names(model),
            'fitted_curve': {
                'x': x_data.tolist(),
                'y': y_fitted.tolist()
            },
            'r_squared': float(r_squared),
            'rmse': float(rmse),
            'residuals': (y_data - y_fitted).tolist()
        }
        
        # 添加模型特定的解释参数
        result.update(self._interpret_parameters(model, popt))
        
        return result
    
    def _fit_multiple_peaks_model(self, x_data: np.ndarray, y_data: np.ndarray,
                                 peak_positions: List[float], model: str) -> Dict[str, Any]:
        """拟合多峰模型"""
        n_peaks = len(peak_positions)
        model_func = self.models[model]
        
        # 为多峰创建组合函数
        def multi_peak_func(x, *params):
            """多峰组合函数"""
            n_params_per_peak = len(self._get_parameter_names(model))
            y_total = np.zeros_like(x)
            
            for i in range(n_peaks):
                start_idx = i * n_params_per_peak
                end_idx = start_idx + n_params_per_peak
                peak_params = params[start_idx:end_idx]
                y_total += model_func(x, *peak_params)
            
            return y_total
        
        # 初始参数估计
        initial_params = []
        bounds_lower = []
        bounds_upper = []
        
        for pos in peak_positions:
            # 为每个峰估计初始参数
            peak_height = np.interp(pos, x_data, y_data)
            
            if model == 'gaussian':
                # [amplitude, center, sigma]
                sigma = 0.1  # 初始宽度估计
                initial_params.extend([peak_height, pos, sigma])
                bounds_lower.extend([0, pos - 0.5, 0.01])
                bounds_upper.extend([peak_height * 2, pos + 0.5, 1.0])
            elif model == 'lorentzian':
                # [amplitude, center, gamma]
                gamma = 0.1
                initial_params.extend([peak_height, pos, gamma])
                bounds_lower.extend([0, pos - 0.5, 0.01])
                bounds_upper.extend([peak_height * 2, pos + 0.5, 1.0])
            # 可以添加更多模型...
        
        # 执行拟合
        try:
            popt, pcov = optimize.curve_fit(
                multi_peak_func, x_data, y_data,
                p0=initial_params,
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000
            )
        except Exception as e:
            return {'success': False, 'error': f'Fitting failed: {str(e)}'}
        
        # 计算拟合质量
        y_fitted = multi_peak_func(x_data, *popt)
        r_squared = self._calculate_r_squared(y_data, y_fitted)
        rmse = np.sqrt(np.mean((y_data - y_fitted) ** 2))
        
        # 分解参数到各个峰
        n_params_per_peak = len(self._get_parameter_names(model))
        individual_peaks = []
        
        for i in range(n_peaks):
            start_idx = i * n_params_per_peak
            end_idx = start_idx + n_params_per_peak
            peak_params = popt[start_idx:end_idx]
            
            # 计算单个峰的拟合曲线
            y_individual = model_func(x_data, *peak_params)
            
            peak_result = {
                'peak_index': i,
                'parameters': peak_params.tolist(),
                'parameter_names': self._get_parameter_names(model),
                'fitted_curve': {
                    'x': x_data.tolist(),
                    'y': y_individual.tolist()
                }
            }
            
            # 添加解释参数
            peak_result.update(self._interpret_parameters(model, peak_params))
            
            individual_peaks.append(peak_result)
        
        return {
            'success': True,
            'n_peaks': n_peaks,
            'overall_fit': {
                'x': x_data.tolist(),
                'y': y_fitted.tolist(),
                'r_squared': float(r_squared),
                'rmse': float(rmse)
            },
            'individual_peaks': individual_peaks,
            'residuals': (y_data - y_fitted).tolist()
        }
    
    def _gaussian_model(self, x, amplitude, center, sigma):
        """高斯模型"""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    def _lorentzian_model(self, x, amplitude, center, gamma):
        """洛伦兹模型"""
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2)
    
    def _voigt_model(self, x, amplitude, center, sigma, gamma):
        """Voigt模型（高斯和洛伦兹的卷积）"""
        # 简化的Voigt近似
        gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        lorentzian = gamma**2 / ((x - center)**2 + gamma**2)
        return amplitude * (0.5 * gaussian + 0.5 * lorentzian)
    
    def _exp_gaussian_model(self, x, amplitude, center, sigma, tau):
        """指数修饰高斯模型"""
        gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        exponential = np.exp(-(x - center) / tau)
        return amplitude * gaussian * exponential
    
    def _skewed_gaussian_model(self, x, amplitude, center, sigma, alpha):
        """偏斜高斯模型"""
        from scipy.stats import skewnorm
        return amplitude * skewnorm.pdf(x, alpha, loc=center, scale=sigma)
    
    def _estimate_initial_params(self, x_data: np.ndarray, y_data: np.ndarray,
                               model: str, peak: Peak) -> List[float]:
        """估计初始参数"""
        max_intensity = np.max(y_data)
        max_position = x_data[np.argmax(y_data)]
        
        if model == 'gaussian':
            # [amplitude, center, sigma]
            sigma = peak.fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma
            return [max_intensity, max_position, max(sigma, 0.01)]
        
        elif model == 'lorentzian':
            # [amplitude, center, gamma]
            gamma = peak.fwhm / 2  # FWHM to gamma
            return [max_intensity, max_position, max(gamma, 0.01)]
        
        elif model == 'voigt':
            # [amplitude, center, sigma, gamma]
            sigma = peak.fwhm / (2 * np.sqrt(2 * np.log(2))) * 0.5
            gamma = peak.fwhm / 2 * 0.5
            return [max_intensity, max_position, max(sigma, 0.01), max(gamma, 0.01)]
        
        elif model == 'exponential_gaussian':
            # [amplitude, center, sigma, tau]
            sigma = peak.fwhm / (2 * np.sqrt(2 * np.log(2)))
            tau = peak.fwhm  # 简单估计
            return [max_intensity, max_position, max(sigma, 0.01), max(tau, 0.01)]
        
        elif model == 'skewed_gaussian':
            # [amplitude, center, sigma, alpha]
            sigma = peak.fwhm / (2 * np.sqrt(2 * np.log(2)))
            alpha = 0.0  # 开始时假设无偏斜
            return [max_intensity, max_position, max(sigma, 0.01), alpha]
        
        else:
            return [max_intensity, max_position, 0.1]
    
    def _get_parameter_bounds(self, x_data: np.ndarray, y_data: np.ndarray,
                            model: str, initial_params: List[float]) -> Tuple[List[float], List[float]]:
        """获取参数边界"""
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        if model == 'gaussian':
            lower = [0, x_min, 0.001]
            upper = [y_max * 2, x_max, (x_max - x_min)]
        elif model == 'lorentzian':
            lower = [0, x_min, 0.001]
            upper = [y_max * 2, x_max, (x_max - x_min)]
        elif model == 'voigt':
            lower = [0, x_min, 0.001, 0.001]
            upper = [y_max * 2, x_max, (x_max - x_min), (x_max - x_min)]
        elif model == 'exponential_gaussian':
            lower = [0, x_min, 0.001, 0.001]
            upper = [y_max * 2, x_max, (x_max - x_min), (x_max - x_min) * 10]
        elif model == 'skewed_gaussian':
            lower = [0, x_min, 0.001, -10]
            upper = [y_max * 2, x_max, (x_max - x_min), 10]
        else:
            lower = [-np.inf] * len(initial_params)
            upper = [np.inf] * len(initial_params)
        
        return lower, upper
    
    def _get_parameter_names(self, model: str) -> List[str]:
        """获取参数名称"""
        if model == 'gaussian':
            return ['amplitude', 'center', 'sigma']
        elif model == 'lorentzian':
            return ['amplitude', 'center', 'gamma']
        elif model == 'voigt':
            return ['amplitude', 'center', 'sigma', 'gamma']
        elif model == 'exponential_gaussian':
            return ['amplitude', 'center', 'sigma', 'tau']
        elif model == 'skewed_gaussian':
            return ['amplitude', 'center', 'sigma', 'alpha']
        else:
            return ['param_' + str(i) for i in range(3)]
    
    def _interpret_parameters(self, model: str, params: np.ndarray) -> Dict[str, Any]:
        """解释拟合参数"""
        result = {}
        
        if model == 'gaussian':
            amplitude, center, sigma = params
            result.update({
                'peak_center': float(center),
                'peak_height': float(amplitude),
                'fwhm_fitted': float(2 * sigma * np.sqrt(2 * np.log(2))),
                'area_fitted': float(amplitude * sigma * np.sqrt(2 * np.pi))
            })
        
        elif model == 'lorentzian':
            amplitude, center, gamma = params
            result.update({
                'peak_center': float(center),
                'peak_height': float(amplitude),
                'fwhm_fitted': float(2 * gamma),
                'area_fitted': float(np.pi * amplitude * gamma)
            })
        
        # 可以为其他模型添加更多解释...
        
        return result
    
    def _calculate_r_squared(self, y_observed: np.ndarray, y_fitted: np.ndarray) -> float:
        """计算R平方值"""
        ss_res = np.sum((y_observed - y_fitted) ** 2)
        ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, r_squared)  # 确保非负
    
    def get_available_models(self) -> List[str]:
        """获取可用的拟合模型"""
        return list(self.models.keys())
    
    def get_model_description(self, model: str) -> Dict[str, Any]:
        """获取模型描述"""
        descriptions = {
            'gaussian': {
                'name': '高斯模型',
                'equation': 'A * exp(-0.5 * ((x - μ) / σ)²)',
                'parameters': ['振幅 (A)', '中心 (μ)', '标准差 (σ)'],
                'best_for': '对称峰形'
            },
            'lorentzian': {
                'name': '洛伦兹模型',
                'equation': 'A * γ² / ((x - x₀)² + γ²)',
                'parameters': ['振幅 (A)', '中心 (x₀)', '半宽 (γ)'],
                'best_for': '宽峰或有拖尾的峰'
            },
            'voigt': {
                'name': 'Voigt模型',
                'equation': '高斯和洛伦兹的卷积',
                'parameters': ['振幅 (A)', '中心 (x₀)', '高斯宽度 (σ)', '洛伦兹宽度 (γ)'],
                'best_for': '复杂峰形'
            },
            'exponential_gaussian': {
                'name': '指数修饰高斯',
                'equation': 'A * exp(-0.5 * ((x - μ) / σ)²) * exp(-(x - μ) / τ)',
                'parameters': ['振幅 (A)', '中心 (μ)', '高斯宽度 (σ)', '指数常数 (τ)'],
                'best_for': '有拖尾的峰'
            },
            'skewed_gaussian': {
                'name': '偏斜高斯',
                'equation': '偏斜正态分布',
                'parameters': ['振幅 (A)', '中心 (μ)', '标准差 (σ)', '偏斜参数 (α)'],
                'best_for': '不对称峰'
            }
        }
        
        return descriptions.get(model, {'name': '未知模型'})
