"""
平滑处理模块
"""
import streamlit as st
import numpy as np
from typing import Dict, Any
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from core.curve import Curve
from core.state_manager import state_manager


class SmoothingProcessor:
    """平滑处理器"""
    
    def __init__(self):
        self.methods = {
            '移动平均': self._moving_average,
            'Savitzky-Golay': self._savitzky_golay,
            '高斯滤波': self._gaussian_filter,
            '中值滤波': self._median_filter,
            '低通滤波': self._lowpass_filter
        }
    
    def render_smoothing(self, curve: Curve) -> bool:
        """渲染平滑处理界面并执行处理"""
        st.markdown("### 🔧 平滑处理")
        
        if not curve or not curve.y_values.size:
            st.warning("请先加载曲线数据")
            return False
        
        # 方法选择
        method = st.selectbox(
            "平滑方法",
            options=list(self.methods.keys()),
            key="smoothing_method"
        )
        
        # 参数配置
        params = self._render_method_params(method)
        
        # 操作按钮 - 垂直布局
        if st.button("🔧 执行平滑处理", key="apply_smoothing", width='stretch'):
            return self._execute_smoothing_with_confirm(curve, method, params)
        
        if st.button("⏭️ 跳过", key="skip_smoothing", width='stretch'):
            st.info("已跳过平滑处理")
            return False
        
        return False
    
    def _render_method_params(self, method: str) -> Dict[str, Any]:
        """渲染方法特定参数"""
        params = {}
        
        if method == '移动平均':
            params['window_size'] = st.slider("窗口大小", 3, 100, 5, 2)
            
        elif method == 'Savitzky-Golay':
            params['window_length'] = st.slider("窗口长度", 5, 101, 11, 2)
            # 动态调整多项式阶数范围
            window_length = params.get('window_length', 11)
            max_polyorder = min(15, window_length - 1)
            params['polyorder'] = st.slider("多项式阶数", 2, max_polyorder, 3)
            
        elif method == '高斯滤波':
            params['sigma'] = st.slider("标准差", 0.1, 20.0, 1.0, 0.1)
            
        elif method == '中值滤波':
            params['kernel_size'] = st.slider("核大小", 3, 51, 5, 2)
            
        elif method == '低通滤波':
            params['cutoff'] = st.slider("截止频率", 0.001, 0.5, 0.1, 0.001)
            params['order'] = st.slider("滤波器阶数", 1, 20, 4, 1)
        
        return params
    
    def _preview_smoothing(self, curve: Curve, method: str, params: Dict[str, Any]):
        """预览平滑效果"""
        try:
            smoothed_y = self.methods[method](curve.y_values, params)
            st.success(f"✅ 预览完成 - 方法: {method}")
            st.info(f"原始数据范围: {curve.y_values.min():.0f} - {curve.y_values.max():.0f}")
            st.info(f"平滑后范围: {smoothed_y.min():.0f} - {smoothed_y.max():.0f}")
            noise = curve.y_values - smoothed_y
            st.info(f"噪声水平: {noise.std():.0f}")
        except Exception as e:
            st.error(f"❌ 预览失败: {str(e)}")
    
    def _execute_smoothing_with_confirm(self, curve: Curve, method: str, params: Dict[str, Any]) -> bool:
        """执行平滑处理并允许确认应用"""
        try:
            # 使用session_state管理工作副本
            working_key = f"working_curve_{curve.curve_id}"
            if working_key not in st.session_state:
                st.error("❌ 工作副本未找到，请重新选择曲线")
                return False
            
            # 获取当前工作副本的原始数据
            working_data = st.session_state[working_key]
            original_y = working_data["original_y"].copy()
            
            # 保存原始数据到曲线对象中用于对比显示
            curve._original_y_values = original_y.copy()
            
            # 基于原始数据执行平滑处理
            smoothed_y = self.methods[method](original_y, params)
            
            # 更新工作副本（临时显示）
            curve.y_values = smoothed_y.copy()
            curve.is_smoothed = True
            working_data["is_modified"] = True
            
            # 显示处理结果统计
            st.success(f"✅ 平滑处理执行完成 - 方法: {method}")
            st.info(f"原始数据范围: {original_y.min():.0f} - {original_y.max():.0f}")
            st.info(f"平滑后范围: {smoothed_y.min():.0f} - {smoothed_y.max():.0f}")
            noise = original_y - smoothed_y
            st.info(f"噪声水平: {noise.std():.0f}")
            
            # 确认应用选项
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ 确认应用", key="confirm_smoothing"):
                    # 确认应用 - 将工作副本写入存储数据
                    try:
                        # 获取存储的曲线
                        stored_curve = state_manager.get_curve(curve.curve_id)
                        if stored_curve is None:
                            st.error("❌ 无法获取存储的曲线数据")
                            return False
                        
                        # 将工作副本的数据写入存储数据
                        stored_curve.y_values = curve.y_values.copy()
                        stored_curve.is_smoothed = curve.is_smoothed
                        state_manager.update_curve(stored_curve)
                        
                        # 更新工作副本的原始数据为新的存储数据
                        working_data["original_y"] = stored_curve.y_values.copy()
                        working_data["is_modified"] = False
                        working_data["last_applied"] = True
                        
                        # 清除对比数据，不再显示原始曲线虚线
                        curve._original_y_values = None
                        if hasattr(curve, '_original_peaks'):
                            curve._original_peaks = None
                        
                        # 确保工作副本也清除对比数据
                        working_data_curve = working_data["curve"]
                        working_data_curve._original_y_values = None
                        if hasattr(working_data_curve, '_original_peaks'):
                            working_data_curve._original_peaks = None
                        
                        st.success(f"✅ 平滑处理已确认应用 (方法: {method})")
                        st.rerun()
                        return True
                        
                    except Exception as e:
                        st.error(f"❌ 确认应用失败: {str(e)}")
                        return False
            
            with col2:
                if st.button("❌ 撤销", key="cancel_smoothing"):
                    # 撤销操作 - 恢复到工作副本的原始数据
                    curve.y_values = original_y.copy()
                    curve.is_smoothed = False
                    curve._original_y_values = None  # 清除对比数据
                    working_data["is_modified"] = False
                    st.info("已撤销平滑处理，恢复到原始数据")
                    st.rerun()
                    return False
            
            return False
            
        except Exception as e:
            st.error(f"❌ 平滑处理失败: {str(e)}")
            return False
    
    def _moving_average(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """移动平均平滑"""
        window_size = params.get('window_size', 5)
        
        if window_size >= len(y):
            return y.copy()
        
        # 使用卷积进行移动平均
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(y, kernel, mode='same')
        
        return smoothed
    
    def _savitzky_golay(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Savitzky-Golay平滑"""
        window_length = params.get('window_length', 11)
        polyorder = params.get('polyorder', 3)
        
        # 确保窗口长度是奇数
        if window_length % 2 == 0:
            window_length += 1
        
        # 确保多项式阶数小于窗口长度
        polyorder = min(polyorder, window_length - 1)
        
        smoothed = signal.savgol_filter(y, window_length, polyorder)
        return smoothed
    
    def _gaussian_filter(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """高斯滤波平滑"""
        sigma = params.get('sigma', 1.0)
        smoothed = gaussian_filter1d(y, sigma)
        return smoothed
    
    def _median_filter(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """中值滤波平滑"""
        kernel_size = params.get('kernel_size', 5)
        smoothed = signal.medfilt(y, kernel_size)
        return smoothed
    
    def _lowpass_filter(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """低通滤波平滑"""
        cutoff = params.get('cutoff', 0.1)
        order = params.get('order', 4)
        
        # 设计低通滤波器
        nyquist = 0.5  # 假设采样频率为1
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        # 应用滤波器
        smoothed = signal.filtfilt(b, a, y)
        return smoothed
