"""
峰检测处理模块
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
from scipy.signal import find_peaks, peak_prominences, peak_widths
from core.curve import Curve
from core.state_manager import state_manager
from peak_analysis.peak_detector import PeakDetector


class PeakDetectionProcessor:
    """峰检测处理器"""
    
    def __init__(self):
        self.peak_detector = PeakDetector()
    
    def render_peak_detection(self, curve: Curve) -> bool:
        """渲染峰检测界面并执行处理"""
        st.markdown("### 🔍 峰检测")
        
        if not curve or not curve.y_values.size:
            st.warning("请先加载曲线数据")
            return False
        
        # 峰检测方法选择
        available_methods = self.peak_detector.get_available_methods()
        method = st.selectbox(
            "检测方法",
            options=available_methods,
            key="peak_detection_method"
        )
        
        # 根据选择的方法显示参数
        params = self._render_method_params(method)
        
        # 操作按钮 - 垂直布局
        if st.button("🔍 执行峰检测", key="detect_peaks", width='stretch'):
            return self._execute_peak_detection(curve, method, params)
        
        if st.button("⏭️ 跳过", key="skip_peak_detection", width='stretch'):
            st.info("已跳过峰检测")
            return False
        
        return False
    
    def _render_method_params(self, method: str) -> Dict[str, Any]:
        """根据选择的方法渲染参数"""
        params = {}
        
        if method == 'scipy_find_peaks':
            # SciPy find_peaks 参数
            col1, col2 = st.columns(2)
            with col1:
                height = st.slider("高度阈值", 0.0, 10000.0, 0.0, step=100.0, help="最小峰高度，0表示自动计算")
            with col2:
                prominence = st.slider("突出度", 0.0, 1000.0, 0.0, step=10.0, help="最小突出度，0表示自动计算")
            
            col3, col4 = st.columns(2)
            with col3:
                distance = st.slider("最小距离(索引)", 1, 100, 1, help="峰之间最小距离（数据点）")
            with col4:
                width = st.slider("最小峰宽度", 0.0, 50.0, 0.0, step=1.0, help="最小峰宽度，0表示自动计算")
            
            params = {
                'height': height if height > 0 else None,
                'prominence': prominence if prominence > 0 else None,
                'distance': distance if distance > 1 else None,
                'width': width if width > 0 else None
            }
            
        elif method == 'cwt':
            # CWT 参数
            col1, col2 = st.columns(2)
            with col1:
                min_snr = st.slider("最小信噪比", 0.1, 10.0, 1.0, 0.1, help="连续小波变换的最小信噪比")
            with col2:
                noise_perc = st.slider("噪声百分位", 1.0, 50.0, 10.0, 1.0, help="用于估计噪声水平的百分位数")
            
            # 宽度范围自动设置，不需要用户输入
            params = {
                'min_snr': min_snr,
                'noise_perc': noise_perc
            }
            
        elif method == 'derivative':
            # 导数方法参数
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("导数阈值", 0.0, 10000.0, 0.0, step=100.0, help="导数强度阈值，0表示自动计算")
            with col2:
                min_distance = st.slider("最小距离(索引)", 1, 100, 1, help="峰之间最小距离（数据点）")
            
            params = {
                'threshold': threshold if threshold > 0 else None,
                'min_distance': min_distance if min_distance > 1 else None
            }
            
        elif method == 'threshold':
            # 阈值方法参数
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("强度阈值", 0.0, 10000.0, 0.0, step=100.0, help="峰强度阈值，0表示自动计算")
            with col2:
                min_distance = st.slider("最小距离(索引)", 1, 100, 1, help="峰之间最小距离（数据点）")
            
            params = {
                'threshold': threshold if threshold > 0 else None,
                'min_distance': min_distance if min_distance > 1 else None
            }
        
        return params
    
    def _preview_peak_detection(self, curve: Curve, method: str, params: Dict[str, Any]):
        """预览峰检测结果"""
        try:
            # 使用PeakDetector进行峰检测
            peaks = self.peak_detector.detect_peaks(
                curve=curve,
                method=method,
                **params
            )
            
            # 显示预览结果
            self._show_peak_detection_result(curve, peaks, preview=True)
            
        except Exception as e:
            st.error(f"❌ 预览失败: {str(e)}")
    
    def _execute_peak_detection(self, curve: Curve, method: str, params: Dict[str, Any]) -> bool:
        """执行峰检测并直接应用结果"""
        try:
            # 使用当前工作副本进行峰检测（包含所有已应用的处理）
            detected_peaks = self.peak_detector.detect_peaks(
                curve=curve,  # 使用当前工作副本，包含所有已应用的处理
                method=method,
                **params
            )
            
            # 直接应用峰检测结果到存储数据
            import copy
            stored_curve = state_manager.get_curve(curve.curve_id)
            stored_curve.peaks.clear()
            stored_curve.peaks.extend(copy.deepcopy(detected_peaks))
            stored_curve.is_peaks_detected = True
            state_manager.update_curve(stored_curve)
            
            # 更新当前显示的曲线数据
            curve.peaks.clear()
            curve.peaks.extend(copy.deepcopy(detected_peaks))
            
            st.success(f"✅ 峰检测完成，检测到 {len(detected_peaks)} 个峰")
            
            # 显示峰检测结果
            self._show_peak_detection_result(curve, detected_peaks, preview=False)
            
            return True
            
        except Exception as e:
            st.error(f"❌ 峰检测失败: {str(e)}")
            return False
    
    def _show_peak_detection_result(self, curve: Curve, peaks: List = None, preview: bool = False):
        """显示峰检测结果"""
        st.markdown("**峰检测结果**")
        
        peaks_to_show = peaks if peaks is not None else curve.peaks
        
        if not peaks_to_show:
            st.info("未检测到峰")
            return
        
        # 峰信息表格 - 峰检测阶段只显示基本信息，不显示面积
        peak_data = []
        for i, peak in enumerate(peaks_to_show):
            peak_data.append({
                '峰序号': i+1,
                '保留时间 (min)': f"{peak.rt:.3f}",
                '强度': f"{peak.intensity:.0f}",
                '信噪比': f"{peak.signal_to_noise:.1f}" if hasattr(peak, 'signal_to_noise') else "N/A"
            })
        
        import pandas as pd
        df = pd.DataFrame(peak_data)
        
        # 使用全宽表格布局
        st.dataframe(
            df, 
            width='stretch',
            height=min(400, 50 + len(df) * 35)  # 动态高度
        )
        
        # 显示统计信息
        st.markdown("**检测统计**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("检测到峰数", len(peaks_to_show))
        with col2:
            avg_rt = np.mean([peak.rt for peak in peaks_to_show])
            st.metric("平均保留时间", f"{avg_rt:.3f} min")
        with col3:
            max_intensity = max([peak.intensity for peak in peaks_to_show])
            st.metric("最大强度", f"{max_intensity:.0f}")
        with col4:
            avg_snr = np.mean([peak.signal_to_noise for peak in peaks_to_show if hasattr(peak, 'signal_to_noise')])
            st.metric("平均信噪比", f"{avg_snr:.1f}")
