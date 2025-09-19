"""
峰分析处理模块
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
from core.curve import Curve
from core.state_manager import state_manager
from peak_analysis.peak_analyzer import PeakAnalyzer

class PeakAnalysisProcessor:
    """峰分析处理器"""
    
    def __init__(self):
        self.peak_analyzer = PeakAnalyzer()
    
    def render_peak_analysis(self, curve: Curve) -> bool:
        """渲染峰分析界面并执行处理"""
        st.markdown("### 📊 峰分析")
        
        if not curve or not curve.peaks:
            st.warning("请先进行峰检测")
            return False
        
        # 使用垂直布局，适应窄侧边栏
        # 分析参数
        extend_range = st.slider(
            "扩展范围倍数", 
            min_value=1.0, 
            max_value=5.0, 
            value=2.0, 
            step=0.1,
            help="相对于FWHM的扩展分析范围倍数"
        )
        
        # 色谱峰积分方法（行业标准）
        st.markdown("**峰积分方法**")
        integration_method = st.selectbox(
            "积分方法",
            ["垂直分割法", "谷到谷积分", "切线基线法", "指数衰减基线", "水平基线法"],
            index=0,
            help="选择符合色谱分析标准的积分方法"
        )
        
        # 基线处理策略
        baseline_method = st.selectbox(
            "基线处理",
            ["自动基线", "线性基线", "多项式基线", "指数基线", "手动基线"],
            index=0,
            help="选择基线校正方法"
        )
        
        # 峰边界检测方法选择
        st.markdown("**🔧 峰边界检测方法**")
        
        # 提供专业的色谱方法选择
        boundary_method = st.selectbox(
            "边界检测方法",
            [
                "自动选择（基于灵敏度）",
                "切线撇取法 (Tangent Skim)",
                "指数撇取法 (Exponential Skim)",
                "谷到谷法 (Valley-to-Valley)",
                "垂直分割法 (Perpendicular Drop)"
            ],
            index=0,
            help="选择色谱分析标准的峰边界检测方法"
        )
        
        peak_sensitivity = st.slider(
            "检测灵敏度",
            min_value=1,
            max_value=10,
            value=5,
            help="控制边界检测的严格程度。低值=保守(抗噪声)，高值=宽松(包含更多信号)"
        )
        
        # 显示方法说明
        method_descriptions = {
            "切线撇取法 (Tangent Skim)": "📐 最保守和准确，适用于基线漂移和重叠峰",
            "指数撇取法 (Exponential Skim)": "📈 适用于拖尾峰，处理不对称峰形",
            "谷到谷法 (Valley-to-Valley)": "🏔️ 经典方法，适用于基线平稳的情况",
            "垂直分割法 (Perpendicular Drop)": "📏 最简单，适用于对称峰形",
            "自动选择（基于灵敏度）": "🤖 根据灵敏度自动选择最佳方法"
        }
        
        if boundary_method in method_descriptions:
            st.info(method_descriptions[boundary_method])
        
        # 添加额外的鲁棒性控制参数
        noise_tolerance = st.slider(
            "噪声容忍度",
            min_value=1,
            max_value=10,
            value=5,
            help="控制对小波动的容忍程度。较高值可以更好地忽略基线噪声"
        )
        
        boundary_smoothing = st.checkbox(
            "边界平滑处理",
            value=True,
            help="启用边界检测前的数据平滑，减少小波动对边界检测的影响"
        )
        
        # 色谱峰质量参数（行业标准）
        st.markdown("**色谱峰质量参数**")
        col1, col2 = st.columns(2)
        with col1:
            calc_theoretical_plates = st.checkbox("理论塔板数 (N)", value=True)
            calc_tailing_factor = st.checkbox("拖尾因子 (Tf)", value=True)
            calc_asymmetry_factor = st.checkbox("不对称因子 (As)", value=True)
        with col2:
            calc_resolution = st.checkbox("分离度 (Rs)", value=True)
            calc_capacity_factor = st.checkbox("容量因子 (k')", value=False)
            calc_selectivity = st.checkbox("选择性因子 (α)", value=False)
        
        # 执行按钮
        if st.button("📊 开始色谱峰分析", key="analyze_peaks", width='stretch'):
            return self._analyze_peaks_inplace(curve, {
                'extend_range': extend_range,
            'integration_method': integration_method,
            'baseline_method': baseline_method,
            'boundary_method': boundary_method,
            'peak_sensitivity': peak_sensitivity,
            'noise_tolerance': noise_tolerance,
            'boundary_smoothing': boundary_smoothing,
            'calc_theoretical_plates': calc_theoretical_plates,
            'calc_tailing_factor': calc_tailing_factor,
            'calc_asymmetry_factor': calc_asymmetry_factor,
            'calc_resolution': calc_resolution,
            'calc_capacity_factor': calc_capacity_factor,
            'calc_selectivity': calc_selectivity
            })
        
        return False
    
    def _analyze_peaks_inplace(self, curve: Curve, params: Dict[str, Any]) -> bool:
        """就地执行峰分析"""
        try:
            # 对每个峰进行分析（使用当前工作副本，包含所有已应用的处理）
            for peak in curve.peaks:
                # 使用色谱分析标准方法（所有计算逻辑都在peak_analyzer中）
                updated_peak = self.peak_analyzer.analyze_peak(
                    curve=curve,
                    peak=peak,
                    extend_range=params.get('extend_range', 2.0),
                    integration_method=params.get('integration_method', '垂直分割法'),
                    baseline_method=params.get('baseline_method', '自动基线'),
                    boundary_method=params.get('boundary_method', '自动选择（基于灵敏度）'),
                    peak_sensitivity=params.get('peak_sensitivity', 5),
                    noise_tolerance=params.get('noise_tolerance', 5),
                    boundary_smoothing=params.get('boundary_smoothing', True),
                    calc_theoretical_plates=params.get('calc_theoretical_plates', True),
                    calc_tailing_factor=params.get('calc_tailing_factor', True),
                    calc_asymmetry_factor=params.get('calc_asymmetry_factor', True),
                    calc_resolution=params.get('calc_resolution', True),
                    calc_capacity_factor=params.get('calc_capacity_factor', False),
                    calc_selectivity=params.get('calc_selectivity', False)
                )
                
                # 直接替换峰对象
                curve.peaks[curve.peaks.index(peak)] = updated_peak
            
            # 更新存储数据
            stored_curve = state_manager.get_curve(curve.curve_id)
            stored_curve.peaks = curve.peaks.copy()
            state_manager.update_curve(stored_curve)
            
            # 显示结果
            st.success(f"✅ 色谱峰分析完成，分析了 {len(curve.peaks)} 个峰")
            st.info(f"📊 使用方法: 积分={params['integration_method']}, 基线={params['baseline_method']}")
            
            # 显示峰分析结果
            self._show_peak_analysis_result(curve)
            
            return True
            
        except Exception as e:
            st.error(f"❌ 峰分析失败: {str(e)}")
            return False
    
    def _show_peak_analysis_result(self, curve: Curve):
        """显示峰分析结果"""
        st.markdown("**峰分析结果**")
        
        if not curve.peaks:
            st.info("暂无峰数据")
            return
        
        # 峰分析结果表格 - 包含详细的三维参数
        analysis_data = []
        for i, peak in enumerate(curve.peaks):
            analysis_data.append({
                '峰': i+1,
                'RT': f"{peak.rt:.3f}",
                '强度': f"{peak.intensity:.0f}",
                '面积': f"{peak.area:.2e}" if peak.area > 1000 else f"{peak.area:.0f}",
                'FWHM': f"{peak.fwhm:.3f}",
                '起始': f"{peak.rt_start:.3f}" if hasattr(peak, 'rt_start') else "N/A",
                '结束': f"{peak.rt_end:.3f}" if hasattr(peak, 'rt_end') else "N/A",
                'SNR': f"{peak.signal_to_noise:.1f}",
                '置信度': f"{peak.confidence:.2f}" if hasattr(peak, 'confidence') else "N/A"
            })
        
        import pandas as pd
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, width='stretch', height=min(300, len(analysis_data) * 35 + 40))
        
        # 显示峰分析统计信息
        if len(curve.peaks) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("分析峰数", len(curve.peaks))
            with col2:
                avg_fwhm = sum(peak.fwhm for peak in curve.peaks) / len(curve.peaks)
                st.metric("平均FWHM", f"{avg_fwhm:.3f} min")
            with col3:
                avg_snr = sum(peak.signal_to_noise for peak in curve.peaks) / len(curve.peaks)
                st.metric("平均SNR", f"{avg_snr:.1f}")
        
        # 详细峰信息展示 - 紧凑布局
        st.markdown("**详细峰信息**")
        for i, peak in enumerate(curve.peaks):
            with st.expander(f"峰 {i+1} - RT: {peak.rt:.3f} min", expanded=False):
                # 使用更紧凑的布局
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RT", f"{peak.rt:.3f} min")
                    st.metric("强度", f"{peak.intensity:.0f}")
                
                with col2:
                    st.metric("面积", f"{peak.area:.0f}")
                    st.metric("FWHM", f"{peak.fwhm:.3f} min")
                
                with col3:
                    st.metric("SNR", f"{peak.signal_to_noise:.1f}")
                    st.metric("置信度", f"{peak.confidence:.2f}")
                
                # 峰边界和FWHM信息
                st.markdown("**边界信息:**")
                if hasattr(peak, 'rt_start') and hasattr(peak, 'rt_end'):
                    st.write(f"峰范围: {peak.rt_start:.3f} - {peak.rt_end:.3f} min")
                    st.write(f"峰宽度: {peak.rt_end - peak.rt_start:.3f} min")
                
                if hasattr(peak, 'fwhm') and peak.fwhm > 0:
                    fwhm_left = peak.rt - peak.fwhm / 2
                    fwhm_right = peak.rt + peak.fwhm / 2
                    st.write(f"FWHM范围: [{fwhm_left:.3f}, {fwhm_right:.3f}] min")
                
                # 色谱峰质量参数
                params = []
                if hasattr(peak, 'theoretical_plates') and peak.theoretical_plates is not None:
                    params.append(f"理论塔板数 (N): {peak.theoretical_plates:.0f}")
                if hasattr(peak, 'asymmetry_factor') and peak.asymmetry_factor is not None:
                    params.append(f"不对称因子 (As): {peak.asymmetry_factor:.3f}")
                if hasattr(peak, 'tailing_factor') and peak.tailing_factor is not None:
                    params.append(f"拖尾因子 (Tf): {peak.tailing_factor:.3f}")
                if hasattr(peak, 'resolution') and peak.resolution is not None:
                    params.append(f"分离度 (Rs): {peak.resolution:.1f}")
                if hasattr(peak, 'capacity_factor') and peak.capacity_factor is not None:
                    params.append(f"容量因子 (k'): {peak.capacity_factor:.2f}")
                if hasattr(peak, 'selectivity') and peak.selectivity is not None:
                    params.append(f"选择性因子 (α): {peak.selectivity:.2f}")
                
                if params:
                    st.markdown("**色谱质量参数:**")
                    st.write(" • ".join(params))
