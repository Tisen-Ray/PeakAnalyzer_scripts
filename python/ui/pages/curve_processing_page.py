"""
曲线处理页面 - 重构版本
"""
import streamlit as st
import numpy as np
from typing import Optional, Tuple, Dict
from core.curve import Curve, Peak
from core.state_manager import state_manager
from ui.pages.processing import (
    BaselineCorrectionProcessor,
    SmoothingProcessor,
    PeakDetectionProcessor,
    PeakAnalysisProcessor,
    PeakFittingProcessor
)


class CurveProcessingPage:
    """曲线处理页面"""
    
    def __init__(self):
        self.baseline_processor = BaselineCorrectionProcessor()
        self.smoothing_processor = SmoothingProcessor()
        self.peak_detection_processor = PeakDetectionProcessor()
        self.peak_analysis_processor = PeakAnalysisProcessor()
        self.peak_fitting_processor = PeakFittingProcessor()
    
    def render(self):
        """渲染曲线处理页面"""
        st.title("🔧 曲线处理")
        
        # 获取所有曲线
        all_curves_dict = state_manager.get_all_curves()
        
        if not all_curves_dict:
            st.info("请先在数据提取页面加载曲线数据")
            return
        
        # 曲线选择 - 只存储curve_id，避免直接引用对象
        curve_options = {f"{curve.curve_type} (ID: {curve.curve_id})": curve.curve_id for curve in all_curves_dict.values()}
        selected_curve_name = st.selectbox(
            "选择要处理的曲线",
            options=list(curve_options.keys()),
            key="selected_curve_for_processing"
        )
        
        if not selected_curve_name:
            return
        
        # 获取曲线ID并创建副本用于处理
        selected_curve_id = curve_options[selected_curve_name]
        original_curve = state_manager.get_curve(selected_curve_id)
        
        # 使用session_state管理工作副本，确保完整的数据保护
        working_key = f"working_curve_{selected_curve_id}"
        if working_key not in st.session_state or st.button("🔄 重置工作副本", key="reset_working_copy"):
            # 首次访问或手动重置，创建完整的工作副本
            import copy
            st.session_state[working_key] = {
                "curve": copy.deepcopy(original_curve),
                "original_y": original_curve.y_values.copy(),
                "original_peaks": copy.deepcopy(original_curve.peaks),
                "original_state": {
                    "is_baseline_corrected": original_curve.is_baseline_corrected,
                    "is_smoothed": original_curve.is_smoothed,
                    "has_peaks": len(original_curve.peaks) > 0
                },
                "is_modified": False,
                "last_applied": False
            }
            if st.button("🔄 重置工作副本", key="reset_working_copy"):
                st.success("✅ 工作副本已重置到原始状态")
                st.rerun()
        
        # 获取工作副本 - 每次都创建新的引用避免状态污染
        import copy
        working_data = st.session_state[working_key]
        selected_curve = copy.deepcopy(working_data["curve"])
        
        # 显示曲线基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("曲线类型", selected_curve.curve_type)
        with col2:
            st.metric("数据点数", len(selected_curve.x_values))
        with col3:
            st.metric("最大强度", f"{selected_curve.max_intensity:.0f}")
        with col4:
            st.metric("检测到的峰数", len(selected_curve.peaks))
        
        # 处理流程
        st.markdown("---")
        st.markdown("## 📋 处理流程")
        
        # 使用标签页组织处理步骤
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔧 基线校正", 
            "🔧 平滑处理", 
            "🔍 峰检测", 
            "📊 峰分析", 
            "📈 峰拟合"
        ])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                self.baseline_processor.render_baseline_correction(selected_curve)
            with col2:
                self._render_curve_plot(selected_curve, "baseline")
        
        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                self.smoothing_processor.render_smoothing(selected_curve)
            with col2:
                self._render_curve_plot(selected_curve, "smoothing")
        
        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                self.peak_detection_processor.render_peak_detection(selected_curve)
            with col2:
                self._render_curve_plot(selected_curve, "peak_detection")
        
        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                self.peak_analysis_processor.render_peak_analysis(selected_curve)
            with col2:
                self._render_curve_plot(selected_curve, "peak_analysis")
        
        with tab5:
            col1, col2 = st.columns([1, 2])
        with col1:
                self.peak_fitting_processor.render_peak_fitting(selected_curve)
        with col2:
                self._render_curve_plot(selected_curve, "peak_fitting")
    
    def _reset_to_stored_data(self, curve: Curve):
        """重置曲线数据到存储的原始状态"""
        # 首先无条件清除所有对比数据
        curve._original_y_values = None
        if hasattr(curve, '_original_peaks'):
            curve._original_peaks = None
        
        try:
            # 从状态管理器获取存储的原始数据
            stored_curve = state_manager.get_curve(curve.curve_id)
            if stored_curve is None:
                return
            
            working_key = f"working_curve_{curve.curve_id}"
            if working_key not in st.session_state:
                return
            
            # 同时清除工作副本中的对比数据
            working_curve = st.session_state[working_key]["curve"]
            working_curve._original_y_values = None
            if hasattr(working_curve, '_original_peaks'):
                working_curve._original_peaks = None
            
            # 检查是否有最近应用的更改
            if st.session_state[working_key].get("last_applied", False):
                # 如果有最近应用的更改，清除标记并更新工作副本以反映存储的数据
                st.session_state[working_key]["last_applied"] = False
                
                # 创建深拷贝并更新工作副本以反映存储的数据
                import copy
                fresh_curve = copy.deepcopy(stored_curve)
                
                # 更新工作副本的所有属性以反映存储的数据
                st.session_state[working_key]["curve"] = fresh_curve
                st.session_state[working_key]["original_y"] = stored_curve.y_values.copy()
                st.session_state[working_key]["is_modified"] = False
                
                # 更新当前曲线对象以反映工作副本
                curve.y_values = fresh_curve.y_values.copy()
                curve.peaks = fresh_curve.peaks.copy()
                curve.is_baseline_corrected = fresh_curve.is_baseline_corrected
                curve.is_smoothed = fresh_curve.is_smoothed
                # curve类没有is_peaks_detected属性，通过peaks列表判断
                
                # 清除任何临时对比数据
                curve._original_y_values = None
                if hasattr(curve, '_original_peaks'):
                    curve._original_peaks = None
                
                # 确保工作副本也清除对比数据
                fresh_curve._original_y_values = None
                if hasattr(fresh_curve, '_original_peaks'):
                    fresh_curve._original_peaks = None
            else:
                # 即使没有最近应用的更改，也要确保清除对比数据
                curve._original_y_values = None
                if hasattr(curve, '_original_peaks'):
                    curve._original_peaks = None
                
                # 同时清除工作副本中的对比数据
                working_curve = st.session_state[working_key]["curve"]
                working_curve._original_y_values = None
                if hasattr(working_curve, '_original_peaks'):
                    working_curve._original_peaks = None
            return
            
        except Exception as e:
            # 如果获取存储数据失败，保持当前状态
            print(f"Warning: Failed to reset curve data: {e}")
            pass
    
    def _render_curve_plot(self, curve: Curve, tab_name: str = ""):
        """渲染统一的曲线显示组件"""
        import plotly.graph_objects as go
        
        # 创建Plotly图表
        fig = go.Figure()
        
        # 检查是否有处理前后的对比数据
        has_original_data = (hasattr(curve, '_original_y_values') and 
                           curve._original_y_values is not None and 
                           len(curve._original_y_values) > 0)
        
        if has_original_data:
            # 显示原始曲线
            fig.add_trace(go.Scatter(
                x=curve.x_values,
                y=curve._original_y_values,
                mode='lines',
                name='原始曲线',
                line=dict(color='lightblue', width=2, dash='dash'),
                hovertemplate='<b>原始曲线</b><br>RT: %{x:.2f} min<br>强度: %{y:.0f}<extra></extra>'
            ))
            
            # 显示处理后的曲线
            fig.add_trace(go.Scatter(
                x=curve.x_values,
                y=curve.y_values,
                mode='lines',
                name='处理后曲线',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>处理后曲线</b><br>RT: %{x:.2f} min<br>强度: %{y:.0f}<extra></extra>'
            ))
        else:
            # 只显示当前曲线
            fig.add_trace(go.Scatter(
                x=curve.x_values,
                y=curve.y_values,
                mode='lines',
                name='当前曲线',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='RT: %{x:.2f} min<br>强度: %{y:.0f}<extra></extra>'
            ))
        
        # 如果有峰数据，添加峰标注
        if curve.peaks:
            peak_x = [peak.rt for peak in curve.peaks]
            peak_y = [peak.intensity for peak in curve.peaks]
            
            fig.add_trace(go.Scatter(
                x=peak_x,
                y=peak_y,
                mode='markers',
                name='检测到的峰',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='triangle-up',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate='<b>峰</b><br>RT: %{x:.3f} min<br>强度: %{y:.0f}<extra></extra>'
            ))
            
            # 添加峰标注和FWHM
            for i, peak in enumerate(curve.peaks):
                # 峰标注 - 使用数据坐标，确保随缩放移动
                if tab_name == "peak_analysis" and hasattr(peak, 'area') and peak.area > 0:
                    # 峰分析阶段显示面积信息
                    label_text = f"峰{i+1}\n面积: {peak.area:.2e}"
                    label_color = ['red', 'green', 'purple', 'orange', 'brown', 'pink'][i % 6]
                elif tab_name == "peak_fitting" and 'fit_result' in peak.metadata:
                    # 峰拟合阶段显示拟合信息
                    fit_result = peak.metadata['fit_result']
                    model = peak.metadata.get('fit_model', '未知')
                    if fit_result.get('success', False):
                        label_text = f"峰{i+1}\n{model}\nR²={fit_result.get('r_squared', 0):.3f}"
                    else:
                        label_text = f"峰{i+1}\n{model}\n拟合失败"
                    label_color = ['red', 'green', 'purple', 'orange', 'brown', 'pink'][i % 6]
                else:
                    # 其他阶段显示基本信息
                    label_text = f"峰{i+1}"
                    label_color = "red"
                
                # 使用数据坐标的标注点，确保随缩放移动
                label_y = peak.intensity * 1.1  # 标注位置在峰顶上方10%
                fig.add_trace(go.Scatter(
                    x=[peak.rt],
                    y=[label_y],
                    mode='text',
                    text=[label_text],
                    textposition='top center',
                    textfont=dict(size=10, color=label_color),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # 只在峰分析阶段显示FWHM标注
                if (hasattr(peak, 'fwhm') and peak.fwhm > 0 and 
                    tab_name == "peak_analysis"):  # 只在峰分析阶段显示
                    
                    # 从峰分析结果获取精确的FWHM交点信息
                    fwhm_data = self._get_precise_fwhm_points(curve, peak)
                    if fwhm_data:
                        fwhm_left, fwhm_right, fwhm_height = fwhm_data
                    else:
                        # 备用：使用简单计算
                        fwhm_left = peak.rt - peak.fwhm / 2
                        fwhm_right = peak.rt + peak.fwhm / 2
                        fwhm_height = peak.intensity / 2
                    
                    # 添加FWHM横线（使用精确的交点坐标）
                    fig.add_trace(go.Scatter(
                        x=[fwhm_left, fwhm_right],
                        y=[fwhm_height, fwhm_height],
                        mode='lines+text',
                        line=dict(color="purple", width=2),
                        text=['', f'FWHM: {peak.fwhm:.3f}min'],
                        textposition='top center',
                        textfont=dict(size=8, color="purple"),
                        name=f'峰{i+1}FWHM',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # 添加FWHM边界标记点（显示精确的交点位置）
                    fig.add_trace(go.Scatter(
                        x=[fwhm_left, fwhm_right],
                        y=[fwhm_height, fwhm_height],
                        mode='markers+text',
                        marker=dict(color="purple", size=8, symbol="diamond"),
                        text=['FWHM左', 'FWHM右'],
                        textposition='bottom center',
                        textfont=dict(size=8, color="purple"),
                        showlegend=False,
                        hovertemplate='FWHM交点<br>RT: %{x:.4f} min<br>强度: %{y:.1f}<extra></extra>'
                    ))
                    
                    # 添加半高水平线延伸到曲线边缘，显示完整的半高线
                    curve_x_min = np.min(curve.x_values)
                    curve_x_max = np.max(curve.x_values)
                    fig.add_trace(go.Scatter(
                        x=[max(curve_x_min, fwhm_left - 0.1), min(curve_x_max, fwhm_right + 0.1)],
                        y=[fwhm_height, fwhm_height],
                        mode='lines',
                        line=dict(color="purple", width=1, dash="dot"),
                        name=f'峰{i+1}半高线',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # 只在峰分析阶段显示峰边界标注
                if (hasattr(peak, 'rt_start') and hasattr(peak, 'rt_end') and 
                    hasattr(peak, 'area') and peak.area > 0 and  # 确保已进行峰分析
                    tab_name == "peak_analysis"):  # 只在峰分析标签页显示
                    
                    # 检查是否有峰分析的可视化数据
                    if 'visualization_data' in peak.metadata:
                        viz_data = peak.metadata['visualization_data']
                        boundaries = viz_data['boundaries']
                        integration_method = viz_data['integration_method']
                        baseline_method = viz_data['baseline_method']
                        
                        peak_color = ['red', 'green', 'purple', 'orange', 'brown', 'pink'][i % 6]
                        
                        # 使用完整曲线坐标系显示基线（向后兼容）
                        if 'full_curve_baseline' in viz_data:
                            full_baseline = np.array(viz_data['full_curve_baseline'])
                        else:
                            # 向后兼容：如果没有完整基线数据，使用旧的峰区域数据
                            peak_x_data = np.array(viz_data.get('x_data', viz_data.get('peak_region_x', [])))
                            peak_baseline_y = np.array(viz_data.get('baseline_y', viz_data.get('peak_region_baseline', [])))
                            
                            # 临时创建完整基线
                            full_baseline = np.zeros_like(curve.y_values)
                            if len(peak_x_data) > 0 and len(peak_baseline_y) > 0:
                                # 找到峰区域在完整曲线中的索引
                                start_idx = np.argmin(np.abs(curve.x_values - start_rt))
                                end_idx = np.argmin(np.abs(curve.x_values - end_rt))
                                start_idx = max(0, min(start_idx, len(curve.x_values) - 1))
                                end_idx = max(start_idx + 1, min(end_idx, len(curve.x_values) - 1))
                                
                                # 插值映射基线
                                full_x_region = curve.x_values[start_idx:end_idx+1]
                                interpolated_baseline = np.interp(full_x_region, peak_x_data, peak_baseline_y)
                                full_baseline[start_idx:end_idx+1] = interpolated_baseline
                        
                        # 只显示峰区域的基线（避免显示整条基线）
                        start_rt, end_rt = boundaries
                        baseline_mask = (curve.x_values >= start_rt) & (curve.x_values <= end_rt)
                        baseline_x = curve.x_values[baseline_mask]
                        baseline_y = full_baseline[baseline_mask]
                        
                        if len(baseline_x) > 0:
                            fig.add_trace(go.Scatter(
                                x=baseline_x,
                                y=baseline_y,
                                mode='lines',
                                name=f'峰{i+1}基线 ({baseline_method})',
                                line=dict(color=peak_color, width=2, dash='dot'),
                                hovertemplate=f'<b>峰{i+1}基线</b><br>' +
                                             f'方法: {baseline_method}<br>' +
                                             'RT: %{x:.3f} min<br>' +
                                             '基线值: %{y:.0f}<extra></extra>'
                            ))
                        
                        # 显示积分区域的面积填充（使用完整曲线坐标）
                        curve_mask = (curve.x_values >= start_rt) & (curve.x_values <= end_rt)
                        x_fill = curve.x_values[curve_mask]
                        y_fill = curve.y_values[curve_mask]  # 使用原始曲线数据
                        baseline_fill = full_baseline[curve_mask]
                        
                        if len(x_fill) > 0:
                            # 创建填充区域：从基线到原始曲线
                            x_fill_area = np.concatenate([x_fill, x_fill[::-1]])
                            y_fill_area = np.concatenate([baseline_fill, y_fill[::-1]])
                            
                            fig.add_trace(go.Scatter(
                                x=x_fill_area,
                                y=y_fill_area,
                                fill='toself',
                                fillcolor=f'rgba({self._get_rgb_values(peak_color)}, 0.3)',
                                mode='none',
                                name=f'峰{i+1}面积 ({integration_method})',
                                hoverinfo='skip',
                                showlegend=True
                            ))
                        
                        # 显示峰边界线（使用数据坐标）
                        boundary_height = np.max(curve.y_values) * 0.1  # 使用完整曲线的最大值
                        
                        # 起始边界线
                        fig.add_trace(go.Scatter(
                            x=[start_rt, start_rt],
                            y=[0, boundary_height],
                            mode='lines+text',
                            line=dict(color=peak_color, width=2),
                            text=['', f'峰{i+1}起始'],
                            textposition='top center',
                            textfont=dict(size=8, color=peak_color),
                            name=f'峰{i+1}边界',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # 结束边界线
                        fig.add_trace(go.Scatter(
                            x=[end_rt, end_rt],
                            y=[0, boundary_height],
                            mode='lines+text',
                            line=dict(color=peak_color, width=2),
                            text=['', f'峰{i+1}结束'],
                            textposition='top center',
                            textfont=dict(size=8, color=peak_color),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                # 峰拟合阶段的可视化
                if (tab_name == "peak_fitting" and 'fit_result' in peak.metadata):
                    fit_result = peak.metadata['fit_result']
                    model = peak.metadata.get('fit_model', '未知')
                    
                    if fit_result.get('success', False):
                        # 获取拟合曲线数据
                        fit_curve_data = fit_result.get('fitted_curve', {})
                        if isinstance(fit_curve_data, dict) and 'x' in fit_curve_data and 'y' in fit_curve_data:
                            fit_x = np.array(fit_curve_data['x'])
                            fit_y = np.array(fit_curve_data['y'])
                            
                            peak_color = ['red', 'green', 'purple', 'orange', 'brown', 'pink'][i % 6]
                            
                            # 添加拟合曲线到统一图表中
                            fig.add_trace(go.Scatter(
                                x=fit_x,
                                y=fit_y,
                                mode='lines',
                                name=f"峰{i+1}拟合 ({model})",
                                line=dict(color=peak_color, width=3),
                                fill='tozeroy',  # 填充到y=0
                                fillcolor=f'rgba({self._get_rgb_values(peak_color)}, 0.3)',
                                hovertemplate=f'<b>峰{i+1}拟合</b><br>' +
                                             f'模型: {model}<br>' +
                                             f'R²: {fit_result.get("r_squared", 0):.4f}<br>' +
                                             'RT: %{x:.3f} min<br>' +
                                             '强度: %{y:.0f}<extra></extra>'
                            ))
        
        # 更新布局
        title = f"当前曲线 - {curve.curve_type}"
        if has_original_data:
            title += " (对比模式)"
        
        fig.update_layout(
            title=title,
            xaxis_title="保留时间 (分钟)",
            yaxis_title="强度",
            showlegend=True,
            height=500,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        
        # 显示图表 - 使用唯一key避免重复ID
        unique_key = f"curve_plot_{curve.curve_id}_{tab_name}" if tab_name else f"curve_plot_{curve.curve_id}"
        st.plotly_chart(fig, width='stretch', key=unique_key)
    
    def _get_precise_fwhm_points(self, curve: Curve, peak: Peak) -> Optional[Tuple[float, float, float]]:
        """
        获取精确的FWHM交点信息
        
        返回: (左交点RT, 右交点RT, 半高强度)
        """
        try:
            # 首先尝试从峰的可视化数据中获取
            if 'visualization_data' in peak.metadata and 'fwhm_info' in peak.metadata['visualization_data']:
                fwhm_info = peak.metadata['visualization_data']['fwhm_info']
                return (
                    fwhm_info['left_intersection'],
                    fwhm_info['right_intersection'], 
                    fwhm_info['half_height']
                )
            
            # 如果没有保存的数据，重新计算
            from peak_analysis.peak_analyzer import PeakAnalyzer
            analyzer = PeakAnalyzer()
            
            # 获取平滑数据
            smoothed_curve_y = analyzer._get_smoothed_curve_data(curve, True)
            
            # 提取峰区域
            region_data = analyzer._extract_peak_region_clean(curve, peak, 2.0, smoothed_curve_y)
            if not region_data:
                return None
            
            x_data, y_original, y_smoothed, peak_idx = region_data
            
            # 计算精确的FWHM信息
            fwhm_info = analyzer._calculate_precise_fwhm_info(x_data, y_original, peak_idx)
            
            return (
                fwhm_info['left_intersection'],
                fwhm_info['right_intersection'],
                fwhm_info['half_height']
            )
            
        except Exception as e:
            print(f"❌ 获取精确FWHM点失败: {e}")
        
        return None
    
    def _get_rgb_values(self, color_name):
        """获取颜色的RGB值"""
        color_map = {
            'red': '255, 0, 0',
            'green': '0, 128, 0', 
            'purple': '128, 0, 128',
            'orange': '255, 165, 0',
            'brown': '165, 42, 42',
            'pink': '255, 192, 203',
            'gray': '128, 128, 128',
            'olive': '128, 128, 0'
        }
        return color_map.get(color_name, '128, 128, 128')