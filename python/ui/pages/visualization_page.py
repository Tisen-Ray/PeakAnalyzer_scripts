"""
结果可视化页面 - 多曲线展示和峰分析
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from core.curve import Curve, Peak
from core.state_manager import state_manager
from peak_analysis.peak_detector import PeakDetector


class VisualizationPage:
    """结果可视化页面类"""
    
    def __init__(self):
        self.peak_detector = PeakDetector()
    
    def render(self):
        """渲染页面内容"""
        st.header("📊 结果可视化")
        st.markdown("多曲线可视化展示和峰分析结果")
        
        curves = state_manager.get_all_curves()
        if not curves:
            st.info("请先在'数据提取'页面提取曲线数据")
            return
        
        # 添加批量处理选项卡
        tab1, tab2 = st.tabs(["📊 可视化分析", "⚙️ 批量处理"])
        
        with tab1:
        # 主要布局：左侧曲线列表，右侧图表展示
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_curve_list_panel()
        
        with col2:
            self._render_visualization_panel()
        
        with tab2:
            self._render_batch_processing_panel()
    
    def _render_curve_list_panel(self):
        """渲染左侧曲线列表面板"""
        st.subheader("📋 曲线列表")
        
        # 曲线过滤选项
        self._render_curve_filters()
        
        st.markdown("---")
        
        # 曲线列表
        self._render_curve_list()
        
        st.markdown("---")
        
        # 导出选项
        self._render_export_options()
    
    def _render_curve_filters(self):
        """渲染曲线过滤选项"""
        st.markdown("**🔍 过滤选项**")
        
        # 按曲线类型过滤
        curves = state_manager.get_all_curves()
        if not curves:
            return
            
        all_types = set(curve.curve_type for curve in curves.values())
        selected_types = st.multiselect(
            "曲线类型",
            options=list(all_types),
            default=list(all_types),
            key="filter_curve_types"
        )
        
        # 按处理状态过滤（简化）
        processing_filter = st.selectbox(
            "处理状态",
            ["全部", "已处理", "有峰数据"],
            key="filter_processing"
        )
        
        # 保存过滤条件
        st.session_state.curve_filters = {
            'types': selected_types,
            'processing': processing_filter
        }
    
    def _render_curve_list(self):
        """渲染曲线列表"""
        st.markdown("**📊 选择显示的曲线**")
        
        # 获取过滤后的曲线
        filtered_curves = self._get_filtered_curves()
        
        if not filtered_curves:
            st.warning("没有符合过滤条件的曲线")
            return
        
        # 初始化选中的曲线列表
        if 'selected_curves_for_viz' not in st.session_state:
            st.session_state.selected_curves_for_viz = []
        
        # 全选/全不选按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 全选", key="select_all_curves"):
                st.session_state.selected_curves_for_viz = list(filtered_curves.keys())
                st.rerun()
        
        with col2:
            if st.button("❌ 清空", key="clear_all_curves"):
                st.session_state.selected_curves_for_viz = []
                st.rerun()
        
        # 曲线列表
        for curve_id, curve in filtered_curves.items():
            # 创建显示名称
            display_name = self._create_curve_display_name(curve)
            
            # 曲线选择框
            is_selected = curve_id in st.session_state.selected_curves_for_viz
            
            if st.checkbox(
                display_name,
                value=is_selected,
                key=f"curve_select_{curve_id}"
            ):
                if curve_id not in st.session_state.selected_curves_for_viz:
                    st.session_state.selected_curves_for_viz.append(curve_id)
            else:
                if curve_id in st.session_state.selected_curves_for_viz:
                    st.session_state.selected_curves_for_viz.remove(curve_id)
            
            # 曲线详细信息（可展开）
            if is_selected:
                with st.expander(f"📋 {curve.curve_type} 详情", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("数据点", len(curve.x_values))
                        st.metric("峰数量", len(curve.peaks))
                    
                    with col2:
                        st.metric("最大强度", f"{curve.max_intensity:.0f}")
                        st.metric("总面积", f"{curve.total_area:.2e}")
                    
                    if curve.peaks:
                        st.write("**主要峰位置:**")
                        main_peaks = sorted(curve.peaks, key=lambda p: p.intensity, reverse=True)[:3]
                        for i, peak in enumerate(main_peaks, 1):
                            st.write(f"{i}. RT {peak.rt:.2f} min (强度: {peak.intensity:.0f})")
    
    
    def _render_export_options(self):
        """渲染导出选项"""
        st.markdown("**📤 导出选项**")
        
        selected_curves = st.session_state.get('selected_curves_for_viz', [])
        
        if not selected_curves:
            st.info("请先选择曲线")
            return
        
        export_format = st.selectbox(
            "导出格式",
            ["CSV数据", "Excel报告", "JSON数据"],
            key="export_format"
        )
        
        if st.button("📥 导出选中数据", type="secondary"):
            self._export_selected_data(selected_curves, export_format)
    
    def _render_visualization_panel(self):
        """渲染右侧可视化面板"""
        selected_curves = st.session_state.get('selected_curves_for_viz', [])
        
        if not selected_curves:
            st.info("请在左侧选择要显示的曲线")
            return
        
        # 可视化选项
        viz_options = self._render_visualization_options()
        
        # 根据选项显示不同的图表
        if viz_options['view_mode'] == "叠加视图":
            self._render_overlay_plot(selected_curves, viz_options)
        elif viz_options['view_mode'] == "分离视图":
            self._render_separated_plot(selected_curves, viz_options)
        elif viz_options['view_mode'] == "对比视图":
            self._render_comparison_plot(selected_curves, viz_options)
        
        # 显示峰分析结果
        if viz_options['show_peak_analysis']:
            self._render_peak_analysis(selected_curves)
        
        # 显示详细峰数据表格
        if viz_options['show_peak_details']:
            self._render_detailed_peak_table(selected_curves)
    
    def _render_visualization_options(self):
        """渲染可视化选项"""
        st.subheader("🎨 可视化选项")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view_mode = st.selectbox(
                "视图模式",
                ["叠加视图", "分离视图", "对比视图"],
                key="viz_view_mode"
            )
        
        with col2:
            show_peaks = st.checkbox("显示峰标记", value=True, key="viz_show_peaks")
            show_legend = st.checkbox("显示图例", value=True, key="viz_show_legend")
        
        with col3:
            show_peak_analysis = st.checkbox("显示峰分析", value=True, key="viz_show_peak_analysis")
            show_peak_details = st.checkbox("显示峰详情", value=True, key="viz_show_peak_details")
        
        return {
            'view_mode': view_mode,
            'show_peaks': show_peaks,
            'show_legend': show_legend,
            'show_peak_analysis': show_peak_analysis,
            'show_peak_details': show_peak_details
        }
    
    def _render_overlay_plot(self, selected_curves: List[str], options: Dict[str, Any]):
        """渲染叠加视图"""
        st.subheader("📈 曲线叠加视图")
        
        fig = go.Figure()
        
        # 颜色循环
        colors = px.colors.qualitative.Set1
        
        for i, curve_id in enumerate(selected_curves):
            curve = state_manager.get_curve(curve_id)
            if not curve:
                continue
            color = colors[i % len(colors)]
            
            # 处理数据
            x_data = curve.x_values
            y_data = curve.y_values
            
            # 添加曲线
            display_name = self._create_curve_display_name(curve)
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=display_name,
                line=dict(color=color, width=2),
                hovertemplate='RT: %{x:.2f} min<br>强度: %{y:.0f}<extra></extra>'
            ))
            
            # 添加峰标记
            if options['show_peaks'] and curve.peaks:
                peak_x = [peak.rt for peak in curve.peaks]
                peak_y = [peak.intensity for peak in curve.peaks]
                
                
                fig.add_trace(go.Scatter(
                    x=peak_x,
                    y=peak_y,
                    mode='markers',
                    name=f'{display_name} - 峰',
                    marker=dict(
                        color=color,
                        size=8,
                        symbol='triangle-up',
                        line=dict(color='white', width=1)
                    ),
                    showlegend=False,
                    hovertemplate='峰 RT: %{x:.2f} min<br>强度: %{y:.0f}<extra></extra>'
                ))
        
        fig.update_layout(
            title="曲线叠加视图",
            xaxis_title="保留时间 (分钟)",
            yaxis_title="强度",
            height=600,
            hovermode='x unified',
            showlegend=options['show_legend']
        )
        
        st.plotly_chart(fig, width='stretch')
    
    def _render_separated_plot(self, selected_curves: List[str], options: Dict[str, Any]):
        """渲染分离视图"""
        st.subheader("📊 曲线分离视图")
        
        n_curves = len(selected_curves)
        
        # 创建子图
        fig = make_subplots(
            rows=n_curves, cols=1,
            subplot_titles=[
                self._create_curve_display_name(state_manager.get_curve(cid)) 
                for cid in selected_curves if state_manager.get_curve(cid)
            ],
            vertical_spacing=0.02,
            shared_xaxes=True
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, curve_id in enumerate(selected_curves):
            curve = state_manager.get_curve(curve_id)
            if not curve:
                continue
            color = colors[i % len(colors)]
            
            # 处理数据
            x_data = curve.x_values
            y_data = curve.y_values
            
            # 添加曲线
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=f'曲线 {i+1}',
                    line=dict(color=color, width=2),
                    hovertemplate='RT: %{x:.2f} min<br>强度: %{y:.0f}<extra></extra>'
                ),
                row=i+1, col=1
            )
            
            # 添加峰标记
            if options['show_peaks'] and curve.peaks:
                peak_x = [peak.rt for peak in curve.peaks]
                peak_y = [peak.intensity for peak in curve.peaks]
                
                
                fig.add_trace(
                    go.Scatter(
                        x=peak_x,
                        y=peak_y,
                        mode='markers',
                        name=f'峰 {i+1}',
                        marker=dict(
                            color=color,
                            size=6,
                            symbol='triangle-up'
                        ),
                        showlegend=False,
                        hovertemplate='峰 RT: %{x:.2f} min<br>强度: %{y:.0f}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title="曲线分离视图",
            height=200 * n_curves,
            showlegend=options['show_legend']
        )
        
        fig.update_xaxes(title_text="保留时间 (分钟)", row=n_curves, col=1)
        
        for i in range(n_curves):
            fig.update_yaxes(
                title_text="强度" + (" (归一化)" if options['normalize_curves'] else ""),
                row=i+1, col=1
            )
        
        st.plotly_chart(fig, width='stretch')
    
    def _render_comparison_plot(self, selected_curves: List[str], options: Dict[str, Any]):
        """渲染对比视图"""
        st.subheader("⚖️ 曲线对比视图")
        
        if len(selected_curves) < 2:
            st.warning("对比视图需要至少选择2条曲线")
            return
        
        # 选择对比的两条曲线
        col1, col2 = st.columns(2)
        
        with col1:
            curve1_id = st.selectbox(
                "曲线 1",
                options=selected_curves,
                format_func=lambda x: self._create_curve_display_name(state_manager.get_curve(x)),
                key="compare_curve1"
            )
        
        with col2:
            curve2_id = st.selectbox(
                "曲线 2", 
                options=selected_curves,
                format_func=lambda x: self._create_curve_display_name(state_manager.get_curve(x)),
                key="compare_curve2"
            )
        
        if curve1_id == curve2_id:
            st.warning("请选择不同的曲线进行对比")
            return
        
        curve1 = state_manager.get_curve(curve1_id)
        curve2 = state_manager.get_curve(curve2_id)
        
        # 创建对比图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('曲线对比', '差值分析'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # 曲线1
        fig.add_trace(
            go.Scatter(
                x=curve1.x_values,
                y=curve1.y_values,
                mode='lines',
                name=self._create_curve_display_name(curve1),
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # 曲线2
        fig.add_trace(
            go.Scatter(
                x=curve2.x_values,
                y=curve2.y_values,
                mode='lines',
                name=self._create_curve_display_name(curve2),
                line=dict(color='#ff7f0e', width=2)
            ),
            row=1, col=1
        )
        
        # 计算差值（如果x轴相同）
        if len(curve1.x_values) == len(curve2.x_values) and np.allclose(curve1.x_values, curve2.x_values):
            diff_y = curve1.y_values - curve2.y_values
            
            fig.add_trace(
                go.Scatter(
                    x=curve1.x_values,
                    y=diff_y,
                    mode='lines',
                    name='差值 (曲线1 - 曲线2)',
                    line=dict(color='#2ca02c', width=1)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="曲线对比分析"
        )
        
        fig.update_xaxes(title_text="保留时间 (分钟)", row=2, col=1)
        fig.update_yaxes(title_text="强度", row=1, col=1)
        fig.update_yaxes(title_text="强度差值", row=2, col=1)
        
        st.plotly_chart(fig, width='stretch')
        
        # 显示对比统计
        self._show_comparison_stats(curve1, curve2)
    
    def _render_peak_analysis(self, selected_curves: List[str]):
        """渲染峰分析结果"""
        st.subheader("🔍 峰分析结果")
        
        all_peaks = []
        for curve_id in selected_curves:
            curve = state_manager.get_curve(curve_id)
            if curve:
                for peak in curve.peaks:
                    peak_data = {
                        '曲线': self._create_curve_display_name(curve),
                        '峰ID': peak.peak_id,
                        'RT (min)': f"{peak.rt:.2f}",
                        '强度': f"{peak.intensity:.0f}",
                        '面积': f"{peak.area:.2e}",
                        'FWHM': f"{peak.fwhm:.3f}",
                        '信噪比': f"{peak.signal_to_noise:.1f}",
                        '置信度': f"{peak.confidence:.2f}"
                    }
                    all_peaks.append(peak_data)
        
        if not all_peaks:
            st.info("没有检测到峰，请先进行峰检测")
            return
        
        # 峰数据表格
        peaks_df = pd.DataFrame(all_peaks)
        st.dataframe(peaks_df, width='stretch')
        
        # 峰统计
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总峰数", len(all_peaks))
        
        with col2:
            avg_intensity = np.mean([float(p['强度']) for p in all_peaks])
            st.metric("平均强度", f"{avg_intensity:.0f}")
        
        with col3:
            avg_fwhm = np.mean([float(p['FWHM']) for p in all_peaks])
            st.metric("平均FWHM", f"{avg_fwhm:.3f}")
    
    def _get_filtered_curves(self) -> Dict[str, Curve]:
        """获取过滤后的曲线"""
        filters = st.session_state.get('curve_filters', {})
        filtered = {}
        
        curves = state_manager.get_all_curves()
        for curve_id, curve in curves.items():
            # 类型过滤
            if filters.get('types') and curve.curve_type not in filters['types']:
                continue
            
            # 处理状态过滤
            processing_filter = filters.get('processing', '全部')
            if processing_filter == '已处理' and not (curve.is_baseline_corrected or curve.is_smoothed):
                continue
            elif processing_filter == '有峰数据' and not curve.peaks:
                continue
            
            filtered[curve_id] = curve
        
        return filtered
    
    def _get_global_processing_params(self) -> Dict[str, Dict[str, Any]]:
        """获取全局处理参数状态"""
        params = {
            'baseline': {'available': False, 'method': 'N/A', 'params': {}},
            'smoothing': {'available': False, 'method': 'N/A', 'params': {}},
            'peak_detection': {'available': False, 'method': 'N/A', 'params': {}},
            'peak_analysis': {'available': False, 'baseline_method': 'N/A', 'boundary_method': 'N/A', 'params': {}},
            'peak_fitting': {'available': False, 'model': 'N/A', 'params': {}}
        }
        
        # 从session_state中查找参数
        # 1. 基线校正参数
        for key in st.session_state:
            if key.startswith('baseline_') and not key.startswith('batch_'):
                if 'method' in key and st.session_state[key]:
                    params['baseline']['available'] = True
                    params['baseline']['method'] = st.session_state[key]
                elif key.endswith('_degree') or key.endswith('_lam') or key.endswith('_p'):
                    params['baseline']['params'][key.replace('baseline_', '')] = st.session_state[key]
        
        # 2. 平滑处理参数
        for key in st.session_state:
            if key.startswith('smooth_') and not key.startswith('batch_'):
                if 'method' in key and st.session_state[key]:
                    params['smoothing']['available'] = True
                    params['smoothing']['method'] = st.session_state[key]
        else:
                    param_name = key.replace('smooth_', '').replace('smoothing_', '')
                    params['smoothing']['params'][param_name] = st.session_state[key]
        
        # 3. 峰检测参数
        for key in st.session_state:
            if key.startswith('peak_detect_') and not key.startswith('batch_'):
                if 'method' in key and st.session_state[key]:
                    params['peak_detection']['available'] = True
                    params['peak_detection']['method'] = st.session_state[key]
                else:
                    param_name = key.replace('peak_detect_', '').replace('detection_', '')
                    params['peak_detection']['params'][param_name] = st.session_state[key]
        
        # 4. 峰分析参数
        for key in st.session_state:
            if key.startswith('analysis_') and not key.startswith('batch_'):
                if 'baseline' in key and st.session_state[key]:
                    params['peak_analysis']['available'] = True
                    params['peak_analysis']['baseline_method'] = st.session_state[key]
                elif 'boundary' in key and st.session_state[key]:
                    params['peak_analysis']['boundary_method'] = st.session_state[key]
                else:
                    param_name = key.replace('analysis_', '')
                    params['peak_analysis']['params'][param_name] = st.session_state[key]
        
        # 5. 峰拟合参数
        for key in st.session_state:
            if key.startswith('fitting_') and not key.startswith('batch_'):
                if 'model' in key and st.session_state[key]:
                    params['peak_fitting']['available'] = True
                    params['peak_fitting']['model'] = st.session_state[key]
                else:
                    param_name = key.replace('fitting_', '')
                    params['peak_fitting']['params'][param_name] = st.session_state[key]
        
        # 从已处理的曲线中推断参数
        curves = state_manager.get_all_curves()
        for curve in curves.values():
            # 基线校正
            if curve.is_baseline_corrected and not params['baseline']['available']:
                params['baseline']['available'] = True
                params['baseline']['method'] = getattr(curve, 'baseline_method', '线性')
            
            # 平滑处理
            if curve.is_smoothed and not params['smoothing']['available']:
                params['smoothing']['available'] = True
                params['smoothing']['method'] = getattr(curve, 'smoothing_method', '移动平均')
                params['smoothing']['params'] = getattr(curve, 'smoothing_params', {})
            
            # 峰检测和分析
        if curve.peaks:
                if not params['peak_detection']['available']:
                    params['peak_detection']['available'] = True
                    params['peak_detection']['method'] = 'scipy_find_peaks'  # 默认方法
                
                # 检查峰分析参数
                for peak in curve.peaks:
                    if hasattr(peak, 'area') and peak.area > 0:
                        params['peak_analysis']['available'] = True
                        params['peak_analysis']['baseline_method'] = '线性基线'
                        params['peak_analysis']['boundary_method'] = '自动选择'
                        break
                    
                    if 'fit_result' in peak.metadata:
                        params['peak_fitting']['available'] = True
                        params['peak_fitting']['model'] = peak.metadata.get('fit_model', 'gaussian')
                        break
        
        return params
    
    def _create_curve_display_name(self, curve: Curve) -> str:
        """创建曲线显示名称 - 使用文件名+提取方式"""
        # 基础名称：提取方式
        name = f"{curve.curve_type}"
        
        # 获取文件名（优先级：config_name > original_filename > curve_id）
        filename = ""
        if 'config_name' in curve.metadata and curve.metadata['config_name']:
            filename = curve.metadata['config_name']
        elif 'original_filename' in curve.metadata and curve.metadata['original_filename']:
            full_filename = curve.metadata['original_filename']
            # 只显示文件名，不显示路径
            if '\\' in full_filename:
                filename = full_filename.split('\\')[-1]
            elif '/' in full_filename:
                filename = full_filename.split('/')[-1]
            else:
                filename = full_filename
            
            # 移除文件扩展名
            if '.' in filename:
                filename = '.'.join(filename.split('.')[:-1])
        else:
            # 备用：使用curve_id的前8位
            filename = f"ID_{curve.curve_id[:8]}"
        
        # 截断过长的文件名
        if len(filename) > 20:
            filename = filename[:17] + "..."
        
        # 组合名称：文件名_提取方式
        if filename:
            name = f"{filename}_{curve.curve_type}"
        
        # 添加处理状态标识
        status_indicators = []
        if curve.is_baseline_corrected:
            status_indicators.append("B")
        if curve.is_smoothed:
            status_indicators.append("S")
        if curve.peaks:
            status_indicators.append(f"P{len(curve.peaks)}")
        
        if status_indicators:
            name += f" [{'/'.join(status_indicators)}]"
        
        return name
    
    
    def _show_comparison_stats(self, curve1: Curve, curve2: Curve):
        """显示对比统计"""
        st.subheader("📊 对比统计")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max1, max2 = curve1.max_intensity, curve2.max_intensity
            max_ratio = max2 / max1 if max1 > 0 else 0
            st.metric("最大强度比", f"{max_ratio:.2f}", 
                     delta=f"差值: {max2 - max1:.0f}")
        
        with col2:
            area1, area2 = curve1.total_area, curve2.total_area
            area_ratio = area2 / area1 if area1 > 0 else 0
            st.metric("总面积比", f"{area_ratio:.2f}",
                     delta=f"差值: {area2 - area1:.2e}")
        
        with col3:
            peaks1, peaks2 = len(curve1.peaks), len(curve2.peaks)
            st.metric("峰数量比", f"{peaks2}/{peaks1}",
                     delta=f"差值: {peaks2 - peaks1}")
        
        with col4:
            rt1 = curve1.x_range[1] - curve1.x_range[0]
            rt2 = curve2.x_range[1] - curve2.x_range[0]
            rt_ratio = rt2 / rt1 if rt1 > 0 else 0
            st.metric("RT范围比", f"{rt_ratio:.2f}",
                     delta=f"差值: {rt2 - rt1:.2f} min")
    
    def _export_selected_data(self, curve_ids: List[str], format_type: str):
        """导出选中的数据"""
        try:
            from ...export.report_generator import ReportGenerator
            
            curves = [state_manager.get_curve(cid) for cid in curve_ids 
                     if state_manager.get_curve(cid)]
            
            if not curves:
                st.error("没有可导出的曲线")
                return
            
            report_gen = ReportGenerator()
            
            export_options = {
                'include_peaks': True,
                'include_metadata': True,
                'include_plots': True,
                'include_statistics': True
            }
            
            if format_type == "CSV数据":
                file_path = report_gen.export_to_csv(curves, export_options)
                mime_type = 'text/csv'
            elif format_type == "Excel报告":
                file_path = report_gen.export_to_excel(curves, export_options)
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif format_type == "JSON数据":
                file_path = report_gen.export_to_json(curves, export_options)
                mime_type = 'application/json'
            else:
                st.error("不支持的导出格式")
                return
            
            st.success(f"✅ 导出完成: {file_path}")
            
            # 提供下载
            with open(file_path, 'rb') as f:
                file_content = f.read()
                filename = file_path.split('\\')[-1] if '\\' in file_path else file_path.split('/')[-1]
                st.download_button(
                    label=f"📥 下载 {format_type}",
                    data=file_content,
                    file_name=filename,
                    mime=mime_type
                )
        
        except Exception as e:
            st.error(f"导出失败: {str(e)}")
    
    def _render_detailed_peak_table(self, selected_curves: List[str]):
        """渲染详细的峰数据表格"""
        st.subheader("📋 详细峰数据表格")
        
        all_peak_data = []
        for curve_id in selected_curves:
            curve = state_manager.get_curve(curve_id)
            if not curve or not curve.peaks:
                continue
            
            curve_name = self._create_curve_display_name(curve)
            
            for i, peak in enumerate(curve.peaks):
                peak_data = {
                    '曲线': curve_name,
                    '峰序号': i+1,
                    'RT (min)': f"{peak.rt:.3f}",
                    '强度': f"{peak.intensity:.0f}",
                    '面积': f"{peak.area:.2e}" if hasattr(peak, 'area') and peak.area > 1000 else f"{getattr(peak, 'area', 0):.0f}",
                    'FWHM (min)': f"{peak.fwhm:.3f}" if hasattr(peak, 'fwhm') else "N/A",
                    '信噪比': f"{peak.signal_to_noise:.1f}" if hasattr(peak, 'signal_to_noise') else "N/A"
                }
                
                # 添加色谱质量参数（如果存在）
                if hasattr(peak, 'theoretical_plates') and peak.theoretical_plates is not None:
                    peak_data['理论塔板数'] = f"{peak.theoretical_plates:.0f}"
                if hasattr(peak, 'asymmetry_factor') and peak.asymmetry_factor is not None:
                    peak_data['不对称因子'] = f"{peak.asymmetry_factor:.3f}"
                if hasattr(peak, 'tailing_factor') and peak.tailing_factor is not None:
                    peak_data['拖尾因子'] = f"{peak.tailing_factor:.3f}"
                if hasattr(peak, 'resolution') and peak.resolution is not None:
                    peak_data['分离度'] = f"{peak.resolution:.1f}"
                
                # 添加拟合信息（如果存在）
                if 'fit_result' in peak.metadata:
                    fit_result = peak.metadata['fit_result']
                    if fit_result.get('success', False):
                        peak_data['拟合模型'] = peak.metadata.get('fit_model', 'N/A')
                        peak_data['拟合R²'] = f"{fit_result.get('r_squared', 0):.4f}"
                    else:
                        peak_data['拟合模型'] = "失败"
                        peak_data['拟合R²'] = "N/A"
                
                all_peak_data.append(peak_data)
        
        if not all_peak_data:
            st.info("没有峰数据可显示")
            return
        
        # 显示数据表格
        df = pd.DataFrame(all_peak_data)
        st.dataframe(df, width='stretch', height=min(600, len(all_peak_data) * 35 + 50))
        
        # 显示统计汇总
        st.markdown("### 📊 数据汇总")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_peaks = len(all_peak_data)
            st.metric("总峰数", total_peaks)
        
        with col2:
            if total_peaks > 0:
                areas = [float(p['面积']) for p in all_peak_data if p['面积'] != 'N/A']
                total_area = sum(areas) if areas else 0
                st.metric("总面积", f"{total_area:.2e}")
        
        with col3:
            if total_peaks > 0:
                intensities = [float(p['强度']) for p in all_peak_data]
                avg_intensity = np.mean(intensities)
                st.metric("平均强度", f"{avg_intensity:.0f}")
        
        with col4:
            curves_with_peaks = len(set(p['曲线'] for p in all_peak_data))
            st.metric("含峰曲线数", curves_with_peaks)
    
    def _render_batch_processing_panel(self):
        """渲染批量处理面板"""
        st.subheader("⚙️ 批量处理中心")
        st.markdown("对多条曲线进行统一参数配置和批量处理")
        
        # 左右分栏：左侧选择曲线，右侧配置参数
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_batch_curve_selection()
        
        with col2:
            self._render_batch_processing_config()
    
    def _render_batch_curve_selection(self):
        """渲染批量处理的曲线选择"""
        st.markdown("### 📋 选择处理曲线")
        
        # 获取所有曲线
        curves = state_manager.get_all_curves()
        if not curves:
            st.warning("没有可用的曲线")
            return
        
        # 按状态分组显示
        raw_curves = {}
        processed_curves = {}
        
        for curve_id, curve in curves.items():
            if not (curve.is_baseline_corrected or curve.is_smoothed or curve.peaks):
                raw_curves[curve_id] = curve
            else:
                processed_curves[curve_id] = curve
        
        # 初始化批量选择状态
        if 'batch_selected_curves' not in st.session_state:
            st.session_state.batch_selected_curves = []
        
        # 快速选择按钮
        st.markdown("**🎯 快速选择**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 全部曲线", key="batch_select_all"):
                st.session_state.batch_selected_curves = list(curves.keys())
                st.rerun()
        
        with col2:
            if st.button("🔄 原始曲线", key="batch_select_raw"):
                st.session_state.batch_selected_curves = list(raw_curves.keys())
                st.rerun()
        
        with col3:
            if st.button("❌ 清空选择", key="batch_clear_all"):
                st.session_state.batch_selected_curves = []
                st.rerun()
        
        st.markdown("---")
        
        # 原始曲线列表
        if raw_curves:
            st.markdown("**📈 原始曲线**")
            for curve_id, curve in raw_curves.items():
                display_name = self._create_curve_display_name(curve)
                is_selected = curve_id in st.session_state.batch_selected_curves
                
                if st.checkbox(display_name, value=is_selected, key=f"batch_raw_{curve_id}"):
                    if curve_id not in st.session_state.batch_selected_curves:
                        st.session_state.batch_selected_curves.append(curve_id)
                else:
                    if curve_id in st.session_state.batch_selected_curves:
                        st.session_state.batch_selected_curves.remove(curve_id)
        
        # 已处理曲线列表
        if processed_curves:
            st.markdown("**🔧 已处理曲线**")
            for curve_id, curve in processed_curves.items():
                display_name = self._create_curve_display_name(curve)
                is_selected = curve_id in st.session_state.batch_selected_curves
                
                if st.checkbox(display_name, value=is_selected, key=f"batch_processed_{curve_id}"):
                    if curve_id not in st.session_state.batch_selected_curves:
                        st.session_state.batch_selected_curves.append(curve_id)
                else:
                    if curve_id in st.session_state.batch_selected_curves:
                        st.session_state.batch_selected_curves.remove(curve_id)
        
        # 显示选择统计
        selected_count = len(st.session_state.batch_selected_curves)
        st.info(f"已选择 {selected_count} 条曲线进行批量处理")
    
    def _render_batch_processing_config(self):
        """渲染批量处理配置"""
        st.markdown("### ⚙️ 批量处理配置")
        
        selected_curves = st.session_state.get('batch_selected_curves', [])
        if not selected_curves:
            st.warning("请先在左侧选择要处理的曲线")
            return
        
        # 获取全局参数状态
        global_params = self._get_global_processing_params()
        
        st.info("💡 **智能参数复用**：批量处理将自动使用您在前面步骤中设置的参数，确保处理一致性")
        
        # 处理步骤选择
        st.markdown("**📋 处理步骤选择**")
        
        # 使用更简洁的选择方式
        col1, col2 = st.columns(2)
        
        with col1:
            enable_baseline = st.checkbox(
                f"🔧 基线校正 {'✓' if global_params['baseline']['available'] else '⚠️'}", 
                value=global_params['baseline']['available'],
                help=f"当前方法: {global_params['baseline']['method']}" if global_params['baseline']['available'] else "未检测到基线校正参数",
                key="batch_enable_baseline"
            )
            
            enable_smoothing = st.checkbox(
                f"🌊 平滑处理 {'✓' if global_params['smoothing']['available'] else '⚠️'}", 
                value=global_params['smoothing']['available'],
                help=f"当前方法: {global_params['smoothing']['method']}" if global_params['smoothing']['available'] else "未检测到平滑参数",
                key="batch_enable_smoothing"
            )
            
            enable_peak_detection = st.checkbox(
                f"🔍 峰检测 {'✓' if global_params['peak_detection']['available'] else '⚠️'}", 
                value=global_params['peak_detection']['available'],
                help=f"当前方法: {global_params['peak_detection']['method']}" if global_params['peak_detection']['available'] else "未检测到峰检测参数",
                key="batch_enable_peak_detection"
            )
        
        with col2:
            enable_peak_analysis = st.checkbox(
                f"📊 峰分析 {'✓' if global_params['peak_analysis']['available'] else '⚠️'}", 
                value=global_params['peak_analysis']['available'],
                help=f"当前基线: {global_params['peak_analysis']['baseline_method']}" if global_params['peak_analysis']['available'] else "未检测到峰分析参数",
                key="batch_enable_peak_analysis"
            )
            
            enable_peak_fitting = st.checkbox(
                f"📈 峰拟合 {'✓' if global_params['peak_fitting']['available'] else '⚠️'}", 
                value=global_params['peak_fitting']['available'],
                help=f"当前模型: {global_params['peak_fitting']['model']}" if global_params['peak_fitting']['available'] else "未检测到峰拟合参数",
                key="batch_enable_peak_fitting"
            )
        
        # 显示参数详情
        if any([enable_baseline, enable_smoothing, enable_peak_detection, enable_peak_analysis, enable_peak_fitting]):
            st.markdown("---")
            st.markdown("**🔧 当前参数配置预览**")
            
            with st.expander("📋 查看详细参数", expanded=False):
                if enable_baseline and global_params['baseline']['available']:
                    st.write(f"**基线校正**: {global_params['baseline']['method']}")
                    if global_params['baseline']['params']:
                        st.json(global_params['baseline']['params'])
                
                if enable_smoothing and global_params['smoothing']['available']:
                    st.write(f"**平滑处理**: {global_params['smoothing']['method']}")
                    if global_params['smoothing']['params']:
                        st.json(global_params['smoothing']['params'])
                
                if enable_peak_detection and global_params['peak_detection']['available']:
                    st.write(f"**峰检测**: {global_params['peak_detection']['method']}")
                    if global_params['peak_detection']['params']:
                        st.json(global_params['peak_detection']['params'])
                
                if enable_peak_analysis and global_params['peak_analysis']['available']:
                    st.write(f"**峰分析**: 基线={global_params['peak_analysis']['baseline_method']}, 边界={global_params['peak_analysis']['boundary_method']}")
                    if global_params['peak_analysis']['params']:
                        st.json(global_params['peak_analysis']['params'])
                
                if enable_peak_fitting and global_params['peak_fitting']['available']:
                    st.write(f"**峰拟合**: {global_params['peak_fitting']['model']}")
                    if global_params['peak_fitting']['params']:
                        st.json(global_params['peak_fitting']['params'])
        
        # 存储批量处理配置
        st.session_state.batch_config = {
            'enable_baseline': enable_baseline,
            'enable_smoothing': enable_smoothing,
            'enable_peak_detection': enable_peak_detection,
            'enable_peak_analysis': enable_peak_analysis,
            'enable_peak_fitting': enable_peak_fitting,
            'global_params': global_params
        }
        
        # 执行批量处理
        st.markdown("---")
        st.markdown("### 🚀 执行批量处理")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ 开始批量处理", type="primary", key="start_batch_processing"):
                batch_config = st.session_state.get('batch_config', {})
                self._execute_batch_processing(selected_curves, batch_config)
        
        with col2:
            if st.button("📊 预览处理配置", key="preview_batch_config"):
                batch_config = st.session_state.get('batch_config', {})
                self._preview_batch_config(selected_curves, batch_config)
    
    def _execute_batch_processing(self, selected_curves: List[str], config: Dict[str, Any]):
        """执行批量处理 - 使用全局参数"""
        if not selected_curves:
            st.error("请先选择要处理的曲线")
            return
        
        global_params = config.get('global_params', {})
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        enabled_steps = [
            config.get('enable_baseline', False),
            config.get('enable_smoothing', False), 
            config.get('enable_peak_detection', False),
            config.get('enable_peak_analysis', False),
            config.get('enable_peak_fitting', False)
        ]
        
        total_steps = len(selected_curves) * sum(enabled_steps)
        
        if total_steps == 0:
            st.warning("请至少启用一个处理步骤")
            return
        
        current_step = 0
        results = {"成功": 0, "失败": 0, "跳过": 0}
        
        # 处理每条曲线
        for i, curve_id in enumerate(selected_curves):
            curve = state_manager.get_curve(curve_id)
            if not curve:
                results["跳过"] += 1
                continue
            
            curve_name = self._create_curve_display_name(curve)
            status_text.text(f"正在处理: {curve_name}")
            
            try:
                # 1. 基线校正
                if config.get('enable_baseline', False):
                    status_text.text(f"基线校正: {curve_name}")
                    success = self._batch_apply_baseline_correction(curve, global_params['baseline'])
                    results["成功" if success else "失败"] += 1
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                # 2. 平滑处理
                if config.get('enable_smoothing', False):
                    status_text.text(f"平滑处理: {curve_name}")
                    success = self._batch_apply_smoothing(curve, global_params['smoothing'])
                    results["成功" if success else "失败"] += 1
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                # 3. 峰检测
                if config.get('enable_peak_detection', False):
                    status_text.text(f"峰检测: {curve_name}")
                    success = self._batch_apply_peak_detection(curve, global_params['peak_detection'])
                    results["成功" if success else "失败"] += 1
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                # 4. 峰分析
                if config.get('enable_peak_analysis', False):
                    status_text.text(f"峰分析: {curve_name}")
                    success = self._batch_apply_peak_analysis(curve, global_params['peak_analysis'])
                    results["成功" if success else "失败"] += 1
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                # 5. 峰拟合
                if config.get('enable_peak_fitting', False):
                    status_text.text(f"峰拟合: {curve_name}")
                    success = self._batch_apply_peak_fitting(curve, global_params['peak_fitting'])
                    results["成功" if success else "失败"] += 1
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
            except Exception as e:
                st.error(f"处理曲线 {curve_name} 时出错: {str(e)}")
                results["失败"] += 1
        
        # 完成处理
        progress_bar.progress(1.0)
        status_text.text("批量处理完成！")
        
        # 显示结果统计
        st.success(f"🎉 批量处理完成！")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ 成功", results["成功"])
        with col2:
            st.metric("❌ 失败", results["失败"])
        with col3:
            st.metric("⏭️ 跳过", results["跳过"])
    
    def _preview_batch_config(self, selected_curves: List[str], config: Dict[str, Any]):
        """预览批量处理配置"""
        st.markdown("### 📋 批量处理配置预览")
        
        if not selected_curves:
            st.warning("请先选择要处理的曲线")
            return
        
        # 显示选中的曲线
        st.markdown(f"**📊 选中曲线数量: {len(selected_curves)}**")
        
        # 显示处理步骤
        enabled_steps = []
        if config.get('enable_baseline', False):
            enabled_steps.append(f"🔧 基线校正 ({config.get('baseline_method', 'N/A')})")
        if config.get('enable_smoothing', False):
            enabled_steps.append(f"🌊 平滑处理 ({config.get('smoothing_method', 'N/A')})")
        if config.get('enable_peak_detection', False):
            enabled_steps.append(f"🔍 峰检测 ({config.get('detection_method', 'N/A')})")
        if config.get('enable_peak_analysis', False):
            enabled_steps.append(f"📊 峰分析 ({config.get('analysis_params', {}).get('boundary_method', 'N/A')})")
        if config.get('enable_peak_fitting', False):
            enabled_steps.append(f"📈 峰拟合 ({config.get('fitting_params', {}).get('model', 'N/A')})")
        
        if enabled_steps:
            st.markdown("**📋 启用的处理步骤:**")
            for step in enabled_steps:
                st.write(f"• {step}")
        else:
            st.warning("没有启用任何处理步骤")
        
        # 估算处理时间
        total_operations = len(selected_curves) * len(enabled_steps)
        estimated_time = total_operations * 2  # 每个操作估算2秒
        st.info(f"⏱️ 预计处理时间: {estimated_time} 秒 ({total_operations} 个操作)")
    
    # 批量处理的具体实现方法
    def _batch_apply_baseline_correction(self, curve: Curve, params: Dict[str, Any]) -> bool:
        """批量应用基线校正"""
        try:
            from ui.pages.processing.baseline_correction import BaselineCorrectionProcessor
            processor = BaselineCorrectionProcessor()
            
            method = params.get('method', '线性')
            method_params = params.get('params', {})
            
            # 直接应用处理（跳过确认步骤）
            corrected_y = processor.methods[method](curve.y_values, method_params)
            
            # 更新曲线数据
            curve.y_values = corrected_y
            curve.is_baseline_corrected = True
            
            # 更新存储数据
            stored_curve = state_manager.get_curve(curve.curve_id)
            stored_curve.y_values = corrected_y.copy()
            stored_curve.is_baseline_corrected = True
            state_manager.update_curve(stored_curve)
            
            return True
        except Exception as e:
            print(f"基线校正失败: {e}")
            return False
    
    def _batch_apply_smoothing(self, curve: Curve, params: Dict[str, Any]) -> bool:
        """批量应用平滑处理"""
        try:
            from ui.pages.processing.smoothing import SmoothingProcessor
            processor = SmoothingProcessor()
            
            method = params.get('method', '移动平均')
            method_params = params.get('params', {})
            
            # 保存原始数据
            if not hasattr(curve, '_original_y_values') or curve._original_y_values is None:
                curve._original_y_values = curve.y_values.copy()
            
            # 直接应用处理
            smoothed_y = processor.methods[method](curve._original_y_values, method_params)
            
            # 更新曲线数据
            curve.y_values = smoothed_y
            curve.is_smoothed = True
            curve.smoothing_method = method
            curve.smoothing_params = method_params
            
            # 更新存储数据
            stored_curve = state_manager.get_curve(curve.curve_id)
            stored_curve.y_values = smoothed_y.copy()
            stored_curve.is_smoothed = True
            stored_curve._original_y_values = curve._original_y_values.copy()
            stored_curve.smoothing_method = method
            stored_curve.smoothing_params = method_params
            state_manager.update_curve(stored_curve)
            
            return True
        except Exception as e:
            print(f"平滑处理失败: {e}")
            return False
    
    def _batch_apply_peak_detection(self, curve: Curve, params: Dict[str, Any]) -> bool:
        """批量应用峰检测"""
        try:
            method = params.get('method', 'scipy_find_peaks')
            method_params = params.get('params', {})
            
            # 执行峰检测
            detected_peaks = self.peak_detector.detect_peaks(
                curve=curve,
                method=method,
                **method_params
            )
            
            # 更新曲线数据
            curve.peaks = detected_peaks
            
            # 更新存储数据
            import copy
            stored_curve = state_manager.get_curve(curve.curve_id)
            stored_curve.peaks = copy.deepcopy(detected_peaks)
            state_manager.update_curve(stored_curve)
            
            return True
        except Exception as e:
            print(f"峰检测失败: {e}")
            return False
    
    def _batch_apply_peak_analysis(self, curve: Curve, params: Dict[str, Any]) -> bool:
        """批量应用峰分析"""
        try:
            from peak_analysis.peak_analyzer import PeakAnalyzer
            analyzer = PeakAnalyzer()
            
            if not curve.peaks:
                return False
            
            # 构建分析参数
            analysis_params = {
                'extend_range': 2.0,
                'baseline_method': params.get('baseline_method', '线性基线'),
                'boundary_method': params.get('boundary_method', '自动选择'),
                'peak_sensitivity': 5,
                'noise_tolerance': 5,
                'boundary_smoothing': True,
                'calc_theoretical_plates': True,
                'calc_tailing_factor': True,
                'calc_asymmetry_factor': True,
                'calc_resolution': True,
                'calc_capacity_factor': False,
                'calc_selectivity': False
            }
            analysis_params.update(params.get('params', {}))
            
            # 分析每个峰
            analyzed_peaks = []
            for peak in curve.peaks:
                updated_peak = analyzer.analyze_peak(
                    curve=curve,
                    peak=peak,
                    **analysis_params
                )
                analyzed_peaks.append(updated_peak)
            
            # 更新曲线数据
            curve.peaks = analyzed_peaks
            
            # 更新存储数据
            import copy
            stored_curve = state_manager.get_curve(curve.curve_id)
            stored_curve.peaks = copy.deepcopy(analyzed_peaks)
            state_manager.update_curve(stored_curve)
            
            return True
        except Exception as e:
            print(f"峰分析失败: {e}")
            return False
    
    def _batch_apply_peak_fitting(self, curve: Curve, params: Dict[str, Any]) -> bool:
        """批量应用峰拟合"""
        try:
            from peak_analysis.peak_fitter import PeakFitter
            fitter = PeakFitter()
            
            if not curve.peaks:
                return False
            
            model = params.get('model', 'gaussian')
            extend_range = params.get('params', {}).get('extend_range', 3.0)
            
            # 拟合每个峰
            for peak in curve.peaks:
                fit_result = fitter.fit_peak(
                    curve=curve,
                    peak=peak,
                    model=model,
                    extend_range=extend_range
                )
                
                # 存储拟合结果
                peak.metadata['fit_result'] = fit_result
                peak.metadata['fit_model'] = model
            
            # 更新存储数据
            import copy
            stored_curve = state_manager.get_curve(curve.curve_id)
            stored_curve.peaks = copy.deepcopy(curve.peaks)
            state_manager.update_curve(stored_curve)
            
            return True
        except Exception as e:
            print(f"峰拟合失败: {e}")
            return False
