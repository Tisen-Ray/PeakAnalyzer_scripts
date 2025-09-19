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
        
        # 主要布局：左侧曲线列表，右侧图表展示
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_curve_list_panel()
        
        with col2:
            self._render_visualization_panel()
    
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
    
    def _create_curve_display_name(self, curve: Curve) -> str:
        """创建曲线显示名称"""
        name = f"{curve.curve_type}"
        
        # 优先显示配置名称，然后是文件名
        if 'config_name' in curve.metadata and curve.metadata['config_name']:
            name += f" - {curve.metadata['config_name']}"
        elif 'original_filename' in curve.metadata and curve.metadata['original_filename']:
            filename = curve.metadata['original_filename']
            # 只显示文件名，不显示路径
            if '\\' in filename:
                filename = filename.split('\\')[-1]
            if '/' in filename:
                filename = filename.split('/')[-1]
            # 移除文件扩展名
            if '.' in filename:
                filename = '.'.join(filename.split('.')[:-1])
            # 截断长文件名
            if len(filename) > 15:
                filename = filename[:12] + "..."
            name += f" - {filename}"
        else:
            name += f" - ID:{curve.curve_id[:8]}"
        
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
