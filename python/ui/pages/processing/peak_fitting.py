"""
峰拟合处理模块
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any
from core.curve import Curve
from core.state_manager import state_manager
from peak_analysis.peak_fitter import PeakFitter


class PeakFittingProcessor:
    """峰拟合处理器"""
    
    def __init__(self):
        self.peak_fitter = PeakFitter()
        # 使用正确的模型名称
        self.model_mapping = {
            '高斯': 'gaussian',
            '洛伦兹': 'lorentzian', 
            'Voigt': 'voigt',
            '指数修正高斯': 'exponential_gaussian',
            '偏斜高斯': 'skewed_gaussian'
        }
        self.models = list(self.model_mapping.keys())
    
    def render_peak_fitting(self, curve: Curve) -> bool:
        """渲染峰拟合界面并执行处理"""
        st.markdown("### 📈 峰拟合")
        
        if not curve or not curve.peaks:
            st.warning("请先进行峰分析")
            return False
        
        # 拟合参数 - 垂直布局
        model = st.selectbox(
            "拟合模型",
            options=self.models,
            key="fitting_model"
        )
        
        extend_range = st.slider(
            "扩展范围倍数", 
            min_value=1.0, 
            max_value=5.0, 
            value=3.0,
            step=0.1,
            help="相对于FWHM的扩展拟合范围倍数"
        )
        
        # 高级参数 - 紧凑布局
        st.markdown("**高级选项**")
        col1, col2 = st.columns(2)
        with col1:
            show_residuals = st.checkbox("显示残差", value=False)
            show_fit_params = st.checkbox("显示拟合参数", value=True)
        with col2:
            show_statistics = st.checkbox("显示拟合统计", value=True)
        
        # 执行按钮
        if st.button("📈 开始峰拟合", key="fit_peaks", width='stretch'):
            return self._fit_peaks_with_confirm(curve, {
                'model': self.model_mapping[model],  # 使用英文模型名
                'extend_range': extend_range,
                'show_residuals': show_residuals,
                'show_fit_params': show_fit_params,
                'show_statistics': show_statistics
            })
        
        return False
    
    def _fit_peaks_with_confirm(self, curve: Curve, params: Dict[str, Any]) -> bool:
        """执行峰拟合并允许确认应用"""
        try:
            # 使用session_state管理工作副本
            working_key = f"working_curve_{curve.curve_id}"
            if working_key not in st.session_state:
                st.error("❌ 工作副本未找到，请重新选择曲线")
                return False
            
            # 获取当前工作副本数据
            working_data = st.session_state[working_key]
            
            # 保存原始峰数据
            import copy
            original_peaks = copy.deepcopy(curve.peaks)
            
            fitted_count = 0
            
            # 对每个峰进行拟合
            for peak in curve.peaks:
                fit_result = self.peak_fitter.fit_peak(
                    curve=curve,
                    peak=peak,
                    model=params['model'],
                    extend_range=params['extend_range']
                )
                
                # 临时存储拟合结果到峰的元数据中
                peak.metadata['fit_result'] = fit_result
                peak.metadata['fit_model'] = params['model']
                
                if fit_result.get('success', False):
                    fitted_count += 1
            
            # 更新工作副本状态
            working_data["is_modified"] = True
            
            st.success(f"✅ 峰拟合执行完成 - 模型: {params['model']}")
            st.info(f"成功拟合了 {fitted_count}/{len(curve.peaks)} 个峰")
            
            # 显示峰拟合结果
            self._show_peak_fitting_result(curve)
            
            # 确认应用选项
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ 确认应用", key="confirm_peak_fitting"):
                    # 确认应用 - 将拟合结果写入存储数据
                    try:
                        stored_curve = state_manager.get_curve(curve.curve_id)
                        if stored_curve is None:
                            st.error("❌ 无法获取存储的曲线数据")
                            return False
                        
                        # 将拟合结果写入存储数据
                        stored_curve.peaks = copy.deepcopy(curve.peaks)
                        state_manager.update_curve(stored_curve)
                        
                        # 更新工作副本状态
                        working_data["is_modified"] = False
                        working_data["last_applied"] = True
                        
                        st.success(f"✅ 峰拟合已确认应用 (拟合了 {fitted_count} 个峰)")
                        st.rerun()
                        return True
                        
                    except Exception as e:
                        st.error(f"❌ 确认应用失败: {str(e)}")
                        return False
            
            with col2:
                if st.button("❌ 撤销", key="cancel_peak_fitting"):
                    # 撤销操作 - 完全恢复到拟合前状态
                    try:
                        import copy
                        
                        # 恢复曲线对象的峰数据（使用深拷贝确保独立）
                        curve.peaks = copy.deepcopy(original_peaks)
                        
                        # 恢复工作副本中的峰数据（使用深拷贝确保独立）
                        working_data_curve = working_data["curve"]
                        working_data_curve.peaks = copy.deepcopy(original_peaks)
                        
                        # 更新工作副本状态
                        working_data["is_modified"] = False
                        
                        st.info("✅ 已撤销峰拟合，完全恢复到拟合前状态")
                        st.rerun()
                        return False
                    except Exception as e:
                        st.error(f"❌ 撤销失败: {str(e)}")
                        return False
            
            return False
            
        except Exception as e:
            st.error(f"❌ 峰拟合失败: {str(e)}")
            return False
    
    def _show_peak_fitting_result(self, curve: Curve):
        """显示峰拟合结果"""
        st.markdown("**峰拟合结果**")
        
        if not curve.peaks:
            st.info("暂无峰数据")
            return
        
        # 显示拟合的峰
        fitted_peaks = [peak for peak in curve.peaks if 'fit_result' in peak.metadata]
        
        if not fitted_peaks:
            st.info("暂无拟合结果")
            return
        
        # 拟合结果表格
        fit_data = []
        for i, peak in enumerate(fitted_peaks):
            fit_result = peak.metadata['fit_result']
            fit_data.append({
                '峰序号': i+1,
                'RT (min)': f"{peak.rt:.3f}",
                '拟合模型': peak.metadata.get('fit_model', '未知'),
                'R²': f"{fit_result.get('r_squared', 0):.4f}",
                '拟合状态': '成功' if fit_result['success'] else '失败'
            })
        
        df = pd.DataFrame(fit_data)
        st.dataframe(df, width='stretch')
        
        # 峰拟合结果现在显示在统一图表中，不再单独创建图表
        
        # 详细拟合结果
        st.markdown("**详细拟合结果**")
        for i, peak in enumerate(fitted_peaks):
            fit_result = peak.metadata['fit_result']
            model = peak.metadata.get('fit_model', '未知')
            
            with st.expander(f"峰 {i+1} - {model} 拟合结果", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("拟合模型", model)
                    st.metric("R² 决定系数", f"{fit_result.get('r_squared', 0):.4f}")
                    st.metric("拟合状态", "✅ 成功" if fit_result['success'] else "❌ 失败")
                
                with col2:
                    if 'parameters' in fit_result:
                        params = fit_result['parameters']
                        st.markdown("**拟合参数:**")
                        # 检查params是否为字典
                        if isinstance(params, dict):
                            for param_name, param_value in params.items():
                                if isinstance(param_value, (int, float)):
                                    st.write(f"• {param_name}: {param_value:.4f}")
                                else:
                                    st.write(f"• {param_name}: {param_value}")
                        elif isinstance(params, (list, tuple)):
                            for j, param_value in enumerate(params):
                                if isinstance(param_value, (int, float)):
                                    st.write(f"• 参数{j+1}: {param_value:.4f}")
                                else:
                                    st.write(f"• 参数{j+1}: {param_value}")
                        else:
                            st.write(f"• 参数: {params}")
                    
                    if 'confidence_intervals' in fit_result:
                        ci = fit_result['confidence_intervals']
                        st.markdown("**置信区间:**")
                        if isinstance(ci, dict):
                            for param_name, interval in ci.items():
                                if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                                    st.write(f"• {param_name}: [{interval[0]:.4f}, {interval[1]:.4f}]")
                                else:
                                    st.write(f"• {param_name}: {interval}")
                        else:
                            st.write(f"• 置信区间: {ci}")
                
                # 拟合质量评估
                r_squared = fit_result.get('r_squared', 0)
                if r_squared >= 0.95:
                    st.success(f"🎉 拟合质量优秀 (R² = {r_squared:.4f})")
                elif r_squared >= 0.90:
                    st.info(f"✅ 拟合质量良好 (R² = {r_squared:.4f})")
                elif r_squared >= 0.80:
                    st.warning(f"⚠️ 拟合质量一般 (R² = {r_squared:.4f})")
                else:
                    st.error(f"❌ 拟合质量较差 (R² = {r_squared:.4f})")
                
                # 拟合统计信息
                if 'statistics' in fit_result:
                    stats = fit_result['statistics']
                    st.markdown("**拟合统计:**")
                    for stat_name, stat_value in stats.items():
                        if isinstance(stat_value, (int, float)):
                            st.write(f"• {stat_name}: {stat_value:.4f}")
                        else:
                            st.write(f"• {stat_name}: {stat_value}")
    
    def _hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB"""
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"{r}, {g}, {b}"
        except:
            return "255, 0, 0"  # 默认红色
