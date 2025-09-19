"""
状态同步工具 - 确保页面间状态同步和参数不丢失
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from .state_manager import state_manager


class StateSync:
    """状态同步管理器"""
    
    def __init__(self):
        """初始化状态同步器"""
        self._sync_callbacks = {}
    
    def register_sync_callback(self, key: str, callback):
        """注册状态同步回调函数"""
        self._sync_callbacks[key] = callback
    
    def sync_processing_params(self):
        """同步处理参数状态"""
        # 从UI状态同步到状态管理器
        if 'enable_baseline' in st.session_state:
            baseline_params = {
                'enabled': st.session_state.get('enable_baseline', False),
                'method': st.session_state.get('baseline_method', 'als'),
                'parameters': {}
            }
            
            if baseline_params['method'] == 'als':
                baseline_params['parameters'] = {
                    'lam': st.session_state.get('baseline_lam', 1e6),
                    'p': st.session_state.get('baseline_p', 0.01)
                }
            elif baseline_params['method'] == 'polynomial':
                baseline_params['parameters'] = {
                    'degree': st.session_state.get('baseline_degree', 3)
                }
            elif baseline_params['method'] == 'rolling_ball':
                baseline_params['parameters'] = {
                    'window_size': st.session_state.get('baseline_window', 50)
                }
            
            state_manager.update_processing_params('baseline_correction', baseline_params)
        
        # 同步平滑参数
        if 'enable_smoothing' in st.session_state:
            smoothing_params = {
                'enabled': st.session_state.get('enable_smoothing', False),
                'method': st.session_state.get('smooth_method', 'savgol'),
                'parameters': {}
            }
            
            if smoothing_params['method'] == 'savgol':
                smoothing_params['parameters'] = {
                    'window_length': st.session_state.get('smooth_window', 11),
                    'polyorder': st.session_state.get('smooth_polyorder', 3)
                }
            elif smoothing_params['method'] == 'gaussian':
                smoothing_params['parameters'] = {
                    'sigma': st.session_state.get('smooth_sigma', 1.0)
                }
            elif smoothing_params['method'] == 'moving_average':
                smoothing_params['parameters'] = {
                    'window_size': st.session_state.get('smooth_window_ma', 5)
                }
            
            state_manager.update_processing_params('smoothing', smoothing_params)
        
        # 同步归一化参数
        if 'enable_normalization' in st.session_state:
            normalization_params = {
                'enabled': st.session_state.get('enable_normalization', False),
                'method': st.session_state.get('norm_method', 'minmax'),
                'parameters': {}
            }
            
            state_manager.update_processing_params('normalization', normalization_params)
    
    def restore_processing_params(self):
        """从状态管理器恢复处理参数到UI状态"""
        # 恢复基线校正参数
        baseline_params = state_manager.get_processing_params('baseline_correction')
        if baseline_params:
            st.session_state.enable_baseline = baseline_params.get('enabled', False)
            st.session_state.baseline_method = baseline_params.get('method', 'als')
            
            params = baseline_params.get('parameters', {})
            if baseline_params['method'] == 'als':
                st.session_state.baseline_lam = params.get('lam', 1e6)
                st.session_state.baseline_p = params.get('p', 0.01)
            elif baseline_params['method'] == 'polynomial':
                st.session_state.baseline_degree = params.get('degree', 3)
            elif baseline_params['method'] == 'rolling_ball':
                st.session_state.baseline_window = params.get('window_size', 50)
        
        # 恢复平滑参数
        smoothing_params = state_manager.get_processing_params('smoothing')
        if smoothing_params:
            st.session_state.enable_smoothing = smoothing_params.get('enabled', False)
            st.session_state.smooth_method = smoothing_params.get('method', 'savgol')
            
            params = smoothing_params.get('parameters', {})
            if smoothing_params['method'] == 'savgol':
                st.session_state.smooth_window = params.get('window_length', 11)
                st.session_state.smooth_polyorder = params.get('polyorder', 3)
            elif smoothing_params['method'] == 'gaussian':
                st.session_state.smooth_sigma = params.get('sigma', 1.0)
            elif smoothing_params['method'] == 'moving_average':
                st.session_state.smooth_window_ma = params.get('window_size', 5)
        
        # 恢复归一化参数
        normalization_params = state_manager.get_processing_params('normalization')
        if normalization_params:
            st.session_state.enable_normalization = normalization_params.get('enabled', False)
            st.session_state.norm_method = normalization_params.get('method', 'minmax')
    
    def sync_visualization_params(self):
        """同步可视化参数状态"""
        viz_params = {}
        
        # 同步视图模式
        if 'viz_view_mode' in st.session_state:
            viz_params['view_mode'] = st.session_state.viz_view_mode
        
        # 同步显示选项
        if 'viz_show_peaks' in st.session_state:
            viz_params['show_peaks'] = st.session_state.viz_show_peaks
        
        if 'viz_show_legend' in st.session_state:
            viz_params['show_legend'] = st.session_state.viz_show_legend
        
        if 'viz_normalize' in st.session_state:
            viz_params['normalize_curves'] = st.session_state.viz_normalize
        
        if 'viz_show_peak_analysis' in st.session_state:
            viz_params['show_peak_analysis'] = st.session_state.viz_show_peak_analysis
        
        # 同步选中的曲线
        if 'selected_curves_for_viz' in st.session_state:
            viz_params['selected_curves_for_viz'] = st.session_state.selected_curves_for_viz
        
        # 同步过滤条件
        if 'curve_filters' in st.session_state:
            viz_params['curve_filters'] = st.session_state.curve_filters
        
        # 更新到状态管理器
        for key, value in viz_params.items():
            state_manager.update_visualization_state(key, value)
    
    def restore_visualization_params(self):
        """从状态管理器恢复可视化参数到UI状态"""
        # 恢复视图模式
        view_mode = state_manager.get_visualization_state('view_mode')
        if view_mode:
            st.session_state.viz_view_mode = view_mode
        
        # 恢复显示选项
        show_peaks = state_manager.get_visualization_state('show_peaks')
        if show_peaks is not None:
            st.session_state.viz_show_peaks = show_peaks
        
        show_legend = state_manager.get_visualization_state('show_legend')
        if show_legend is not None:
            st.session_state.viz_show_legend = show_legend
        
        normalize_curves = state_manager.get_visualization_state('normalize_curves')
        if normalize_curves is not None:
            st.session_state.viz_normalize = normalize_curves
        
        show_peak_analysis = state_manager.get_visualization_state('show_peak_analysis')
        if show_peak_analysis is not None:
            st.session_state.viz_show_peak_analysis = show_peak_analysis
        
        # 恢复选中的曲线
        selected_curves = state_manager.get_visualization_state('selected_curves_for_viz')
        if selected_curves is not None:
            st.session_state.selected_curves_for_viz = selected_curves
        
        # 恢复过滤条件
        curve_filters = state_manager.get_visualization_state('curve_filters')
        if curve_filters:
            st.session_state.curve_filters = curve_filters
    
    def sync_extraction_params(self):
        """同步提取参数状态"""
        # 同步单文件配置
        if 'single_file_configs' in st.session_state:
            state_manager.update_extraction_config('single_file_configs', 
                                                 st.session_state.single_file_configs)
        
        # 同步批量配置
        if 'batch_configs' in st.session_state:
            state_manager.update_extraction_config('batch_configs', 
                                                 st.session_state.batch_configs)
        
        # 同步最近文件
        if 'recent_files' in st.session_state:
            state_manager.update_extraction_config('recent_files', 
                                                 st.session_state.recent_files)
        
        # 同步最近目录
        if 'recent_directories' in st.session_state:
            state_manager.update_extraction_config('recent_directories', 
                                                 st.session_state.recent_directories)
    
    def restore_extraction_params(self):
        """从状态管理器恢复提取参数到UI状态"""
        # 恢复单文件配置
        single_file_configs = state_manager.get_extraction_config('single_file_configs')
        if single_file_configs:
            st.session_state.single_file_configs = single_file_configs
        
        # 恢复批量配置
        batch_configs = state_manager.get_extraction_config('batch_configs')
        if batch_configs:
            st.session_state.batch_configs = batch_configs
        
        # 恢复最近文件
        recent_files = state_manager.get_extraction_config('recent_files')
        if recent_files:
            st.session_state.recent_files = recent_files
        
        # 恢复最近目录
        recent_directories = state_manager.get_extraction_config('recent_directories')
        if recent_directories:
            st.session_state.recent_directories = recent_directories
    
    def auto_sync_on_page_change(self, current_page: str):
        """页面切换时自动同步状态"""
        # 记录当前页面
        state_manager.update_ui_state('active_page', current_page)
        
        # 根据当前页面同步相应的状态
        if current_page == "🔧 曲线处理":
            self.sync_processing_params()
        elif current_page == "📊 结果可视化":
            self.sync_visualization_params()
        elif current_page == "📂 数据提取":
            self.sync_extraction_params()
    
    def auto_restore_on_page_load(self, target_page: str):
        """页面加载时自动恢复状态"""
        # 根据目标页面恢复相应的状态
        if target_page == "🔧 曲线处理":
            self.restore_processing_params()
        elif target_page == "📊 结果可视化":
            self.restore_visualization_params()
        elif target_page == "📂 数据提取":
            self.restore_extraction_params()
    
    def force_sync_all(self):
        """强制同步所有状态"""
        self.sync_processing_params()
        self.sync_visualization_params()
        self.sync_extraction_params()
    
    def force_restore_all(self):
        """强制恢复所有状态"""
        self.restore_processing_params()
        self.restore_visualization_params()
        self.restore_extraction_params()
    
    def get_sync_status(self) -> Dict[str, Any]:
        """获取同步状态信息"""
        return {
            'last_sync': state_manager.get_ui_state('last_updated'),
            'current_page': state_manager.get_ui_state('active_page'),
            'processing_params_synced': bool(state_manager.get_processing_params('baseline_correction')),
            'visualization_params_synced': bool(state_manager.get_visualization_state('view_mode')),
            'extraction_params_synced': bool(state_manager.get_extraction_config('single_file_configs'))
        }


# 全局状态同步器实例
state_sync = StateSync()
