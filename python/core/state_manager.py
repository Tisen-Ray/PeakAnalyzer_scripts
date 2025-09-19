"""
简化的状态管理器 - 只管理配置和曲线对象
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from core.curve import Curve


class SimpleStateManager:
    """简化的状态管理器 - 只管理必要的状态"""
    
    def __init__(self):
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """确保session_state已初始化"""
        if 'curves' not in st.session_state:
            st.session_state.curves = {}
        
        if 'extraction_configs' not in st.session_state:
            st.session_state.extraction_configs = {}
        
        if 'processing_configs' not in st.session_state:
            st.session_state.processing_configs = {}
        
        if 'visualization_configs' not in st.session_state:
            st.session_state.visualization_configs = {}
    
    def add_curve(self, curve: Curve):
        """添加曲线"""
        self._ensure_initialized()
        st.session_state.curves[curve.curve_id] = curve
    
    def get_curve(self, curve_id: str) -> Optional[Curve]:
        """获取曲线"""
        self._ensure_initialized()
        return st.session_state.curves.get(curve_id)
    
    def get_all_curves(self) -> Dict[str, Curve]:
        """获取所有曲线"""
        self._ensure_initialized()
        return st.session_state.curves
    
    def remove_curve(self, curve_id: str):
        """移除曲线"""
        self._ensure_initialized()
        if curve_id in st.session_state.curves:
            del st.session_state.curves[curve_id]
    
    def clear_all_curves(self):
        """清除所有曲线"""
        self._ensure_initialized()
        st.session_state.curves = {}
    
    def update_curve(self, curve: Curve):
        """更新曲线"""
        self._ensure_initialized()
        if curve.curve_id in st.session_state.curves:
            st.session_state.curves[curve.curve_id] = curve
    
    # 配置管理
    def get_extraction_config(self, config_id: str) -> Dict[str, Any]:
        """获取提取配置"""
        self._ensure_initialized()
        return st.session_state.extraction_configs.get(config_id, {})
    
    def update_extraction_config(self, config_id: str, config: Dict[str, Any]):
        """更新提取配置"""
        self._ensure_initialized()
        st.session_state.extraction_configs[config_id] = config
    
    def get_processing_config(self, config_id: str) -> Dict[str, Any]:
        """获取处理配置"""
        self._ensure_initialized()
        return st.session_state.processing_configs.get(config_id, {})
    
    def update_processing_config(self, config_id: str, config: Dict[str, Any]):
        """更新处理配置"""
        self._ensure_initialized()
        st.session_state.processing_configs[config_id] = config
    
    def get_visualization_config(self, config_id: str) -> Dict[str, Any]:
        """获取可视化配置"""
        self._ensure_initialized()
        return st.session_state.visualization_configs.get(config_id, {})
    
    def update_visualization_config(self, config_id: str, config: Dict[str, Any]):
        """更新可视化配置"""
        self._ensure_initialized()
        st.session_state.visualization_configs[config_id] = config
    
    def get_visualization_state(self, key: str) -> Any:
        """获取可视化状态"""
        self._ensure_initialized()
        return st.session_state.visualization_configs.get(key)
    
    def set_visualization_state(self, key: str, value: Any):
        """设置可视化状态"""
        self._ensure_initialized()
        st.session_state.visualization_configs[key] = value


# 全局状态管理器实例
state_manager = SimpleStateManager()

# 为了向后兼容，也导出simple_state_manager
simple_state_manager = state_manager
