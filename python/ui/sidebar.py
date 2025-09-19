"""
侧边栏模块 - 统一的侧边栏管理
"""

import streamlit as st
from core.state_manager import state_manager
from core.rust_bridge import rust_bridge


def show_sidebar():
    """显示统一的侧边栏"""
    # 初始化rust_bridge到session_state
    if 'rust_bridge' not in st.session_state:
        st.session_state.rust_bridge = rust_bridge
    
    with st.sidebar:
        st.title("📈 PeakAnalyzer")
        st.markdown("---")
        
        # Rust后端状态
        if st.session_state.rust_bridge.is_available():
            st.success("🟢 Rust后端已连接")
        else:
            st.warning("🟡 Rust后端不可用")
            st.info("请运行: maturin develop")
        
        st.markdown("---")
        
        # 数据概览
        curves = state_manager.get_all_curves()
        st.metric("曲线数量", len(curves))
        
        # 曲线类型分布
        if curves:
            curve_types = {}
            for curve in curves.values():
                curve_type = curve.curve_type
                curve_types[curve_type] = curve_types.get(curve_type, 0) + 1
            
            st.markdown("**曲线类型分布:**")
            for curve_type, count in curve_types.items():
                st.write(f"• {curve_type}: {count}")
            
            # 处理状态统计
            processed_count = sum(1 for curve in curves.values() if curve.processing_history)
            st.metric("已处理曲线", processed_count)
        
        st.markdown("---")
        
        # 清理按钮
        if st.button("🧹 清理所有数据"):
            state_manager.clear_all_curves()
            st.success("数据已清理")
            st.rerun()
        
        st.markdown("---")
        
        # 快速导航提示
        st.markdown("**💡 使用提示:**")
        st.markdown("""
        - 使用页面顶部的导航栏切换功能
        - 所有处理结果会自动保存
        - 支持多曲线同时处理
        """)
