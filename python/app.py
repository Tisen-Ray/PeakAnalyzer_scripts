"""
PeakAnalyzer 主应用 - 使用Streamlit官方多页面结构
"""

import streamlit as st
from core.state_manager import state_manager
from core.rust_bridge import rust_bridge
from ui.sidebar import show_sidebar


def setup_page_config():
    """设置页面配置"""
    st.set_page_config(
        page_title="PeakAnalyzer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def initialize_session_state():
    """初始化会话状态"""
    if 'rust_bridge' not in st.session_state:
        st.session_state.rust_bridge = rust_bridge


def main():
    """主应用入口"""
    setup_page_config()
    initialize_session_state()
    show_sidebar()
    
    # 主内容区域显示欢迎信息
    st.title("📈 PeakAnalyzer")
    st.markdown("""
    ### 功能页面
    - **📂 数据提取**: 从质谱文件中提取曲线数据
    - **🔧 曲线处理**: 完整的峰分析流程
    - **📊 结果可视化**: 进行结果的统一展示
    """)
    
    # 显示当前数据状态
    curves = state_manager.get_all_curves()
    if curves:
        st.markdown("### 📊 当前数据状态")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("曲线总数", len(curves))
        
        with col2:
            curve_types = set(curve.curve_type for curve in curves.values())
            st.metric("曲线类型", len(curve_types))
        
        with col3:
            total_peaks = sum(len(curve.peaks) for curve in curves.values())
            st.metric("检测到的峰", total_peaks)


if __name__ == "__main__":
    main()
