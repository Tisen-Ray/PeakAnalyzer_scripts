"""
PeakAnalyzer - 多页面主应用
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入核心模块
from core.curve_wrapper import wrap_pycurve
from core.state_manager import state_manager
from core.state_sync import state_sync

# 导入页面模块
from ui.pages.data_extraction_page import DataExtractionPage
from ui.pages.curve_processing_page import CurveProcessingPage
from ui.pages.visualization_page import VisualizationPage


def initialize_session_state():
    """初始化会话状态"""
    # 状态管理器会自动初始化所有必要的状态
    # 这里只需要初始化Rust桥接
    if 'rust_bridge' not in st.session_state:
        # 创建一个简单的rust_bridge对象来模拟RustBridge类
        class SimpleRustBridge:
            def __init__(self):
                try:
                    import peakanalyzer_scripts
                    self.module = peakanalyzer_scripts
                    self.available = True
                    st.write("✅ Rust模块加载成功")
                except ImportError as e:
                    self.module = None
                    self.available = False
                    st.write(f"⚠️ Rust模块加载失败: {e}")
                except Exception as e:
                    self.module = None
                    self.available = False
                    st.write(f"❌ Rust模块初始化错误: {e}")
            
            def is_available(self):
                return self.available
            
            def get_file_info(self, file_path):
                if self.available:
                    return dict(self.module.get_file_info(file_path))
                else:
                    return {'status': 'error', 'error': 'Rust模块不可用'}
            
            def extract_tic_curve(self, file_path, mz_min, mz_max, rt_min, rt_max, ms_level):
                if self.available:
                    py_curve = self.module.extract_tic_curve(file_path, mz_min, mz_max, rt_min, rt_max, ms_level)
                    return wrap_pycurve(py_curve)
                else:
                    raise RuntimeError("Rust模块不可用")
            
            def extract_eic_curve(self, file_path, target_mz, mz_tolerance, rt_min, rt_max, ms_level):
                if self.available:
                    py_curve = self.module.extract_eic_curve(file_path, target_mz, mz_tolerance, rt_min, rt_max, ms_level)
                    return wrap_pycurve(py_curve)
                else:
                    raise RuntimeError("Rust模块不可用")
            
            def extract_bpc_curve(self, file_path, mz_min, mz_max, rt_min, rt_max, ms_level):
                if self.available:
                    py_curve = self.module.extract_bpc_curve(file_path, mz_min, mz_max, rt_min, rt_max, ms_level)
                    return wrap_pycurve(py_curve)
                else:
                    raise RuntimeError("Rust模块不可用")
        
        st.session_state.rust_bridge = SimpleRustBridge()
        st.session_state.rust_available = st.session_state.rust_bridge.is_available()
        
        # 更新系统状态
        state_manager.update_ui_state('rust_available', st.session_state.rust_available)


def setup_page_config():
    """设置页面配置"""
    st.set_page_config(
        page_title="PeakAnalyzer - 质谱峰分析工具",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def show_sidebar():
    """显示侧边栏导航"""
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
        
        # 页面导航
        page = st.radio(
            "选择功能页面",
            ["📂 数据提取", "🔧 曲线处理", "📊 结果可视化", "⚙️ 项目管理"]
        )
        
        st.markdown("---")
        
        # 数据概览
        stats = state_manager.get_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("曲线数量", stats['curves']['total'])
        with col2:
            st.metric("文件数量", stats['files'])
        
        # 曲线类型分布
        if stats['curves']['by_type']:
            st.markdown("**曲线类型分布:**")
            for curve_type, count in stats['curves']['by_type'].items():
                st.write(f"• {curve_type}: {count}")
        
        # 会话信息
        st.markdown("---")
        st.markdown("**会话信息:**")
        st.write(f"持续时间: {stats['session_duration']}")
        
        # 项目管理快捷操作
        st.markdown("---")
        st.markdown("**项目管理:**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 保存项目", help="保存当前工作状态"):
                project_name = st.text_input("项目名称", key="save_project_name")
                if project_name and st.button("确认保存", key="confirm_save"):
                    project_id = state_manager.save_project(project_name)
                    st.success(f"项目已保存: {project_name}")
                    st.rerun()
        
        with col2:
            projects = state_manager.get_projects()
            if projects:
                project_names = [p['name'] for p in projects]
                selected_project = st.selectbox("加载项目", [""] + project_names, key="load_project_select")
                if selected_project and st.button("加载", key="confirm_load"):
                    project_id = next(p['id'] for p in projects if p['name'] == selected_project)
                    if state_manager.load_project(project_id):
                        st.success(f"项目已加载: {selected_project}")
                        st.rerun()
        
        # 清理缓存按钮
        if st.button("🧹 清理缓存"):
            state_manager.clear_all_data()
            st.success("缓存已清理")
            st.rerun()
        
        return page


def render_project_management_page():
    """渲染项目管理页面"""
    st.header("⚙️ 项目管理")
    st.markdown("管理您的工作流程，保存和恢复分析状态")
    
    # 项目列表
    projects = state_manager.get_projects()
    
    if projects:
        st.subheader("📋 现有项目")
        
        for project in projects:
            with st.expander(f"📁 {project['name']} ({project['curve_count']} 条曲线)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**创建时间:** {project['created_at'][:19]}")
                    st.write(f"**更新时间:** {project['updated_at'][:19]}")
                
                with col2:
                    st.write(f"**描述:** {project.get('description', '无描述')}")
                    st.write(f"**曲线数量:** {project['curve_count']}")
                
                with col3:
                    if st.button(f"加载 {project['name']}", key=f"load_{project['id']}"):
                        if state_manager.load_project(project['id']):
                            st.success(f"项目 '{project['name']}' 已加载")
                            st.rerun()
                    
                    if st.button(f"删除 {project['name']}", key=f"delete_{project['id']}"):
                        if project['id'] in st.session_state.projects:
                            del st.session_state.projects[project['id']]
                        st.success(f"项目 '{project['name']}' 已删除")
                        st.rerun()
    else:
        st.info("还没有保存的项目")
    
    # 创建新项目
    st.markdown("---")
    st.subheader("💾 创建新项目")
    
    col1, col2 = st.columns(2)
    
    with col1:
        project_name = st.text_input("项目名称", placeholder="输入项目名称")
        project_description = st.text_area("项目描述", placeholder="输入项目描述（可选）")
    
    with col2:
        # 显示当前状态统计
        stats = state_manager.get_statistics()
        st.write("**当前状态:**")
        st.write(f"• 曲线数量: {stats['curves']['total']}")
        st.write(f"• 文件数量: {stats['files']}")
        st.write(f"• 会话持续时间: {stats['session_duration']}")
    
    if st.button("💾 保存为新项目", type="primary"):
        if project_name:
            project_id = state_manager.save_project(project_name, project_description)
            st.success(f"项目 '{project_name}' 已保存")
            st.rerun()
        else:
            st.error("请输入项目名称")
    
    # 导入导出功能
    st.markdown("---")
    st.subheader("📤 导入导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**导出当前状态**")
        export_options = st.multiselect(
            "导出内容",
            ["曲线数据", "处理参数", "提取配置", "可视化设置"],
            default=["曲线数据", "处理参数"]
        )
        
        if st.button("📥 导出状态"):
            include_data = "曲线数据" in export_options
            export_data = state_manager.export_state(include_data)
            
            # 创建下载文件
            import json
            import io
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            
            st.download_button(
                label="下载状态文件",
                data=json_str,
                file_name=f"peakanalyzer_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("**导入状态文件**")
        uploaded_file = st.file_uploader(
            "选择状态文件",
            type=['json'],
            help="选择之前导出的状态文件"
        )
        
        if uploaded_file is not None:
            try:
                import json
                state_data = json.load(uploaded_file)
                
                if st.button("📤 导入状态"):
                    if state_manager.import_state(state_data):
                        st.success("状态导入成功")
                        st.rerun()
                    else:
                        st.error("状态导入失败")
                        
            except Exception as e:
                st.error(f"文件解析失败: {str(e)}")


def main():
    """主应用入口"""
    setup_page_config()
    initialize_session_state()
    
    selected_page = show_sidebar()
    
    # 页面切换时同步状态
    state_sync.auto_sync_on_page_change(selected_page)
    
    # 根据选择显示相应页面
    if selected_page == "📂 数据提取":
        # 恢复提取页面状态
        state_sync.auto_restore_on_page_load(selected_page)
        data_extraction_page = DataExtractionPage()
        data_extraction_page.render()
    
    elif selected_page == "🔧 曲线处理":
        # 恢复处理页面状态
        state_sync.auto_restore_on_page_load(selected_page)
        curve_processing_page = CurveProcessingPage()
        curve_processing_page.render()
    
    elif selected_page == "📊 结果可视化":
        # 恢复可视化页面状态
        state_sync.auto_restore_on_page_load(selected_page)
        visualization_page = VisualizationPage()
        visualization_page.render()
    
    elif selected_page == "⚙️ 项目管理":
        render_project_management_page()


if __name__ == "__main__":
    main()