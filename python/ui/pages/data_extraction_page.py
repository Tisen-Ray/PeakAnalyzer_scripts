"""
数据提取页面 - 支持两种模式的批量曲线提取
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import tempfile
import uuid
from typing import List, Dict, Any
from pathlib import Path
from core.state_manager import state_manager
from utils.file_dialog import show_file_selection_button, show_directory_selection_button


class DataExtractionPage:
    """数据提取页面类"""
    
    def __init__(self):
        self.temp_files = []
    
    def render(self):
        """渲染页面内容"""
        st.header("📂 数据提取")
        st.markdown("支持两种模式的批量曲线提取：单文件多曲线 或 多文件同参数提取")
        
        # 选择提取模式
        extraction_mode = st.radio(
            "选择提取模式",
            ["🎯 单文件多曲线提取", "📁 多文件批量提取"],
            horizontal=True
        )
        
        if extraction_mode == "🎯 单文件多曲线提取":
            self._render_single_file_mode()
        else:
            self._render_batch_file_mode()
        
        # 显示提取历史
        self._render_extraction_history()
        
        # 显示系统状态检查
        self._render_system_status()
    
    def _render_single_file_mode(self):
        """渲染单文件多曲线提取模式"""
        st.subheader("🎯 单文件多曲线提取")
        st.info("从一个文件中提取多条不同参数的曲线（如不同m/z范围的EIC曲线）")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 文件路径输入
            st.markdown("**文件选择**")
            
            # 文件路径输入方式选择
            input_method = st.radio(
                "选择输入方式",
                ["📁 浏览选择", "✏️ 直接输入路径"],
                horizontal=True
            )
            
            file_path = None
            
            if input_method == "📁 浏览选择":
                # 使用tkinter文件选择对话框
                selected_files = show_file_selection_button(
                    button_text="📁 选择质谱文件",
                    title="选择质谱文件",
                    filetypes=[
                        ("质谱文件", "*.mzML *.mzXML *.raw"),
                        ("Excel文件", "*.xlsx *.xls"),
                        ("CSV文件", "*.csv"),
                        ("所有文件", "*.*")
                    ],
                    multiple=False,
                    key="single_file_selection"
                )
                
                if selected_files:
                    if isinstance(selected_files, list):
                        file_path = selected_files[0]
                    else:
                        file_path = selected_files
                    st.session_state.selected_file_path = file_path
                
                # 显示当前选择的文件
                if 'selected_file_path' in st.session_state:
                    current_file = st.session_state.selected_file_path
                    if os.path.exists(current_file):
                        st.success(f"✅ 当前文件: {os.path.basename(current_file)}")
                        file_path = current_file
                    else:
                        st.error(f"❌ 文件不存在: {current_file}")
                        if st.button("清除文件选择"):
                            del st.session_state.selected_file_path
                    
            else:  # 直接输入路径
                file_path = st.text_input(
                    "完整文件路径",
                    value=st.session_state.get('selected_file_path', ''),
                    placeholder="D:/data/sample.mzML",
                    help="输入MS数据文件的完整路径"
                )
                if file_path:
                    st.session_state.selected_file_path = file_path
            
            if file_path:
                # 验证文件是否存在
                if Path(file_path).exists():
                    st.success(f"✅ 文件找到: {Path(file_path).name}")
                    
                    # 显示文件信息
                    if st.button("📋 获取文件信息"):
                        self._show_file_info_by_path(file_path)
                else:
                    st.error(f"❌ 文件不存在: {file_path}")
                    file_path = None
        
        with col2:
            # 曲线配置表格
            st.markdown("**曲线提取配置**")
            
            # 从状态管理器获取配置
            single_file_configs = state_manager.get_extraction_config('single_file_configs')
            if not single_file_configs:
                single_file_configs = [
                    {
                        'name': 'TIC_全范围',
                        'type': 'TIC',
                        'mz_min': 50.0,
                        'mz_max': 1000.0,
                        'rt_min': None,
                        'rt_max': None,
                        'ms_level': 1
                    }
                ]
                state_manager.update_extraction_config('single_file_configs', single_file_configs)
            
            # 配置编辑器
            self._render_curve_config_editor('single_file_configs', single_file_configs)
        
        # 提取按钮
        if file_path and st.button("🚀 开始单文件多曲线提取", type="primary"):
            self._execute_single_file_extraction_by_path(file_path)
    
    def _render_batch_file_mode(self):
        """渲染多文件批量提取模式"""
        st.subheader("📁 多文件批量提取")
        st.info("使用相同参数从多个文件中提取曲线")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 批量文件路径输入
            st.markdown("**批量文件选择**")
            
            # 目录选择 - 使用tkinter对话框
            directory = show_directory_selection_button(
                button_text="📂 选择数据目录",
                title="选择包含MS数据文件的目录",
                key="batch_directory_selection"
            )
            
            if directory:
                st.session_state.selected_batch_directory = directory
            
            # 显示当前选择的目录
            current_dir = st.session_state.get('selected_batch_directory', '')
            base_directory = st.text_input(
                "数据目录",
                value=current_dir,
                placeholder="D:/ms_data/",
                help="包含MS数据文件的目录路径"
            )
            if base_directory:
                st.session_state.selected_batch_directory = base_directory
            
            # 文件模式匹配
            file_pattern = st.text_input(
                "文件模式",
                value="*.raw",
                help="文件匹配模式，如 *.mzML, sample*.mzXML 等"
            )
            
            # 扫描文件
            if st.button("🔍 扫描文件"):
                self._scan_files(base_directory, file_pattern)
            
            # 显示扫描到的文件
            if 'scanned_files' in st.session_state and st.session_state.scanned_files:
                st.success(f"找到 {len(st.session_state.scanned_files)} 个文件")
                
                # 文件选择
                selected_files = []
                with st.expander("📋 选择要处理的文件", expanded=True):
                    select_all = st.checkbox("全选")
                    
                    for file_path in st.session_state.scanned_files:
                        file_name = Path(file_path).name
                        if select_all or st.checkbox(file_name, key=f"batch_file_{file_name}"):
                            selected_files.append(file_path)
                
                st.session_state.selected_batch_files = selected_files
                st.info(f"已选择 {len(selected_files)} 个文件")
        
        with col2:
            # 统一提取参数
            st.markdown("**统一提取参数**")
            
            curve_type = st.selectbox(
                "曲线类型",
                ["TIC", "EIC", "BPC"],
                help="TIC: 总离子流图, EIC: 提取离子流图, BPC: 基峰图"
            )
            
            if curve_type == "EIC":
                target_mz = st.number_input("目标 m/z", value=100.0, min_value=0.0)
                mz_tolerance = st.number_input("m/z 容差", value=0.1, min_value=0.001)
                mz_min = target_mz - mz_tolerance
                mz_max = target_mz + mz_tolerance
            else:
                mz_min = st.number_input("m/z 最小值", value=50.0, min_value=0.0)
                mz_max = st.number_input("m/z 最大值", value=1000.0, min_value=0.0)
            
            # RT范围（可选）
            use_rt_filter = st.checkbox("限制保留时间范围")
            if use_rt_filter:
                rt_min = st.number_input("RT 最小值 (分钟)", value=0.0, min_value=0.0)
                rt_max = st.number_input("RT 最大值 (分钟)", value=30.0, min_value=0.0)
            else:
                rt_min = rt_max = None
            
            # MS级别
            ms_level = st.selectbox("MS级别", [1, 2, 3, 4, 5], index=0)
        
        # 批量提取按钮
        selected_files = st.session_state.get('selected_batch_files', [])
        if selected_files and st.button("🚀 开始批量提取", type="primary"):
            batch_config = {
                'curve_type': curve_type,
                'mz_min': mz_min,
                'mz_max': mz_max,
                'rt_min': rt_min,
                'rt_max': rt_max,
                'ms_level': ms_level
            }
            self._execute_batch_extraction_by_paths(selected_files, batch_config)
    
    def _render_curve_config_editor(self, config_key: str, configs: List[Dict[str, Any]]):
        """渲染曲线配置编辑器"""
        
        # 显示当前配置
        for i, config in enumerate(configs):
            with st.expander(f"配置 {i+1}: {config['name']}", expanded=i==0):
                col1, col2 = st.columns(2)
                
                with col1:
                    config['name'] = st.text_input(f"配置名称", value=config['name'], key=f"{config_key}_name_{i}")
                    config['type'] = st.selectbox(f"曲线类型", ["TIC", "EIC", "BPC"], 
                                                index=["TIC", "EIC", "BPC"].index(config['type']), 
                                                key=f"{config_key}_type_{i}")
                    config['mz_min'] = st.number_input(f"m/z 最小值", value=config['mz_min'], key=f"{config_key}_mz_min_{i}")
                    config['mz_max'] = st.number_input(f"m/z 最大值", value=config['mz_max'], key=f"{config_key}_mz_max_{i}")
                
                with col2:
                    config['rt_min'] = st.number_input(f"RT 最小值 (可选)", value=config['rt_min'], key=f"{config_key}_rt_min_{i}")
                    config['rt_max'] = st.number_input(f"RT 最大值 (可选)", value=config['rt_max'], key=f"{config_key}_rt_max_{i}")
                    config['ms_level'] = st.selectbox(f"MS级别", [1, 2, 3, 4, 5], 
                                                    index=config['ms_level']-1, 
                                                    key=f"{config_key}_ms_level_{i}")
                
                # 删除配置按钮
                if len(configs) > 1:
                    if st.button(f"🗑️ 删除配置 {i+1}", key=f"{config_key}_delete_{i}"):
                        configs.pop(i)
                        state_manager.update_extraction_config(config_key, configs)
                        st.rerun()
        
        # 添加新配置按钮
        if st.button(f"➕ 添加新配置", key=f"{config_key}_add"):
            new_config = {
                'name': f'配置_{len(configs)+1}',
                'type': 'TIC',
                'mz_min': 50.0,
                'mz_max': 1000.0,
                'rt_min': None,
                'rt_max': None,
                'ms_level': 1
            }
            configs.append(new_config)
            state_manager.update_extraction_config(config_key, configs)
            st.rerun()
    
    def _scan_files(self, directory: str, pattern: str):
        """扫描目录中的文件"""
        try:
            from glob import glob
            
            if not Path(directory).exists():
                st.error(f"目录不存在: {directory}")
                return
            
            # 构建搜索模式
            search_pattern = str(Path(directory) / pattern)
            found_files = glob(search_pattern)
            
            # 过滤出存在的文件
            valid_files = [f for f in found_files if Path(f).is_file()]
            
            st.session_state.scanned_files = valid_files
            
            if valid_files:
                st.success(f"找到 {len(valid_files)} 个匹配的文件")
            else:
                st.warning(f"在目录 {directory} 中没有找到匹配 {pattern} 的文件")
                
        except Exception as e:
            st.error(f"扫描文件时出错: {e}")
    
    def _show_file_info_by_path(self, file_path: str):
        """通过文件路径显示文件信息"""
        try:
            # 获取文件信息
            file_info = st.session_state.rust_bridge.get_file_info(file_path)
            
            if file_info.get('status') == 'success':
                st.success("✅ 文件信息获取成功")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("光谱数量", file_info.get('spectrum_count', 0))
                    st.metric("RT范围 (分钟)", file_info.get('rt_range', 'N/A'))
                
                with col2:
                    st.metric("m/z范围", file_info.get('mz_range', 'N/A'))
                    st.metric("MS级别分布", file_info.get('ms_levels', 'N/A'))
            elif file_info.get('status') == 'mock':
                st.warning("⚠️ 使用模拟数据（文件格式可能不受支持）")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("光谱数量", file_info.get('spectrum_count', 0))
                    st.metric("RT范围 (分钟)", file_info.get('rt_range', 'N/A'))
                
                with col2:
                    st.metric("m/z范围", file_info.get('mz_range', 'N/A'))
                    st.metric("MS级别分布", file_info.get('ms_levels', 'N/A'))
            elif file_info.get('status') == 'error':
                st.error(f"❌ 获取文件信息失败: {file_info.get('error', '未知错误')}")
                if 'suggestion' in file_info:
                    st.info(f"💡 建议: {file_info['suggestion']}")
            else:
                st.error(f"❌ 获取文件信息失败: {file_info.get('error', '未知错误')}")
        
        except Exception as e:
            st.error(f"❌ 处理文件时出错: {str(e)}")
    
    def _show_file_info(self, uploaded_file):
        """显示文件信息"""
        try:
            # 保存临时文件
            temp_path = self._save_temp_file(uploaded_file)
            
            # 获取文件信息
            file_info = st.session_state.rust_bridge.get_file_info(temp_path)
            
            if file_info.get('status') == 'success':
                st.success("✅ 文件信息获取成功")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("光谱数量", file_info.get('spectrum_count', 0))
                    if 'rt_range' in file_info:
                        rt_range = file_info['rt_range']
                        st.metric("RT范围 (分钟)", f"{rt_range[0]:.2f} - {rt_range[1]:.2f}")
                
                with col2:
                    if 'mz_range' in file_info:
                        mz_range = file_info['mz_range']
                        st.metric("m/z范围", f"{mz_range[0]:.1f} - {mz_range[1]:.1f}")
                    
                    if 'ms_levels' in file_info:
                        ms_levels = file_info['ms_levels']
                        st.write("**MS级别分布:**")
                        for level, count in ms_levels.items():
                            st.write(f"MS{level}: {count} 个光谱")
            else:
                st.error(f"❌ 获取文件信息失败: {file_info.get('error', '未知错误')}")
        
        except Exception as e:
            st.error(f"❌ 处理文件时出错: {str(e)}")
    
    def _save_temp_file(self, uploaded_file) -> str:
        """保存临时文件"""
        temp_dir = Path(tempfile.gettempdir()) / "peakanalyzer_temp"
        temp_dir.mkdir(exist_ok=True)
        
        temp_path = temp_dir / f"{uuid.uuid4().hex}_{uploaded_file.name}"
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        self.temp_files.append(str(temp_path))
        return str(temp_path)
    
    def _execute_single_file_extraction(self, uploaded_file):
        """执行单文件多曲线提取"""
        configs = st.session_state.single_file_configs
        
        if not configs:
            st.warning("请至少配置一个曲线提取参数")
            return
        
        # 保存临时文件
        temp_path = self._save_temp_file(uploaded_file)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        extracted_curves = []
        
        for i, config in enumerate(configs):
            try:
                status_text.text(f"提取曲线: {config['name']}")
                
                # 执行提取
                if config['type'] == 'TIC':
                    curve = st.session_state.rust_bridge.extract_tic_curve(
                        temp_path, config['mz_min'], config['mz_max'],
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                elif config['type'] == 'EIC':
                    target_mz = (config['mz_min'] + config['mz_max']) / 2
                    mz_tolerance = (config['mz_max'] - config['mz_min']) / 2
                    curve = st.session_state.rust_bridge.extract_eic_curve(
                        temp_path, target_mz, mz_tolerance,
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                elif config['type'] == 'BPC':
                    curve = st.session_state.rust_bridge.extract_bpc_curve(
                        temp_path, config['mz_min'], config['mz_max'],
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                
                # 更新曲线元数据
                curve.metadata['original_filename'] = uploaded_file.name
                curve.metadata['config_name'] = config['name']
                curve.metadata['extraction_mode'] = 'single_file_multi_curve'
                
                extracted_curves.append(curve)
                
            except Exception as e:
                st.error(f"❌ 提取曲线 {config['name']} 时出错: {str(e)}")
            
            progress_bar.progress((i + 1) / len(configs))
        
        # 保存到状态管理器
        for curve in extracted_curves:
            state_manager.add_curve(curve)
        
        progress_bar.empty()
        status_text.empty()
        
        if extracted_curves:
            st.success(f"🎉 成功提取了 {len(extracted_curves)} 条曲线")
        # 清理临时文件
        self._cleanup_temp_files()
    
    def _execute_batch_extraction(self, uploaded_files, config: Dict[str, Any]):
        """执行批量文件提取"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        extracted_curves = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"处理文件 {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # 保存临时文件
                temp_path = self._save_temp_file(uploaded_file)
                
                # 执行提取
                if config['curve_type'] == 'TIC':
                    curve = st.session_state.rust_bridge.extract_tic_curve(
                        temp_path, config['mz_min'], config['mz_max'],
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                elif config['curve_type'] == 'EIC':
                    target_mz = (config['mz_min'] + config['mz_max']) / 2
                    mz_tolerance = (config['mz_max'] - config['mz_min']) / 2
                    curve = st.session_state.rust_bridge.extract_eic_curve(
                        temp_path, target_mz, mz_tolerance,
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                elif config['curve_type'] == 'BPC':
                    curve = st.session_state.rust_bridge.extract_bpc_curve(
                        temp_path, config['mz_min'], config['mz_max'],
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                
                # 更新曲线元数据
                curve.metadata['original_filename'] = uploaded_file.name
                curve.metadata['extraction_mode'] = 'batch_files'
                curve.metadata.update(config)
                
                extracted_curves.append(curve)
                
            except Exception as e:
                st.error(f"❌ 处理文件 {uploaded_file.name} 时出错: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
                # 保存到状态管理器
        for curve in extracted_curves:
            state_manager.add_curve(curve)
        
        progress_bar.empty()
        status_text.empty()
        
        if extracted_curves:
            st.success(f"🎉 成功提取了 {len(extracted_curves)} 条曲线")
            self._show_extraction_preview(extracted_curves)
        
        # 清理临时文件
        self._cleanup_temp_files()
            
    def _render_extraction_history(self):
        """显示提取历史"""
        curves = state_manager.get_all_curves()
        if curves:
            st.subheader("📋 提取历史")
            
            # 按提取模式分组显示
            single_file_curves = []
            batch_curves = []
            
            for curve in curves.values():
                extraction_mode = curve.metadata.get('extraction_mode', 'unknown')
                if extraction_mode == 'single_file_multi_curve':
                    single_file_curves.append(curve)
                elif extraction_mode == 'batch_files':
                    batch_curves.append(curve)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("单文件多曲线", len(single_file_curves))
                if single_file_curves:
                    with st.expander("查看详情"):
                        for curve in single_file_curves[:5]:  # 显示前5个
                            st.write(f"• {curve.metadata.get('config_name', curve.curve_id)} - {curve.curve_type}")
            
            with col2:
                st.metric("批量文件提取", len(batch_curves))
                if batch_curves:
                    with st.expander("查看详情"):
                        files = set(curve.metadata.get('original_filename', '未知') for curve in batch_curves)
                        st.write(f"涉及文件: {len(files)} 个")
                        for filename in list(files)[:3]:  # 显示前3个文件名
                            st.write(f"• {filename}")
    
    def _render_system_status(self):
        """显示系统状态检查"""
        with st.expander("🔧 系统状态检查", expanded=False):
            st.markdown("**RAW文件支持状态**")
            
            if st.button("🔍 检查.NET运行时状态"):
                try:
                    dotnet_status = st.session_state.rust_bridge.check_system_dotnet_status()
                    
                    if dotnet_status.get('status') == 'available':
                        st.success(f"✅ {dotnet_status.get('message', '.NET可用')}")
                        st.info("📄 系统支持直接读取Thermo RAW文件")
                    else:
                        st.warning("⚠️ .NET 8.0运行时不可用")
                        st.error(f"❌ 错误: {dotnet_status.get('error', '未知错误')}")
                        
                        if 'info' in dotnet_status:
                            st.info(f"💡 {dotnet_status['info']}")
                        
                        if 'download_url' in dotnet_status:
                            st.markdown(f"🔗 **下载.NET 8.0**: {dotnet_status['download_url']}")
                        
                        st.markdown("""
                        **替代方案:**
                        1. 使用ProteoWizard msconvert转换RAW文件为mzML格式
                        2. mzML格式无需额外依赖，兼容性更好
                        """)
                        
                except Exception as e:
                    st.error(f"❌ 检查系统状态时出错: {str(e)}")
            
            st.markdown("---")
            st.markdown("""
            **支持的文件格式:**
            - ✅ **mzML** - 推荐格式，无需额外依赖
            - ✅ **mzXML** - 标准开放格式  
            - ✅ **MGF** - 质谱数据格式
            - ⚠️ **RAW** - 需要.NET 8.0运行时支持
            """)

    def _cleanup_temp_files(self):
        """清理临时文件"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass  # 忽略删除失败
        self.temp_files.clear()
    
    def _execute_single_file_extraction_by_path(self, file_path: str):
        """通过文件路径执行单文件多曲线提取"""
        configs = state_manager.get_extraction_config('single_file_configs')
        
        if not configs:
            st.warning("请至少配置一个曲线提取参数")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        extracted_curves = []
        
        for i, config in enumerate(configs):
            try:
                status_text.text(f"提取曲线: {config['name']}")
                
                # 执行提取
                if config['type'] == 'TIC':
                    curve = st.session_state.rust_bridge.extract_tic_curve(
                        file_path, config['mz_min'], config['mz_max'],
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                elif config['type'] == 'EIC':
                    target_mz = (config['mz_min'] + config['mz_max']) / 2
                    mz_tolerance = (config['mz_max'] - config['mz_min']) / 2
                    curve = st.session_state.rust_bridge.extract_eic_curve(
                        file_path, target_mz, mz_tolerance,
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                elif config['type'] == 'BPC':
                    curve = st.session_state.rust_bridge.extract_bpc_curve(
                        file_path, config['mz_min'], config['mz_max'],
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                
                # 保存曲线到状态管理器
                state_manager.add_curve(curve)
                extracted_curves.append(curve)
                
            except Exception as e:
                st.error(f"❌ 提取曲线 {config['name']} 时出错: {str(e)}")
            
            progress_bar.progress((i + 1) / len(configs))
        
        progress_bar.empty()
        status_text.empty()
    
    def _execute_batch_extraction_by_paths(self, file_paths: List[str], config: Dict[str, Any]):
        """通过文件路径执行批量提取"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        extracted_curves = []
        
        for i, file_path in enumerate(file_paths):
            try:
                file_name = Path(file_path).name
                status_text.text(f"处理文件 {i+1}/{len(file_paths)}: {file_name}")
                
                # 执行提取
                if config['curve_type'] == 'TIC':
                    curve = st.session_state.rust_bridge.extract_tic_curve(
                        file_path, config['mz_min'], config['mz_max'],
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                elif config['curve_type'] == 'EIC':
                    target_mz = (config['mz_min'] + config['mz_max']) / 2
                    mz_tolerance = (config['mz_max'] - config['mz_min']) / 2
                    curve = st.session_state.rust_bridge.extract_eic_curve(
                        file_path, target_mz, mz_tolerance,
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                elif config['curve_type'] == 'BPC':
                    curve = st.session_state.rust_bridge.extract_bpc_curve(
                        file_path, config['mz_min'], config['mz_max'],
                        config['rt_min'], config['rt_max'], config['ms_level']
                    )
                
                # 保存曲线到状态管理器
                state_manager.add_curve(curve)
                extracted_curves.append(curve)
                
            except Exception as e:
                st.error(f"❌ 处理文件 {Path(file_path).name} 时出错: {str(e)}")
            
            progress_bar.progress((i + 1) / len(file_paths))
        
        progress_bar.empty()
        status_text.empty()
        
        if extracted_curves:
            st.success(f"🎉 成功提取了 {len(extracted_curves)} 条曲线")
            self._show_extraction_preview(extracted_curves)
