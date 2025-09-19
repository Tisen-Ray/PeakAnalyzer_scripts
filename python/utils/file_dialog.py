"""
文件对话框工具 - 使用streamlit_file_dialog或streamlit原生组件
"""

import streamlit as st
import threading
from typing import List, Optional, Tuple
import os

# 尝试导入tkinter
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False


class FileDialogManager:
    """文件对话框管理器 - 支持多种后端"""
    
    def __init__(self):
        self.backend = self._detect_backend()
    
    def _detect_backend(self) -> str:
        """检测可用的文件对话框后端"""
        if HAS_TKINTER:
            return "tkinter"
        else:
            # 如果没有tkinter，使用streamlit原生组件
            return "streamlit_native"
    
    
    def _show_dialog_streamlit_native(self, dialog_type: str, title: str, 
                                    filetypes: List[Tuple[str, str]], 
                                    multiple: bool = False) -> Optional[str]:
        """使用Streamlit原生文件上传组件"""
        try:
            if dialog_type == "open":
                # 构建accept参数
                accept_types = []
                for name, pattern in filetypes:
                    if pattern != "*.*":
                        # 转换文件类型模式
                        extensions = pattern.replace("*", "").split(" ")
                        accept_types.extend(extensions)
                
                # 使用streamlit的file_uploader
                uploaded_files = st.file_uploader(
                    title,
                    accept_multiple_files=multiple,
                    type=accept_types if accept_types else None,
                    key=f"file_upload_{hash(title)}"
                )
                
                if uploaded_files:
                    if multiple:
                        # 保存临时文件并返回路径
                        temp_paths = []
                        for uploaded_file in uploaded_files:
                            temp_path = f"temp_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_paths.append(temp_path)
                        return temp_paths
                    else:
                        # 单个文件
                        temp_path = f"temp_{uploaded_files.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_files.getbuffer())
                        return temp_path
                return None
            else:
                st.info("请手动输入保存路径")
                return st.text_input("保存路径:")
        except Exception as e:
            st.error(f"文件上传错误: {str(e)}")
            return None
    
    def _show_dialog_tkinter(self, dialog_type: str, title: str, filetypes: List[Tuple[str, str]], 
                           multiple: bool = False) -> Optional[str]:
        """使用tkinter显示文件对话框 - 线程安全版本"""
        import queue
        import threading
        
        result_queue = queue.Queue()
        
        def run_dialog():
            try:
                import tkinter as tk
                from tkinter import filedialog
                
                # 创建隐藏的根窗口
                root = tk.Tk()
                root.withdraw()  # 隐藏主窗口
                root.attributes('-topmost', True)  # 保持在最前面
                
                try:
                    if dialog_type == "open":
                        if multiple:
                            files = filedialog.askopenfilenames(
                                title=title,
                                filetypes=filetypes,
                                parent=root
                            )
                            result = list(files) if files else None
                        else:
                            result = filedialog.askopenfilename(
                                title=title,
                                filetypes=filetypes,
                                parent=root
                            )
                            result = result if result else None
                    elif dialog_type == "save":
                        result = filedialog.asksaveasfilename(
                            title=title,
                            filetypes=filetypes,
                            defaultextension=filetypes[0][1].split(';')[0] if filetypes else '.txt',
                            parent=root
                        )
                        result = result if result else None
                    elif dialog_type == "directory":
                        result = filedialog.askdirectory(title=title, parent=root)
                        result = result if result else None
                    else:
                        result = None
                        
                    result_queue.put(('success', result))
                    
                except Exception as e:
                    result_queue.put(('error', str(e)))
                finally:
                    root.quit()
                    root.destroy()
                    
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        # 在新线程中运行对话框
        dialog_thread = threading.Thread(target=run_dialog, daemon=True)
        dialog_thread.start()
        
        # 等待结果，最多等待30秒
        try:
            status, result = result_queue.get(timeout=30)
            if status == 'success':
                return result
            else:
                st.error(f"文件对话框错误: {result}")
                return None
        except queue.Empty:
            st.error("文件对话框超时")
            return None
        except Exception as e:
            st.error(f"文件对话框错误: {str(e)}")
            return None
    
    def _show_dialog(self, dialog_type: str, title: str, filetypes: List[Tuple[str, str]], 
                    multiple: bool = False) -> Optional[str]:
        """显示文件对话框 - 自动选择后端"""
        if self.backend == "tkinter":
            return self._show_dialog_tkinter(dialog_type, title, filetypes, multiple)
        else:
            return self._show_dialog_streamlit_native(dialog_type, title, filetypes, multiple)
    
    def select_files(self, title: str = "选择文件", 
                    filetypes: List[Tuple[str, str]] = None,
                    multiple: bool = True) -> Optional[List[str]]:
        """选择文件"""
        if filetypes is None:
            filetypes = [
                ("所有文件", "*.*"),
                ("质谱文件", "*.mzML *.mzXML *.raw"),
                ("Excel文件", "*.xlsx *.xls"),
                ("CSV文件", "*.csv"),
                ("文本文件", "*.txt")
            ]
        
        result = self._show_dialog("open", title, filetypes, multiple)
        return result
    
    def select_save_file(self, title: str = "保存文件",
                        filetypes: List[Tuple[str, str]] = None,
                        default_name: str = "") -> Optional[str]:
        """选择保存文件"""
        if filetypes is None:
            filetypes = [
                ("JSON文件", "*.json"),
                ("Excel文件", "*.xlsx"),
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ]
        
        # 设置默认文件名
        if default_name:
            initial_file = default_name
        else:
            initial_file = ""
        
        result = self._show_dialog("save", title, filetypes)
        return result
    
    def select_directory(self, title: str = "选择文件夹") -> Optional[str]:
        """选择文件夹"""
        return self._show_dialog("directory", title, [])


# 全局文件对话框管理器实例
file_dialog_manager = FileDialogManager()


def select_files(title: str = "选择文件", 
                filetypes: List[Tuple[str, str]] = None,
                multiple: bool = True) -> Optional[List[str]]:
    """选择文件的便捷函数"""
    return file_dialog_manager.select_files(title, filetypes, multiple)


def select_save_file(title: str = "保存文件",
                    filetypes: List[Tuple[str, str]] = None,
                    default_name: str = "") -> Optional[str]:
    """选择保存文件的便捷函数"""
    return file_dialog_manager.select_save_file(title, filetypes, default_name)


def select_directory(title: str = "选择文件夹") -> Optional[str]:
    """选择文件夹的便捷函数"""
    return file_dialog_manager.select_directory(title)


def show_file_selection_button(button_text: str = "📁 选择文件",
                              title: str = "选择文件",
                              filetypes: List[Tuple[str, str]] = None,
                              multiple: bool = True,
                              key: str = None) -> Optional[List[str]]:
    """显示文件选择按钮的便捷函数"""
    import streamlit as st
    
    if st.button(button_text, key=key):
        selected_files = select_files(title, filetypes, multiple)
        if selected_files:
            if multiple and isinstance(selected_files, list):
                st.success(f"✅ 已选择 {len(selected_files)} 个文件")
                for i, file_path in enumerate(selected_files):
                    st.write(f"📄 {i+1}. {os.path.basename(file_path)}")
            else:
                if isinstance(selected_files, list):
                    selected_files = selected_files[0] if selected_files else None
                if selected_files:
                    st.success(f"✅ 已选择文件: {os.path.basename(selected_files)}")
                    selected_files = [selected_files]  # 转换为列表格式
            return selected_files
        else:
            st.warning("未选择任何文件")
            return None
    return None


def show_save_file_button(button_text: str = "💾 保存文件",
                         title: str = "保存文件",
                         filetypes: List[Tuple[str, str]] = None,
                         default_name: str = "",
                         key: str = None) -> Optional[str]:
    """显示保存文件按钮的便捷函数"""
    import streamlit as st
    
    if st.button(button_text, key=key):
        save_path = select_save_file(title, filetypes, default_name)
        if save_path:
            st.success(f"将保存到: {os.path.basename(save_path)}")
            return save_path
        else:
            st.warning("未选择保存路径")
            return None
    return None


def show_directory_selection_button(button_text: str = "📂 选择文件夹",
                                   title: str = "选择文件夹",
                                   key: str = None) -> Optional[str]:
    """显示文件夹选择按钮的便捷函数"""
    import streamlit as st
    
    if st.button(button_text, key=key):
        directory = select_directory(title)
        if directory:
            st.success(f"已选择文件夹: {os.path.basename(directory)}")
            return directory
        else:
            st.warning("未选择文件夹")
            return None
    return None
