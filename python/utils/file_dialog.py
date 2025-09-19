"""
文件对话框工具 - 使用tkinter提供文件选择功能
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from typing import List, Optional, Tuple
import os


class FileDialogManager:
    """文件对话框管理器"""
    
    def __init__(self):
        self.root = None
        self.result = None
        self.dialog_thread = None
    
    def _show_dialog(self, dialog_type: str, title: str, filetypes: List[Tuple[str, str]], 
                    multiple: bool = False) -> Optional[str]:
        """显示文件对话框"""
        try:
            # 创建隐藏的根窗口
            self.root = tk.Tk()
            self.root.withdraw()  # 隐藏主窗口
            
            # 设置窗口图标（如果有的话）
            try:
                self.root.iconbitmap(default='icon.ico')
            except:
                pass
            
            if dialog_type == "open":
                if multiple:
                    files = filedialog.askopenfilenames(
                        title=title,
                        filetypes=filetypes
                    )
                    return list(files) if files else None
                else:
                    return filedialog.askopenfilename(
                        title=title,
                        filetypes=filetypes
                    )
            elif dialog_type == "save":
                return filedialog.asksaveasfilename(
                    title=title,
                    filetypes=filetypes,
                    defaultextension=filetypes[0][1].split(';')[0] if filetypes else '.txt'
                )
            elif dialog_type == "directory":
                return filedialog.askdirectory(title=title)
                
        except Exception as e:
            messagebox.showerror("错误", f"文件对话框错误: {str(e)}")
            return None
        finally:
            if self.root:
                self.root.destroy()
                self.root = None
    
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
                st.success(f"已选择 {len(selected_files)} 个文件")
                for i, file_path in enumerate(selected_files):
                    st.write(f"{i+1}. {os.path.basename(file_path)}")
            else:
                st.success(f"已选择文件: {os.path.basename(selected_files)}")
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
