"""
曲线处理页面
"""

import streamlit as st
from core.state_manager import state_manager
from ui.pages.curve_processing_page import CurveProcessingPage
from ui.sidebar import show_sidebar


def main():
    """曲线处理页面主函数"""
    # 显示侧边栏
    show_sidebar()

    # 创建曲线处理页面实例
    curve_processing_page = CurveProcessingPage()
    curve_processing_page.render()


if __name__ == "__main__":
    main()
