"""
结果可视化页面
"""

import streamlit as st
from core.state_manager import state_manager
from ui.pages.visualization_page import VisualizationPage
from ui.sidebar import show_sidebar


def main():
    """结果可视化页面主函数"""
    # 显示侧边栏
    show_sidebar()
    # 创建可视化页面实例
    visualization_page = VisualizationPage()
    visualization_page.render()


if __name__ == "__main__":
    main()
