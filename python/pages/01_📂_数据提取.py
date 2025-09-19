"""
数据提取页面
"""

import streamlit as st
from core.state_manager import state_manager
from core.rust_bridge import rust_bridge
from ui.pages.data_extraction_page import DataExtractionPage
from ui.sidebar import show_sidebar


def main():
    """数据提取页面主函数"""
    # 显示侧边栏
    show_sidebar()

    # 创建数据提取页面实例
    data_extraction_page = DataExtractionPage()
    data_extraction_page.render()


if __name__ == "__main__":
    main()
