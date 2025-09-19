#!/usr/bin/env python3
"""
PeakAnalyzer - 质谱数据峰分析工具
主入口文件，启动Streamlit应用
"""

import streamlit as st
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数入口"""
    from app import main as app_main
    app_main()

if __name__ == "__main__":
    main()
