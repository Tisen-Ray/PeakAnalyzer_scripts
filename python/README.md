# PeakAnalyzer - 质谱峰分析工具

一个基于Rust+Python的质谱数据峰分析工具，提供高性能的曲线提取和智能的峰检测分析功能。

## 🌟 主要特性

### 🔧 后端特性 (Rust)
- **高性能数据提取**: 基于mzdata库，支持多种MS数据格式
- **多种曲线类型**: TIC、EIC、BPC曲线提取
- **批量处理**: 支持大规模数据文件的批量处理
- **内存优化**: 高效的数据结构和内存管理

### 🎨 前端特性 (Python + Streamlit)
- **直观的Web界面**: 基于Streamlit的现代化用户界面
- **数据处理功能**: 基线校正、平滑、归一化等
- **峰检测算法**: 多种峰检测方法和参数调优
- **交互式可视化**: 基于Plotly的高质量图表
- **缓存管理**: 智能缓存系统，避免重复计算
- **多格式导出**: CSV、Excel、JSON、PDF报告

## 📁 项目结构

```
PeakAnalyzer_scripts/
├── src/                    # Rust后端源码
│   ├── lib.rs             # Python接口定义
│   ├── curve.rs           # 曲线数据结构
│   ├── data_loader.rs     # mzdata数据加载器
│   ├── curve_extractor.rs # 曲线提取器
│   └── errors.rs          # 错误处理
├── python_app/            # Python前端应用
│   ├── core/              # 核心模块
│   │   ├── curve.py       # Python曲线数据结构
│   │   ├── rust_bridge.py # Rust桥接
│   │   └── data_processor.py # 数据处理
│   ├── ui/                # 用户界面
│   │   └── app.py         # Streamlit主应用
│   ├── peak_analysis/     # 峰分析模块
│   │   ├── peak_detector.py # 峰检测器
│   │   ├── peak_analyzer.py # 峰分析器
│   │   └── peak_fitter.py   # 峰拟合器
│   ├── cache/             # 缓存管理
│   ├── export/            # 结果导出
│   └── utils/             # 工具模块
└── Cargo.toml             # Rust项目配置
```

## 🚀 快速开始

### 1. 环境准备

#### 系统要求
- Python 3.8+
- Rust 1.70+
- CMake (用于编译mzdata依赖)

#### 安装依赖

```bash
# 安装Python依赖
cd python_app
pip install -r requirements.txt

# 编译Rust模块 (可选，不编译将使用模拟数据)
cd ..
pip install maturin
maturin develop
```

### 2. 启动应用

```bash
# 方法1: 使用启动脚本
cd python_app
python run.py

# 方法2: 直接启动Streamlit
streamlit run main.py
```

应用将在 http://localhost:8501 打开

### 3. 使用指南

#### 数据提取
1. 在"数据提取"页面上传MS数据文件
2. 设置提取参数（曲线类型、m/z范围、RT范围等）
3. 点击"开始提取"获得曲线数据

#### 曲线分析
1. 在"曲线分析"页面选择要分析的曲线
2. 配置数据处理选项（基线校正、平滑、归一化）
3. 查看处理前后的曲线对比

#### 峰检测
1. 在"峰检测"页面选择曲线
2. 选择检测方法并调整参数
3. 查看检测结果和峰标注图

#### 批量处理
1. 在"批量处理"页面上传多个文件
2. 设置统一的处理参数
3. 一键完成批量提取和分析

## 🔬 功能详解

### 曲线提取
- **TIC (Total Ion Current)**: 总离子流图，显示所有离子的总强度随时间变化
- **EIC (Extracted Ion Current)**: 提取离子流图，显示特定m/z离子的强度变化
- **BPC (Base Peak Chromatogram)**: 基峰图，显示每个时间点的最强峰

### 数据处理
- **基线校正**: ALS、多项式拟合、滚动球算法
- **平滑处理**: Savitzky-Golay、高斯滤波、移动平均
- **归一化**: Min-Max、Z-score、面积归一化

### 峰检测算法
- **scipy.find_peaks**: 基于scipy的经典峰检测
- **CWT**: 连续小波变换峰检测
- **导数法**: 基于一阶导数的峰检测
- **阈值法**: 简单阈值峰检测

### 峰分析参数
- **峰面积**: 使用梯形积分计算
- **FWHM**: 半峰宽，峰宽度指标
- **信噪比**: 信号与噪声的比值
- **不对称因子**: 峰形对称性指标
- **拖尾因子**: 峰拖尾程度
- **分辨率**: 相邻峰的分离度

### 缓存系统
- **智能缓存**: 基于文件和参数的缓存键
- **内存管理**: LRU缓存策略
- **持久化**: 磁盘缓存支持
- **缓存统计**: 详细的缓存使用情况

## 📊 导出格式

### 支持的导出格式
- **CSV**: 曲线数据和峰信息的表格格式
- **Excel**: 多工作表的详细报告
- **JSON**: 完整的数据结构，包含元数据
- **PDF**: 图文并茂的分析报告

### 导出内容
- 曲线摘要信息
- 峰检测结果
- 处理参数记录
- 统计分析数据
- 可视化图表

## 🛠️ 开发指南

### 添加新的峰检测算法

1. 在 `peak_analysis/peak_detector.py` 中添加新方法
2. 在 `detection_methods` 字典中注册
3. 实现 `_detect_peaks_[method_name]` 函数

### 添加新的数据处理方法

1. 在 `core/data_processor.py` 中添加静态方法
2. 在 `process_curve` 方法中添加处理逻辑
3. 更新UI中的选项

### 扩展Rust后端

1. 在 `src/` 目录下修改相应模块
2. 更新 `src/lib.rs` 中的Python接口
3. 运行 `maturin develop` 重新编译

## 🐛 调试和故障排除

### 常见问题

**Q: Rust模块加载失败**
A: 确保已安装CMake，然后运行 `maturin develop` 编译模块

**Q: 内存使用过高**
A: 调整缓存设置，减少同时处理的文件数量

**Q: 峰检测结果不理想**
A: 尝试不同的检测方法，调整参数，或先进行数据预处理

### 调试模式

```bash
# 启用详细日志
export RUST_LOG=debug
python run.py

# 查看缓存状态
# 在应用中访问"缓存管理"页面
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情

## 🙏 致谢

- [mzdata](https://github.com/mobiusklein/mzdata) - Rust质谱数据处理库
- [Streamlit](https://streamlit.io/) - Python Web应用框架
- [Plotly](https://plotly.com/) - 交互式可视化库
- [SciPy](https://scipy.org/) - 科学计算库

---

**开发团队**: PeakAnalyzer Development Team  
**版本**: 1.0.0  
**最后更新**: 2024年9月
