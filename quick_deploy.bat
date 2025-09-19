@echo off
echo ========================================
echo   PeakAnalyzer 快速部署 (Windows)
echo ========================================

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 检查 uv
uv --version >nul 2>&1
if errorlevel 1 (
    echo 📦 安装 uv 包管理器...
    curl -LsSf https://astral.sh/uv/install.ps1 | powershell
    if errorlevel 1 (
        echo ❌ uv 安装失败
        pause
        exit /b 1
    )
)

:: 检查 Rust
rustc --version >nul 2>&1
if errorlevel 1 (
    echo 🦀 安装 Rust 工具链...
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    call %USERPROFILE%\.cargo\env.bat
)

:: 创建虚拟环境
echo 🐍 创建 Python 虚拟环境...
uv venv
if errorlevel 1 (
    echo ❌ 虚拟环境创建失败
    pause
    exit /b 1
)

:: 安装依赖
echo 📦 安装 Python 依赖...
.venv\Scripts\uv pip install -r python\requirements.txt
.venv\Scripts\uv pip install maturin

:: 编译 Rust 扩展
echo 🔨 编译 Rust 扩展...
cd python
..\.venv\Scripts\maturin develop --release
if errorlevel 1 (
    echo ❌ Rust 扩展编译失败
    cd ..
    pause
    exit /b 1
)
cd ..

:: 验证安装
echo ✅ 验证安装...
.venv\Scripts\python -c "import peakanalyzer_scripts; print('✅ 安装成功')"
if errorlevel 1 (
    echo ❌ 安装验证失败
    pause
    exit /b 1
)

:: 创建启动脚本
echo @echo off > start_app.bat
echo cd /d "%%~dp0python" >> start_app.bat
echo ..\\.venv\\Scripts\\streamlit run main.py >> start_app.bat
echo pause >> start_app.bat

echo.
echo 🎉 部署完成！
echo ========================================
echo 启动方式：
echo   1. 双击 start_app.bat
echo   2. 或运行: start_app.bat
echo.
echo 应用将在 http://localhost:8501 启动
echo ========================================

pause
