#!/usr/bin/env python3
"""
PeakAnalyzer 一键部署脚本
支持 Windows, Linux, macOS
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """执行命令并处理错误"""
    print(f"🔧 执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_command_exists(cmd):
    """检查命令是否存在"""
    return shutil.which(cmd) is not None

def install_uv():
    """安装 uv 包管理器"""
    print("📦 安装 uv 包管理器...")
    
    system = platform.system()
    if system == "Windows":
        cmd = "curl -LsSf https://astral.sh/uv/install.ps1 | powershell"
    else:
        cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    
    run_command(cmd, check=False)

def install_rust():
    """安装 Rust 工具链"""
    print("🦀 安装 Rust 工具链...")
    
    if check_command_exists("rustc"):
        print("✅ Rust 已安装")
        return
    
    cmd = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    run_command(cmd, check=False)
    
    # 添加 Rust 到 PATH
    if platform.system() != "Windows":
        os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:{os.environ['PATH']}"

def check_dotnet():
    """检查 .NET Runtime"""
    print("🔍 检查 .NET 8.0 Runtime...")
    
    try:
        result = run_command("dotnet --version", check=False)
        if result.returncode == 0:
            version = result.stdout.strip()
            if version.startswith("8."):
                print(f"✅ .NET Runtime {version} 已安装")
                return True
            else:
                print(f"⚠️  发现 .NET {version}，但需要 8.x 版本")
        else:
            print("⚠️  未找到 .NET Runtime")
    except:
        print("⚠️  .NET Runtime 检查失败")
    
    print("💡 请手动安装 .NET 8.0 Runtime:")
    print("   Windows: https://dotnet.microsoft.com/download/dotnet/8.0")
    print("   Linux:   sudo apt-get install dotnet-runtime-8.0")
    print("   macOS:   brew install dotnet")
    
    return False

def setup_python_environment():
    """设置 Python 环境"""
    print("🐍 设置 Python 环境...")
    
    # 检查 uv
    if not check_command_exists("uv"):
        print("❌ uv 未安装，正在安装...")
        install_uv()
    
    # 创建虚拟环境
    print("📁 创建虚拟环境...")
    run_command("uv venv", cwd=".")
    
    # 激活虚拟环境并安装依赖
    print("📦 安装 Python 依赖...")
    
    system = platform.system()
    if system == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\uv pip install"
    else:
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/uv pip install"
    
    # 安装 Python 依赖
    run_command(f"{pip_cmd} -r python/requirements.txt")
    
    # 安装 maturin
    run_command(f"{pip_cmd} maturin")

def compile_rust_extension():
    """编译 Rust 扩展"""
    print("🔨 编译 Rust 扩展...")
    
    system = platform.system()
    if system == "Windows":
        maturin_cmd = ".venv\\Scripts\\maturin"
    else:
        maturin_cmd = ".venv/bin/maturin"
    
    # 编译 Rust 扩展
    run_command(f"{maturin_cmd} develop --release", cwd="python")

def verify_installation():
    """验证安装"""
    print("✅ 验证安装...")
    
    system = platform.system()
    if system == "Windows":
        python_cmd = ".venv\\Scripts\\python"
    else:
        python_cmd = ".venv/bin/python"
    
    # 测试 Rust 模块导入
    test_cmd = f'{python_cmd} -c "import peakanalyzer_scripts; print(\\"✅ Rust 模块安装成功\\")"'
    run_command(test_cmd, cwd="python")

def create_start_script():
    """创建启动脚本"""
    print("📝 创建启动脚本...")
    
    system = platform.system()
    
    if system == "Windows":
        script_content = """@echo off
cd /d "%~dp0python"
..\\.venv\\Scripts\\streamlit run main.py
pause
"""
        script_path = "start.bat"
    else:
        script_content = """#!/bin/bash
cd "$(dirname "$0")/python"
../.venv/bin/streamlit run main.py
"""
        script_path = "start.sh"
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    if system != "Windows":
        os.chmod(script_path, 0o755)
    
    print(f"✅ 启动脚本已创建: {script_path}")

def main():
    """主函数"""
    print("🚀 PeakAnalyzer 一键部署脚本")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("Cargo.toml").exists():
        print("❌ 请在项目根目录运行此脚本")
        sys.exit(1)
    
    # 检查 Python 版本
    if sys.version_info < (3, 10):
        print("❌ 需要 Python 3.10 或更高版本")
        sys.exit(1)
    
    print(f"✅ Python {sys.version}")
    print(f"✅ 系统: {platform.system()} {platform.release()}")
    
    try:
        # 1. 安装 Rust
        install_rust()
        
        # 2. 检查 .NET
        check_dotnet()
        
        # 3. 设置 Python 环境
        setup_python_environment()
        
        # 4. 编译 Rust 扩展
        compile_rust_extension()
        
        # 5. 验证安装
        verify_installation()
        
        # 6. 创建启动脚本
        create_start_script()
        
        print("\n🎉 部署完成！")
        print("=" * 50)
        print("启动方式:")
        
        system = platform.system()
        if system == "Windows":
            print("  双击 start.bat")
            print("  或运行: .\\start.bat")
        else:
            print("  运行: ./start.sh")
            print("  或手动: cd python && ../.venv/bin/streamlit run main.py")
        
        print("\n应用将在 http://localhost:8501 启动")
        print("\n🔧 如需帮助，请查看 README 文档")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断部署")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 部署失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
