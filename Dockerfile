# PeakAnalyzer Docker 部署文件
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 安装 .NET 8.0 Runtime (可选，用于 RAW 文件支持)
RUN curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 8.0 --runtime dotnet
ENV PATH="/root/.dotnet:${PATH}"

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 创建虚拟环境并安装依赖
RUN uv venv && \
    .venv/bin/uv pip install -r python/requirements.txt && \
    .venv/bin/uv pip install maturin

# 编译 Rust 扩展
WORKDIR /app/python
RUN ../.venv/bin/maturin develop --release

# 暴露端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 启动命令
CMD ["../.venv/bin/streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
