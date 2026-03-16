# 基础镜像：PyTorch 2.1.0 + CUDA 12.1 + cuDNN 8
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 避免交互式前端
ENV DEBIAN_FRONTEND=noninteractive

# 创建工作目录并设置权限（稍后由用户qing拥有）
WORKDIR /workspace

# 安装系统依赖（包括 Qt xcb 插件所需的所有库）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    vim \
    curl \
    # Qt xcb 插件依赖
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libfontconfig1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libdbus-1-3 \
    dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# 创建用户 qing（UID=1000，与宿主机默认用户一致）
ARG USERNAME=qing
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /workspace && chown $USER_UID:$USER_GID /workspace

# 切换到用户 qing
USER $USERNAME

# 将用户本地 bin 目录加入 PATH（便于使用 --user 安装的工具）
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# 安装 Python 包（使用 --user 安装到用户目录）
RUN pip install --user --no-cache-dir \
    ultralytics \
    opencv-python \
    pyqt5 \
    pyqt5-tools \
    scikit-image \
    pandas \
    matplotlib \
    seaborn \
    jupyter \
    tqdm \
    labelImg \
    comet_ml

# 降级 NumPy 到 <2（兼容 PyTorch 2.1.0）
RUN pip install --user --upgrade "numpy<2"

# 设置工作目录
WORKDIR /workspace

# 默认命令
CMD ["/bin/bash"]