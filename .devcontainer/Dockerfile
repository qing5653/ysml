FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖（增加 bash-completion）
RUN apt-get update && apt-get install -y \
    sudo \
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
    bash-completion \
    # Qt xcb 插件完整依赖
    libxcb-util1 \
    libxcb-image0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxcb-xv0 \
    libxcb-xtest0 \
    libxcb-glx0 \
    libxcb-present0 \
    libxcb-dri2-0 \
    libxcb-dri3-0 \
    libxcb-composite0 \
    libxcb-damage0 \
    libxcb-xinput0 \
    libxcb-xrm0 \
    libxcb-cursor0 \
    libfontconfig1 \
    libfreetype6 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libdbus-1-3 \
    dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# 创建用户 qing（UID=1000）
ARG USERNAME=qing
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /workspace && chown $USER_UID:$USER_GID /workspace

USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# 安装 Python 包（固定 numpy<2，使用 opencv-headless）
RUN pip install --user --no-cache-dir \
    "numpy<2" \
    ultralytics \
    opencv-python-headless \
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

# 启用 bash-completion（需要添加到 .bashrc）
RUN echo 'if [ -f /etc/bash_completion ]; then . /etc/bash_completion; fi' >> /home/$USERNAME/.bashrc \
    && echo 'if command -v python >/dev/null 2>&1; then QT_PLUGIN_DIR=$(python -c "import os,PyQt5;print(os.path.join(os.path.dirname(PyQt5.__file__),\"Qt5\",\"plugins\"))" 2>/dev/null); if [ -n "$QT_PLUGIN_DIR" ]; then export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_DIR"; export QT_PLUGIN_PATH="$QT_PLUGIN_DIR"; fi; fi' >> /home/$USERNAME/.bashrc \
    && echo 'export PS1="\u@\h:\w\$ "' >> /home/$USERNAME/.bashrc

WORKDIR /ysml
CMD ["/bin/bash"]