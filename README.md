# ysml

基于深度学习的工业轴承缺陷检测系统，已整理为可直接开展 YOLO11 训练、验证、推理与导出工作的工程结构。

## 环境要求
- Docker Engine >= 19.03
- NVIDIA Container Toolkit（用于 GPU 支持）
- 支持 X11 转发的 Linux 系统（用于 GUI 显示）

## 已完成优化
- 训练权重路径从脚本目录解析，避免受当前工作目录影响。
- 测试脚本移除无关顶层导入，减少启动失败面。
- Qt 插件路径改为运行时动态检测，避免 Python 小版本升级导致失效。
- devcontainer 挂载路径改为相对 .devcontainer 的稳定写法。

## 项目目录
```text
ysml/
├── .devcontainer/
├── configs/
│   ├── dataset.yaml
│   └── train.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── datasets/
├── experiments/
├── models/
│   ├── exports/
│   └── weights/
├── notebooks/
├── scripts/
│   ├── train.py
│   ├── val.py
│   ├── predict.py
│   └── export.py
├── src/
│   ├── test.py
│   ├── yolo11n.pt
│   └── yolo11_project/
│       ├── __init__.py
│       └── paths.py
├── tests/
│   └── test_project_structure.py
├── .gitignore
├── requirements.txt
└── README.md
```

## YOLO11 使用流程
1. 准备数据集目录
	按 configs/dataset.yaml 约定放置数据：
	datasets/bearing_defect/images/train
	datasets/bearing_defect/images/val
	datasets/bearing_defect/images/test
	以及对应 labels 目录

2. 根据任务修改配置
	configs/dataset.yaml：数据路径与类别名
	configs/train.yaml：模型、轮数、批量大小、设备号等

3. 训练
```bash
python scripts/train.py
```

4. 验证
```bash
python scripts/val.py
```

5. 推理
```bash
python scripts/predict.py
```

6. 导出 ONNX
```bash
python scripts/export.py
```

## 依赖安装（非容器）
```bash
pip install -r requirements.txt
```

