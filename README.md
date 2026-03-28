# ysml

工业轴承缺陷检测工程（YOLO11），包含数据准备、训练评估、蒸馏剪枝、报告生成与 PyQt5 可视化系统。

## 项目目标

- 可复现实验链路：数据 -> 训练 -> 评估 -> 报告
- 轻量化验证链路：蒸馏 -> 剪枝 -> 稀疏率与性能证据
- 可演示系统：实时检测 + 批处理检测 + FPS 反馈

## 当前目录（核心）

```text
configs/                     # 全部配置入口
scripts/
  cli.py                     # 统一命令入口
  app.py                     # GUI 启动入口（含 Qt/字体运行时修复）
  ops/
    common.py                # 公共工具与共享路径解析
    data.py                  # 数据检查与增强构建
    model.py                 # 训练/验证/推理/导出/FPS基准
    compress.py              # 蒸馏与剪枝
    report.py                # 实验报告生成
    system.py                # doctor/clean 维护命令
  ui/window.py               # PyQt5 主窗口
src/yolo11_project/
  spot_guided.py             # Spot-Guided 候选区域增强
docs/
  README.md                  # 文档索引
  experiment_report.md       # 自动生成实验报告
  thesis_experiment_section_template.md
  defense_full_deck_template.md
assets/fonts/
  NotoSansCJKsc-Regular.otf  # 内置中文字体（避免方块字）
tests/                       # 配置与结构基础测试
```

## 环境准备

1. Python 3.10+（建议与当前工程一致）
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行诊断：

```bash
python3 scripts/cli.py doctor
```

## 统一命令

所有流程统一使用：

```bash
python3 scripts/cli.py <command>
```

支持命令：

- `doctor`：环境与关键配置自检（依赖、数据配置、字体文件）
- `check`：数据集结构与标签质量检查
- `prepare`：构建增强数据集
- `train`：训练基线模型
- `val`：验证模型指标
- `predict`：批量推理并保存可视化结果
- `benchmark`：离线 FPS 基准测试
- `report`：生成实验报告 `docs/experiment_report.md`
- `export`：导出 ONNX
- `distill`：蒸馏训练
- `prune`：结构化剪枝与微调
- `clean`：清理缓存与临时产物

## 推荐执行顺序

```bash
python3 scripts/cli.py doctor
python3 scripts/cli.py check
python3 scripts/cli.py train
python3 scripts/cli.py val
python3 scripts/cli.py benchmark
python3 scripts/cli.py report
```

轻量化链路：

```bash
python3 scripts/cli.py distill
python3 scripts/cli.py prune
python3 scripts/cli.py report
```

## GUI 使用

启动：

```bash
python3 scripts/app.py
```

功能：

- 摄像头实时检测
- 文件夹批处理检测
- 实时 FPS 与目标提示（25 FPS）
- Spot-Guided 推理开关

说明：项目已内置中文字体，容器环境下无需额外安装系统中文字体即可正常显示中文 UI。

## 关键配置文件

- `configs/train.yaml`：训练主配置、预测配置、benchmark 配置、research claims
- `configs/prepare.yaml`：增强与 Spot-Guided 参数
- `configs/distill.yaml`：蒸馏配置（复用数据、热启动、Comet 开关）
- `configs/prune.yaml`：剪枝与微调配置（稀疏率证据输出、热启动、标签去重）
- `configs/dataset.yaml`：当前激活数据集
- `configs/datasets_registry.yaml`：可用数据集登记

## 输出产物约定

- 训练产物：`experiments/<run_name>/weights/best.pt`
- 推理图像：`experiments/predict/`
- FPS 基准：`experiments/benchmark/fps.csv`
- 剪枝证据：`experiments/benchmark/prune_stats.json`
- 实验报告：`docs/experiment_report.md`

## 维护约定

- 尽量通过 `scripts/ops/common.py` 复用通用逻辑，避免跨模块重复实现。
- 保持命令入口单一（`scripts/cli.py`），减少“孤立脚本”。
- 报告优先引用自动生成证据（CSV/JSON）而非手填数据。
- 运行结束建议执行：

```bash
python3 scripts/cli.py clean
```
