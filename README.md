# ysml

基于 YOLO11 的工业轴承缺陷检测工程，面向科研与工程落地，覆盖：
- 数据体检与增强（旋转、亮度、MixUp、Spot-Guided）
- 训练、验证、推理、导出
- 蒸馏与剪枝（轻量化）
- PyQt5 实时/批处理检测界面

当前主链路仅维护 NEU-DET（fa031-main）。

## 1. 环境准备

### 1.1 Python 版本
建议 Python 3.10 或 3.11。

### 1.2 安装依赖
```bash
pip install -r requirements.txt
```

### 1.3 快速自检
```bash
python3 scripts/cli.py check
```

## 2. 项目结构

```text
configs/                 # 训练、数据与增强等配置
scripts/
  cli.py                 # 统一命令入口
  app.py                 # UI 启动入口
  ops/
    common.py            # 公共工具与数据结构
    data.py              # 数据体检与增强
    model.py             # 训练/验证/推理/导出
    compress.py          # 蒸馏与剪枝
    system.py            # 清理命令
  ui/
    window.py            # PyQt5 主界面
data/fa031-main/NEU-DET/ # 当前默认数据集
datasets/                # 增强/蒸馏中间数据集输出
experiments/             # 训练、推理与UI输出
models/                  # 导出模型与压缩模型
src/                     # 业务模块（含 spot_guided）
```

## 3. 统一命令入口

所有命令统一通过 [scripts/cli.py](scripts/cli.py) 执行：

```bash
python3 scripts/cli.py <command>
```

可选 `command`：
- `check`：数据集体检
- `prepare`：生成增强数据集
- `train`：训练
- `val`：验证
- `predict`：批量推理（默认走验证集路径）
- `benchmark`：离线FPS基准（支持 guided on/off 对比）
- `report`：生成实验报告（精度/体积/FPS汇总）
- `export`：导出 ONNX
- `distill`：蒸馏训练
- `prune`：结构化剪枝 + 微调
- `clean`：清理缓存

## 4. 快速开始（推荐顺序）

### 4.1 数据体检
```bash
python3 scripts/cli.py check
```
检查项包括：目录完整性、图像/标签计数、当前正在使用的数据集路径。

### 4.2 基线训练
```bash
python3 scripts/cli.py train
```
默认读取 [configs/train.yaml](configs/train.yaml)。

### 4.3 验证
```bash
python3 scripts/cli.py val
```
默认读取 `experiments/<name>/weights/best.pt` 进行验证。

### 4.4 推理
```bash
python3 scripts/cli.py predict
```
输出到 `experiments/predict/`。

### 4.5 导出
```bash
python3 scripts/cli.py export
```
导出 ONNX（dynamic + simplify）。

### 4.6 FPS 基准
```bash
python3 scripts/cli.py benchmark
```
输出 CSV 默认路径：`experiments/benchmark/fps.csv`。

### 4.7 生成实验报告
```bash
python3 scripts/cli.py report
```
输出 Markdown 报告：`docs/experiment_report.md`。

## 5. 数据增强链路

执行：
```bash
python3 scripts/cli.py prepare
```

默认配置位于 [configs/prepare.yaml](configs/prepare.yaml)，会在 `datasets/bearing_defect_aug/` 生成增强数据。增强策略包括：
- 90 度旋转
- 亮度扰动
- MixUp
- Spot-Guided 候选区域增强

如需训练增强数据，请在 [configs/train.yaml](configs/train.yaml) 中开启：
- `use_prepared_dataset: true`
- `prepared_dataset_yaml: configs/dataset_prepared.yaml`

## 6. 轻量化链路（蒸馏 + 剪枝）

### 6.1 蒸馏
```bash
python3 scripts/cli.py distill
```
配置文件：[configs/distill.yaml](configs/distill.yaml)

断点续跑相关开关：
- `reuse_generated_data`：复用已生成蒸馏数据，避免重复构建伪标签数据集
- `train.warm_start_from_last`：自动从 `experiments/<name*>/weights/last.pt` 热启动

### 6.2 剪枝
```bash
python3 scripts/cli.py prune
```
配置文件：[configs/prune.yaml](configs/prune.yaml)

断点续跑相关开关：
- `finetune.warm_start_from_last`：自动从 `experiments/<name*>/weights/last.pt` 热启动微调
- `finetune.dedupe_labels_on_start`：微调前自动去重标签重复行

证据链相关开关：
- `stats_output`：剪枝统计输出文件（默认 `experiments/benchmark/prune_stats.json`）

## 7. UI 系统

启动界面：
```bash
python3 scripts/app.py
```

功能：
- 摄像头实时检测
- 批量图片检测
- 实时 FPS 显示与 25 FPS 目标提示
- 可选 Spot-Guided 推理增强

实现位于 [scripts/ui/window.py](scripts/ui/window.py)。

## 8. 核心配置说明

### 8.1 训练配置
文件：[configs/train.yaml](configs/train.yaml)
- `model`：初始权重路径
- `data`：默认数据集 yaml
- `name`：实验名（影响输出目录）
- `adaptive_loss`：类别不平衡自适应损失参数
- `predict`：预测阈值、分辨率、Spot-Guided 开关
- `benchmark`：FPS 基准输入 split、样本数、输出路径
- `claims.research_route`：科研叙事路线，`A`（声明 BiFPN）或 `B`（不声明未实现结构）
- `claims.target_effective_sparsity`：剪枝有效稀疏率目标（用于 report checklist）

### 8.2 数据集配置
文件：[configs/dataset.yaml](configs/dataset.yaml)
- `path`：数据集根目录
- `train` / `val`：训练与验证图像子目录
- `names`：类别定义

### 8.3 增强配置
文件：[configs/prepare.yaml](configs/prepare.yaml)
- 数据源与目标路径
- 旋转、亮度、MixUp 参数
- Spot-Guided 参数

## 9. 产物目录说明

- `experiments/<name>/weights/best.pt`：训练最佳权重
- `experiments/predict/`：批量推理可视化结果
- `experiments/ui_batch_results/`：UI 批处理输出
- `models/weights/`：剪枝后模型

## 10. 常见问题

### 10.1 执行 `python3 scripts/cli.py ...` 报找不到模块
通常是工作目录不在项目根目录。请先：
```bash
cd /ysml
```

### 10.2 验证时报未找到 `best.pt`
请先执行训练：
```bash
python3 scripts/cli.py train
```

### 10.3 UI 无法打开摄像头
检查系统摄像头权限或设备占用；可先使用批处理模式验证模型。

### 10.4 训练速度慢或显存不足
可在 [configs/train.yaml](configs/train.yaml) 下调 `imgsz`、`batch`，或更换设备配置。

## 11. 科研实验建议（可直接用于论文）

建议至少形成三组对比：
- 基线模型
- 增强数据训练模型
- 蒸馏/剪枝轻量化模型

每组记录：
- mAP / Precision / Recall
- 参数量或模型大小
- 平均 FPS

用以上指标即可支撑“样本稀缺优化、精度提升、轻量化与实时性”的研究结论。

## 12. 论文与答辩模板

- 实验章节模板：[docs/thesis_experiment_section_template.md](docs/thesis_experiment_section_template.md)
- 答辩单页模板：[docs/defense_slide_template.md](docs/defense_slide_template.md)
- 答辩第二页（架构与创新）模板：[docs/defense_architecture_slide_template.md](docs/defense_architecture_slide_template.md)
- 答辩第三页（不足与未来工作）模板：[docs/defense_limitations_future_slide_template.md](docs/defense_limitations_future_slide_template.md)
- 答辩全套单文件模板：[docs/defense_full_deck_template.md](docs/defense_full_deck_template.md)
