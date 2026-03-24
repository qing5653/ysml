# ysml

工业缺陷检测工程（YOLO11），包含：数据增强、训练/验证/推理、蒸馏、剪枝、PyQt5 可视化系统。

## 快速开始
```bash
pip install -r requirements.txt
python3 scripts/check_datasets.py
python3 scripts/train.py
python3 scripts/val.py
python3 scripts/predict.py
```

## 常用命令
也可以统一用一个入口（更简洁）：
```bash
python3 scripts/run.py check
python3 scripts/run.py merge
python3 scripts/run.py train
python3 scripts/run.py val
python3 scripts/run.py predict
```

1. 数据体检
```bash
python3 scripts/check_datasets.py
```

2. 合并公开数据集
```bash
python3 scripts/merge_datasets.py
```

3. 数据增强（旋转/亮度/MixUp/Spot-Guided）
```bash
python3 scripts/prepare_dataset.py
```

4. 训练与评估
```bash
python3 scripts/train.py
python3 scripts/val.py
```

5. 推理与导出
```bash
python3 scripts/predict.py
python3 scripts/export.py
```

6. 模型压缩
```bash
python3 scripts/distill.py
python3 scripts/prune.py
```

7. UI 系统
```bash
python3 scripts/app.py
```

## 当前数据集状态
1. `data/fa031-main/NEU-DET`：结构完整，可直接训练。
2. `data/Quality-Control367-main`：当前缺少 train/val 标注结构，尚不能直接训练。
3. 当前训练默认使用合并配置：`configs/dataset_merged.yaml`。

## 最小目录约定
1. `data/`：原始公开数据与转换脚本。
2. `datasets/`：可直接训练的数据副本（增强集/合并集）。
3. `experiments/`：训练与推理输出（不提交）。
4. `configs/`：执行入口配置。
5. `scripts/`：流程脚本。

## 核心配置
1. `configs/train.yaml`：训练主配置。
2. `configs/dataset_merged.yaml`：合并后训练数据。
3. `configs/merge.yaml`：合并策略。
4. `configs/prepare.yaml`：增强策略。

## 备注
合并脚本按检测标签格式 `cls x y w h` 处理；不符合格式或缺失标注的样本会自动跳过并统计。

