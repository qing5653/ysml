# Bearing Defect Experiment Report

- Generated at: 2026-03-28 13:09:47
- Run name: yolo11_spot_bifpn
- Results CSV: /ysml/experiments/yolo11_spot_bifpn/results.csv

## Detection Metrics
- Best mAP50: 0.8469 (epoch=100)
- Best mAP50-95: 0.5097 (epoch=58)
- Last precision: 0.7484
- Last recall: 0.7866
- Last mAP50: 0.8469
- Last mAP50-95: 0.4970

## Model Size (MB)
- Baseline best.pt: 5.23
- Baseline best.onnx: 9.94
- Pruned student pt: 5.25
- Compression ratio (baseline pt -> pruned pt): -0.49%

## Pruning Evidence
- Prune stats file: /ysml/experiments/benchmark/prune_stats.json
- Effective sparsity: 29.70% (target: 29.50%)

## Pipeline Status
- Distill best.pt: /ysml/experiments/yolo11_student_distill/weights/best.pt
- Prune output model: /ysml/models/weights/yolo11_student_pruned.pt
- Prune finetune best.pt: /ysml/experiments/yolo11_student_pruned_finetune/weights/best.pt

## Runtime Benchmark (FPS)
- guided=False | images=30 | imgsz=640 | conf=0.25 | fps=40.1406
- guided=True | images=30 | imgsz=640 | conf=0.25 | fps=9.9085
- max_fps: 40.1406 (target: 25.00)

## Readiness Checklist
- [x] Route B consistency: report narrative matches implemented pipeline (no unimplemented BiFPN claim)
- [x] Distillation run completed with saved weights
- [x] Pruning run completed with saved weights
- [x] Effective sparsity >= 29.5% is validated
- [x] FPS >= 25 validated on target hardware
