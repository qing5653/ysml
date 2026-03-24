from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class YoloObject:
    class_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class BoxLabel:
    cls_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_labels_yolo(label_path: Path) -> list[YoloObject]:
    if not label_path.exists():
        return []
    objects: list[YoloObject] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x, y, w, h = parts
        objects.append(YoloObject(int(cls_id), float(x), float(y), float(w), float(h)))
    return objects


def write_labels_yolo(label_path: Path, objects: list[YoloObject]) -> None:
    lines = [f"{obj.class_id} {obj.x:.6f} {obj.y:.6f} {obj.w:.6f} {obj.h:.6f}" for obj in objects]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def to_xyxy(box: BoxLabel) -> tuple[float, float, float, float]:
    x1 = box.x - box.w / 2
    y1 = box.y - box.h / 2
    x2 = box.x + box.w / 2
    y2 = box.y + box.h / 2
    return x1, y1, x2, y2


def iou(a: BoxLabel, b: BoxLabel) -> float:
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter + 1e-8
    return inter / union


def read_labels_box(label_file: Path) -> list[BoxLabel]:
    if not label_file.exists():
        return []
    labels: list[BoxLabel] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        labels.append(BoxLabel(int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), 1.0))
    return labels


def write_labels_box(label_file: Path, labels: list[BoxLabel]) -> None:
    content = "\n".join(f"{l.cls_id} {l.x:.6f} {l.y:.6f} {l.w:.6f} {l.h:.6f}" for l in labels)
    label_file.write_text(content, encoding="utf-8")


def merge_labels(gt_labels: list[BoxLabel], teacher_labels: list[BoxLabel], iou_thr: float = 0.5) -> list[BoxLabel]:
    merged = list(gt_labels)
    for teacher in teacher_labels:
        duplicate = any((gt.cls_id == teacher.cls_id and iou(gt, teacher) >= iou_thr) for gt in gt_labels)
        if not duplicate:
            merged.append(teacher)
    return merged


def count_params(module: nn.Module) -> tuple[int, int]:
    total = 0
    non_zero = 0
    for param in module.parameters():
        total += param.numel()
        non_zero += int(torch.count_nonzero(param).item())
    return total, non_zero


def apply_channel_prune(module: nn.Module, ratio: float) -> None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d) and child.weight.shape[0] >= 8:
            prune.ln_structured(child, name="weight", amount=ratio, n=2, dim=0)
            prune.remove(child, "weight")
