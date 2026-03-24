from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from ultralytics import YOLO
import yaml


@dataclass
class BoxLabel:
    cls_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0


def _read_dataset_yaml(path: Path, project_root: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = project_root / cfg["path"]
    if not root.exists():
        root = Path(cfg["path"])
    return {"root": root, "train": cfg["train"], "val": cfg["val"], "names": cfg["names"]}


def _read_labels(label_file: Path) -> list[BoxLabel]:
    if not label_file.exists():
        return []
    labels: list[BoxLabel] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        labels.append(BoxLabel(int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), 1.0))
    return labels


def _to_xyxy(box: BoxLabel) -> tuple[float, float, float, float]:
    x1 = box.x - box.w / 2
    y1 = box.y - box.h / 2
    x2 = box.x + box.w / 2
    y2 = box.y + box.h / 2
    return x1, y1, x2, y2


def _iou(a: BoxLabel, b: BoxLabel) -> float:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
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


def _merge_labels(gt_labels: list[BoxLabel], teacher_labels: list[BoxLabel], iou_thr: float = 0.5) -> list[BoxLabel]:
    merged = list(gt_labels)
    for t in teacher_labels:
        duplicate = any((g.cls_id == t.cls_id and _iou(g, t) >= iou_thr) for g in gt_labels)
        if not duplicate:
            merged.append(t)
    return merged


def _write_labels(label_file: Path, labels: list[BoxLabel]) -> None:
    content = "\n".join(f"{l.cls_id} {l.x:.6f} {l.y:.6f} {l.w:.6f} {l.h:.6f}" for l in labels)
    label_file.write_text(content, encoding="utf-8")


def _build_distill_dataset(project_root: Path, cfg: dict) -> None:
    base_data_path = project_root / cfg["base_data"]
    if not base_data_path.exists():
        raise FileNotFoundError(f"未找到基础数据配置: {base_data_path}")

    data_cfg = _read_dataset_yaml(base_data_path, project_root)
    distill_yaml_path = project_root / cfg["distill_data"]
    out_root = distill_yaml_path.parent
    if out_root.exists():
        shutil.rmtree(out_root)

    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    teacher = YOLO(str(project_root / cfg["teacher_model"]))
    teacher_cfg = cfg["teacher_predict"]

    train_images_dir = data_cfg["root"] / data_cfg["train"]
    train_labels_dir = data_cfg["root"] / data_cfg["train"].replace("images", "labels")
    val_images_dir = data_cfg["root"] / data_cfg["val"]
    val_labels_dir = data_cfg["root"] / data_cfg["val"].replace("images", "labels")

    for image_path in sorted(train_images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        result = teacher.predict(
            source=str(image_path),
            conf=float(teacher_cfg["conf"]),
            iou=float(teacher_cfg["iou"]),
            imgsz=int(teacher_cfg["imgsz"]),
            verbose=False,
        )[0]
        teacher_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for cls_id, conf, xywhn in zip(result.boxes.cls, result.boxes.conf, result.boxes.xywhn):
                teacher_boxes.append(
                    BoxLabel(int(cls_id.item()), float(xywhn[0].item()), float(xywhn[1].item()), float(xywhn[2].item()), float(xywhn[3].item()), float(conf.item()))
                )

        gt = _read_labels(train_labels_dir / f"{image_path.stem}.txt")
        merged = _merge_labels(gt, teacher_boxes)
        shutil.copy2(image_path, out_root / "train" / "images" / image_path.name)
        _write_labels(out_root / "train" / "labels" / f"{image_path.stem}.txt", merged)

    for image_path in sorted(val_images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        shutil.copy2(image_path, out_root / "val" / "images" / image_path.name)
        src_label = val_labels_dir / f"{image_path.stem}.txt"
        _write_labels(out_root / "val" / "labels" / f"{image_path.stem}.txt", _read_labels(src_label))

    data_yaml = {
        "path": str(out_root.relative_to(project_root)),
        "train": "train/images",
        "val": "val/images",
        "names": data_cfg["names"],
    }
    distill_yaml_path.write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "distill.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到蒸馏配置: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    teacher_path = root / cfg["teacher_model"]
    if not teacher_path.exists():
        raise FileNotFoundError(f"未找到教师模型: {teacher_path}")

    student_path = root / cfg["student_model"]
    if not student_path.exists():
        raise FileNotFoundError(f"未找到学生模型初始化权重: {student_path}")

    _build_distill_dataset(root, cfg)

    student = YOLO(str(student_path))
    train_cfg = cfg["train"]
    student.train(
        data=str(root / cfg["distill_data"]),
        epochs=int(train_cfg["epochs"]),
        imgsz=int(train_cfg["imgsz"]),
        batch=int(train_cfg["batch"]),
        device=train_cfg["device"],
        workers=int(train_cfg["workers"]),
        project=str(root / train_cfg["project"]),
        name=str(train_cfg["name"]),
    )
    print("蒸馏训练完成。")


if __name__ == "__main__":
    main()