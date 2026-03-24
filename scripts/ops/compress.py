from __future__ import annotations

from pathlib import Path
import shutil

from ultralytics import YOLO
import yaml

from scripts.ops.common import (
    ROOT,
    BoxLabel,
    apply_channel_prune,
    count_params,
    load_yaml,
    merge_labels,
    read_labels_box,
    write_labels_box,
)


def cmd_distill() -> None:
    cfg = load_yaml(ROOT / "configs" / "distill.yaml")
    teacher_path = ROOT / cfg["teacher_model"]
    if not teacher_path.exists():
        raise FileNotFoundError(f"未找到教师模型: {teacher_path}")
    student_path = ROOT / cfg["student_model"]
    if not student_path.exists():
        raise FileNotFoundError(f"未找到学生模型初始化权重: {student_path}")

    base_data_path = ROOT / cfg["base_data"]
    if not base_data_path.exists():
        raise FileNotFoundError(f"未找到基础数据配置: {base_data_path}")

    base_cfg = load_yaml(base_data_path)
    base_root = ROOT / base_cfg["path"]
    if not base_root.exists():
        base_root = Path(base_cfg["path"])

    distill_yaml_path = ROOT / cfg["distill_data"]
    out_root = distill_yaml_path.parent
    if out_root.exists():
        shutil.rmtree(out_root)

    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    teacher = YOLO(str(teacher_path))
    teacher_cfg = cfg["teacher_predict"]
    train_images_dir = base_root / base_cfg["train"]
    train_labels_dir = base_root / base_cfg["train"].replace("images", "labels")
    val_images_dir = base_root / base_cfg["val"]
    val_labels_dir = base_root / base_cfg["val"].replace("images", "labels")

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

        teacher_boxes: list[BoxLabel] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for cls_id, conf, xywhn in zip(result.boxes.cls, result.boxes.conf, result.boxes.xywhn):
                teacher_boxes.append(
                    BoxLabel(
                        int(cls_id.item()),
                        float(xywhn[0].item()),
                        float(xywhn[1].item()),
                        float(xywhn[2].item()),
                        float(xywhn[3].item()),
                        float(conf.item()),
                    )
                )

        gt = read_labels_box(train_labels_dir / f"{image_path.stem}.txt")
        merged = merge_labels(gt, teacher_boxes)
        shutil.copy2(image_path, out_root / "train" / "images" / image_path.name)
        write_labels_box(out_root / "train" / "labels" / f"{image_path.stem}.txt", merged)

    for image_path in sorted(val_images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        shutil.copy2(image_path, out_root / "val" / "images" / image_path.name)
        src_label = val_labels_dir / f"{image_path.stem}.txt"
        write_labels_box(out_root / "val" / "labels" / f"{image_path.stem}.txt", read_labels_box(src_label))

    distill_data_yaml = {
        "path": str(out_root.relative_to(ROOT)),
        "train": "train/images",
        "val": "val/images",
        "names": base_cfg["names"],
    }
    distill_yaml_path.write_text(yaml.safe_dump(distill_data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    student = YOLO(str(student_path))
    train_cfg = cfg["train"]
    student.train(
        data=str(distill_yaml_path),
        epochs=int(train_cfg["epochs"]),
        imgsz=int(train_cfg["imgsz"]),
        batch=int(train_cfg["batch"]),
        device=train_cfg["device"],
        workers=int(train_cfg["workers"]),
        project=str(ROOT / train_cfg["project"]),
        name=str(train_cfg["name"]),
    )
    print("蒸馏训练完成。")


def cmd_prune() -> None:
    cfg = load_yaml(ROOT / "configs" / "prune.yaml")
    input_model = ROOT / cfg["input_model"]
    output_model = ROOT / cfg["output_model"]
    output_model.parent.mkdir(parents=True, exist_ok=True)

    if not input_model.exists():
        raise FileNotFoundError(f"未找到待剪枝模型: {input_model}")

    yolo_model = YOLO(str(input_model))
    torch_model = yolo_model.model

    total_before, non_zero_before = count_params(torch_model)
    apply_channel_prune(torch_model, float(cfg["channel_prune_ratio"]))
    total_after, non_zero_after = count_params(torch_model)

    yolo_model.save(str(output_model))
    print(
        "剪枝完成: "
        f"total={total_before} -> {total_after}, "
        f"non_zero={non_zero_before} -> {non_zero_after}, "
        f"effective_sparsity={(1.0 - non_zero_after / max(non_zero_before, 1)) * 100:.2f}%"
    )

    ft = cfg["finetune"]
    finetune_model = YOLO(str(input_model))
    finetune_model.model.load_state_dict(torch_model.state_dict(), strict=False)
    finetune_model.train(
        data=str(ROOT / ft["data"]),
        epochs=int(ft["epochs"]),
        imgsz=int(ft["imgsz"]),
        batch=int(ft["batch"]),
        device=ft["device"],
        workers=int(ft["workers"]),
        project=str(ROOT / ft["project"]),
        name=str(ft["name"]),
    )
