from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

from ultralytics import YOLO
import yaml

from scripts.ops.common import (
    IMAGE_EXTS,
    ROOT,
    BoxLabel,
    apply_channel_prune,
    count_params,
    load_yaml,
    merge_labels,
    read_labels_box,
    resolve_dataset_root,
    resolve_latest_weight,
    write_labels_box,
)


def _is_shm_like_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "No space left on device" in msg
        or "Unexpected bus error" in msg
        or "insufficient shared memory" in msg.lower()
    )


def _train_with_workers_fallback(model: YOLO, train_kwargs: dict, stage_name: str) -> None:
    try:
        model.train(**train_kwargs)
    except Exception as exc:  # noqa: BLE001
        current_workers = int(train_kwargs.get("workers", 0))
        if _is_shm_like_error(exc) and current_workers > 0:
            print(
                f"检测到 {stage_name} 阶段 DataLoader 共享内存/空间问题，"
                f"自动降级 workers: {current_workers} -> 0 后重试一次。"
            )
            fallback_kwargs = dict(train_kwargs)
            fallback_kwargs["workers"] = 0
            model.train(**fallback_kwargs)
            return
        raise


def _prepare_distill_dataset(cfg: dict, base_cfg: dict, base_root: Path, distill_yaml_path: Path) -> None:
    out_root = distill_yaml_path.parent
    if out_root.exists():
        shutil.rmtree(out_root)

    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    teacher = YOLO(str(ROOT / cfg["teacher_model"]))
    teacher_cfg = cfg["teacher_predict"]
    train_images_dir = base_root / base_cfg["train"]
    train_labels_dir = base_root / base_cfg["train"].replace("images", "labels")
    val_images_dir = base_root / base_cfg["val"]
    val_labels_dir = base_root / base_cfg["val"].replace("images", "labels")

    for image_path in sorted(train_images_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
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
        if image_path.suffix.lower() not in IMAGE_EXTS:
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


def _deduplicate_label_file(label_path: Path) -> tuple[bool, int]:
    if not label_path.exists():
        return False, 0

    original_lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not original_lines:
        return False, 0

    seen: set[str] = set()
    deduped: list[str] = []
    for line in original_lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)

    removed = len(original_lines) - len(deduped)
    if removed <= 0:
        return False, 0

    label_path.write_text("\n".join(deduped) + "\n", encoding="utf-8")
    return True, removed


def _sanitize_distill_labels(out_root: Path) -> tuple[int, int]:
    touched_files = 0
    removed_lines = 0
    for split in ["train", "val"]:
        labels_dir = out_root / split / "labels"
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            touched, removed = _deduplicate_label_file(label_file)
            if touched:
                touched_files += 1
                removed_lines += removed
    return touched_files, removed_lines


def _resolve_data_root_and_labels(data_yaml_path: Path) -> tuple[Path, list[Path]]:
    data_cfg = load_yaml(data_yaml_path)
    data_root = ROOT / str(data_cfg["path"])
    if not data_root.exists():
        data_root = Path(str(data_cfg["path"]))

    label_dirs: list[Path] = []
    for split_key in ["train", "val", "valid"]:
        split_rel = data_cfg.get(split_key)
        if not split_rel:
            continue
        split_path = str(split_rel)
        if "images" in split_path:
            label_rel = split_path.replace("images", "labels")
        else:
            label_rel = split_path
        label_dirs.append(data_root / label_rel)
    return data_root, label_dirs


def _sanitize_yolo_dataset_labels(data_yaml_path: Path) -> tuple[int, int]:
    _, label_dirs = _resolve_data_root_and_labels(data_yaml_path)
    touched_files = 0
    removed_lines = 0
    for labels_dir in label_dirs:
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            touched, removed = _deduplicate_label_file(label_file)
            if touched:
                touched_files += 1
                removed_lines += removed
    return touched_files, removed_lines


def _set_comet_disabled(disable: bool) -> bool | None:
    if not disable:
        return None

    old_comet = None
    try:
        from ultralytics import settings as ul_settings

        if "comet" in ul_settings:
            old_comet = bool(ul_settings.get("comet"))
            ul_settings.update({"comet": False})
    except Exception:  # noqa: BLE001
        pass

    os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
    return old_comet


def _restore_comet_setting(old_comet: bool | None) -> None:
    if old_comet is None:
        return
    try:
        from ultralytics import settings as ul_settings

        ul_settings.update({"comet": old_comet})
    except Exception:  # noqa: BLE001
        pass


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
    base_root = resolve_dataset_root(base_cfg)

    distill_yaml_path = ROOT / cfg["distill_data"]
    out_root = distill_yaml_path.parent
    reuse_generated_data = bool(cfg.get("reuse_generated_data", True))
    can_reuse = reuse_generated_data and distill_yaml_path.exists() and out_root.exists()

    if can_reuse:
        print(f"复用已有蒸馏数据: {distill_yaml_path}")
        touched_files, removed_lines = _sanitize_distill_labels(out_root)
        if removed_lines > 0:
            print(f"复用数据预清洗: 去重标签 {removed_lines} 行，涉及文件 {touched_files} 个。")
    else:
        print(f"生成蒸馏数据集: {distill_yaml_path}")
        _prepare_distill_dataset(cfg, base_cfg, base_root, distill_yaml_path)

    train_cfg = cfg["train"]
    train_project = ROOT / str(train_cfg["project"])
    run_name = str(train_cfg["name"])

    init_weight = student_path
    if bool(train_cfg.get("warm_start_from_last", True)):
        latest_last = resolve_latest_weight(train_project, run_name, "last.pt")
        if latest_last is not None:
            init_weight = latest_last
            print(f"distill 热启动: {latest_last}")

    student = YOLO(str(init_weight))
    old_comet = _set_comet_disabled(bool(cfg.get("disable_comet", True)))
    train_kwargs = {
        "data": str(distill_yaml_path),
        "epochs": int(train_cfg["epochs"]),
        "imgsz": int(train_cfg["imgsz"]),
        "batch": int(train_cfg["batch"]),
        "device": train_cfg["device"],
        "workers": int(train_cfg["workers"]),
        "project": str(train_project),
        "name": run_name,
    }
    try:
        _train_with_workers_fallback(student, train_kwargs, stage_name="distill")
    finally:
        _restore_comet_setting(old_comet)
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
    effective_sparsity = (1.0 - non_zero_after / max(non_zero_before, 1)) * 100.0

    yolo_model.save(str(output_model))
    print(
        "剪枝完成: "
        f"total={total_before} -> {total_after}, "
        f"non_zero={non_zero_before} -> {non_zero_after}, "
        f"effective_sparsity={effective_sparsity:.2f}%"
    )

    stats_output = ROOT / str(cfg.get("stats_output", "experiments/benchmark/prune_stats.json"))
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    prune_stats = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "channel_prune_ratio": float(cfg["channel_prune_ratio"]),
        "total_before": int(total_before),
        "total_after": int(total_after),
        "non_zero_before": int(non_zero_before),
        "non_zero_after": int(non_zero_after),
        "effective_sparsity": float(effective_sparsity),
    }
    stats_output.write_text(json.dumps(prune_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"剪枝统计已写入: {stats_output}")

    ft = cfg["finetune"]
    ft_project = ROOT / str(ft["project"])
    ft_name = str(ft["name"])

    resume_weight: Path | None = None
    if bool(ft.get("warm_start_from_last", True)):
        resume_weight = resolve_latest_weight(ft_project, ft_name, "last.pt")

    if resume_weight is not None:
        print(f"prune-finetune 热启动: {resume_weight}")
        finetune_model = YOLO(str(resume_weight))
    else:
        finetune_model = YOLO(str(input_model))
        finetune_model.model.load_state_dict(torch_model.state_dict(), strict=False)

    ft_data_yaml = ROOT / str(ft["data"])
    if bool(ft.get("dedupe_labels_on_start", True)):
        touched_files, removed_lines = _sanitize_yolo_dataset_labels(ft_data_yaml)
        if removed_lines > 0:
            print(f"prune-finetune 预清洗: 去重标签 {removed_lines} 行，涉及文件 {touched_files} 个。")

    old_comet = _set_comet_disabled(bool(cfg.get("disable_comet", True)))
    train_kwargs = {
        "data": str(ft_data_yaml),
        "epochs": int(ft["epochs"]),
        "imgsz": int(ft["imgsz"]),
        "batch": int(ft["batch"]),
        "device": ft["device"],
        "workers": int(ft["workers"]),
        "project": str(ft_project),
        "name": ft_name,
    }
    try:
        _train_with_workers_fallback(finetune_model, train_kwargs, stage_name="prune-finetune")
    finally:
        _restore_comet_setting(old_comet)
