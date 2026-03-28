from __future__ import annotations

import csv
from pathlib import Path
import time

import cv2
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics import YOLO

from scripts.ops.common import IMAGE_EXTS, ROOT, load_yaml, resolve_active_data_cfg_path, resolve_latest_weight
from src.yolo11_project.spot_guided import SpotGuidedConfig, apply_spot_guided_attention


def _compute_class_distribution(dataset_yaml: Path, project_root: Path) -> list[int]:
    data_cfg = load_yaml(dataset_yaml)
    root = project_root / data_cfg["path"]
    if not root.exists():
        root = Path(data_cfg["path"])

    train_labels_dir = root / data_cfg["train"].replace("images", "labels")
    if not train_labels_dir.exists():
        return []

    counts: dict[int, int] = {}
    for label_file in train_labels_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            counts[cls_id] = counts.get(cls_id, 0) + 1

    if not counts:
        return []

    max_cls = max(counts)
    return [counts.get(i, 0) for i in range(max_cls + 1)]


def _adaptive_loss_kwargs(train_cfg: dict, dataset_yaml: Path, project_root: Path) -> dict:
    adaptive = train_cfg.get("adaptive_loss", {})
    if not adaptive or not adaptive.get("enabled", False):
        return {}

    class_counts = _compute_class_distribution(dataset_yaml, project_root)
    if not class_counts:
        return {}

    non_zero = [v for v in class_counts if v > 0]
    if not non_zero:
        return {}

    imbalance_ratio = max(non_zero) / max(min(non_zero), 1)
    ratio_norm = min(max((imbalance_ratio - 1.0) / 20.0, 0.0), 1.0)

    cls_gain = adaptive["cls_gain_min"] + (adaptive["cls_gain_max"] - adaptive["cls_gain_min"]) * ratio_norm
    fl_gamma = adaptive["fl_gamma_min"] + (adaptive["fl_gamma_max"] - adaptive["fl_gamma_min"]) * ratio_norm

    # 兼容不同版本 Ultralytics 参数集合，避免传入未知参数导致训练直接报错。
    kwargs = {"cls": float(cls_gain), "fl_gamma": float(fl_gamma)}
    supported_keys = set(DEFAULT_CFG_DICT.keys())
    return {k: v for k, v in kwargs.items() if k in supported_keys}


def _resolve_latest_best_ckpt(exp_root: Path, run_name: str) -> Path:
    selected = resolve_latest_weight(exp_root, run_name, "best.pt")
    if selected is None:
        expected = exp_root / run_name / "weights" / "best.pt"
        raise FileNotFoundError(f"未找到验证权重: {expected} (以及 {run_name}* 下的 best.pt)")
    if selected != exp_root / run_name / "weights" / "best.pt":
        print(f"自动选择最新权重: {selected}")
    return selected


def cmd_train() -> None:
    cfg = load_yaml(ROOT / "configs" / "train.yaml")
    model_path = ROOT / cfg["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    data_path = resolve_active_data_cfg_path(cfg)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据集配置: {data_path}")

    model = YOLO(str(model_path))
    train_kwargs = {
        k: v
        for k, v in cfg.items()
        if k not in {"model", "adaptive_loss", "use_prepared_dataset", "prepared_dataset_yaml"}
    }
    train_kwargs["data"] = str(data_path)
    train_kwargs["project"] = str(ROOT / train_kwargs["project"])
    train_kwargs.update(_adaptive_loss_kwargs(cfg, data_path, ROOT))

    if "cls" in train_kwargs and "fl_gamma" in train_kwargs:
        print(f"自适应损失参数: cls={train_kwargs['cls']:.3f}, fl_gamma={train_kwargs['fl_gamma']:.3f}")
    elif "cls" in train_kwargs:
        print(f"自适应损失参数: cls={train_kwargs['cls']:.3f} (当前版本不支持 fl_gamma，已自动跳过)")

    try:
        model.train(**train_kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        is_shm_like_error = (
            "No space left on device" in msg
            or "Unexpected bus error" in msg
            or "insufficient shared memory" in msg.lower()
        )
        current_workers = int(train_kwargs.get("workers", 0))
        if is_shm_like_error and current_workers > 0:
            print(
                f"检测到 DataLoader 共享内存/空间问题，自动降级 workers: {current_workers} -> 0 后重试一次。"
            )
            train_kwargs["workers"] = 0
            model.train(**train_kwargs)
            return
        raise


def cmd_val() -> None:
    cfg = load_yaml(ROOT / "configs" / "train.yaml")
    ckpt_path = _resolve_latest_best_ckpt(ROOT / "experiments", str(cfg["name"]))

    data_path = resolve_active_data_cfg_path(cfg)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据集配置: {data_path}")

    model = YOLO(str(ckpt_path))
    val_kwargs = {
        "data": str(data_path),
        "imgsz": cfg["imgsz"],
        "device": cfg["device"],
        "workers": int(cfg.get("workers", 0)),
    }
    try:
        metrics = model.val(**val_kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        is_shm_like_error = (
            "No space left on device" in msg
            or "Unexpected bus error" in msg
            or "insufficient shared memory" in msg.lower()
        )
        current_workers = int(val_kwargs.get("workers", 0))
        if is_shm_like_error and current_workers > 0:
            print(
                f"检测到验证阶段 DataLoader 共享内存/空间问题，自动降级 workers: {current_workers} -> 0 后重试一次。"
            )
            val_kwargs["workers"] = 0
            metrics = model.val(**val_kwargs)
        else:
            raise
    summary = getattr(metrics, "results_dict", None)
    if isinstance(summary, dict):
        print("验证完成(摘要):")
        for key in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"]:
            if key in summary:
                print(f"  {key}: {summary[key]:.6f}")
    else:
        print(metrics)


def cmd_predict() -> None:
    train_cfg = load_yaml(ROOT / "configs" / "train.yaml")
    best_ckpt = _resolve_latest_best_ckpt(ROOT / "experiments", str(train_cfg["name"]))
    model_path = best_ckpt if best_ckpt.exists() else ROOT / train_cfg["model"]

    data_cfg_path = resolve_active_data_cfg_path(train_cfg)
    data_cfg = load_yaml(data_cfg_path)
    source = ROOT / data_cfg["path"] / data_cfg["val"]

    predict_cfg = train_cfg.get("predict", {})
    predict_conf = float(predict_cfg.get("conf", 0.25))
    predict_imgsz = int(predict_cfg.get("imgsz", train_cfg.get("imgsz", 640)))
    use_guided = bool(predict_cfg.get("use_spot_guided", True))

    prepare_cfg = load_yaml(ROOT / "configs" / "prepare.yaml")
    sg = prepare_cfg.get("spot_guided", {})
    sg_cfg = SpotGuidedConfig(
        slic_segments=int(sg.get("slic_segments", 200)),
        slic_compactness=float(sg.get("slic_compactness", 12.0)),
        glcm_distances=tuple(int(v) for v in sg.get("glcm_distances", [1, 2])),
        glcm_angles=tuple(float(v) for v in sg.get("glcm_angles", [0.0, 0.785398, 1.570796])),
        entropy_threshold_quantile=float(sg.get("entropy_threshold_quantile", 0.75)),
        blend_alpha=float(sg.get("blend_alpha", 0.45)),
    )

    if not model_path.exists():
        raise FileNotFoundError(f"未找到推理模型: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"未找到推理输入目录: {source}")

    model = YOLO(str(model_path))
    image_paths = [p for p in sorted(source.glob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if not image_paths:
        raise RuntimeError(f"目录中没有图像文件: {source}")

    out_dir = ROOT / "experiments" / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        if use_guided:
            image, _ = apply_spot_guided_attention(image, sg_cfg)
        result = model.predict(image, imgsz=predict_imgsz, conf=predict_conf, verbose=False)[0]
        cv2.imwrite(str(out_dir / image_path.name), result.plot())

    elapsed = max(time.perf_counter() - start, 1e-6)
    fps = len(image_paths) / elapsed
    print(f"预测完成: 数量={len(image_paths)}, 平均FPS={fps:.2f}, 输出目录={out_dir}")


def cmd_benchmark() -> None:
    train_cfg = load_yaml(ROOT / "configs" / "train.yaml")
    benchmark_cfg = train_cfg.get("benchmark", {})

    best_ckpt = _resolve_latest_best_ckpt(ROOT / "experiments", str(train_cfg["name"]))
    model_path = best_ckpt if best_ckpt.exists() else ROOT / train_cfg["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"未找到基准模型: {model_path}")

    data_cfg_path = resolve_active_data_cfg_path(train_cfg)
    data_cfg = load_yaml(data_cfg_path)
    source_split = str(benchmark_cfg.get("source_split", "val"))
    source_rel = data_cfg.get(source_split)
    if source_rel is None:
        raise KeyError(f"数据集配置中不存在 split={source_split}")

    source = ROOT / data_cfg["path"] / source_rel
    if not source.exists():
        raise FileNotFoundError(f"未找到基准输入目录: {source}")

    sample_limit = int(benchmark_cfg.get("sample_limit", 100))
    conf = float(benchmark_cfg.get("conf", 0.25))
    imgsz = int(benchmark_cfg.get("imgsz", train_cfg.get("imgsz", 640)))
    guided_options = benchmark_cfg.get("use_spot_guided_options", [False, True])
    output_csv = ROOT / str(benchmark_cfg.get("output_csv", "experiments/benchmark/fps.csv"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    prepare_cfg = load_yaml(ROOT / "configs" / "prepare.yaml")
    sg = prepare_cfg.get("spot_guided", {})
    sg_cfg = SpotGuidedConfig(
        slic_segments=int(sg.get("slic_segments", 200)),
        slic_compactness=float(sg.get("slic_compactness", 12.0)),
        glcm_distances=tuple(int(v) for v in sg.get("glcm_distances", [1, 2])),
        glcm_angles=tuple(float(v) for v in sg.get("glcm_angles", [0.0, 0.785398, 1.570796])),
        entropy_threshold_quantile=float(sg.get("entropy_threshold_quantile", 0.75)),
        blend_alpha=float(sg.get("blend_alpha", 0.45)),
    )

    image_paths = [p for p in sorted(source.glob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if not image_paths:
        raise RuntimeError(f"目录中没有图像文件: {source}")
    if sample_limit > 0:
        image_paths = image_paths[:sample_limit]

    model = YOLO(str(model_path))
    rows: list[dict[str, float | int | str]] = []

    for use_guided in guided_options:
        begin = time.perf_counter()
        processed = 0

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            if bool(use_guided):
                image, _ = apply_spot_guided_attention(image, sg_cfg)
            _ = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
            processed += 1

        elapsed = max(time.perf_counter() - begin, 1e-6)
        fps = processed / elapsed
        rows.append(
            {
                "timestamp": int(time.time()),
                "model": str(model_path),
                "source": str(source),
                "images": processed,
                "imgsz": imgsz,
                "conf": conf,
                "use_spot_guided": bool(use_guided),
                "fps": round(fps, 4),
                "elapsed_sec": round(elapsed, 4),
            }
        )
        print(
            f"benchmark: guided={bool(use_guided)} | images={processed} | imgsz={imgsz} | conf={conf} | fps={fps:.2f}"
        )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "model",
                "source",
                "images",
                "imgsz",
                "conf",
                "use_spot_guided",
                "fps",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"基准完成，结果已写入: {output_csv}")


def cmd_export() -> None:
    train_cfg = load_yaml(ROOT / "configs" / "train.yaml")
    best_ckpt = _resolve_latest_best_ckpt(ROOT / "experiments", str(train_cfg["name"]))

    model = YOLO(str(best_ckpt))
    exported = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"导出完成: {exported}")
