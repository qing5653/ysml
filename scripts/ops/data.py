from __future__ import annotations

from pathlib import Path
import random
import shutil

import cv2
import numpy as np

from scripts.ops.common import ROOT, YoloObject, load_yaml, read_labels_yolo, write_labels_yolo
from src.yolo11_project.spot_guided import SpotGuidedConfig, apply_spot_guided_attention


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in exts)


def _count_labels(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.txt"))


def _has_readme(path: Path) -> bool:
    for base in [path, path.parent]:
        if (base / "README.md").exists() or (base / "readme.md").exists() or (base / "README.docx").exists():
            return True
    return False


def _rot90_clockwise_objects(objects: list[YoloObject]) -> list[YoloObject]:
    return [YoloObject(obj.class_id, 1.0 - obj.y, obj.x, obj.h, obj.w) for obj in objects]


def _brightness(image: np.ndarray, gain: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _mixup(
    img_a: np.ndarray,
    objs_a: list[YoloObject],
    img_b: np.ndarray,
    objs_b: list[YoloObject],
    alpha: float,
) -> tuple[np.ndarray, list[YoloObject]]:
    mixed = cv2.addWeighted(img_a, alpha, img_b, 1.0 - alpha, 0.0)
    return mixed, objs_a + objs_b


def _copy_split(source_root: Path, target_root: Path, split_src: str, split_dst: str) -> None:
    src_images = source_root / split_src / "images"
    src_labels = source_root / split_src / "labels"
    dst_images = target_root / split_dst / "images"
    dst_labels = target_root / split_dst / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(src_images.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = src_labels / f"{image_path.stem}.txt"
        shutil.copy2(image_path, dst_images / image_path.name)
        if label_path.exists():
            shutil.copy2(label_path, dst_labels / label_path.name)


def cmd_check() -> None:
    train_cfg = load_yaml(ROOT / "configs" / "train.yaml")
    data_key = "prepared_dataset_yaml" if train_cfg.get("use_prepared_dataset", False) else "data"
    active_yaml = ROOT / train_cfg[data_key]
    if not active_yaml.exists():
        raise FileNotFoundError(f"训练配置中指定的数据集文件不存在: {active_yaml}")

    active_cfg = load_yaml(active_yaml)
    active_root = ROOT / active_cfg["path"]
    if not active_root.exists():
        active_root = Path(active_cfg["path"])

    registry = load_yaml(ROOT / "configs" / "datasets_registry.yaml")

    print("=" * 72)
    print("数据集体检报告")
    print("=" * 72)
    print(f"当前训练数据配置: {active_yaml}")
    print(f"当前训练数据根目录: {active_root}")
    print()

    active_train_img = _count_images(active_root / "train" / "images")
    active_train_lbl = _count_labels(active_root / "train" / "labels")
    active_val_img = _count_images(active_root / "val" / "images") + _count_images(active_root / "valid" / "images")
    active_val_lbl = _count_labels(active_root / "val" / "labels") + _count_labels(active_root / "valid" / "labels")
    print("[active_dataset]")
    print(f"  train images/labels: {active_train_img}/{active_train_lbl}")
    print(f"  val(valid) images/labels: {active_val_img}/{active_val_lbl}")
    print()

    datasets = registry.get("datasets", {})
    for name, info in datasets.items():
        ds_path = Path(info["path"])
        ds_root = ds_path if ds_path.is_absolute() else ROOT / ds_path
        required = [Path(p) for p in info.get("required", [])]

        print(f"[{name}] task={info.get('task', 'unknown')}")
        print(f"  路径: {ds_root}")
        print(f"  说明: {info.get('notes', '')}")

        exists = ds_root.exists()
        print(f"  根目录存在: {'YES' if exists else 'NO'}")
        print(f"  README 存在: {'YES' if _has_readme(ds_root) else 'NO'}")

        missing = [str(p) for p in required if not (ds_root / p).exists()]
        if missing:
            print("  结构完整: NO")
            print(f"  缺失目录: {missing}")
        else:
            print("  结构完整: YES")

        train_img = _count_images(ds_root / "train" / "images")
        train_lbl = _count_labels(ds_root / "train" / "labels")
        val_img = _count_images(ds_root / "val" / "images") + _count_images(ds_root / "valid" / "images")
        val_lbl = _count_labels(ds_root / "val" / "labels") + _count_labels(ds_root / "valid" / "labels")
        print(f"  train images/labels: {train_img}/{train_lbl}")
        print(f"  val(valid) images/labels: {val_img}/{val_lbl}")

        in_use = active_root.resolve() == ds_root.resolve() if exists else False
        print(f"  当前是否在用: {'YES' if in_use else 'NO'}")
        print()

    print("结论建议:")
    print("1. 结构完整且当前在用的数据集可直接训练。")
    print("2. 若结构不完整，请先按对应 README 下载并整理标注，再更新 configs/dataset.yaml。")


def cmd_prepare() -> None:
    random.seed(42)
    np.random.seed(42)

    cfg = load_yaml(ROOT / "configs" / "prepare.yaml")
    source_root = ROOT / cfg["source"]["root"]
    target_root = ROOT / cfg["target"]["root"]

    if not source_root.exists():
        raise FileNotFoundError(f"未找到数据源目录: {source_root}")

    if target_root.exists():
        shutil.rmtree(target_root)

    _copy_split(source_root, target_root, "train", "train")
    _copy_split(source_root, target_root, "valid", "valid")

    aug_cfg = cfg["augmentation"]
    spot_cfg = cfg["spot_guided"]
    image_dir = target_root / "train" / "images"
    label_dir = target_root / "train" / "labels"
    image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    sg_cfg = SpotGuidedConfig(
        slic_segments=int(spot_cfg["slic_segments"]),
        slic_compactness=float(spot_cfg["slic_compactness"]),
        glcm_distances=tuple(int(v) for v in spot_cfg["glcm_distances"]),
        glcm_angles=tuple(float(v) for v in spot_cfg["glcm_angles"]),
        entropy_threshold_quantile=float(spot_cfg["entropy_threshold_quantile"]),
        blend_alpha=float(spot_cfg["blend_alpha"]),
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        objects = read_labels_yolo(label_dir / f"{image_path.stem}.txt")

        if aug_cfg.get("rotate_90", True):
            rot = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(str(image_dir / f"{image_path.stem}_rot90{image_path.suffix}"), rot)
            write_labels_yolo(label_dir / f"{image_path.stem}_rot90.txt", _rot90_clockwise_objects(objects))

        if aug_cfg.get("brightness", True):
            gain = random.uniform(float(aug_cfg["brightness_gain_min"]), float(aug_cfg["brightness_gain_max"]))
            bright = _brightness(image, gain)
            cv2.imwrite(str(image_dir / f"{image_path.stem}_bright{image_path.suffix}"), bright)
            write_labels_yolo(label_dir / f"{image_path.stem}_bright.txt", objects)

        if bool(spot_cfg.get("enabled", False)):
            guided, _ = apply_spot_guided_attention(image, sg_cfg)
            cv2.imwrite(str(image_dir / f"{image_path.stem}_guided{image_path.suffix}"), guided)
            write_labels_yolo(label_dir / f"{image_path.stem}_guided.txt", objects)

    if aug_cfg.get("mixup", True) and len(image_paths) >= 2:
        alpha = float(aug_cfg["mixup_alpha"])
        shuffled = image_paths[:]
        random.shuffle(shuffled)
        pair_count = len(shuffled) // 2
        for idx in range(pair_count):
            img_a_path = shuffled[idx * 2]
            img_b_path = shuffled[idx * 2 + 1]
            img_a = cv2.imread(str(img_a_path))
            img_b = cv2.imread(str(img_b_path))
            if img_a is None or img_b is None:
                continue
            if img_a.shape[:2] != img_b.shape[:2]:
                img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

            objs_a = read_labels_yolo(label_dir / f"{img_a_path.stem}.txt")
            objs_b = read_labels_yolo(label_dir / f"{img_b_path.stem}.txt")
            mixed, objs = _mixup(img_a, objs_a, img_b, objs_b, alpha)
            cv2.imwrite(str(image_dir / f"mixup_{idx:05d}.jpg"), mixed)
            write_labels_yolo(label_dir / f"mixup_{idx:05d}.txt", objs)

    print(f"数据准备完成: {target_root}")
