from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo11_project.spot_guided import SpotGuidedConfig, apply_spot_guided_attention


@dataclass
class YoloObject:
    class_id: int
    x: float
    y: float
    w: float
    h: float


def _read_labels(label_path: Path) -> list[YoloObject]:
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


def _write_labels(label_path: Path, objects: list[YoloObject]) -> None:
    lines = [f"{obj.class_id} {obj.x:.6f} {obj.y:.6f} {obj.w:.6f} {obj.h:.6f}" for obj in objects]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _rot90_clockwise_objects(objects: list[YoloObject]) -> list[YoloObject]:
    rotated: list[YoloObject] = []
    for obj in objects:
        rotated.append(YoloObject(obj.class_id, 1.0 - obj.y, obj.x, obj.h, obj.w))
    return rotated


def _brightness(image: np.ndarray, gain: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _mixup(img_a: np.ndarray, objs_a: list[YoloObject], img_b: np.ndarray, objs_b: list[YoloObject], alpha: float) -> tuple[np.ndarray, list[YoloObject]]:
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


def _augment_train_images(target_root: Path, cfg: dict) -> None:
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

        objects = _read_labels(label_dir / f"{image_path.stem}.txt")

        if aug_cfg.get("rotate_90", True):
            rot = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rot_name = f"{image_path.stem}_rot90{image_path.suffix}"
            cv2.imwrite(str(image_dir / rot_name), rot)
            _write_labels(label_dir / f"{image_path.stem}_rot90.txt", _rot90_clockwise_objects(objects))

        if aug_cfg.get("brightness", True):
            gain = random.uniform(float(aug_cfg["brightness_gain_min"]), float(aug_cfg["brightness_gain_max"]))
            bright = _brightness(image, gain)
            bright_name = f"{image_path.stem}_bright{image_path.suffix}"
            cv2.imwrite(str(image_dir / bright_name), bright)
            _write_labels(label_dir / f"{image_path.stem}_bright.txt", objects)

        if bool(spot_cfg.get("enabled", False)):
            guided, _ = apply_spot_guided_attention(image, sg_cfg)
            guided_name = f"{image_path.stem}_guided{image_path.suffix}"
            cv2.imwrite(str(image_dir / guided_name), guided)
            _write_labels(label_dir / f"{image_path.stem}_guided.txt", objects)

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

            objs_a = _read_labels(label_dir / f"{img_a_path.stem}.txt")
            objs_b = _read_labels(label_dir / f"{img_b_path.stem}.txt")
            mixed, objs = _mixup(img_a, objs_a, img_b, objs_b, alpha)
            name = f"mixup_{idx:05d}.jpg"
            cv2.imwrite(str(image_dir / name), mixed)
            _write_labels(label_dir / f"mixup_{idx:05d}.txt", objs)


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "prepare.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到数据准备配置: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_root = root / cfg["source"]["root"]
    target_root = root / cfg["target"]["root"]

    if not source_root.exists():
        raise FileNotFoundError(f"未找到数据源目录: {source_root}")

    if target_root.exists():
        shutil.rmtree(target_root)

    _copy_split(source_root, target_root, "train", "train")
    _copy_split(source_root, target_root, "valid", "valid")
    _augment_train_images(target_root, cfg)

    print(f"数据准备完成: {target_root}")


if __name__ == "__main__":
    main()