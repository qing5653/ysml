from pathlib import Path
import time
import sys

import cv2
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo11_project.spot_guided import SpotGuidedConfig, apply_spot_guided_attention


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_cfg_path = root / "configs" / "train.yaml"
    train_cfg = yaml.safe_load(train_cfg_path.read_text(encoding="utf-8"))

    best_ckpt = root / "experiments" / train_cfg["name"] / "weights" / "best.pt"
    model_path = best_ckpt if best_ckpt.exists() else root / train_cfg["model"]

    data_key = "prepared_dataset_yaml" if train_cfg.get("use_prepared_dataset", False) else "data"
    data_cfg_path = root / train_cfg.get(data_key, "configs/dataset.yaml")
    data_cfg = yaml.safe_load(data_cfg_path.read_text(encoding="utf-8"))
    source = root / data_cfg["path"] / data_cfg["val"]

    prepare_cfg = yaml.safe_load((root / "configs" / "prepare.yaml").read_text(encoding="utf-8"))
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
    image_paths = [p for p in sorted(source.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not image_paths:
        raise RuntimeError(f"目录中没有图像文件: {source}")

    out_dir = root / "experiments" / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image, _ = apply_spot_guided_attention(image, sg_cfg)
        result = model.predict(image, imgsz=640, conf=0.25, verbose=False)[0]
        cv2.imwrite(str(out_dir / image_path.name), result.plot())
    elapsed = max(time.perf_counter() - start, 1e-6)
    fps = len(image_paths) / elapsed
    print(f"预测完成: 数量={len(image_paths)}, 平均FPS={fps:.2f}, 输出目录={out_dir}")


if __name__ == "__main__":
    main()
