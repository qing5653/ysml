from pathlib import Path

from ultralytics import YOLO
import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "train.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到训练配置: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Use the best checkpoint from the latest run by default.
    ckpt_path = root / "experiments" / cfg["name"] / "weights" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到验证权重: {ckpt_path}")

    data_path = root / cfg["data"]
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据集配置: {data_path}")

    model = YOLO(str(ckpt_path))
    metrics = model.val(data=str(data_path), imgsz=cfg["imgsz"], device=cfg["device"])
    print(metrics)


if __name__ == "__main__":
    main()
