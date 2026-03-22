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

    model_path = root / cfg["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    data_path = root / cfg["data"]
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据集配置: {data_path}")

    model = YOLO(str(model_path))
    train_kwargs = {k: v for k, v in cfg.items() if k != "model"}
    train_kwargs["data"] = str(data_path)
    train_kwargs["project"] = str(root / train_kwargs["project"])
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
