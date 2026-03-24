from pathlib import Path

from ultralytics import YOLO
import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_cfg = yaml.safe_load((root / "configs" / "train.yaml").read_text(encoding="utf-8"))
    best_ckpt = root / "experiments" / train_cfg["name"] / "weights" / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"未找到待导出权重: {best_ckpt}")

    model = YOLO(str(best_ckpt))
    exported = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"导出完成: {exported}")


if __name__ == "__main__":
    main()
