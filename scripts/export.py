from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    best_ckpt = root / "experiments" / "yolo11_baseline" / "weights" / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"未找到待导出权重: {best_ckpt}")

    model = YOLO(str(best_ckpt))
    exported = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"导出完成: {exported}")


if __name__ == "__main__":
    main()
