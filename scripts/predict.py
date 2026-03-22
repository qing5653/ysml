from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "src" / "yolo11n.pt"
    source = root / "datasets" / "bearing_defect" / "images" / "test"

    if not model_path.exists():
        raise FileNotFoundError(f"未找到推理模型: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"未找到推理输入目录: {source}")

    model = YOLO(str(model_path))
    results = model.predict(source=str(source), imgsz=640, conf=0.25, save=True, project=str(root / "experiments"), name="predict")
    print(f"预测完成，结果数量: {len(results)}")


if __name__ == "__main__":
    main()
