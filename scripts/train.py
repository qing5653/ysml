from pathlib import Path

from ultralytics import YOLO
import yaml


def _compute_class_distribution(dataset_yaml: Path, project_root: Path) -> list[int]:
    data_cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
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
    return {"cls": float(cls_gain), "fl_gamma": float(fl_gamma)}


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

    data_key = "prepared_dataset_yaml" if cfg.get("use_prepared_dataset", False) else "data"
    data_path = root / cfg[data_key]
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据集配置: {data_path}")

    model = YOLO(str(model_path))
    train_kwargs = {
        k: v
        for k, v in cfg.items()
        if k not in {"model", "adaptive_loss", "use_prepared_dataset", "prepared_dataset_yaml"}
    }
    train_kwargs["data"] = str(data_path)
    train_kwargs["project"] = str(root / train_kwargs["project"])
    train_kwargs.update(_adaptive_loss_kwargs(cfg, data_path, root))

    if "cls" in train_kwargs and "fl_gamma" in train_kwargs:
        print(f"自适应损失参数: cls={train_kwargs['cls']:.3f}, fl_gamma={train_kwargs['fl_gamma']:.3f}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
