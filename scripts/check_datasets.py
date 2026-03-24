from __future__ import annotations

from pathlib import Path
import sys

import yaml


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in exts)


def _count_labels(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*.txt"))


def _has_readme(path: Path) -> bool:
    candidates = [path, path.parent]
    for base in candidates:
        if (base / "README.md").exists() or (base / "readme.md").exists() or (base / "README.docx").exists():
            return True
    return False


def _resolve_active_dataset_yaml(root: Path) -> Path:
    train_cfg = _load_yaml(root / "configs" / "train.yaml")
    data_key = "prepared_dataset_yaml" if train_cfg.get("use_prepared_dataset", False) else "data"
    dataset_yaml = root / train_cfg[data_key]
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"训练配置中指定的数据集文件不存在: {dataset_yaml}")
    return dataset_yaml


def _resolve_dataset_root(project_root: Path, dataset_yaml: Path) -> Path:
    data_cfg = _load_yaml(dataset_yaml)
    rel = Path(data_cfg["path"])
    if rel.is_absolute():
        return rel
    return project_root / rel


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    # 读取注册表（全局视角）与当前 train.yaml 指向的数据（实际运行视角）。
    registry = _load_yaml(root / "configs" / "datasets_registry.yaml")
    active_yaml = _resolve_active_dataset_yaml(root)
    active_root = _resolve_dataset_root(root, active_yaml)

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
    if not datasets:
        print("未在 configs/datasets_registry.yaml 中找到 datasets 定义。")
        return

    # 逐个数据集做结构检查、样本计数、是否在用判断。
    for name, info in datasets.items():
        ds_path = Path(info["path"])
        ds_root = ds_path if ds_path.is_absolute() else root / ds_path
        required = [Path(p) for p in info.get("required", [])]

        print(f"[{name}] task={info.get('task', 'unknown')}")
        print(f"  路径: {ds_root}")
        print(f"  说明: {info.get('notes', '')}")

        exists = ds_root.exists()
        print(f"  根目录存在: {'YES' if exists else 'NO'}")

        readme_ok = _has_readme(ds_root)
        print(f"  README 存在: {'YES' if readme_ok else 'NO'}")

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


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"体检失败: {exc}")
        sys.exit(1)