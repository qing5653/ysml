from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import yaml


@dataclass
class SourceCfg:
    name: str
    dataset_yaml: Path


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"配置不存在: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_names(names: dict | list) -> list[str]:
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        items = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda x: x[0])
        return [v for _, v in items]
    raise TypeError("names 字段必须是 list 或 dict")


def _ensure_dirs(root: Path) -> None:
    (root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "val" / "images").mkdir(parents=True, exist_ok=True)
    (root / "val" / "labels").mkdir(parents=True, exist_ok=True)


def _resolve_split_label_dir(ds_root: Path, split_images_rel: str) -> Path:
    return ds_root / split_images_rel.replace("images", "labels")


def _split_candidates(data_cfg: dict, split: str) -> str | None:
    if split in data_cfg:
        return str(data_cfg[split])
    if split == "val" and "valid" in data_cfg:
        return str(data_cfg["valid"])
    return None


def _iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for p in sorted(path.glob("*")):
        if p.suffix.lower() in exts:
            yield p


def _parse_detection_lines(lines: list[str], class_name_list: list[str], merged_class_to_id: dict[str, int]) -> tuple[list[str], int]:
    converted: list[str] = []
    invalid = 0
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        # 仅接收检测标注: cls x y w h
        if len(parts) != 5:
            invalid += 1
            continue

        src_cls = int(parts[0])
        if src_cls < 0 or src_cls >= len(class_name_list):
            invalid += 1
            continue

        cls_name = class_name_list[src_cls]
        dst_cls = merged_class_to_id[cls_name]
        converted.append(f"{dst_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

    return converted, invalid


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = _load_yaml(root / "configs" / "merge.yaml")

    sources = [SourceCfg(name=s["name"], dataset_yaml=root / s["dataset_yaml"]) for s in cfg["sources"]]
    output_root = root / cfg["output"]["dataset_root"]
    output_yaml = root / cfg["output"]["dataset_yaml"]
    clean_output = bool(cfg.get("options", {}).get("clean_output", True))
    copy_unlabeled = bool(cfg.get("options", {}).get("copy_unlabeled_images", False))

    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    _ensure_dirs(output_root)

    merged_class_to_id: dict[str, int] = {}
    used_sources = []
    per_source_stats: dict[str, dict[str, int]] = {}

    # 先扫描来源配置并汇总全局类别空间
    source_runtime = []
    for src in sources:
        if not src.dataset_yaml.exists():
            print(f"[WARN] 跳过 {src.name}: dataset_yaml 不存在 {src.dataset_yaml}")
            continue
        data_cfg = _load_yaml(src.dataset_yaml)
        class_names = _normalize_names(data_cfg["names"])
        for name in class_names:
            if name not in merged_class_to_id:
                merged_class_to_id[name] = len(merged_class_to_id)

        ds_root = root / data_cfg["path"]
        if not ds_root.exists():
            print(f"[WARN] 跳过 {src.name}: 数据根目录不存在 {ds_root}")
            continue

        train_rel = _split_candidates(data_cfg, "train")
        val_rel = _split_candidates(data_cfg, "val")
        if not train_rel or not val_rel:
            print(f"[WARN] 跳过 {src.name}: 缺少 train/val 定义")
            continue

        source_runtime.append((src.name, ds_root, train_rel, val_rel, class_names))
        used_sources.append(src.name)
        per_source_stats[src.name] = {
            "copied_train": 0,
            "copied_val": 0,
            "invalid_label_lines": 0,
            "skipped_missing_label": 0,
            "skipped_invalid_label": 0,
        }

    # 二阶段: 按 split 拷贝图像并重映射标签类别 ID。
    for src_name, ds_root, train_rel, val_rel, class_names in source_runtime:
        for split, rel in [("train", train_rel), ("val", val_rel)]:
            src_img_dir = ds_root / rel
            src_lbl_dir = _resolve_split_label_dir(ds_root, rel)
            if not src_img_dir.exists() or not src_lbl_dir.exists():
                print(f"[WARN] {src_name}:{split} 跳过，缺少 images/labels 目录")
                continue

            dst_img_dir = output_root / split / "images"
            dst_lbl_dir = output_root / split / "labels"

            for img_path in _iter_images(src_img_dir):
                label_path = src_lbl_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    per_source_stats[src_name]["skipped_missing_label"] += 1
                    if not copy_unlabeled:
                        continue
                    converted_lines: list[str] = []
                else:
                    raw_lines = label_path.read_text(encoding="utf-8").splitlines()
                    converted_lines, invalid = _parse_detection_lines(raw_lines, class_names, merged_class_to_id)
                    per_source_stats[src_name]["invalid_label_lines"] += invalid

                    if invalid > 0 and not converted_lines:
                        per_source_stats[src_name]["skipped_invalid_label"] += 1
                        continue

                dst_name = f"{src_name}_{img_path.stem}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dst_img_dir / dst_name)
                (dst_lbl_dir / f"{src_name}_{img_path.stem}.txt").write_text("\n".join(converted_lines), encoding="utf-8")

                key = "copied_train" if split == "train" else "copied_val"
                per_source_stats[src_name][key] += 1

    # 生成统一 data.yaml，供 train.py 直接使用。
    merged_names = {i: name for name, i in sorted(merged_class_to_id.items(), key=lambda x: x[1])}
    out_yaml = {
        "path": str(output_root.relative_to(root)),
        "train": "train/images",
        "val": "val/images",
        "names": merged_names,
    }
    output_yaml.write_text(yaml.safe_dump(out_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print("=" * 72)
    print("数据集合并完成")
    print("=" * 72)
    print(f"输出数据集目录: {output_root}")
    print(f"输出数据集配置: {output_yaml}")
    print(f"已纳入来源: {used_sources}")
    print(f"合并后类别: {merged_names}")
    print("-" * 72)
    for name, stats in per_source_stats.items():
        print(f"[{name}] {stats}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"合并失败: {exc}")
        sys.exit(1)