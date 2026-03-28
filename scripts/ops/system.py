from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import sys

from scripts.ops.common import ROOT, load_yaml, resolve_active_data_cfg_path


def _safe_remove_dir(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def _safe_remove_file(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink(missing_ok=True)
    return True


def cmd_doctor() -> None:
    checks: list[tuple[str, bool, str]] = []

    train_yaml = ROOT / "configs" / "train.yaml"
    checks.append(("train.yaml", train_yaml.exists(), str(train_yaml)))

    if train_yaml.exists():
        cfg = load_yaml(train_yaml)
        data_yaml = resolve_active_data_cfg_path(cfg)
        checks.append(("active_data_yaml", data_yaml.exists(), str(data_yaml)))

        model_path = ROOT / str(cfg.get("model", ""))
        checks.append(("train_init_model", model_path.exists(), str(model_path)))

    pyqt_ok = importlib.util.find_spec("PyQt5") is not None
    yolo_ok = importlib.util.find_spec("ultralytics") is not None
    cv2_ok = importlib.util.find_spec("cv2") is not None
    checks.append(("dependency:PyQt5", pyqt_ok, "PyQt5"))
    checks.append(("dependency:ultralytics", yolo_ok, "ultralytics"))
    checks.append(("dependency:opencv-python", cv2_ok, "cv2"))

    bundled_font = ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
    checks.append(("bundled_cjk_font", bundled_font.exists(), str(bundled_font)))

    print("=" * 72)
    print("环境与配置自检")
    print("=" * 72)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {ROOT}")
    print()

    failed = 0
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status:<4}] {name}: {detail}")

    print()
    if failed:
        print(f"诊断完成：{failed} 项失败，请按 FAIL 条目修复后重试。")
        raise SystemExit(1)
    print("诊断完成：全部通过。")


def cmd_clean() -> None:
    removed_dirs = 0
    removed_files = 0

    for cache_dir in ROOT.rglob("__pycache__"):
        if _safe_remove_dir(cache_dir):
            removed_dirs += 1

    for cache_dir in [ROOT / ".pytest_cache", ROOT / ".mypy_cache", ROOT / ".ruff_cache"]:
        if _safe_remove_dir(cache_dir):
            removed_dirs += 1

    for cache_file in (ROOT / "data").rglob("*.cache"):
        if _safe_remove_file(cache_file):
            removed_files += 1

    print(f"清理完成: 删除目录 {removed_dirs} 个，删除缓存文件 {removed_files} 个")
