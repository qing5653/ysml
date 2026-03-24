from __future__ import annotations

from pathlib import Path
import subprocess
import sys


# 统一入口命令映射，减少记忆多个脚本名的负担。
COMMAND_MAP = {
    "check": "check_datasets.py",
    "merge": "merge_datasets.py",
    "prepare": "prepare_dataset.py",
    "train": "train.py",
    "val": "val.py",
    "predict": "predict.py",
    "export": "export.py",
    "distill": "distill.py",
    "prune": "prune.py",
    "app": "app.py",
}


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python3 scripts/run.py <command>")
        print(f"可选: {', '.join(COMMAND_MAP)}")
        sys.exit(1)

    cmd = sys.argv[1].strip().lower()
    if cmd == "clean":
        root = Path(__file__).resolve().parents[1]
        # 仅清理可再生缓存，不触碰原始数据与训练结果。
        for cache_dir in [root / "scripts" / "__pycache__", root / "src" / "__pycache__", root / "tests" / "__pycache__"]:
            if cache_dir.exists():
                subprocess.run(["rm", "-rf", str(cache_dir)], check=False)
        subprocess.run(["bash", "-lc", f"rm -f {root}/data/fa031-main/NEU-DET/*.cache"], check=False)
        print("清理完成")
        return

    script = COMMAND_MAP.get(cmd)
    if script is None:
        print(f"未知命令: {cmd}")
        print(f"可选: {', '.join(COMMAND_MAP)}")
        sys.exit(1)

    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / script
    # 使用当前解释器继续执行，避免 python 与 python3 混用导致环境不一致。
    result = subprocess.run([sys.executable, str(script_path)])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
