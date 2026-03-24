from __future__ import annotations

import shutil
import subprocess

from scripts.ops.common import ROOT


def cmd_clean() -> None:
    cache_dirs = [ROOT / "scripts" / "__pycache__", ROOT / "src" / "__pycache__", ROOT / "tests" / "__pycache__"]
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
    subprocess.run(["bash", "-lc", f"rm -f {ROOT}/data/fa031-main/NEU-DET/*.cache"], check=False)
    print("清理完成")
