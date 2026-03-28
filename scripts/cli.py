from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops.compress import cmd_distill, cmd_prune
from scripts.ops.data import cmd_check, cmd_prepare
from scripts.ops.model import cmd_benchmark, cmd_export, cmd_predict, cmd_train, cmd_val
from scripts.ops.report import cmd_report
from scripts.ops.system import cmd_clean


COMMANDS = {
    "check": cmd_check,
    "prepare": cmd_prepare,
    "train": cmd_train,
    "val": cmd_val,
    "predict": cmd_predict,
    "benchmark": cmd_benchmark,
    "report": cmd_report,
    "export": cmd_export,
    "distill": cmd_distill,
    "prune": cmd_prune,
    "clean": cmd_clean,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 轴承缺陷检测统一命令入口")
    parser.add_argument("command", choices=sorted(COMMANDS), help="执行命令")
    args = parser.parse_args()

    try:
        COMMANDS[args.command]()
    except Exception as exc:  # noqa: BLE001
        print(f"执行失败({args.command}): {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()