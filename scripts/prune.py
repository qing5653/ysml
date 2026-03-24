from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO
import yaml


def _count_params(module: nn.Module) -> tuple[int, int]:
    total = 0
    non_zero = 0
    for p in module.parameters():
        total += p.numel()
        non_zero += int(torch.count_nonzero(p).item())
    return total, non_zero


def _apply_channel_prune(module: nn.Module, ratio: float) -> None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d) and child.weight.shape[0] >= 8:
            prune.ln_structured(child, name="weight", amount=ratio, n=2, dim=0)
            prune.remove(child, "weight")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "prune.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到剪枝配置: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    input_model = root / cfg["input_model"]
    output_model = root / cfg["output_model"]
    output_model.parent.mkdir(parents=True, exist_ok=True)

    if not input_model.exists():
        raise FileNotFoundError(f"未找到待剪枝模型: {input_model}")

    yolo_model = YOLO(str(input_model))
    torch_model = yolo_model.model

    total_before, non_zero_before = _count_params(torch_model)
    _apply_channel_prune(torch_model, float(cfg["channel_prune_ratio"]))
    total_after, non_zero_after = _count_params(torch_model)

    # 通过 YOLO 对象保存 checkpoint，确保后续可直接加载。
    yolo_model.save(str(output_model))
    print(
        "剪枝完成: "
        f"total={total_before} -> {total_after}, "
        f"non_zero={non_zero_before} -> {non_zero_after}, "
        f"effective_sparsity={(1.0 - non_zero_after / max(non_zero_before, 1)) * 100:.2f}%"
    )

    ft = cfg["finetune"]
    finetune_model = YOLO(str(input_model))
    finetune_model.model.load_state_dict(torch_model.state_dict(), strict=False)
    finetune_model.train(
        data=str(root / ft["data"]),
        epochs=int(ft["epochs"]),
        imgsz=int(ft["imgsz"]),
        batch=int(ft["batch"]),
        device=ft["device"],
        workers=int(ft["workers"]),
        project=str(root / ft["project"]),
        name=str(ft["name"]),
    )


if __name__ == "__main__":
    main()