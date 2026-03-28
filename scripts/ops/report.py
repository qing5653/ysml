from __future__ import annotations

import csv
import json
from pathlib import Path
import time

from scripts.ops.common import ROOT, load_yaml


def _read_best_metrics(results_csv: Path) -> dict[str, float | int]:
    if not results_csv.exists():
        return {}

    max_map50 = -1.0
    max_map5095 = -1.0
    e_map50 = -1
    e_map5095 = -1
    last_row: dict[str, str] | None = None

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
            epoch = int(float(row.get("epoch", "0")))
            map50 = float(row.get("metrics/mAP50(B)", "0") or 0.0)
            map5095 = float(row.get("metrics/mAP50-95(B)", "0") or 0.0)
            if map50 > max_map50:
                max_map50 = map50
                e_map50 = epoch
            if map5095 > max_map5095:
                max_map5095 = map5095
                e_map5095 = epoch

    if last_row is None:
        return {}

    return {
        "max_map50": max_map50,
        "max_map50_epoch": e_map50,
        "max_map5095": max_map5095,
        "max_map5095_epoch": e_map5095,
        "last_precision": float(last_row.get("metrics/precision(B)", "0") or 0.0),
        "last_recall": float(last_row.get("metrics/recall(B)", "0") or 0.0),
        "last_map50": float(last_row.get("metrics/mAP50(B)", "0") or 0.0),
        "last_map5095": float(last_row.get("metrics/mAP50-95(B)", "0") or 0.0),
    }


def _read_latest_benchmark(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_latest_best_ckpt(exp_root: Path, run_name: str) -> Path | None:
    exact = exp_root / run_name / "weights" / "best.pt"
    if exact.exists():
        return exact

    candidates = []
    for run_dir in exp_root.glob(f"{run_name}*"):
        ckpt = run_dir / "weights" / "best.pt"
        if ckpt.exists():
            candidates.append(ckpt)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _is_checked(v: bool) -> str:
    return "x" if v else " "


def _max_benchmark_fps(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get("fps", "0") or 0.0))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def _file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def _fmt_mb(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}"


def _fmt_ratio(before: float | None, after: float | None) -> str:
    if before is None or after is None or before <= 0:
        return "N/A"
    return f"{(1.0 - after / before) * 100:.2f}%"


def _calc_ratio(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return (1.0 - after / before) * 100.0


def cmd_report() -> None:
    train_cfg = load_yaml(ROOT / "configs" / "train.yaml")
    run_name = str(train_cfg["name"])

    run_dir = ROOT / "experiments" / run_name
    results_csv = run_dir / "results.csv"
    metrics = _read_best_metrics(results_csv)

    baseline_pt = run_dir / "weights" / "best.pt"
    baseline_onnx = run_dir / "weights" / "best.onnx"
    pruned_pt = ROOT / "models" / "weights" / "yolo11_student_pruned.pt"

    distill_cfg = load_yaml(ROOT / "configs" / "distill.yaml")
    prune_cfg = load_yaml(ROOT / "configs" / "prune.yaml")
    distill_best = _resolve_latest_best_ckpt(ROOT / distill_cfg["train"]["project"], str(distill_cfg["train"]["name"]))
    prune_finetune_best = _resolve_latest_best_ckpt(
        ROOT / prune_cfg["finetune"]["project"], str(prune_cfg["finetune"]["name"])
    )

    benchmark_cfg = train_cfg.get("benchmark", {})
    benchmark_csv = ROOT / str(benchmark_cfg.get("output_csv", "experiments/benchmark/fps.csv"))
    benchmark_rows = _read_latest_benchmark(benchmark_csv)
    max_fps = _max_benchmark_fps(benchmark_rows)
    fps_target = float(benchmark_cfg.get("target_fps", 25.0))

    claims_cfg = train_cfg.get("claims", {})
    research_route = str(claims_cfg.get("research_route", "B")).upper()
    bifpn_implemented = bool(claims_cfg.get("bifpn_implemented", False))
    target_effective_sparsity = float(claims_cfg.get("target_effective_sparsity", 30.0))

    prune_stats_path = ROOT / str(prune_cfg.get("stats_output", "experiments/benchmark/prune_stats.json"))
    prune_stats = _read_json(prune_stats_path)
    effective_sparsity = prune_stats.get("effective_sparsity")
    if effective_sparsity is not None:
        try:
            effective_sparsity = float(effective_sparsity)
        except (TypeError, ValueError):
            effective_sparsity = None

    base_pt_mb = _file_size_mb(baseline_pt)
    base_onnx_mb = _file_size_mb(baseline_onnx)
    pruned_pt_mb = _file_size_mb(pruned_pt)
    size_ratio = _calc_ratio(base_pt_mb, pruned_pt_mb)

    lines: list[str] = []
    lines.append("# Bearing Defect Experiment Report")
    lines.append("")
    lines.append(f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Run name: {run_name}")
    lines.append(f"- Results CSV: {results_csv}")
    lines.append("")

    lines.append("## Detection Metrics")
    if metrics:
        lines.append(f"- Best mAP50: {metrics['max_map50']:.4f} (epoch={metrics['max_map50_epoch']})")
        lines.append(f"- Best mAP50-95: {metrics['max_map5095']:.4f} (epoch={metrics['max_map5095_epoch']})")
        lines.append(f"- Last precision: {metrics['last_precision']:.4f}")
        lines.append(f"- Last recall: {metrics['last_recall']:.4f}")
        lines.append(f"- Last mAP50: {metrics['last_map50']:.4f}")
        lines.append(f"- Last mAP50-95: {metrics['last_map5095']:.4f}")
    else:
        lines.append("- N/A: results.csv not found or empty.")
    lines.append("")

    lines.append("## Model Size (MB)")
    lines.append(f"- Baseline best.pt: {_fmt_mb(base_pt_mb)}")
    lines.append(f"- Baseline best.onnx: {_fmt_mb(base_onnx_mb)}")
    lines.append(f"- Pruned student pt: {_fmt_mb(pruned_pt_mb)}")
    lines.append(f"- Compression ratio (baseline pt -> pruned pt): {_fmt_ratio(base_pt_mb, pruned_pt_mb)}")
    lines.append("")

    lines.append("## Pruning Evidence")
    lines.append(f"- Prune stats file: {prune_stats_path if prune_stats_path.exists() else 'N/A'}")
    if effective_sparsity is not None:
        lines.append(f"- Effective sparsity: {effective_sparsity:.2f}% (target: {target_effective_sparsity:.2f}%)")
    else:
        lines.append(f"- Effective sparsity: N/A (target: {target_effective_sparsity:.2f}%)")
    lines.append("")

    lines.append("## Pipeline Status")
    lines.append(f"- Distill best.pt: {distill_best if distill_best else 'N/A'}")
    lines.append(f"- Prune output model: {pruned_pt if pruned_pt.exists() else 'N/A'}")
    lines.append(f"- Prune finetune best.pt: {prune_finetune_best if prune_finetune_best else 'N/A'}")
    lines.append("")

    lines.append("## Runtime Benchmark (FPS)")
    if benchmark_rows:
        for row in benchmark_rows:
            lines.append(
                "- "
                f"guided={row.get('use_spot_guided')} | images={row.get('images')} | "
                f"imgsz={row.get('imgsz')} | conf={row.get('conf')} | fps={row.get('fps')}"
            )
        if max_fps is not None:
            lines.append(f"- max_fps: {max_fps:.4f} (target: {fps_target:.2f})")
    else:
        lines.append("- N/A: run `python3 scripts/cli.py benchmark` first.")
    lines.append("")

    distill_ready = distill_best is not None and distill_best.exists()
    prune_ready = pruned_pt.exists() and prune_finetune_best is not None and prune_finetune_best.exists()
    sparsity_ready = (effective_sparsity is not None) and (effective_sparsity >= target_effective_sparsity)
    fps_ready = (max_fps is not None) and (max_fps >= fps_target)

    if research_route == "A":
        narrative_ready = bifpn_implemented
        narrative_text = "Route A consistency: BiFPN claim is implemented and traceable"
    else:
        narrative_ready = not bifpn_implemented
        narrative_text = "Route B consistency: report narrative matches implemented pipeline (no unimplemented BiFPN claim)"

    lines.append("## Readiness Checklist")
    lines.append(f"- [{_is_checked(narrative_ready)}] {narrative_text}")
    lines.append(f"- [{_is_checked(distill_ready)}] Distillation run completed with saved weights")
    lines.append(f"- [{_is_checked(prune_ready)}] Pruning run completed with saved weights")
    lines.append(
        f"- [{_is_checked(sparsity_ready)}] Effective sparsity >= {target_effective_sparsity:.1f}% is validated"
    )
    lines.append(f"- [{_is_checked(fps_ready)}] FPS >= {fps_target:.0f} validated on target hardware")

    report_path = ROOT / "docs" / "experiment_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"实验报告已生成: {report_path}")
