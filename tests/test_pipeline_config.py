from pathlib import Path

import yaml


def test_train_yaml_has_predict_and_benchmark() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "train.yaml").read_text(encoding="utf-8"))

    assert "predict" in cfg, "train.yaml 缺少 predict 配置"
    assert "benchmark" in cfg, "train.yaml 缺少 benchmark 配置"

    predict = cfg["predict"]
    benchmark = cfg["benchmark"]

    assert isinstance(predict.get("use_spot_guided"), bool)
    assert int(predict.get("imgsz", 0)) > 0
    assert float(predict.get("conf", -1)) >= 0

    assert benchmark.get("source_split") in {"train", "val", "valid"}
    assert int(benchmark.get("sample_limit", 0)) >= 0
    assert isinstance(benchmark.get("use_spot_guided_options"), list)
    assert float(benchmark.get("target_fps", 0)) > 0

    claims = cfg.get("claims", {})
    assert claims.get("research_route") in {"A", "B"}
    assert isinstance(claims.get("bifpn_implemented"), bool)
    assert float(claims.get("target_effective_sparsity", 0)) > 0


def test_dataset_yaml_exists_for_active_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "train.yaml").read_text(encoding="utf-8"))

    key = "prepared_dataset_yaml" if cfg.get("use_prepared_dataset", False) else "data"
    target = root / cfg[key]
    assert target.exists(), f"激活的数据集配置不存在: {target}"


def test_prepare_yaml_copy_original_flag() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "prepare.yaml").read_text(encoding="utf-8"))
    augmentation = cfg.get("augmentation", {})
    assert isinstance(augmentation.get("copy_original"), bool), "prepare.yaml 中 copy_original 应为布尔值"


def test_distill_resume_friendly_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "distill.yaml").read_text(encoding="utf-8"))
    assert isinstance(cfg.get("reuse_generated_data"), bool)
    assert isinstance(cfg.get("disable_comet"), bool)
    train_cfg = cfg.get("train", {})
    assert isinstance(train_cfg.get("warm_start_from_last"), bool)


def test_prune_resume_friendly_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "prune.yaml").read_text(encoding="utf-8"))
    assert isinstance(cfg.get("stats_output"), str)
    assert isinstance(cfg.get("disable_comet"), bool)
    finetune = cfg.get("finetune", {})
    assert isinstance(finetune.get("dedupe_labels_on_start"), bool)
    assert isinstance(finetune.get("warm_start_from_last"), bool)
