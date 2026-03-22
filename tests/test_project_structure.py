from pathlib import Path


def test_project_structure() -> None:
    root = Path(__file__).resolve().parents[1]
    required_dirs = [
        root / "configs",
        root / "data",
        root / "datasets",
        root / "experiments",
        root / "models",
        root / "scripts",
        root / "src",
        root / "tests",
    ]
    for d in required_dirs:
        assert d.exists() and d.is_dir(), f"Missing required directory: {d}"
