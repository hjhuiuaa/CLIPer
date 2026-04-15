from pathlib import Path

from cliper.pipeline import _resolve_experiment_dir


def test_resolve_experiment_dir_auto_increment(tmp_path: Path) -> None:
    first_dir, first_id = _resolve_experiment_dir(tmp_path, prefix="exp")
    first_dir.mkdir(parents=True, exist_ok=True)
    second_dir, second_id = _resolve_experiment_dir(tmp_path, prefix="exp")

    assert first_id == "exp0001"
    assert second_id == "exp0002"
    assert second_dir.name == "exp0002"

