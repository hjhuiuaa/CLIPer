import json
from pathlib import Path

import pytest
import yaml

from cliper.pipeline import train


def _make_record(protein_id: str, length: int, positive_start: int, positive_len: int) -> str:
    sequence = ("ACDEFGHIKLMNPQRSTVWY" * ((length // 20) + 1))[:length]
    labels = ["0"] * length
    for idx in range(positive_start, min(length, positive_start + positive_len)):
        labels[idx] = "1"
    return f">{protein_id}\n{sequence}\n{''.join(labels)}\n"


def _write_stale_split(path: Path) -> None:
    path.write_text(
        (
            "{\n"
            '  "train_ids": ["OLD1"],\n'
            '  "val_ids": ["OLD2"],\n'
            '  "excluded_error_ids": [],\n'
            '  "excluded_holdout_overlap_ids": []\n'
            "}\n"
        ),
        encoding="utf-8",
    )


def test_train_rebuilds_split_manifest_when_ids_missing(tmp_path: Path) -> None:
    train_fasta = tmp_path / "train.fasta"
    train_fasta.write_text(
        _make_record("A", 64, 10, 8) + _make_record("B", 70, 20, 6) + _make_record("C", 66, 0, 0),
        encoding="utf-8",
    )
    caid_fasta = tmp_path / "caid.fasta"
    caid_fasta.write_text(_make_record("X1", 68, 22, 8), encoding="utf-8")
    error_file = tmp_path / "error.txt"
    error_file.write_text("", encoding="utf-8")
    stale_split = tmp_path / "stale_split.json"
    _write_stale_split(stale_split)

    config = {
        "backbone_name": "dummy",
        "window_size": 64,
        "batch_tokens": 128,
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1,
        "early_stop_patience": 1,
        "seed": 42,
        "threshold_search": {"min": 0.1, "max": 0.9, "step": 0.1},
        "train_fasta": str(train_fasta),
        "caid_fasta": str(caid_fasta),
        "error_file": str(error_file),
        "split_manifest": str(stale_split),
        "output_dir": str(tmp_path / "run"),
        "device": "cpu",
        "save_every": 1,
        "print_every": 1,
        "eval_every": 2,
        "auto_start_tensorboard": False,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train(config_path)
    assert Path(result["best_checkpoint"]).exists()
    assert result["split_manifest_resolution"]["status"] == "rebuilt"
    rebuilt_split_path = Path(result["split_manifest"])
    assert rebuilt_split_path.exists()

    metadata = json.loads((Path(result["output_dir"]) / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["split_manifest_resolution"]["status"] == "rebuilt"
    assert Path(metadata["split_manifest"]).exists()


def test_train_raises_when_split_mismatch_and_auto_rebuild_disabled(tmp_path: Path) -> None:
    train_fasta = tmp_path / "train.fasta"
    train_fasta.write_text(_make_record("A", 64, 10, 8) + _make_record("B", 70, 20, 6), encoding="utf-8")
    caid_fasta = tmp_path / "caid.fasta"
    caid_fasta.write_text(_make_record("X1", 68, 22, 8), encoding="utf-8")
    error_file = tmp_path / "error.txt"
    error_file.write_text("", encoding="utf-8")
    stale_split = tmp_path / "stale_split.json"
    _write_stale_split(stale_split)

    config = {
        "backbone_name": "dummy",
        "window_size": 64,
        "batch_tokens": 128,
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1,
        "early_stop_patience": 1,
        "seed": 42,
        "threshold_search": {"min": 0.1, "max": 0.9, "step": 0.1},
        "train_fasta": str(train_fasta),
        "caid_fasta": str(caid_fasta),
        "error_file": str(error_file),
        "split_manifest": str(stale_split),
        "auto_rebuild_split_on_mismatch": False,
        "output_dir": str(tmp_path / "run"),
        "device": "cpu",
        "save_every": 1,
        "print_every": 1,
        "eval_every": 2,
        "auto_start_tensorboard": False,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Split manifest is inconsistent"):
        train(config_path)
