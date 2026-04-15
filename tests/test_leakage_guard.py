from pathlib import Path

import pytest
import yaml

from cliper.pipeline import train


def test_train_raises_on_caid_leakage(tmp_path: Path) -> None:
    train_fasta = tmp_path / "train.fasta"
    train_fasta.write_text(">A\nAAAA\n0000\n>B\nAAAA\n1111\n", encoding="utf-8")

    caid_fasta = tmp_path / "caid.fasta"
    caid_fasta.write_text(">A\nAAAA\n0000\n", encoding="utf-8")

    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        (
            "{\n"
            '  "train_ids": ["A"],\n'
            '  "val_ids": ["B"],\n'
            '  "excluded_error_ids": [],\n'
            '  "excluded_holdout_overlap_ids": []\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config = {
        "backbone_name": "dummy",
        "window_size": 4,
        "batch_tokens": 8,
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1,
        "early_stop_patience": 1,
        "seed": 42,
        "threshold_search": {"min": 0.1, "max": 0.9, "step": 0.2},
        "train_fasta": str(train_fasta),
        "caid_fasta": str(caid_fasta),
        "split_manifest": str(split_manifest),
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "auto_start_tensorboard": False,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="leakage"):
        train(config_path)
