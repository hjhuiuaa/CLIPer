from pathlib import Path

import pytest
import yaml

from cliper.pipeline import load_config


def _base_config() -> dict:
    return {
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
    }


def test_load_config_normalizes_wandb_tags_string(tmp_path: Path) -> None:
    config = _base_config()
    config.update({"use_wandb": True, "wandb_tags": "stage2"})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(config_path)
    assert loaded["wandb_tags"] == ["stage2"]
    assert loaded["wandb_project"] == "CLIPer"


def test_load_config_rejects_invalid_wandb_mode(tmp_path: Path) -> None:
    config = _base_config()
    config.update({"wandb_mode": "invalid"})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="wandb_mode"):
        load_config(config_path)


def test_load_config_requires_project_when_wandb_enabled(tmp_path: Path) -> None:
    config = _base_config()
    config.update({"use_wandb": True, "wandb_project": ""})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="wandb_project"):
        load_config(config_path)
