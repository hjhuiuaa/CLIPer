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


def test_load_config_accepts_stage3_mlp12_defaults(tmp_path: Path) -> None:
    config = _base_config()
    config.update(
        {
            "stage": "stage3",
            "classifier_head": {"type": "mlp12"},
            "contrastive": {"enabled": False},
        }
    )
    config_path = tmp_path / "config_stage3_mlp12.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(config_path)
    assert loaded["stage"] == "stage3"
    assert loaded["classifier_head"]["type"] == "mlp12"
    assert len(loaded["classifier_head"]["hidden_dims"]) == 11


def test_load_config_accepts_stage3_mlp3_defaults(tmp_path: Path) -> None:
    config = _base_config()
    config.update(
        {
            "stage": "stage3",
            "classifier_head": {"type": "mlp3"},
            "contrastive": {"enabled": False},
        }
    )
    config_path = tmp_path / "config_stage3_mlp3.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(config_path)
    assert loaded["classifier_head"]["type"] == "mlp3"
    assert loaded["classifier_head"]["hidden_dims"] == [128, 64, 32]


def test_load_config_rejects_mlp3_hidden_dim_above_128(tmp_path: Path) -> None:
    config = _base_config()
    config.update(
        {
            "stage": "stage3",
            "classifier_head": {"type": "mlp3", "hidden_dims": [256, 64, 32]},
            "contrastive": {"enabled": False},
        }
    )
    config_path = tmp_path / "config_stage3_mlp3_bad.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="mlp3"):
        load_config(config_path)


def test_load_config_rejects_invalid_mlp12_hidden_dims_length(tmp_path: Path) -> None:
    config = _base_config()
    config.update(
        {
            "stage": "stage3",
            "classifier_head": {"type": "mlp12", "hidden_dims": [128] * 10},
            "contrastive": {"enabled": False},
        }
    )
    config_path = tmp_path / "config_stage3_mlp12_bad.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="mlp12"):
        load_config(config_path)


def test_load_config_accepts_stage3_transformer_defaults(tmp_path: Path) -> None:
    config = _base_config()
    config.update(
        {
            "stage": "stage3",
            "classifier_head": {"type": "transformer"},
            "contrastive": {"enabled": False},
        }
    )
    config_path = tmp_path / "config_stage3_transformer.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(config_path)
    assert loaded["stage"] == "stage3"
    assert loaded["classifier_head"]["type"] == "transformer"
    assert loaded["classifier_head"]["num_layers"] == 2
    assert loaded["classifier_head"]["num_heads"] == 4
    assert loaded["classifier_head"]["ffn_dim"] == 2048
    assert loaded["classifier_head"]["use_positional_encoding"] is True


def test_load_config_rejects_invalid_transformer_params(tmp_path: Path) -> None:
    config = _base_config()
    config.update(
        {
            "stage": "stage3",
            "classifier_head": {"type": "transformer", "num_layers": 0},
            "contrastive": {"enabled": False},
        }
    )
    config_path = tmp_path / "config_stage3_transformer_bad.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="num_layers"):
        load_config(config_path)


def test_stage3_forces_contrastive_disabled_even_if_enabled_in_config(tmp_path: Path) -> None:
    config = _base_config()
    config.update(
        {
            "stage": "stage3",
            "classifier_head": {"type": "transformer"},
            "contrastive": {"enabled": True},
        }
    )
    config_path = tmp_path / "config_stage3_force_off.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(config_path)
    assert loaded["contrastive"]["enabled"] is False
    assert loaded["stage3_contrastive_forced_off"] is True
