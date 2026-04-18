from pathlib import Path

import pytest
import torch
import yaml

from cliper.pipeline import prepare_data, train


def _make_record(protein_id: str, length: int, positive_start: int, positive_len: int) -> str:
    sequence = ("ACDEFGHIKLMNPQRSTVWY" * ((length // 20) + 1))[:length]
    labels = ["0"] * length
    for idx in range(positive_start, min(length, positive_start + positive_len)):
        labels[idx] = "1"
    return f">{protein_id}\n{sequence}\n{''.join(labels)}\n"


def _build_base_config(tmp_path: Path, *, max_epochs: int) -> tuple[Path, dict]:
    train_fasta = tmp_path / "disprot.fasta"
    train_fasta.write_text(
        (
            _make_record("R1", 80, 10, 8)
            + _make_record("R2", 84, 22, 11)
            + _make_record("R3", 72, 6, 9)
            + _make_record("R4", 76, 30, 7)
            + _make_record("R5", 78, 18, 6)
            + _make_record("R6", 74, 40, 5)
        ),
        encoding="utf-8",
    )
    caid_fasta = tmp_path / "caid.fasta"
    caid_fasta.write_text(_make_record("C1", 70, 24, 8) + _make_record("C2", 66, 16, 7), encoding="utf-8")
    error_file = tmp_path / "error.txt"
    error_file.write_text("", encoding="utf-8")

    split_out = tmp_path / "split.json"
    exclusion_out = tmp_path / "exclude.json"
    prepare_data(
        fasta_path=train_fasta,
        error_file=error_file,
        caid_fasta=caid_fasta,
        seed=42,
        val_ratio=0.25,
        split_out=split_out,
        exclusion_out=exclusion_out,
    )

    config = {
        "backbone_name": "dummy",
        "window_size": 64,
        "batch_tokens": 128,
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": max_epochs,
        "early_stop_patience": 10,
        "seed": 42,
        "threshold_search": {"min": 0.1, "max": 0.9, "step": 0.1},
        "train_fasta": str(train_fasta),
        "caid_fasta": str(caid_fasta),
        "split_manifest": str(split_out),
        "error_file": str(error_file),
        "output_dir": str(tmp_path / "run"),
        "device": "cpu",
        "eval_stride": 32,
        "top_k_heuristic": 2,
        "save_every": 1,
        "print_every": 1,
        "eval_every": 2,
        "auto_start_tensorboard": False,
    }
    config_path = tmp_path / f"config_max{max_epochs}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def test_resume_training_success_same_experiment_dir(tmp_path: Path) -> None:
    first_config_path, _ = _build_base_config(tmp_path, max_epochs=1)
    first_result = train(first_config_path)
    first_last = Path(first_result["last_checkpoint"])
    first_steps = int(first_result["global_steps"])
    first_output_dir = Path(first_result["output_dir"])

    resume_config_path, _ = _build_base_config(tmp_path, max_epochs=3)
    resumed_result = train(resume_config_path, resume_checkpoint=first_last)

    assert resumed_result["resumed"] is True
    assert Path(resumed_result["output_dir"]) == first_output_dir
    assert int(resumed_result["global_steps"]) > first_steps
    assert int(resumed_result["resume_start_epoch"]) == 1
    assert int(resumed_result["resume_start_global_step"]) == first_steps


def test_resume_training_legacy_checkpoint_without_optimizer_scaler(tmp_path: Path) -> None:
    first_config_path, _ = _build_base_config(tmp_path, max_epochs=1)
    first_result = train(first_config_path)
    last_ckpt = Path(first_result["last_checkpoint"])

    legacy_ckpt = last_ckpt.parent / "legacy_last.pt"
    payload = torch.load(last_ckpt, map_location="cpu")
    payload.pop("optimizer_state", None)
    payload.pop("scaler_state", None)
    payload.pop("train_state", None)
    torch.save(payload, legacy_ckpt)

    resume_config_path, _ = _build_base_config(tmp_path, max_epochs=2)
    resumed_result = train(resume_config_path, resume_checkpoint=legacy_ckpt)
    assert resumed_result["resumed"] is True
    train_log = Path(resumed_result["log_file"]).read_text(encoding="utf-8")
    assert "optimizer_state missing in checkpoint" in train_log
    assert "scaler_state missing in checkpoint" in train_log


def test_resume_training_raises_when_max_epochs_not_increased(tmp_path: Path) -> None:
    first_config_path, _ = _build_base_config(tmp_path, max_epochs=1)
    first_result = train(first_config_path)
    last_ckpt = Path(first_result["last_checkpoint"])

    same_epoch_config_path, _ = _build_base_config(tmp_path, max_epochs=1)
    with pytest.raises(ValueError, match="Increase max_epochs"):
        train(same_epoch_config_path, resume_checkpoint=last_ckpt)
