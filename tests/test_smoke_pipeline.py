import json
from pathlib import Path

import yaml

from cliper.pipeline import evaluate, prepare_data, train


def _make_record(protein_id: str, length: int, positive_start: int, positive_len: int) -> str:
    sequence = ("ACDEFGHIKLMNPQRSTVWY" * ((length // 20) + 1))[:length]
    labels = ["0"] * length
    for idx in range(positive_start, min(length, positive_start + positive_len)):
        labels[idx] = "1"
    return f">{protein_id}\n{sequence}\n{''.join(labels)}\n"


def test_prepare_train_eval_smoke(tmp_path: Path) -> None:
    train_fasta = tmp_path / "disprot.fasta"
    train_fasta.write_text(
        (
            _make_record("T1", 60, 10, 8)
            + _make_record("T2", 64, 20, 12)
            + _make_record("T3", 72, 5, 10)
            + _make_record("T4", 80, 40, 6)
            + _make_record("T5", 70, 30, 5)
            + _make_record("T6", 75, 45, 7)
        ),
        encoding="utf-8",
    )

    caid_fasta = tmp_path / "caid.fasta"
    caid_fasta.write_text(_make_record("T6", 75, 45, 7) + _make_record("C1", 68, 22, 8), encoding="utf-8")

    error_file = tmp_path / "error.txt"
    error_file.write_text(">T5\n", encoding="utf-8")

    split_out = tmp_path / "split.json"
    exclusion_out = tmp_path / "exclude.json"
    prepare_result = prepare_data(
        fasta_path=train_fasta,
        error_file=error_file,
        caid_fasta=caid_fasta,
        seed=42,
        val_ratio=0.25,
        split_out=split_out,
        exclusion_out=exclusion_out,
    )
    assert Path(prepare_result["split_manifest_path"]).exists()
    assert Path(prepare_result["exclusion_report_path"]).exists()

    config = {
        "backbone_name": "dummy",
        "window_size": 64,
        "batch_tokens": 128,
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 2,
        "early_stop_patience": 2,
        "seed": 42,
        "threshold_search": {"min": 0.1, "max": 0.9, "step": 0.1},
        "train_fasta": str(train_fasta),
        "caid_fasta": str(caid_fasta),
        "split_manifest": str(split_out),
        "output_dir": str(tmp_path / "run"),
        "device": "cpu",
        "eval_stride": 32,
        "top_k_heuristic": 2,
        "save_every": 1,
        "print_every": 1,
        "eval_every": 2,
        "auto_start_tensorboard": False,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train_result = train(config_path)
    checkpoint_path = Path(train_result["best_checkpoint"])
    assert checkpoint_path.exists()
    assert Path(train_result["output_dir"]).exists()
    assert train_result["run_id"].startswith("exp")
    assert Path(train_result["log_file"]).exists()
    assert Path(train_result["tensorboard_dir"]).exists()
    assert train_result["wandb_service"]["status"] == "disabled"
    assert Path(train_result["last_checkpoint"]).exists()
    step_ckpts = list((Path(train_result["output_dir"]) / "checkpoints").glob("step_*.pt"))
    assert step_ckpts

    eval_result = evaluate(
        checkpoint_path=checkpoint_path,
        fasta_path=caid_fasta,
    )
    predictions_path = Path(eval_result["predictions_path"])
    metrics_path = Path(eval_result["metrics_path"])
    assert predictions_path.exists()
    assert metrics_path.exists()
    assert eval_result["eval_id"].startswith("eval")
    assert Path(eval_result["output_dir"]).is_dir()
    assert Path(eval_result["output_dir"]).parent == Path(train_result["output_dir"]) / "evaluations"
    assert eval_result["tensorboard_service"]["status"] == "disabled"
    assert eval_result["wandb_service"]["status"] == "disabled"

    header = predictions_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "protein_id\tposition_1based\tprobability\tpred_label"


def test_stage2_train_from_scratch_smoke(tmp_path: Path) -> None:
    train_fasta = tmp_path / "disprot.fasta"
    train_fasta.write_text(
        (
            _make_record("S1", 60, 10, 8)
            + _make_record("S2", 64, 20, 12)
            + _make_record("S3", 72, 5, 10)
            + _make_record("S4", 80, 40, 6)
            + _make_record("S5", 70, 30, 5)
            + _make_record("S6", 75, 45, 7)
        ),
        encoding="utf-8",
    )

    caid_fasta = tmp_path / "caid.fasta"
    caid_fasta.write_text(_make_record("S6", 75, 45, 7) + _make_record("C1", 68, 22, 8), encoding="utf-8")

    error_file = tmp_path / "error.txt"
    error_file.write_text(">S5\n", encoding="utf-8")

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
        "stage": "stage2",
        "backbone_name": "dummy",
        "window_size": 64,
        "batch_tokens": 128,
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 2,
        "early_stop_patience": 2,
        "seed": 42,
        "threshold_search": {"min": 0.1, "max": 0.9, "step": 0.1},
        "contrastive": {
            "enabled": True,
            "weight": 0.2,
            "temperature": 0.1,
            "proj_dim": 32,
            "max_samples_per_class": 16,
        },
        "train_fasta": str(train_fasta),
        "caid_fasta": str(caid_fasta),
        "split_manifest": str(split_out),
        "output_dir": str(tmp_path / "run_stage2"),
        "device": "cpu",
        "eval_stride": 32,
        "top_k_heuristic": 2,
        "save_every": 1,
        "print_every": 1,
        "eval_every": 2,
        "auto_start_tensorboard": False,
    }
    config_path = tmp_path / "config_stage2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train_result = train(config_path)
    checkpoint_path = Path(train_result["best_checkpoint"])
    assert checkpoint_path.exists()
    assert train_result["stage"] == "stage2"
    assert train_result["contrastive"]["enabled"] is True
    assert train_result["wandb_service"]["status"] == "disabled"
    assert Path(train_result["last_checkpoint"]).exists()

    run_metadata = json.loads((Path(train_result["output_dir"]) / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["stage"] == "stage2"
    assert run_metadata["contrastive"]["enabled"] is True
    assert run_metadata["wandb_service"]["status"] == "disabled"

    train_history = json.loads(
        (Path(train_result["output_dir"]) / "metrics" / "train_history.json").read_text(encoding="utf-8")
    )
    assert train_history["epoch_history"]
    assert "train_supcon_loss" in train_history["epoch_history"][0]

    eval_result = evaluate(
        checkpoint_path=checkpoint_path,
        fasta_path=caid_fasta,
    )
    assert Path(eval_result["predictions_path"]).exists()
    assert Path(eval_result["metrics_path"]).exists()
    assert eval_result["wandb_service"]["status"] == "disabled"
