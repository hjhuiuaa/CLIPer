from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cliper.caid_io import parse_caid_fasta, write_caid_file, write_timings_csv
from cliper.caid_predict import load_classifier_from_checkpoint, predict_caid, predict_sequence_from_embedding
from disorder.feature_modeling import DisorderFeatureClassifier


pytest.importorskip("torch")


def _make_two_line_fasta(path: Path, protein_id: str, sequence: str) -> None:
    path.write_text(f">{protein_id}\n{sequence}\n", encoding="utf-8")


def _save_stage4_like_checkpoint(path: Path, *, base_hidden: int = 64) -> dict:
    local_context = {
        "enabled": True,
        "radius": 1,
        "mode": "concat_window",
        "include_self": True,
    }
    classifier_head = {
        "type": "cnn",
        "dropout": 0.0,
        "activation": "relu",
        "conv_channels": [16, 16],
        "kernel_size": 3,
        "dilations": [1, 1],
    }
    input_dim = base_hidden * 3
    model = DisorderFeatureClassifier(
        hidden_size=input_dim,
        dropout=0.0,
        classifier_head=classifier_head,
    )
    config = {
        "stage": "stage4",
        "window_size": 64,
        "eval_stride": 32,
        "top_k_heuristic": 2,
        "dropout": 0.0,
        "local_context": local_context,
        "classifier_head": classifier_head,
    }
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "threshold": 0.42,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return config


def test_parse_caid_fasta_sanitizes_ambiguous(tmp_path: Path) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">P1\nACBZXJ\n", encoding="utf-8")
    records = parse_caid_fasta(fasta)
    assert records == [("P1", "ACXXXX")]


def test_caid_predict_smoke(tmp_path: Path) -> None:
    seq = "ACDEFGHIKLMNPQRSTVWY" * 6
    checkpoint = tmp_path / "best.pt"
    config = _save_stage4_like_checkpoint(checkpoint, base_hidden=64)

    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    np.save(emb_dir / "P1.npy", np.random.RandomState(0).randn(len(seq), 64).astype(np.float32))

    two_line = tmp_path / "predict.fasta"
    _make_two_line_fasta(two_line, "P1", seq)

    model, metadata = load_classifier_from_checkpoint(checkpoint, device=torch.device("cpu"))
    assert metadata["threshold"] == 0.42
    probs = predict_sequence_from_embedding(
        model,
        sequence=seq,
        embedding=torch.tensor(np.load(emb_dir / "P1.npy")),
        local_context=config["local_context"],
        window_size=64,
        eval_stride=32,
        top_k_heuristic=2,
        device=torch.device("cpu"),
        window_batch_size=4,
    )
    assert len(probs) == len(seq)
    assert all(0.0 <= p <= 1.0 for p in probs)

    out_dir = tmp_path / "caid_out"
    summary = predict_caid(
        checkpoint_path=checkpoint,
        fasta_path=two_line,
        embeddings_dir=emb_dir,
        output_dir=out_dir,
        device="cpu",
        num_threads=2,
    )
    assert Path(summary["timings_path"]).exists()
    assert len(summary["caid_files"]) == 1
    caid_text = Path(summary["caid_files"][0]).read_text(encoding="utf-8")
    assert caid_text.startswith(">P1\n")
    assert json.loads((out_dir / "predict_summary.json").read_text(encoding="utf-8"))["num_sequences"] == 1

    write_timings_csv(out_dir / "timings_only.csv", [("P1", 42)])
    assert "sequence,milliseconds" in (out_dir / "timings_only.csv").read_text(encoding="utf-8")
    write_caid_file(
        out_dir / "manual.caid",
        protein_id="P1",
        sequence=seq[:3],
        probabilities=probs[:3],
        threshold=0.42,
    )
