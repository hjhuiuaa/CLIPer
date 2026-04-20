"""
Disorder training/eval pipeline: reuses cliper.train/evaluate with patched windowing.

- Training: overlapping grid of crops; prefers windows that do not bisect contiguous positive runs.
- Eval / holdout: sliding windows with overlap, logits merged by mean (same as linker merge_window_logits).

Stage (stage1/2/3), backbone, and classifier head (linear / mlp5 / mlp12 / transformer) are unchanged from
CLIPer: set `stage` and `classifier_head` in the YAML (see disorder/configs/disorder_stage3_mlp5_example.yaml).

Precomputed ProstT5 embeddings (one file per protein, one line per residue): use CLI `extract_features`,
then `train_features` / `eval_features` with YAML `disorder/configs/disorder_feature_stage3_mlp5_example.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

from cliper.data import ProteinRecord
from cliper.metrics import (
    apply_threshold,
    binary_roc_auc,
    f1_score,
    mcc_score,
    precision_recall_auc,
    search_best_threshold,
)
from cliper.pipeline import (
    TrainItem,
    _batched_forward,
    _extract_scored_residues,
    prepare_data,
)
from cliper.windowing import merge_window_logits, sigmoid
from disorder.windowing import build_sliding_eval_starts, pick_training_window
from torch.utils.data import Dataset


@dataclass
class _RuntimeWindowCfg:
    train_window_overlap: int = 256
    eval_window_overlap: int = 256
    train_split_penalty_weight: float = 3.0


_RUNTIME = _RuntimeWindowCfg()


def load_disorder_config(path: str | Path) -> dict[str, Any]:
    """Same as cliper load_config; disorder keys must appear in YAML (see example config)."""
    from cliper.pipeline import load_config

    return load_config(path)


def _sync_runtime_from_yaml(path: str | Path) -> None:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML: {path}")
    wo = int(raw.get("window_overlap", 256))
    eo = int(raw.get("eval_window_overlap", wo))
    sp = float(raw.get("train_split_penalty_weight", 3.0))
    _RUNTIME.train_window_overlap = wo
    _RUNTIME.eval_window_overlap = eo
    _RUNTIME.train_split_penalty_weight = sp


def _sync_runtime_from_checkpoint_config(config: dict[str, Any]) -> None:
    wo = int(config.get("window_overlap", 256))
    eo = int(config.get("eval_window_overlap", wo))
    sp = float(config.get("train_split_penalty_weight", 3.0))
    _RUNTIME.train_window_overlap = wo
    _RUNTIME.eval_window_overlap = eo
    _RUNTIME.train_split_penalty_weight = sp


class DisorderTrainDataset(Dataset):
    """One sample = one overlapping window; long proteins use sliding-grid crop selection."""

    def __init__(self, records: list[ProteinRecord], window_size: int, seed: int) -> None:
        self.records = records
        self.window_size = window_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrainItem:
        record = self.records[index]
        seq, lab, start = pick_training_window(
            record.sequence,
            record.labels,
            self.window_size,
            _RUNTIME.train_window_overlap,
            self.seed + index,
            split_penalty_weight=_RUNTIME.train_split_penalty_weight,
        )
        return TrainItem(protein_id=record.protein_id, sequence=seq, labels=lab, start=start)


def disorder_evaluate_records(
    model: nn.Module,
    tokenizer: Any,
    backbone_name: str,
    records: list[ProteinRecord],
    *,
    window_size: int,
    stride: int | None,
    top_k_heuristic: int,
    device: torch.device,
    batch_size: int,
    threshold: float | None,
    threshold_search: dict[str, float],
    mixed_precision: bool,
) -> dict[str, Any]:
    del stride, top_k_heuristic
    model.eval()
    per_record: list[tuple[str, str, list[float]]] = []
    y_true_all: list[int] = []
    y_prob_all: list[float] = []
    eval_overlap = _RUNTIME.eval_window_overlap
    step_stride = max(1, int(window_size) - int(eval_overlap))

    for record in records:
        starts = build_sliding_eval_starts(len(record.sequence), window_size, step_stride)
        windows = [record.sequence[start : start + window_size] for start in starts]
        window_logits = _batched_forward(
            model,
            tokenizer,
            backbone_name,
            windows,
            device=device,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
        )
        merged_logits = merge_window_logits(
            length=len(record.sequence),
            window_logits=list(zip(starts, window_logits)),
        )
        probs = sigmoid(merged_logits)
        per_record.append((record.protein_id, record.labels, probs))
        true_slice, prob_slice = _extract_scored_residues(record.labels, probs)
        y_true_all.extend(true_slice)
        y_prob_all.extend(prob_slice)

    tuned_threshold = threshold
    best_f1 = 0.0
    best_mcc = 0.0
    if y_true_all and tuned_threshold is None:
        tuned_threshold, best_f1, best_mcc = search_best_threshold(
            y_true_all,
            y_prob_all,
            min_threshold=float(threshold_search["min"]),
            max_threshold=float(threshold_search["max"]),
            step=float(threshold_search["step"]),
        )
    if tuned_threshold is None:
        tuned_threshold = 0.5

    preds = apply_threshold(y_prob_all, tuned_threshold) if y_true_all else []
    auprc = precision_recall_auc(y_true_all, y_prob_all) if y_true_all else 0.0
    auroc = binary_roc_auc(y_true_all, y_prob_all)
    f1 = best_f1 if y_true_all and threshold is None else (f1_score(y_true_all, preds) if y_true_all else 0.0)
    mcc = best_mcc if y_true_all and threshold is None else (mcc_score(y_true_all, preds) if y_true_all else 0.0)

    return {
        "threshold": tuned_threshold,
        "metrics": {
            "num_records": len(records),
            "num_scored_residues": len(y_true_all),
            "auprc": auprc,
            "auroc": auroc,
            "f1": f1,
            "mcc": mcc,
            "threshold": tuned_threshold,
        },
        "per_record": per_record,
    }


def train(config_path: str | Path, resume_checkpoint: str | Path | None = None) -> dict[str, Any]:
    import cliper.pipeline as cp

    _sync_runtime_from_yaml(config_path)
    _orig_ds, _orig_er = cp.TrainDataset, cp.evaluate_records
    cp.TrainDataset = DisorderTrainDataset
    cp.evaluate_records = disorder_evaluate_records
    try:
        return cp.train(config_path, resume_checkpoint=resume_checkpoint)
    finally:
        cp.TrainDataset = _orig_ds
        cp.evaluate_records = _orig_er


def evaluate(
    *,
    checkpoint_path: str | Path,
    fasta_path: str | Path,
    output_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    split_key: str | None = None,
    threshold: float | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    import cliper.pipeline as cp

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _sync_runtime_from_checkpoint_config(checkpoint["config"])
    _orig_er = cp.evaluate_records
    cp.evaluate_records = disorder_evaluate_records
    try:
        return cp.evaluate(
            checkpoint_path=checkpoint_path,
            fasta_path=fasta_path,
            output_dir=output_dir,
            split_manifest_path=split_manifest_path,
            split_key=split_key,
            threshold=threshold,
            batch_size=batch_size,
        )
    finally:
        cp.evaluate_records = _orig_er


__all__ = [
    "prepare_data",
    "train",
    "evaluate",
    "load_disorder_config",
    "DisorderTrainDataset",
    "disorder_evaluate_records",
]
