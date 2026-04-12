from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import yaml

from .data import (
    ProteinRecord,
    build_split_manifest,
    parse_id_lines,
    parse_three_line_fasta,
    read_json,
    select_records,
    write_json,
)
from .metrics import apply_threshold, f1_score, mcc_score, precision_recall_auc, search_best_threshold
from .modeling import ResidueClassifier, encode_sequences, load_backbone_and_tokenizer
from .windowing import build_eval_window_starts, merge_window_logits, sigmoid, training_crop


def _autocast_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def _build_grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


REQUIRED_CONFIG_FIELDS = [
    "backbone_name",
    "window_size",
    "batch_tokens",
    "optimizer",
    "lr",
    "weight_decay",
    "max_epochs",
    "early_stop_patience",
    "seed",
    "threshold_search",
]


DEFAULTS = {
    "dropout": 0.1,
    "freeze_backbone": True,
    "val_ratio": 0.2,
    "eval_stride": None,
    "top_k_heuristic": 4,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_fasta": "dataset/disprot_202312_linker_label.fasta",
    "caid_fasta": "dataset/linker.fasta",
    "split_manifest": "artifacts/splits/disprot_split_seed42.json",
    "output_dir": "artifacts/runs/stage1",
}


@dataclass
class TrainItem:
    protein_id: str
    sequence: str
    labels: str
    start: int


class TrainDataset(Dataset):
    def __init__(self, records: list[ProteinRecord], window_size: int, seed: int) -> None:
        self.records = records
        self.window_size = window_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrainItem:
        record = self.records[index]
        seq, labels, start = training_crop(record.sequence, record.labels, self.window_size, seed=self.seed + index)
        return TrainItem(protein_id=record.protein_id, sequence=seq, labels=labels, start=start)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_required_fields(config: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config file must be a YAML object: {source}")
    _ensure_required_fields(config)
    merged = dict(DEFAULTS)
    merged.update(config)
    if merged["optimizer"].lower() != "adamw":
        raise ValueError("Only optimizer=adamw is supported in Stage 1.")
    threshold_search = merged["threshold_search"]
    if not isinstance(threshold_search, dict):
        raise ValueError("threshold_search must be a dict with min/max/step.")
    for key in ("min", "max", "step"):
        if key not in threshold_search:
            raise ValueError(f"threshold_search missing key: {key}")
    return merged


def prepare_data(
    *,
    fasta_path: str | Path,
    error_file: str | Path,
    caid_fasta: str | Path,
    seed: int,
    val_ratio: float,
    split_out: str | Path,
    exclusion_out: str | Path,
) -> dict:
    records = parse_three_line_fasta(fasta_path)
    error_ids = parse_id_lines(error_file)
    caid_ids = parse_id_lines(caid_fasta)
    split_manifest, exclusion_report = build_split_manifest(
        records,
        source_fasta=fasta_path,
        error_ids=error_ids,
        caid_ids=caid_ids,
        seed=seed,
        val_ratio=val_ratio,
    )
    write_json(split_out, split_manifest)
    write_json(exclusion_out, exclusion_report)
    return {
        "split_manifest_path": str(Path(split_out)),
        "exclusion_report_path": str(Path(exclusion_out)),
        "counts": split_manifest["counts"],
    }


def _labels_to_tensor(label_strings: list[str], residue_lengths: list[int]) -> torch.Tensor:
    max_len = max(residue_lengths)
    labels_tensor = torch.full((len(label_strings), max_len), -100.0, dtype=torch.float32)
    for row, (label_string, length) in enumerate(zip(label_strings, residue_lengths)):
        for idx, char in enumerate(label_string[:length]):
            if char == "1":
                labels_tensor[row, idx] = 1.0
            elif char == "0":
                labels_tensor[row, idx] = 0.0
            else:
                labels_tensor[row, idx] = -100.0
    return labels_tensor


def _batched_forward(
    model: nn.Module,
    tokenizer: Any,
    backbone_name: str,
    sequences: list[str],
    *,
    device: torch.device,
    batch_size: int,
    mixed_precision: bool,
) -> list[list[float]]:
    outputs: list[list[float]] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        encoded = encode_sequences(tokenizer, backbone_name, batch_sequences)
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        residue_lengths = encoded.residue_lengths

        with torch.no_grad():
            with _autocast_context(device_type=device.type, enabled=mixed_precision):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    residue_lengths=residue_lengths,
                )
        logits = logits.detach().float().cpu()
        for row, length in enumerate(residue_lengths):
            outputs.append(logits[row, :length].tolist())
    return outputs


def _extract_scored_residues(labels: str, probs: list[float]) -> tuple[list[int], list[float]]:
    y_true: list[int] = []
    y_prob: list[float] = []
    for char, prob in zip(labels, probs):
        if char == "1":
            y_true.append(1)
            y_prob.append(prob)
        elif char == "0":
            y_true.append(0)
            y_prob.append(prob)
        else:
            continue
    return y_true, y_prob


def evaluate_records(
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
    model.eval()
    per_record: list[tuple[str, str, list[float]]] = []
    y_true_all: list[int] = []
    y_prob_all: list[float] = []

    for record in records:
        starts = build_eval_window_starts(
            record.sequence,
            window_size=window_size,
            stride=stride,
            top_k_heuristic=top_k_heuristic,
        )
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
    f1 = best_f1 if y_true_all and threshold is None else (f1_score(y_true_all, preds) if y_true_all else 0.0)
    mcc = best_mcc if y_true_all and threshold is None else (mcc_score(y_true_all, preds) if y_true_all else 0.0)

    return {
        "threshold": tuned_threshold,
        "metrics": {
            "num_records": len(records),
            "num_scored_residues": len(y_true_all),
            "auprc": auprc,
            "f1": f1,
            "mcc": mcc,
            "threshold": tuned_threshold,
        },
        "per_record": per_record,
    }


def _write_predictions_tsv(path: str | Path, per_record: list[tuple[str, str, list[float]]], threshold: float) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write("protein_id\tposition_1based\tprobability\tpred_label\n")
        for protein_id, _, probs in per_record:
            for index, probability in enumerate(probs, start=1):
                pred = 1 if probability >= threshold else 0
                handle.write(f"{protein_id}\t{index}\t{probability:.8f}\t{pred}\n")


def _compute_pos_weight(records: list[ProteinRecord]) -> tuple[float, dict[str, int]]:
    pos = 0
    neg = 0
    for rec in records:
        pos += rec.labels.count("1")
        neg += rec.labels.count("0")
    if pos == 0:
        raise ValueError("No positive labels found in training split; cannot compute pos_weight.")
    return (neg / pos), {"positive": pos, "negative": neg, "total": pos + neg}


def train(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    set_seed(int(config["seed"]))

    device = torch.device(config["device"])
    use_amp = device.type == "cuda"
    output_dir = Path(config["output_dir"])
    metrics_dir = output_dir / "metrics"
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = read_json(config["split_manifest"])
    all_records = parse_three_line_fasta(config["train_fasta"])
    train_records = select_records(all_records, split_manifest["train_ids"])
    val_records = select_records(all_records, split_manifest["val_ids"])
    caid_records = parse_three_line_fasta(config["caid_fasta"])

    train_val_ids = {rec.protein_id for rec in train_records + val_records}
    caid_ids = {rec.protein_id for rec in caid_records}
    leakage = sorted(train_val_ids.intersection(caid_ids))
    if leakage:
        raise ValueError(f"Detected CAID3 leakage into train/val split. Example ids: {leakage[:10]}")

    pos_weight, class_stats = _compute_pos_weight(train_records)

    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(config["backbone_name"])
    model = ResidueClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        dropout=float(config["dropout"]),
        freeze_backbone=bool(config["freeze_backbone"]),
    ).to(device)

    batch_size = max(1, int(config["batch_tokens"]) // int(config["window_size"]))
    train_dataset = TrainDataset(train_records, window_size=int(config["window_size"]), seed=int(config["seed"]))

    def _collate(batch: list[TrainItem]) -> dict[str, Any]:
        sequences = [item.sequence for item in batch]
        labels = [item.labels for item in batch]
        encoded = encode_sequences(tokenizer, config["backbone_name"], sequences)
        label_tensor = _labels_to_tensor(labels, encoded.residue_lengths)
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "residue_lengths": encoded.residue_lengths,
            "labels": label_tensor,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["num_workers"]),
        collate_fn=_collate,
    )

    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scaler = _build_grad_scaler(enabled=use_amp)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")

    best_auprc = -1.0
    best_epoch = -1
    best_threshold = 0.5
    epochs_without_improve = 0
    history: list[dict[str, Any]] = []
    best_val_metrics: dict[str, Any] = {}

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            residue_lengths = batch["residue_lengths"]

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device_type=device.type, enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, residue_lengths=residue_lengths)
                mask = labels >= 0
                valid_count = int(mask.sum().item())
                if valid_count == 0:
                    continue
                loss_matrix = criterion(logits, labels)
                loss = (loss_matrix * mask).sum() / valid_count

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item())
            steps += 1

        train_loss = running_loss / max(1, steps)
        val_result = evaluate_records(
            model,
            tokenizer,
            config["backbone_name"],
            val_records,
            window_size=int(config["window_size"]),
            stride=config["eval_stride"],
            top_k_heuristic=int(config["top_k_heuristic"]),
            device=device,
            batch_size=batch_size,
            threshold=None,
            threshold_search=config["threshold_search"],
            mixed_precision=use_amp,
        )
        val_metrics = val_result["metrics"]
        val_threshold = float(val_result["threshold"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "val_threshold": val_threshold,
                "epoch_seconds": round(time.time() - start_time, 3),
            }
        )

        if val_metrics["auprc"] > best_auprc:
            best_auprc = float(val_metrics["auprc"])
            best_epoch = epoch
            best_threshold = val_threshold
            best_val_metrics = val_metrics
            epochs_without_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_auprc": best_auprc,
                    "threshold": best_threshold,
                    "config": config,
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= int(config["early_stop_patience"]):
            break

    best_ckpt_path = ckpt_dir / "best.pt"
    if not best_ckpt_path.exists():
        torch.save(
            {
                "epoch": history[-1]["epoch"] if history else 0,
                "model_state": model.state_dict(),
                "best_val_auprc": 0.0,
                "threshold": best_threshold,
                "config": config,
            },
            best_ckpt_path,
        )

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    best_threshold = float(checkpoint.get("threshold", best_threshold))

    caid_eval = evaluate_records(
        model,
        tokenizer,
        config["backbone_name"],
        caid_records,
        window_size=int(config["window_size"]),
        stride=config["eval_stride"],
        top_k_heuristic=int(config["top_k_heuristic"]),
        device=device,
        batch_size=batch_size,
        threshold=best_threshold,
        threshold_search=config["threshold_search"],
        mixed_precision=use_amp,
    )

    write_json(metrics_dir / "train_history.json", {"history": history})
    write_json(metrics_dir / "best_val_metrics.json", {"epoch": best_epoch, "metrics": best_val_metrics})
    write_json(metrics_dir / "caid3_metrics.json", caid_eval["metrics"])
    write_json(
        output_dir / "run_metadata.json",
        {
            "config_path": str(Path(config_path)),
            "split_manifest": str(Path(config["split_manifest"])),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "caid_records": len(caid_records),
            "class_stats_train": class_stats,
            "pos_weight": pos_weight,
            "excluded_error_ids": split_manifest.get("excluded_error_ids", []),
            "excluded_holdout_overlap_ids": split_manifest.get("excluded_holdout_overlap_ids", []),
            "selected_threshold": best_threshold,
        },
    )

    return {
        "output_dir": str(output_dir),
        "best_checkpoint": str(best_ckpt_path),
        "best_epoch": best_epoch,
        "best_val_auprc": best_auprc,
        "caid3_metrics": caid_eval["metrics"],
    }


def evaluate(
    *,
    checkpoint_path: str | Path,
    fasta_path: str | Path,
    output_dir: str | Path,
    split_manifest_path: str | Path | None = None,
    split_key: str | None = None,
    threshold: float | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if batch_size is None:
        batch_size = max(1, int(config["batch_tokens"]) // int(config["window_size"]))

    records = parse_three_line_fasta(fasta_path)
    if split_manifest_path and split_key:
        manifest = read_json(split_manifest_path)
        if split_key not in manifest:
            raise ValueError(f"split_key {split_key!r} not found in split manifest.")
        records = select_records(records, manifest[split_key])

    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(config["backbone_name"])
    model = ResidueClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        dropout=float(config.get("dropout", 0.1)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    resolved_threshold = float(threshold) if threshold is not None else float(checkpoint.get("threshold", 0.5))
    result = evaluate_records(
        model,
        tokenizer,
        config["backbone_name"],
        records,
        window_size=int(config["window_size"]),
        stride=config.get("eval_stride"),
        top_k_heuristic=int(config.get("top_k_heuristic", 4)),
        device=device,
        batch_size=batch_size,
        threshold=resolved_threshold,
        threshold_search=config["threshold_search"],
        mixed_precision=(device.type == "cuda"),
    )

    predictions_path = output / "predictions.tsv"
    metrics_path = output / "metrics.json"
    _write_predictions_tsv(predictions_path, result["per_record"], resolved_threshold)
    write_json(metrics_path, result["metrics"])

    return {
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics": result["metrics"],
    }
