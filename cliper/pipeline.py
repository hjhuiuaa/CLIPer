from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
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
    "experiment_prefix": "exp",
    "save_every": 500,
    "print_every": 20,
    "eval_every": 200,
    "auto_start_tensorboard": True,
    "tensorboard_host": "0.0.0.0",
    "tensorboard_port": 6006,
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
    for key in ("save_every", "print_every", "eval_every"):
        if int(merged[key]) <= 0:
            raise ValueError(f"{key} must be > 0, got {merged[key]}")
    if int(merged["tensorboard_port"]) <= 0:
        raise ValueError(f"tensorboard_port must be > 0, got {merged['tensorboard_port']}")
    return merged


def _resolve_experiment_dir(base_output_dir: str | Path, prefix: str) -> tuple[Path, str]:
    return _resolve_incremental_dir(base_output_dir, prefix=prefix)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _build_logger(log_path: str | Path):
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return _log


def _save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    best_val_auprc: float,
    threshold: float,
    config: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "best_val_auprc": best_val_auprc,
            "threshold": threshold,
            "config": config,
        },
        target,
    )


def _port_probe_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def _is_tcp_port_open(host: str, port: int) -> bool:
    probe_host = _port_probe_host(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe_host, port)) == 0


def _start_tensorboard_if_needed(
    *,
    enabled: bool,
    logdir: str | Path,
    host: str,
    port: int,
    runtime_log_path: str | Path,
    logger,
) -> dict[str, Any]:
    resolved_logdir = Path(logdir)
    resolved_logdir.mkdir(parents=True, exist_ok=True)
    runtime_log = Path(runtime_log_path)
    runtime_log.parent.mkdir(parents=True, exist_ok=True)

    service_url = f"http://{_port_probe_host(host)}:{port}"
    if not enabled:
        logger("[tensorboard] auto start disabled by config.")
        return {
            "status": "disabled",
            "url": service_url,
            "logdir": str(resolved_logdir),
            "host": host,
            "port": port,
            "pid": None,
        }

    if _is_tcp_port_open(host=host, port=port):
        logger(f"[tensorboard] detected existing service on {service_url}, reuse.")
        return {
            "status": "reused",
            "url": service_url,
            "logdir": str(resolved_logdir),
            "host": host,
            "port": port,
            "pid": None,
        }

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(resolved_logdir),
        "--host",
        host,
        "--port",
        str(port),
    ]

    with runtime_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_timestamp()}] launch tensorboard: {' '.join(cmd)}\n")
        kwargs: dict[str, Any] = {"stdout": handle, "stderr": subprocess.STDOUT}
        if sys.platform.startswith("win"):
            creationflags = 0
            creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            creationflags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
            kwargs["creationflags"] = creationflags
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **kwargs)

    time.sleep(0.8)
    if proc.poll() is not None:
        logger(
            f"[tensorboard] failed to start on {service_url}. "
            f"See runtime log: {runtime_log}"
        )
        return {
            "status": "failed",
            "url": service_url,
            "logdir": str(resolved_logdir),
            "host": host,
            "port": port,
            "pid": proc.pid,
        }

    logger(f"[tensorboard] started pid={proc.pid} at {service_url}, logdir={resolved_logdir}")
    return {
        "status": "started",
        "url": service_url,
        "logdir": str(resolved_logdir),
        "host": host,
        "port": port,
        "pid": proc.pid,
    }


def _infer_experiment_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def _resolve_incremental_dir(base_dir: str | Path, prefix: str) -> tuple[Path, str]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_index = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))
    next_index = max_index + 1
    item_id = f"{prefix}{next_index:04d}"
    return base / item_id, item_id


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
    output_base_dir = Path(config["output_dir"])
    output_dir, run_id = _resolve_experiment_dir(output_base_dir, prefix=str(config["experiment_prefix"]))
    metrics_dir = output_dir / "metrics"
    ckpt_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    tensorboard_dir = output_dir / "tensorboard"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    config_resolved = dict(config)
    config_resolved["output_dir"] = str(output_dir)
    config_resolved["run_id"] = run_id
    config_resolved["output_base_dir"] = str(output_base_dir)

    log = _build_logger(logs_dir / "train.log")
    log(f"Run started: run_id={run_id} output_dir={output_dir}")
    tensorboard_info = _start_tensorboard_if_needed(
        enabled=bool(config.get("auto_start_tensorboard", True)),
        logdir=output_base_dir,
        host=str(config.get("tensorboard_host", "0.0.0.0")),
        port=int(config.get("tensorboard_port", 6006)),
        runtime_log_path=logs_dir / "tensorboard_runtime.log",
        logger=log,
    )

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("run/id", run_id, 0)
    writer.add_text("run/config_path", str(Path(config_path)), 0)
    writer.add_text("run/config_resolved", json.dumps(config_resolved, ensure_ascii=False, indent=2), 0)
    writer.add_text("tensorboard/service", json.dumps(tensorboard_info, ensure_ascii=False, indent=2), 0)

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
    writer.add_scalar("data/class_positive", class_stats["positive"], 0)
    writer.add_scalar("data/class_negative", class_stats["negative"], 0)
    writer.add_scalar("train/pos_weight", pos_weight, 0)

    best_auprc = -1.0
    best_epoch = -1
    best_step = -1
    best_threshold = 0.5
    evals_without_improve = 0
    history: list[dict[str, Any]] = []
    print_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    best_val_metrics: dict[str, Any] = {}
    last_eval_metrics: dict[str, Any] = {}
    global_step = 0

    save_every = int(config["save_every"])
    print_every = int(config["print_every"])
    eval_every = int(config["eval_every"])
    early_stop_patience = int(config["early_stop_patience"])
    stop_training = False

    def _run_validation(epoch: int, step: int, reason: str) -> None:
        nonlocal best_auprc, best_epoch, best_step, best_threshold
        nonlocal best_val_metrics, evals_without_improve, last_eval_metrics

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
        last_eval_metrics = val_metrics
        eval_history.append(
            {
                "epoch": epoch,
                "global_step": step,
                "reason": reason,
                "metrics": val_metrics,
                "threshold": val_threshold,
            }
        )

        writer.add_scalar("val/auprc", float(val_metrics["auprc"]), step)
        writer.add_scalar("val/f1", float(val_metrics["f1"]), step)
        writer.add_scalar("val/mcc", float(val_metrics["mcc"]), step)
        writer.add_scalar("val/threshold", float(val_threshold), step)
        log(
            "[eval] "
            f"reason={reason} epoch={epoch} step={step} "
            f"auprc={val_metrics['auprc']:.6f} f1={val_metrics['f1']:.6f} mcc={val_metrics['mcc']:.6f} "
            f"threshold={val_threshold:.4f}"
        )

        if float(val_metrics["auprc"]) > best_auprc:
            best_auprc = float(val_metrics["auprc"])
            best_epoch = epoch
            best_step = step
            best_threshold = val_threshold
            best_val_metrics = val_metrics
            evals_without_improve = 0
            _save_checkpoint(
                ckpt_dir / "best.pt",
                epoch=epoch,
                global_step=step,
                model=model,
                best_val_auprc=best_auprc,
                threshold=best_threshold,
                config=config_resolved,
            )
            log(f"[best] updated best checkpoint at epoch={epoch}, step={step}, auprc={best_auprc:.6f}")
        else:
            evals_without_improve += 1
            log(f"[early-stop] no improvement count={evals_without_improve}/{early_stop_patience}")

    try:
        for epoch in range(1, int(config["max_epochs"]) + 1):
            model.train()
            start_time = time.time()
            running_loss = 0.0
            steps = 0
            window_loss = 0.0
            window_steps = 0
            eval_happened_in_epoch = False

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

                loss_value = float(loss.detach().cpu().item())
                global_step += 1
                running_loss += loss_value
                steps += 1
                window_loss += loss_value
                window_steps += 1

                writer.add_scalar("train/loss_step", loss_value, global_step)
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

                if global_step % print_every == 0:
                    avg_window_loss = window_loss / max(1, window_steps)
                    print_event = {"epoch": epoch, "global_step": global_step, "avg_loss": avg_window_loss}
                    print_history.append(print_event)
                    log(f"[train] epoch={epoch} step={global_step} avg_loss={avg_window_loss:.6f}")
                    writer.add_scalar("train/loss_print_avg", avg_window_loss, global_step)
                    window_loss = 0.0
                    window_steps = 0

                if global_step % save_every == 0:
                    checkpoint_path = ckpt_dir / f"step_{global_step:07d}.pt"
                    _save_checkpoint(
                        checkpoint_path,
                        epoch=epoch,
                        global_step=global_step,
                        model=model,
                        best_val_auprc=best_auprc,
                        threshold=best_threshold,
                        config=config_resolved,
                    )
                    log(f"[save] periodic checkpoint saved: {checkpoint_path.name}")

                if global_step % eval_every == 0:
                    eval_happened_in_epoch = True
                    _run_validation(epoch=epoch, step=global_step, reason="interval")
                    if evals_without_improve >= early_stop_patience:
                        stop_training = True
                        log(
                            f"[early-stop] triggered at epoch={epoch}, step={global_step}, "
                            f"patience={early_stop_patience}"
                        )
                        break

            if window_steps > 0:
                avg_window_loss = window_loss / max(1, window_steps)
                print_event = {"epoch": epoch, "global_step": global_step, "avg_loss": avg_window_loss}
                print_history.append(print_event)
                log(f"[train] epoch={epoch} step={global_step} avg_loss={avg_window_loss:.6f} (epoch_end_flush)")
                writer.add_scalar("train/loss_print_avg", avg_window_loss, global_step)

            train_loss = running_loss / max(1, steps)
            epoch_seconds = round(time.time() - start_time, 3)
            history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "epoch_seconds": epoch_seconds,
                    "train_steps": steps,
                }
            )
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("train/epoch_seconds", epoch_seconds, epoch)
            log(
                f"[epoch] epoch={epoch} steps={steps} global_step={global_step} "
                f"train_loss={train_loss:.6f} epoch_seconds={epoch_seconds:.3f}"
            )

            if stop_training:
                break

            if not eval_happened_in_epoch:
                _run_validation(epoch=epoch, step=global_step, reason="epoch_end")
                if evals_without_improve >= early_stop_patience:
                    stop_training = True
                    log(
                        f"[early-stop] triggered at epoch_end epoch={epoch}, step={global_step}, "
                        f"patience={early_stop_patience}"
                    )
                    break

        if global_step > 0 and global_step % save_every != 0:
            final_step_path = ckpt_dir / f"step_{global_step:07d}.pt"
            if not final_step_path.exists():
                _save_checkpoint(
                    final_step_path,
                    epoch=history[-1]["epoch"] if history else 0,
                    global_step=global_step,
                    model=model,
                    best_val_auprc=best_auprc,
                    threshold=best_threshold,
                    config=config_resolved,
                )
                log(f"[save] final step checkpoint saved: {final_step_path.name}")
    finally:
        writer.flush()
        writer.close()

    best_ckpt_path = ckpt_dir / "best.pt"
    if not best_ckpt_path.exists():
        _save_checkpoint(
            best_ckpt_path,
            epoch=history[-1]["epoch"] if history else 0,
            global_step=global_step,
            model=model,
            best_val_auprc=0.0,
            threshold=best_threshold,
            config=config_resolved,
        )
        log("[best] best checkpoint was missing; fallback best.pt created from latest model state.")

    last_ckpt_path = ckpt_dir / "last.pt"
    _save_checkpoint(
        last_ckpt_path,
        epoch=history[-1]["epoch"] if history else 0,
        global_step=global_step,
        model=model,
        best_val_auprc=best_auprc,
        threshold=best_threshold,
        config=config_resolved,
    )
    log(f"[save] last checkpoint saved: {last_ckpt_path.name}")

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    best_threshold = float(checkpoint.get("threshold", best_threshold))
    best_step = int(checkpoint.get("global_step", best_step))

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
    log(
        "[caid3] "
        f"auprc={caid_eval['metrics']['auprc']:.6f} "
        f"f1={caid_eval['metrics']['f1']:.6f} "
        f"mcc={caid_eval['metrics']['mcc']:.6f} "
        f"threshold={caid_eval['metrics']['threshold']:.4f}"
    )

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_scalar("caid3/auprc", float(caid_eval["metrics"]["auprc"]), global_step)
    writer.add_scalar("caid3/f1", float(caid_eval["metrics"]["f1"]), global_step)
    writer.add_scalar("caid3/mcc", float(caid_eval["metrics"]["mcc"]), global_step)
    writer.flush()
    writer.close()

    write_json(
        metrics_dir / "train_history.json",
        {
            "epoch_history": history,
            "print_history": print_history,
            "eval_history": eval_history,
        },
    )
    write_json(
        metrics_dir / "best_val_metrics.json",
        {
            "epoch": best_epoch,
            "global_step": best_step,
            "metrics": best_val_metrics,
            "threshold": best_threshold,
        },
    )
    write_json(metrics_dir / "last_eval_metrics.json", {"metrics": last_eval_metrics})
    write_json(metrics_dir / "caid3_metrics.json", caid_eval["metrics"])
    write_json(
        output_dir / "run_metadata.json",
        {
            "config_path": str(Path(config_path)),
            "run_id": run_id,
            "output_base_dir": str(output_base_dir),
            "output_dir": str(output_dir),
            "tensorboard_dir": str(tensorboard_dir),
            "log_file": str(logs_dir / "train.log"),
            "checkpoints_dir": str(ckpt_dir),
            "tensorboard_service": tensorboard_info,
            "split_manifest": str(Path(config["split_manifest"])),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "caid_records": len(caid_records),
            "class_stats_train": class_stats,
            "pos_weight": pos_weight,
            "global_steps": global_step,
            "save_every": save_every,
            "print_every": print_every,
            "eval_every": eval_every,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "excluded_error_ids": split_manifest.get("excluded_error_ids", []),
            "excluded_holdout_overlap_ids": split_manifest.get("excluded_holdout_overlap_ids", []),
            "selected_threshold": best_threshold,
        },
    )

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "output_base_dir": str(output_base_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "tensorboard_service": tensorboard_info,
        "log_file": str(logs_dir / "train.log"),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "best_epoch": best_epoch,
        "best_step": best_step,
        "global_steps": global_step,
        "best_val_auprc": best_auprc,
        "caid3_metrics": caid_eval["metrics"],
    }


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    experiment_dir = _infer_experiment_dir_from_checkpoint(checkpoint_path)
    if output_dir is None:
        eval_base = experiment_dir / "evaluations"
        output, eval_id = _resolve_incremental_dir(eval_base, prefix="eval")
    else:
        output = Path(output_dir)
        eval_id = None
    output.mkdir(parents=True, exist_ok=True)

    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log = _build_logger(logs_dir / "eval.log")

    tensorboard_info = _start_tensorboard_if_needed(
        enabled=bool(config.get("auto_start_tensorboard", True)),
        logdir=Path(config.get("output_base_dir", experiment_dir.parent)),
        host=str(config.get("tensorboard_host", "0.0.0.0")),
        port=int(config.get("tensorboard_port", 6006)),
        runtime_log_path=logs_dir / "tensorboard_runtime.log",
        logger=log,
    )

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
    write_json(
        output / "evaluation_metadata.json",
        {
            "checkpoint_path": str(Path(checkpoint_path)),
            "experiment_dir": str(experiment_dir),
            "output_dir": str(output),
            "eval_id": eval_id,
            "tensorboard_service": tensorboard_info,
            "fasta_path": str(Path(fasta_path)),
            "split_manifest_path": str(Path(split_manifest_path)) if split_manifest_path else None,
            "split_key": split_key,
            "threshold": resolved_threshold,
            "batch_size": batch_size,
        },
    )
    log(
        "[eval] "
        f"checkpoint={Path(checkpoint_path).name} output_dir={output} "
        f"auprc={result['metrics']['auprc']:.6f} "
        f"f1={result['metrics']['f1']:.6f} "
        f"mcc={result['metrics']['mcc']:.6f}"
    )

    return {
        "experiment_dir": str(experiment_dir),
        "eval_id": eval_id,
        "output_dir": str(output),
        "tensorboard_service": tensorboard_info,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics": result["metrics"],
    }
