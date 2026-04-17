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
from torch.nn import functional as F
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
    "stage": "stage1",
    "dropout": 0.1,
    "freeze_backbone": True,
    "val_ratio": 0.2,
    "error_file": "dataset/error.txt",
    "auto_rebuild_split_on_mismatch": True,
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
    "use_wandb": False,
    "wandb_entity": "3151599052-nankai-university",
    "wandb_project": "CLIPer",
    "wandb_mode": "online",
    "wandb_run_name": None,
    "wandb_group": None,
    "wandb_tags": [],
    "wandb_dir": None,
    "contrastive": {
        "enabled": False,
        "weight": 0.2,
        "temperature": 0.1,
        "proj_dim": 128,
        "max_samples_per_class": 256,
    },
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


def _resolve_contrastive_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    defaults = dict(DEFAULTS["contrastive"])
    provided = config.get("contrastive", {})
    if provided is None:
        provided = {}
    if not isinstance(provided, dict):
        raise ValueError("contrastive must be a dict when provided.")
    defaults.update(provided)
    if stage == "stage2" and "enabled" not in provided:
        defaults["enabled"] = True
    if float(defaults["weight"]) < 0:
        raise ValueError(f"contrastive.weight must be >= 0, got {defaults['weight']}")
    if float(defaults["temperature"]) <= 0:
        raise ValueError(f"contrastive.temperature must be > 0, got {defaults['temperature']}")
    if int(defaults["proj_dim"]) <= 0:
        raise ValueError(f"contrastive.proj_dim must be > 0, got {defaults['proj_dim']}")
    if int(defaults["max_samples_per_class"]) <= 0:
        raise ValueError(
            "contrastive.max_samples_per_class must be > 0, "
            f"got {defaults['max_samples_per_class']}"
        )
    return defaults


def _resolve_wandb_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        stripped = raw.strip()
        return [stripped] if stripped else []
    if isinstance(raw, (list, tuple)):
        tags: list[str] = []
        for item in raw:
            if item is None:
                continue
            value = str(item).strip()
            if value:
                tags.append(value)
        return tags
    raise ValueError("wandb_tags must be a list/tuple of strings or a single string.")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config file must be a YAML object: {source}")
    _ensure_required_fields(config)
    merged = dict(DEFAULTS)
    merged.update(config)
    stage = str(merged.get("stage", "stage1")).lower()
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"stage must be one of ['stage1', 'stage2'], got {stage!r}")
    merged["stage"] = stage
    merged["contrastive"] = _resolve_contrastive_config(config, stage=stage)
    if merged["optimizer"].lower() != "adamw":
        raise ValueError("Only optimizer=adamw is supported.")
    merged["auto_rebuild_split_on_mismatch"] = bool(merged.get("auto_rebuild_split_on_mismatch", True))
    wandb_mode = str(merged.get("wandb_mode", "online")).lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            "wandb_mode must be one of ['online', 'offline', 'disabled'], "
            f"got {wandb_mode!r}"
        )
    merged["wandb_mode"] = wandb_mode
    merged["wandb_tags"] = _resolve_wandb_tags(merged.get("wandb_tags"))
    if bool(merged.get("use_wandb", False)):
        if not str(merged.get("wandb_project", "")).strip():
            raise ValueError("use_wandb=true requires non-empty wandb_project.")
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


def _load_model_state(
    model: nn.Module,
    state_dict: dict[str, Any],
    *,
    logger=None,
) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = ("projection_head.",)
    disallowed_missing = [name for name in missing if not name.startswith(allowed_missing_prefixes)]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Checkpoint state mismatch. "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    if missing and logger is not None:
        logger(f"[checkpoint] missing optional params during load: {missing}")


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


def _start_wandb_if_needed(
    *,
    enabled: bool,
    project: str | None,
    entity: str | None,
    run_name: str | None,
    group: str | None,
    tags: list[str],
    mode: str,
    run_dir: str | Path | None,
    job_type: str,
    config_payload: dict[str, Any],
    logger,
) -> tuple[Any | None, dict[str, Any]]:
    # W&B integration is temporarily disabled by request.
    logger("[wandb] temporarily commented out (disabled).")
    return None, {
        "status": "disabled",
        "project": project,
        "entity": entity,
        "run_id": None,
        "run_name": None,
        "mode": mode,
        "url": None,
        "dir": None,
    }


def _wandb_log(run: Any | None, payload: dict[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=step)


def _finish_wandb(run: Any | None, logger) -> None:
    if run is None:
        return
    try:
        run.finish()
        logger("[wandb] run finished.")
    except Exception as exc:  # pragma: no cover - defensive cleanup
        logger(f"[wandb] failed to finish run cleanly: {exc}")


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


def _safe_parse_id_lines(path: str | Path, *, kind: str, logger) -> set[str]:
    source = Path(path)
    if not source.exists():
        logger(f"[split] {kind} file not found: {source}. Treat as empty id set during split rebuild.")
        return set()
    return parse_id_lines(source)


def _resolve_train_val_records(
    *,
    all_records: list[ProteinRecord],
    caid_records: list[ProteinRecord],
    split_manifest_path: str | Path,
    source_fasta_path: str | Path,
    error_file_path: str | Path,
    seed: int,
    val_ratio: float,
    auto_rebuild_on_mismatch: bool,
    logger,
) -> tuple[list[ProteinRecord], list[ProteinRecord], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    split_manifest_source = Path(split_manifest_path)
    available_ids = {rec.protein_id for rec in all_records}
    caid_ids = {rec.protein_id for rec in caid_records}

    split_manifest: dict[str, Any] | None = None
    missing_train: list[str] = []
    missing_val: list[str] = []

    if split_manifest_source.exists():
        split_manifest = read_json(split_manifest_source)
        train_ids = split_manifest.get("train_ids")
        val_ids = split_manifest.get("val_ids")
        if not isinstance(train_ids, list) or not isinstance(val_ids, list):
            raise ValueError(
                f"Invalid split manifest format at {split_manifest_source}. "
                "Expected list fields: train_ids and val_ids."
            )
        missing_train = sorted(set(train_ids) - available_ids)
        missing_val = sorted(set(val_ids) - available_ids)
    elif not auto_rebuild_on_mismatch:
        raise FileNotFoundError(
            f"split_manifest file not found: {split_manifest_source}. "
            "Set auto_rebuild_split_on_mismatch=true to rebuild automatically."
        )

    needs_rebuild = (split_manifest is None) or bool(missing_train or missing_val)
    if not needs_rebuild:
        train_records = select_records(all_records, split_manifest["train_ids"])
        val_records = select_records(all_records, split_manifest["val_ids"])
        return (
            train_records,
            val_records,
            split_manifest,
            {
                "status": "used",
                "source_split_manifest_path": str(split_manifest_source),
                "missing_train_ids_count": 0,
                "missing_val_ids_count": 0,
                "missing_train_ids_sample": [],
                "missing_val_ids_sample": [],
            },
            None,
        )

    if not auto_rebuild_on_mismatch:
        raise ValueError(
            "Split manifest is inconsistent with current training FASTA. "
            f"Missing train ids={len(missing_train)}, missing val ids={len(missing_val)}. "
            "Set auto_rebuild_split_on_mismatch=true to rebuild automatically."
        )

    logger(
        "[split] detected split/FASTA mismatch, rebuilding split manifest "
        f"(missing_train={len(missing_train)} missing_val={len(missing_val)})."
    )
    error_ids = _safe_parse_id_lines(error_file_path, kind="error", logger=logger)
    rebuilt_manifest, rebuilt_exclusion = build_split_manifest(
        all_records,
        source_fasta=source_fasta_path,
        error_ids=error_ids,
        caid_ids=caid_ids,
        seed=seed,
        val_ratio=val_ratio,
    )
    train_records = select_records(all_records, rebuilt_manifest["train_ids"])
    val_records = select_records(all_records, rebuilt_manifest["val_ids"])
    return (
        train_records,
        val_records,
        rebuilt_manifest,
        {
            "status": "rebuilt",
            "source_split_manifest_path": str(split_manifest_source),
            "missing_train_ids_count": len(missing_train),
            "missing_val_ids_count": len(missing_val),
            "missing_train_ids_sample": missing_train[:10],
            "missing_val_ids_sample": missing_val[:10],
        },
        rebuilt_exclusion,
    )


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


def _sample_contrastive_residues(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_samples_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = labels >= 0
    if int(valid_mask.sum().item()) == 0:
        empty_embeddings = embeddings.new_zeros((0, embeddings.shape[-1]))
        empty_labels = labels.new_zeros((0,), dtype=torch.long)
        return empty_embeddings, empty_labels

    flat_embeddings = embeddings[valid_mask]
    flat_labels = labels[valid_mask].to(dtype=torch.long)
    sampled_embeddings: list[torch.Tensor] = []
    sampled_labels: list[torch.Tensor] = []

    for class_id in (0, 1):
        class_mask = flat_labels == class_id
        class_count = int(class_mask.sum().item())
        if class_count == 0:
            continue
        class_embeddings = flat_embeddings[class_mask]
        if class_count > max_samples_per_class:
            perm = torch.randperm(class_count, device=class_embeddings.device)[:max_samples_per_class]
            class_embeddings = class_embeddings[perm]
        sampled_embeddings.append(class_embeddings)
        sampled_labels.append(
            torch.full(
                (class_embeddings.shape[0],),
                class_id,
                device=flat_labels.device,
                dtype=torch.long,
            )
        )

    if not sampled_embeddings:
        empty_embeddings = embeddings.new_zeros((0, embeddings.shape[-1]))
        empty_labels = labels.new_zeros((0,), dtype=torch.long)
        return empty_embeddings, empty_labels

    return torch.cat(sampled_embeddings, dim=0), torch.cat(sampled_labels, dim=0)


def _supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings [N, D], got shape={tuple(embeddings.shape)}")
    num_samples = embeddings.shape[0]
    if num_samples < 2:
        return embeddings.new_zeros(())

    normalized = F.normalize(embeddings, p=2, dim=1)
    logits = torch.matmul(normalized, normalized.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).to(dtype=logits.dtype)
    logits_mask = torch.ones_like(positive_mask) - torch.eye(num_samples, device=logits.device, dtype=logits.dtype)
    positive_mask = positive_mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    denominator = exp_logits.sum(dim=1, keepdim=True)
    log_prob = logits - torch.log(denominator + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if int(valid.sum().item()) == 0:
        return embeddings.new_zeros(())

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def _compute_batch_supcon_loss(
    residue_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_samples_per_class: int,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    sampled_embeddings, sampled_labels = _sample_contrastive_residues(
        residue_embeddings,
        labels,
        max_samples_per_class=max_samples_per_class,
    )
    sampled_count = int(sampled_labels.shape[0])
    class_count = int(sampled_labels.unique().shape[0]) if sampled_count > 0 else 0
    if sampled_count < 2 or class_count < 2:
        return residue_embeddings.new_zeros(()), sampled_count
    loss = _supervised_contrastive_loss(
        sampled_embeddings,
        sampled_labels,
        temperature=temperature,
    )
    if not torch.isfinite(loss):
        raise ValueError("SupCon loss became non-finite; please check training hyperparameters.")
    return loss, sampled_count


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
    stage = str(config.get("stage", "stage1"))
    contrastive_cfg = dict(config.get("contrastive", {}))
    contrastive_enabled = bool(contrastive_cfg.get("enabled", False))
    contrastive_weight = float(contrastive_cfg.get("weight", 0.0)) if contrastive_enabled else 0.0
    contrastive_temperature = float(contrastive_cfg.get("temperature", 0.1))
    contrastive_max_samples = int(contrastive_cfg.get("max_samples_per_class", 256))

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
    log(
        f"Run stage={stage} contrastive_enabled={contrastive_enabled} "
        f"contrastive_weight={contrastive_weight:.4f}"
    )
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
    writer.add_text("run/stage", stage, 0)
    writer.add_text("run/config_path", str(Path(config_path)), 0)
    writer.add_text("run/config_resolved", json.dumps(config_resolved, ensure_ascii=False, indent=2), 0)
    writer.add_text("tensorboard/service", json.dumps(tensorboard_info, ensure_ascii=False, indent=2), 0)
    wandb_run, wandb_info = _start_wandb_if_needed(
        enabled=bool(config.get("use_wandb", False)),
        project=_optional_str(config.get("wandb_project")),
        entity=_optional_str(config.get("wandb_entity")),
        run_name=_optional_str(config.get("wandb_run_name")) or run_id,
        group=_optional_str(config.get("wandb_group")),
        tags=list(config.get("wandb_tags", [])),
        mode=str(config.get("wandb_mode", "online")),
        run_dir=config.get("wandb_dir") or (output_dir / "wandb"),
        job_type="train",
        config_payload=config_resolved,
        logger=log,
    )
    writer.add_text("wandb/service", json.dumps(wandb_info, ensure_ascii=False, indent=2), 0)

    all_records = parse_three_line_fasta(config["train_fasta"])
    caid_records = parse_three_line_fasta(config["caid_fasta"])
    train_records, val_records, split_manifest, split_manifest_resolution, rebuilt_exclusion_report = _resolve_train_val_records(
        all_records=all_records,
        caid_records=caid_records,
        split_manifest_path=config["split_manifest"],
        source_fasta_path=config["train_fasta"],
        error_file_path=config["error_file"],
        seed=int(config["seed"]),
        val_ratio=float(config["val_ratio"]),
        auto_rebuild_on_mismatch=bool(config.get("auto_rebuild_split_on_mismatch", True)),
        logger=log,
    )
    active_split_manifest_path = Path(config["split_manifest"])
    rebuilt_exclusion_report_path: Path | None = None
    if split_manifest_resolution["status"] == "rebuilt":
        split_dir = output_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        active_split_manifest_path = split_dir / "split_manifest_rebuilt.json"
        write_json(active_split_manifest_path, split_manifest)
        if rebuilt_exclusion_report is not None:
            rebuilt_exclusion_report_path = split_dir / "exclusion_report_rebuilt.json"
            write_json(rebuilt_exclusion_report_path, rebuilt_exclusion_report)
        log(
            "[split] rebuilt split manifest saved to "
            f"{active_split_manifest_path} (train={len(train_records)} val={len(val_records)})"
        )

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
        projection_dim=int(contrastive_cfg["proj_dim"]),
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
    writer.add_scalar("train/contrastive_weight", contrastive_weight, 0)
    _wandb_log(
        wandb_run,
        {
            "data/class_positive": class_stats["positive"],
            "data/class_negative": class_stats["negative"],
            "train/pos_weight": pos_weight,
            "train/contrastive_weight": contrastive_weight,
            "run/stage": stage,
        },
        step=0,
    )

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
        _wandb_log(
            wandb_run,
            {
                "val/auprc": float(val_metrics["auprc"]),
                "val/f1": float(val_metrics["f1"]),
                "val/mcc": float(val_metrics["mcc"]),
                "val/threshold": float(val_threshold),
            },
            step=step,
        )
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
            running_total_loss = 0.0
            running_bce_loss = 0.0
            running_supcon_loss = 0.0
            steps = 0
            window_total_loss = 0.0
            window_bce_loss = 0.0
            window_supcon_loss = 0.0
            window_steps = 0
            eval_happened_in_epoch = False

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                residue_lengths = batch["residue_lengths"]

                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device_type=device.type, enabled=use_amp):
                    if contrastive_enabled:
                        logits, residue_embeddings = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            residue_lengths=residue_lengths,
                            return_embeddings=True,
                        )
                    else:
                        logits = model(input_ids=input_ids, attention_mask=attention_mask, residue_lengths=residue_lengths)
                        residue_embeddings = None
                    mask = labels >= 0
                    valid_count = int(mask.sum().item())
                    if valid_count == 0:
                        continue
                    loss_matrix = criterion(logits, labels)
                    bce_loss = (loss_matrix * mask).sum() / valid_count

                    contrastive_num_samples = 0
                    if contrastive_enabled and residue_embeddings is not None:
                        supcon_loss, contrastive_num_samples = _compute_batch_supcon_loss(
                            residue_embeddings,
                            labels,
                            max_samples_per_class=contrastive_max_samples,
                            temperature=contrastive_temperature,
                        )
                    else:
                        supcon_loss = bce_loss.new_zeros(())
                    total_loss = bce_loss + contrastive_weight * supcon_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss_value = float(total_loss.detach().cpu().item())
                bce_loss_value = float(bce_loss.detach().cpu().item())
                supcon_loss_value = float(supcon_loss.detach().cpu().item())
                global_step += 1
                running_total_loss += total_loss_value
                running_bce_loss += bce_loss_value
                running_supcon_loss += supcon_loss_value
                steps += 1
                window_total_loss += total_loss_value
                window_bce_loss += bce_loss_value
                window_supcon_loss += supcon_loss_value
                window_steps += 1

                writer.add_scalar("train/loss_step", total_loss_value, global_step)
                writer.add_scalar("train/loss_total", total_loss_value, global_step)
                writer.add_scalar("train/loss_bce", bce_loss_value, global_step)
                writer.add_scalar("train/loss_supcon", supcon_loss_value, global_step)
                writer.add_scalar("train/contrastive_num_samples", contrastive_num_samples, global_step)
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
                _wandb_log(
                    wandb_run,
                    {
                        "train/loss_step": total_loss_value,
                        "train/loss_total": total_loss_value,
                        "train/loss_bce": bce_loss_value,
                        "train/loss_supcon": supcon_loss_value,
                        "train/contrastive_num_samples": contrastive_num_samples,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_step,
                )

                if global_step % print_every == 0:
                    avg_window_total = window_total_loss / max(1, window_steps)
                    avg_window_bce = window_bce_loss / max(1, window_steps)
                    avg_window_supcon = window_supcon_loss / max(1, window_steps)
                    print_event = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "avg_loss": avg_window_total,
                        "avg_bce_loss": avg_window_bce,
                        "avg_supcon_loss": avg_window_supcon,
                    }
                    print_history.append(print_event)
                    log(
                        f"[train] epoch={epoch} step={global_step} "
                        f"avg_total={avg_window_total:.6f} avg_bce={avg_window_bce:.6f} "
                        f"avg_supcon={avg_window_supcon:.6f}"
                    )
                    writer.add_scalar("train/loss_print_avg", avg_window_total, global_step)
                    writer.add_scalar("train/loss_bce_print_avg", avg_window_bce, global_step)
                    writer.add_scalar("train/loss_supcon_print_avg", avg_window_supcon, global_step)
                    window_total_loss = 0.0
                    window_bce_loss = 0.0
                    window_supcon_loss = 0.0
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
                avg_window_total = window_total_loss / max(1, window_steps)
                avg_window_bce = window_bce_loss / max(1, window_steps)
                avg_window_supcon = window_supcon_loss / max(1, window_steps)
                print_event = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "avg_loss": avg_window_total,
                    "avg_bce_loss": avg_window_bce,
                    "avg_supcon_loss": avg_window_supcon,
                }
                print_history.append(print_event)
                log(
                    f"[train] epoch={epoch} step={global_step} avg_total={avg_window_total:.6f} "
                    f"avg_bce={avg_window_bce:.6f} avg_supcon={avg_window_supcon:.6f} (epoch_end_flush)"
                )
                writer.add_scalar("train/loss_print_avg", avg_window_total, global_step)
                writer.add_scalar("train/loss_bce_print_avg", avg_window_bce, global_step)
                writer.add_scalar("train/loss_supcon_print_avg", avg_window_supcon, global_step)

            train_loss = running_total_loss / max(1, steps)
            train_bce_loss = running_bce_loss / max(1, steps)
            train_supcon_loss = running_supcon_loss / max(1, steps)
            epoch_seconds = round(time.time() - start_time, 3)
            history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "train_bce_loss": train_bce_loss,
                    "train_supcon_loss": train_supcon_loss,
                    "epoch_seconds": epoch_seconds,
                    "train_steps": steps,
                }
            )
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("train/loss_bce_epoch", train_bce_loss, epoch)
            writer.add_scalar("train/loss_supcon_epoch", train_supcon_loss, epoch)
            writer.add_scalar("train/epoch_seconds", epoch_seconds, epoch)
            _wandb_log(
                wandb_run,
                {
                    "train/loss_epoch": train_loss,
                    "train/loss_bce_epoch": train_bce_loss,
                    "train/loss_supcon_epoch": train_supcon_loss,
                    "train/epoch_seconds": epoch_seconds,
                    "train/epoch": epoch,
                },
                step=global_step,
            )
            log(
                f"[epoch] epoch={epoch} steps={steps} global_step={global_step} "
                f"train_total={train_loss:.6f} train_bce={train_bce_loss:.6f} "
                f"train_supcon={train_supcon_loss:.6f} epoch_seconds={epoch_seconds:.3f}"
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
    _load_model_state(model, checkpoint["model_state"], logger=log)
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
    _wandb_log(
        wandb_run,
        {
            "caid3/auprc": float(caid_eval["metrics"]["auprc"]),
            "caid3/f1": float(caid_eval["metrics"]["f1"]),
            "caid3/mcc": float(caid_eval["metrics"]["mcc"]),
            "caid3/threshold": float(caid_eval["metrics"]["threshold"]),
        },
        step=global_step,
    )

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
            "wandb_service": wandb_info,
            "split_manifest": str(active_split_manifest_path),
            "split_manifest_resolution": split_manifest_resolution,
            "rebuilt_exclusion_report": str(rebuilt_exclusion_report_path) if rebuilt_exclusion_report_path else None,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "caid_records": len(caid_records),
            "class_stats_train": class_stats,
            "pos_weight": pos_weight,
            "stage": stage,
            "contrastive": contrastive_cfg,
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
    _finish_wandb(wandb_run, logger=log)

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "output_base_dir": str(output_base_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "tensorboard_service": tensorboard_info,
        "wandb_service": wandb_info,
        "split_manifest": str(active_split_manifest_path),
        "split_manifest_resolution": split_manifest_resolution,
        "log_file": str(logs_dir / "train.log"),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "best_epoch": best_epoch,
        "best_step": best_step,
        "global_steps": global_step,
        "best_val_auprc": best_auprc,
        "stage": stage,
        "contrastive": contrastive_cfg,
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
    eval_id_for_name = eval_id if eval_id is not None else output.name
    wandb_eval_config = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "fasta_path": str(Path(fasta_path)),
        "split_manifest_path": str(Path(split_manifest_path)) if split_manifest_path else None,
        "split_key": split_key,
        "batch_size": batch_size,
        "threshold_override": threshold,
        "experiment_dir": str(experiment_dir),
        "eval_output_dir": str(output),
        "train_config": config,
    }
    wandb_run, wandb_info = _start_wandb_if_needed(
        enabled=bool(config.get("use_wandb", False)),
        project=_optional_str(config.get("wandb_project")),
        entity=_optional_str(config.get("wandb_entity")),
        run_name=f"{config.get('run_id', experiment_dir.name)}-{eval_id_for_name}-eval",
        group=_optional_str(config.get("wandb_group")),
        tags=list(config.get("wandb_tags", [])) + ["eval"],
        mode=str(config.get("wandb_mode", "online")),
        run_dir=config.get("wandb_dir") or (experiment_dir / "wandb"),
        job_type="eval",
        config_payload=wandb_eval_config,
        logger=log,
    )

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
        projection_dim=int(config.get("contrastive", {}).get("proj_dim", 128)),
    )
    _load_model_state(model, checkpoint["model_state"], logger=log)
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
    _wandb_log(
        wandb_run,
        {
            "eval/auprc": float(result["metrics"]["auprc"]),
            "eval/f1": float(result["metrics"]["f1"]),
            "eval/mcc": float(result["metrics"]["mcc"]),
            "eval/threshold": float(result["metrics"]["threshold"]),
            "eval/num_records": int(result["metrics"]["num_records"]),
            "eval/num_scored_residues": int(result["metrics"]["num_scored_residues"]),
        },
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
            "wandb_service": wandb_info,
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
    _finish_wandb(wandb_run, logger=log)

    return {
        "experiment_dir": str(experiment_dir),
        "eval_id": eval_id,
        "output_dir": str(output),
        "tensorboard_service": tensorboard_info,
        "wandb_service": wandb_info,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics": result["metrics"],
    }
