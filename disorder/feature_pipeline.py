"""Train and evaluate disorder classifier on precomputed per-residue feature files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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
    _compute_pos_weight,
    _labels_to_tensor,
    _load_model_state,
    _save_checkpoint,
    set_seed,
)
from cliper.windowing import merge_window_logits, sigmoid
from disorder.feature_io import feature_file_path, manifest_hidden_size, read_residue_feature_file
from disorder.feature_modeling import DisorderFeatureClassifier
from disorder.windowing import build_sliding_eval_starts, pick_training_window


DEFAULT_FEATURE_CONFIG: dict[str, Any] = {
    "window_overlap": 256,
    "eval_window_overlap": None,
    "train_split_penalty_weight": 3.0,
    "dropout": 0.1,
    "seed": 42,
    "max_epochs": 30,
    "early_stop_patience": 5,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "batch_size": 4,
    "eval_every": 200,
    "print_every": 20,
    "save_every": 500,
    "num_workers": 0,
    "device": "cuda",
    "mixed_precision": True,
    "threshold_search": {"min": 0.05, "max": 0.95, "step": 0.05},
    "experiment_prefix": "exp",
    "tensorboard_dir": None,
    "resume_checkpoint": None,
    "hidden_size": None,
    "local_context": {
        "enabled": False,
        "radius": 2,
        "mode": "concat_mean",
        "include_self": False,
    },
}

REQUIRED_FEATURE_CONFIG = [
    "features_dir",
    "train_sequence_fasta",
    "train_label_fasta",
    "val_sequence_fasta",
    "val_label_fasta",
    "window_size",
    "classifier_head",
    "output_dir",
]


def _autocast_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def _build_grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


def load_feature_train_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {source}")
    merged = dict(DEFAULT_FEATURE_CONFIG)
    merged.update(raw)
    missing = [k for k in REQUIRED_FEATURE_CONFIG if k not in merged or merged[k] is None]
    if missing:
        raise ValueError(f"Missing required feature-train config keys: {missing}")
    if merged.get("eval_window_overlap") is None:
        merged["eval_window_overlap"] = merged["window_overlap"]
    ts = merged.get("threshold_search")
    if not isinstance(ts, dict) or not all(k in ts for k in ("min", "max", "step")):
        raise ValueError("threshold_search must be a dict with min, max, step.")
    merged["local_context"] = _resolve_local_context(merged)
    return merged


def _resolve_hidden_size(cfg: dict[str, Any]) -> int:
    if cfg.get("hidden_size") is not None:
        base_hidden = int(cfg["hidden_size"])
    else:
        base_hidden = manifest_hidden_size(cfg["features_dir"])
    local_context = cfg.get("local_context", {})
    if isinstance(local_context, dict) and bool(local_context.get("enabled", False)):
        mode = str(local_context.get("mode", "concat_mean")).lower()
        if mode == "concat_mean":
            return base_hidden * 2
        if mode == "concat_window":
            radius = int(local_context.get("radius", 2))
            include_self = bool(local_context.get("include_self", True))
            width = (2 * radius + 1) if include_self else (2 * radius)
            if width <= 0:
                raise ValueError("local_context concat_window produced width <= 0.")
            return base_hidden * width
        raise ValueError(f"Unsupported local_context.mode: {mode!r}")
    return base_hidden


def _resolve_local_context(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("local_context")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("local_context must be a dict when provided.")
    merged = dict(DEFAULT_FEATURE_CONFIG["local_context"])
    merged.update(raw)
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["radius"] = int(merged.get("radius", 2))
    merged["mode"] = str(merged.get("mode", "concat_mean")).lower()
    merged["include_self"] = bool(merged.get("include_self", True))
    if merged["radius"] < 0:
        raise ValueError(f"local_context.radius must be >= 0, got {merged['radius']}")
    if merged["mode"] not in {"concat_mean", "concat_window"}:
        raise ValueError("local_context.mode must be one of ['concat_mean', 'concat_window']")
    if merged["enabled"] and merged["mode"] == "concat_window":
        width = (2 * merged["radius"] + 1) if merged["include_self"] else (2 * merged["radius"])
        if width <= 0:
            raise ValueError("local_context concat_window requires non-empty neighborhood.")
    return merged


def _augment_with_local_context(feats: torch.Tensor, local_context: dict[str, Any]) -> torch.Tensor:
    if feats.dim() != 2:
        raise ValueError(f"Expected [L, D] feature tensor, got shape {tuple(feats.shape)}")
    if not bool(local_context.get("enabled", False)):
        return feats
    radius = int(local_context.get("radius", 2))
    if radius <= 0:
        return feats
    mode = str(local_context.get("mode", "concat_mean")).lower()
    include_self = bool(local_context.get("include_self", False))
    length = int(feats.shape[0])
    if mode == "concat_mean":
        ctx = torch.zeros_like(feats)
        for idx in range(length):
            start = max(0, idx - radius)
            end = min(length, idx + radius + 1)
            window = feats[start:end]
            if not include_self and window.shape[0] > 1:
                center = idx - start
                window = torch.cat([window[:center], window[center + 1 :]], dim=0)
            ctx[idx] = window.mean(dim=0)
        return torch.cat([feats, ctx], dim=1)
    if mode == "concat_window":
        chunks: list[torch.Tensor] = []
        for idx in range(length):
            parts: list[torch.Tensor] = []
            for offset in range(-radius, radius + 1):
                if offset == 0 and not include_self:
                    continue
                neighbor = idx + offset
                if 0 <= neighbor < length:
                    parts.append(feats[neighbor])
                else:
                    parts.append(torch.zeros_like(feats[idx]))
            if not parts:
                raise ValueError("local_context concat_window produced no parts.")
            chunks.append(torch.cat(parts, dim=0))
        return torch.stack(chunks, dim=0)
    raise ValueError(f"Unsupported local_context.mode: {mode!r}")


def _ensure_feature_files(records: list[ProteinRecord], features_dir: str | Path) -> None:
    root = Path(features_dir)
    missing = [r.protein_id for r in records if not feature_file_path(root, r.protein_id).exists()]
    if missing:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} feature file(s) under {root} (e.g. {sample}). Run extract_features first."
        )


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
    return y_true, y_prob


@torch.inference_mode()
def _forward_feature_windows(
    model: DisorderFeatureClassifier,
    windows: list[torch.Tensor],
    *,
    device: torch.device,
    mixed_precision: bool,
) -> list[list[float]]:
    lengths = [w.shape[0] for w in windows]
    max_len = max(lengths)
    hidden = windows[0].shape[1]
    batch_t = torch.zeros(len(windows), max_len, hidden, device=device)
    for i, w in enumerate(windows):
        batch_t[i, : w.shape[0]] = w.to(device)
    use_amp = mixed_precision and device.type == "cuda"
    with _autocast_context(device_type=device.type, enabled=use_amp):
        logits = model(batch_t, lengths)
    logits = logits.float().cpu()
    out: list[list[float]] = []
    for row, ln in enumerate(lengths):
        out.append(logits[row, :ln].tolist())
    return out


def evaluate_feature_records(
    model: DisorderFeatureClassifier,
    records: list[ProteinRecord],
    *,
    features_dir: str | Path,
    window_size: int,
    eval_window_overlap: int,
    device: torch.device,
    window_batch_size: int,
    threshold: float | None,
    threshold_search: dict[str, float],
    mixed_precision: bool,
    local_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    root = Path(features_dir)
    step_stride = max(1, int(window_size) - int(eval_window_overlap))
    per_record: list[tuple[str, str, list[float]]] = []
    y_true_all: list[int] = []
    y_prob_all: list[float] = []

    for record in records:
        path = feature_file_path(root, record.protein_id)
        feats = read_residue_feature_file(path)
        if local_context is not None:
            feats = _augment_with_local_context(feats, local_context)
        length = int(feats.shape[0])
        if length != len(record.sequence):
            raise ValueError(
                f"Feature length mismatch for {record.protein_id}: "
                f"embeddings L={length} sequence L={len(record.sequence)}"
            )
        starts = build_sliding_eval_starts(length, window_size, step_stride)
        window_logits: list[list[float]] = []
        for wb_start in range(0, len(starts), max(1, window_batch_size)):
            chunk_starts = starts[wb_start : wb_start + window_batch_size]
            windows = [feats[s : s + window_size] for s in chunk_starts]
            window_logits.extend(
                _forward_feature_windows(model, windows, device=device, mixed_precision=mixed_precision)
            )
        merged_logits = merge_window_logits(
            length=length,
            window_logits=list(zip(starts, window_logits)),
        )
        probs = sigmoid(merged_logits)
        per_record.append((record.protein_id, record.labels, probs))
        yt, yp = _extract_scored_residues(record.labels, probs)
        y_true_all.extend(yt)
        y_prob_all.extend(yp)

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


@dataclass
class FeatureTrainItem:
    protein_id: str
    embedding: torch.Tensor
    labels: str
    start: int


class DisorderFeatureTrainDataset(Dataset):
    def __init__(
        self,
        records: list[ProteinRecord],
        features_dir: str | Path,
        window_size: int,
        overlap: int,
        seed: int,
        split_penalty_weight: float,
        local_context: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.features_dir = Path(features_dir)
        self.window_size = window_size
        self.overlap = overlap
        self.seed = seed
        self.split_penalty_weight = split_penalty_weight
        self.local_context = local_context or dict(DEFAULT_FEATURE_CONFIG["local_context"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FeatureTrainItem:
        record = self.records[index]
        path = feature_file_path(self.features_dir, record.protein_id)
        full = read_residue_feature_file(path)
        full = _augment_with_local_context(full, self.local_context)
        if full.shape[0] != len(record.sequence):
            raise ValueError(
                f"Feature/sequence length mismatch for {record.protein_id}: "
                f"{full.shape[0]} vs {len(record.sequence)}"
            )
        _, labels, start = pick_training_window(
            record.sequence,
            record.labels,
            self.window_size,
            self.overlap,
            self.seed + index,
            split_penalty_weight=self.split_penalty_weight,
        )
        crop_len = len(labels)
        crop = full[start : start + crop_len].clone()
        return FeatureTrainItem(record.protein_id, crop, labels, start)


def _collate_feature_batch(batch: list[FeatureTrainItem]) -> dict[str, Any]:
    lengths = [item.embedding.shape[0] for item in batch]
    hidden = batch[0].embedding.shape[1]
    max_len = max(lengths)
    emb = torch.zeros(len(batch), max_len, hidden, dtype=torch.float32)
    label_strings: list[str] = []
    for row, item in enumerate(batch):
        ln = item.embedding.shape[0]
        emb[row, :ln] = item.embedding
        label_strings.append(item.labels)
    labels = _labels_to_tensor(label_strings, lengths)
    return {"embeddings": emb, "residue_lengths": lengths, "labels": labels}


def train_features(config_path: str | Path, resume_checkpoint: str | Path | None = None) -> dict[str, Any]:
    from disorder.data import load_disorder_labeled_pair

    cfg = load_feature_train_config(config_path)
    set_seed(int(cfg["seed"]))
    hidden_size = _resolve_hidden_size(cfg)
    cfg["hidden_size"] = hidden_size

    train_records = load_disorder_labeled_pair(cfg["train_sequence_fasta"], cfg["train_label_fasta"])
    val_records = load_disorder_labeled_pair(cfg["val_sequence_fasta"], cfg["val_label_fasta"])
    _ensure_feature_files(train_records, cfg["features_dir"])
    _ensure_feature_files(val_records, cfg["features_dir"])

    pos_weight, class_stats = _compute_pos_weight(train_records)
    want_dev = str(cfg["device"])
    device = torch.device("cuda" if want_dev == "cuda" and torch.cuda.is_available() else "cpu")
    model = DisorderFeatureClassifier(
        hidden_size=hidden_size,
        dropout=float(cfg["dropout"]),
        classifier_head=dict(cfg["classifier_head"]),
    ).to(device)

    window_size = int(cfg["window_size"])
    batch_size = max(1, int(cfg["batch_size"]))
    train_ds = DisorderFeatureTrainDataset(
        train_records,
        cfg["features_dir"],
        window_size=window_size,
        overlap=int(cfg["window_overlap"]),
        seed=int(cfg["seed"]),
        split_penalty_weight=float(cfg["train_split_penalty_weight"]),
        local_context=dict(cfg["local_context"]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        collate_fn=_collate_feature_batch,
    )

    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    use_amp = bool(cfg["mixed_precision"]) and device.type == "cuda"
    scaler = _build_grad_scaler(enabled=use_amp)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
        reduction="none",
    )

    out_root = Path(cfg["output_dir"])
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = cfg.get("tensorboard_dir") or str(out_root / "tensorboard")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    global_step = 0
    start_epoch = 0
    best_auroc = -1.0
    best_threshold = 0.5
    evals_without_improve = 0
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    else:
        resume_path = _optional_resume(cfg.get("resume_checkpoint"))

    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu")
        _load_model_state(model, payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scaler_state" in payload:
            try:
                scaler.load_state_dict(payload["scaler_state"])
            except Exception:
                pass
        ts = payload.get("train_state") or {}
        start_epoch = int(ts.get("epoch", payload.get("epoch", 0)))
        global_step = int(ts.get("global_step", payload.get("global_step", 0)))
        best_auroc = float(ts.get("best_val_auroc", payload.get("best_val_auroc", -1.0)))
        best_threshold = float(ts.get("best_threshold", payload.get("threshold", 0.5)))
        evals_without_improve = int(ts.get("evals_without_improve", 0))

    eval_every = int(cfg["eval_every"])
    print_every = int(cfg["print_every"])
    save_every = int(cfg["save_every"])
    early_stop_patience = int(cfg["early_stop_patience"])
    max_epochs = int(cfg["max_epochs"])
    window_batch_size = int(cfg.get("eval_window_batch_size", 8))

    def log(msg: str) -> None:
        print(msg, flush=True)

    def run_eval(reason: str) -> tuple[float, float]:
        nonlocal best_auroc, best_threshold, evals_without_improve
        model.eval()
        val_result = evaluate_feature_records(
            model,
            val_records,
            features_dir=cfg["features_dir"],
            window_size=window_size,
            eval_window_overlap=int(cfg["eval_window_overlap"]),
            device=device,
            window_batch_size=window_batch_size,
            threshold=None,
            threshold_search=dict(cfg["threshold_search"]),
            mixed_precision=use_amp,
            local_context=dict(cfg["local_context"]),
        )
        vm = val_result["metrics"]
        th = float(val_result["threshold"])
        writer.add_scalar("val/auprc", float(vm["auprc"]), global_step)
        if vm.get("auroc") is not None:
            writer.add_scalar("val/auroc", float(vm["auroc"]), global_step)
        log(
            f"[eval:{reason}] auroc={vm.get('auroc')} auprc={vm['auprc']:.6f} "
            f"f1={vm['f1']:.6f} mcc={vm['mcc']:.6f} threshold={th:.4f}"
        )
        auroc_val = float(vm["auroc"]) if vm.get("auroc") is not None else -1.0
        if auroc_val > best_auroc:
            best_auroc = auroc_val
            best_threshold = th
            evals_without_improve = 0
            best_path = ckpt_dir / "best.pt"
            _save_checkpoint(
                best_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                best_val_auroc=best_auroc,
                threshold=best_threshold,
                config=cfg,
                optimizer=optimizer,
                scaler=scaler,
                train_state={
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_auroc": best_auroc,
                    "best_threshold": best_threshold,
                    "evals_without_improve": evals_without_improve,
                },
            )
            log(f"[best] saved {best_path}")
        else:
            evals_without_improve += 1
        return auroc_val, th

    epoch = start_epoch
    stop = False
    for epoch in range(start_epoch + 1, max_epochs + 1):
        if stop:
            break
        model.train()
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            emb = batch["embeddings"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["residue_lengths"]
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device_type=device.type, enabled=use_amp):
                logits = model(emb, lengths)
                mask = labels >= 0
                if int(mask.sum().item()) == 0:
                    continue
                loss_mat = criterion(logits, labels)
                loss = (loss_mat * mask).sum() / mask.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            lv = float(loss.detach().cpu().item())
            running_loss += lv
            steps += 1
            writer.add_scalar("train/loss_step", lv, global_step)
            if global_step % print_every == 0:
                log(f"[train] epoch={epoch} step={global_step} loss={running_loss / max(1, steps):.6f}")
                running_loss = 0.0
                steps = 0
            if global_step % save_every == 0:
                p = ckpt_dir / f"step_{global_step:07d}.pt"
                _save_checkpoint(
                    p,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    best_val_auroc=best_auroc,
                    threshold=best_threshold,
                    config=cfg,
                    optimizer=optimizer,
                    scaler=scaler,
                    train_state={
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_auroc": best_auroc,
                        "best_threshold": best_threshold,
                        "evals_without_improve": evals_without_improve,
                    },
                )
                log(f"[save] {p.name}")
            if global_step % eval_every == 0:
                run_eval("interval")
                model.train()
                if evals_without_improve >= early_stop_patience:
                    log(f"[early-stop] patience={early_stop_patience}")
                    stop = True
                    break
        if not stop and steps > 0:
            log(f"[train] epoch={epoch} end avg_loss={running_loss / max(1, steps):.6f}")

    if global_step % eval_every != 0 and not stop:
        run_eval("final")

    last_path = ckpt_dir / "last.pt"
    _save_checkpoint(
        last_path,
        epoch=epoch,
        global_step=global_step,
        model=model,
        best_val_auroc=best_auroc,
        threshold=best_threshold,
        config=cfg,
        optimizer=optimizer,
        scaler=scaler,
        train_state={
            "epoch": epoch,
            "global_step": global_step,
            "best_val_auroc": best_auroc,
            "best_threshold": best_threshold,
            "evals_without_improve": evals_without_improve,
        },
    )
    writer.close()
    best_ckpt = ckpt_dir / "best.pt"
    return {
        "output_dir": str(out_root.resolve()),
        "best_checkpoint": str(best_ckpt.resolve()) if best_ckpt.exists() else None,
        "last_checkpoint": str(last_path.resolve()),
        "class_stats": class_stats,
        "best_val_auroc": best_auroc,
        "best_threshold": best_threshold,
        "global_step": global_step,
    }


def _optional_resume(val: Any) -> Path | None:
    if val is None or val == "":
        return None
    p = Path(str(val))
    return p if p.exists() else None


def eval_features_checkpoint(
    *,
    checkpoint_path: str | Path,
    sequence_fasta: str | Path,
    label_fasta: str | Path,
    features_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    window_batch_size: int = 8,
) -> dict[str, Any]:
    from disorder.data import load_disorder_labeled_pair

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = dict(ckpt["config"])
    feats_root = Path(features_dir or cfg["features_dir"])
    records = load_disorder_labeled_pair(sequence_fasta, label_fasta)
    _ensure_feature_files(records, feats_root)
    hidden_size = int(cfg.get("hidden_size") or manifest_hidden_size(feats_root))
    want = str(cfg.get("device", "cuda"))
    device = torch.device("cuda" if want == "cuda" and torch.cuda.is_available() else "cpu")
    model = DisorderFeatureClassifier(
        hidden_size=hidden_size,
        dropout=float(cfg.get("dropout", 0.1)),
        classifier_head=dict(cfg["classifier_head"]),
    )
    _load_model_state(model, ckpt["model_state"])
    model.to(device)
    use_amp = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
    result = evaluate_feature_records(
        model,
        records,
        features_dir=feats_root,
        window_size=int(cfg["window_size"]),
        eval_window_overlap=int(cfg.get("eval_window_overlap", cfg.get("window_overlap", 256))),
        device=device,
        window_batch_size=window_batch_size,
        threshold=threshold,
        threshold_search=dict(cfg.get("threshold_search", DEFAULT_FEATURE_CONFIG["threshold_search"])),
        mixed_precision=use_amp,
        local_context=dict(cfg.get("local_context", DEFAULT_FEATURE_CONFIG["local_context"])),
    )
    out: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "metrics": result["metrics"],
        "threshold": result["threshold"],
    }
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # optional: write per-residue predictions like cliper evaluate
        out_path = Path(output_dir) / "predictions.tsv"
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write("protein_id\tposition\tprobability\tprediction\n")
            th = float(result["threshold"])
            for protein_id, labels, probs in result["per_record"]:
                for index, probability in enumerate(probs, start=1):
                    pred = 1 if probability >= th else 0
                    handle.write(f"{protein_id}\t{index}\t{probability:.8f}\t{pred}\n")
        out["predictions_path"] = str(out_path)
    return out
