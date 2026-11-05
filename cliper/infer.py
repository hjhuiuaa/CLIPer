"""Score sequences from a saved classifier checkpoint and ProstT5 embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from cliper.model import EmbeddingClassifier
from cliper.windowing import build_eval_window_starts, linker_probs_from_logits, merge_window_logits


def _resolve_local_context(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    radius = int(cfg.get("radius", 2))
    mode = str(cfg.get("mode", "concat_window")).lower()
    include_self = bool(cfg.get("include_self", True))
    if radius < 0:
        raise ValueError(f"local_context.radius must be >= 0, got {radius}")
    if mode != "concat_window":
        raise ValueError("Inference only supports local_context.mode='concat_window'.")
    return {
        "enabled": enabled,
        "radius": radius,
        "mode": mode,
        "include_self": include_self,
    }


def _local_context_width(local_context: dict[str, Any]) -> int:
    if not bool(local_context.get("enabled", False)):
        return 1
    radius = int(local_context.get("radius", 2))
    include_self = bool(local_context.get("include_self", True))
    width = (2 * radius + 1) if include_self else (2 * radius)
    if width <= 0:
        raise ValueError("local_context concat_window produced width <= 0.")
    return width


def _augment_with_local_context(feats: torch.Tensor, local_context: dict[str, Any]) -> torch.Tensor:
    if feats.dim() != 2:
        raise ValueError(f"Expected [L, D] feature tensor, got shape {tuple(feats.shape)}")
    if not bool(local_context.get("enabled", False)):
        return feats

    radius = int(local_context.get("radius", 2))
    include_self = bool(local_context.get("include_self", True))
    length = int(feats.shape[0])
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


def infer_classifier_input_dim(model_state: dict[str, torch.Tensor]) -> int:
    preferred_keys = (
        "classifier.conv.0.weight",
        "classifier.input_proj.weight",
        "classifier.network.0.weight",
        "classifier.classifier.weight",
        "classifier.weight",
    )
    for key in preferred_keys:
        tensor = model_state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.dim() in {2, 3}:
            return int(tensor.shape[1])
    for key, tensor in model_state.items():
        if not key.startswith("classifier."):
            continue
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 3 and key.endswith(".weight"):
            return int(tensor.shape[1])
    raise ValueError("Cannot infer classifier input dimension from checkpoint.")


def load_classifier_from_checkpoint(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[EmbeddingClassifier, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in payload:
        raise ValueError(f"Checkpoint missing model_state: {checkpoint_path}")

    config = dict(payload.get("config", {}))
    model_state = payload["model_state"]
    classifier_head = dict(config.get("classifier_head", {}))
    dropout = float(config.get("dropout", 0.1))
    local_context = _resolve_local_context(config.get("local_context"))
    input_dim = infer_classifier_input_dim(model_state)

    model = EmbeddingClassifier(
        hidden_size=input_dim,
        dropout=dropout,
        classifier_head=classifier_head,
    )
    head_state = {key: value for key, value in model_state.items() if key.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(head_state, strict=False)
    bad_missing = [name for name in missing if not name.startswith("dropout.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Classifier checkpoint mismatch: missing={bad_missing}, unexpected={unexpected}"
        )

    model.to(device)
    model.eval()
    model._classifier_input_dim = input_dim  # noqa: SLF001 — runtime metadata for shape checks
    metadata = {
        "config": config,
        "threshold": float(payload.get("threshold", 0.5)),
        "local_context": local_context,
        "classifier_input_dim": input_dim,
        "base_hidden_size": int(input_dim // _local_context_width(local_context)),
    }
    return model, metadata


@torch.inference_mode()
def _forward_feature_windows(
    model: nn.Module,
    windows: list[torch.Tensor],
    *,
    device: torch.device,
) -> list[list[float]]:
    lengths = [int(window.shape[0]) for window in windows]
    max_len = max(lengths)
    hidden = int(windows[0].shape[1])
    batch = torch.zeros(len(windows), max_len, hidden, device=device)
    for row, window in enumerate(windows):
        batch[row, : window.shape[0]] = window.to(device)
    logits = model(batch, lengths).float().cpu()
    return [logits[row, : length].tolist() for row, length in enumerate(lengths)]


@torch.inference_mode()
def predict_sequence_from_embedding(
    model: nn.Module,
    *,
    sequence: str,
    embedding: torch.Tensor,
    local_context: dict[str, Any],
    window_size: int,
    eval_stride: int | None,
    top_k_heuristic: int,
    device: torch.device,
    window_batch_size: int,
    linker_class: int = 1,
    dual_binary: bool = False,
) -> list[float]:
    base_hidden = int(embedding.shape[1])
    expected_input = int(base_hidden * _local_context_width(local_context))
    classifier_input = int(getattr(model, "_classifier_input_dim", expected_input))
    if expected_input != classifier_input:
        raise ValueError(
            f"Embedding hidden size mismatch: base={base_hidden}, "
            f"local_context width={_local_context_width(local_context)}, "
            f"expected classifier input={expected_input}, checkpoint input={classifier_input}"
        )
    if int(embedding.shape[0]) != len(sequence):
        raise ValueError(
            f"Embedding length {int(embedding.shape[0])} != sequence length {len(sequence)}"
        )

    starts = build_eval_window_starts(
        sequence,
        window_size=window_size,
        stride=eval_stride,
        top_k_heuristic=top_k_heuristic,
    )
    window_pairs: list[tuple[int, list[float]]] = []
    batch_size = max(1, int(window_batch_size))
    for batch_start in range(0, len(starts), batch_size):
        chunk_starts = starts[batch_start : batch_start + batch_size]
        windows = [
            _augment_with_local_context(embedding[start : start + window_size], local_context)
            for start in chunk_starts
        ]
        batch_logits = _forward_feature_windows(model, windows, device=device)
        window_pairs.extend(zip(chunk_starts, batch_logits))

    merged_logits = merge_window_logits(length=len(sequence), window_logits=window_pairs)
    return linker_probs_from_logits(
        merged_logits, linker_class=linker_class, dual_binary=dual_binary
    )
