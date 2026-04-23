"""Classifier on precomputed per-residue embeddings (no ProstT5 at train time)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cliper.modeling import MLPClassifierHead, TransformerClassifierHead, _build_padding_mask


class DisorderFeatureClassifier(nn.Module):
    """Same heads as ResidueClassifier, but forward takes [B, L, hidden] embeddings."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        classifier_head: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        classifier_cfg = dict(classifier_head or {})
        head_type = str(classifier_cfg.get("type", "linear")).lower()
        self.classifier_type = head_type
        if head_type == "linear":
            self.classifier = nn.Linear(hidden_size, 1)
        elif head_type in {"mlp3", "mlp5", "mlp12"}:
            default_hidden_dims_map: dict[str, list[int]] = {
                "mlp3": [128, 64, 32],
                "mlp5": [1024, 256, 128, 64],
                "mlp12": [1024, 1024, 768, 768, 512, 512, 256, 256, 128, 128, 64],
            }
            expected_hidden_layers = {"mlp3": 3, "mlp5": 4, "mlp12": 11}[head_type]
            hidden_dims_raw = classifier_cfg.get("hidden_dims", default_hidden_dims_map[head_type])
            if not isinstance(hidden_dims_raw, list) or len(hidden_dims_raw) != expected_hidden_layers:
                raise ValueError(
                    f"classifier_head.hidden_dims for {head_type} must be a list of "
                    f"{expected_hidden_layers} integers, got {hidden_dims_raw!r}"
                )
            hidden_dims = [int(dim) for dim in hidden_dims_raw]
            if any(dim <= 0 for dim in hidden_dims):
                raise ValueError(f"classifier_head.hidden_dims must be > 0, got {hidden_dims_raw!r}")
            if head_type == "mlp3" and any(dim > 128 for dim in hidden_dims):
                raise ValueError(
                    "classifier_head.hidden_dims for mlp3 must use widths <= 128, "
                    f"got {hidden_dims_raw!r}"
                )
            self.classifier = MLPClassifierHead(
                input_dim=hidden_size,
                hidden_dims=hidden_dims,
                dropout=float(classifier_cfg.get("dropout", 0.3)),
                dropout_schedule=(
                    [float(x) for x in classifier_cfg["dropout_schedule"]]
                    if isinstance(classifier_cfg.get("dropout_schedule"), list)
                    else None
                ),
                activation=str(classifier_cfg.get("activation", "relu")),
                use_layernorm=bool(classifier_cfg.get("use_layernorm", True)),
            )
        elif head_type == "transformer":
            self.classifier = TransformerClassifierHead(
                input_dim=hidden_size,
                num_layers=int(classifier_cfg.get("num_layers", 2)),
                num_heads=int(classifier_cfg.get("num_heads", 4)),
                ffn_dim=int(classifier_cfg.get("ffn_dim", 2048)),
                dropout=float(classifier_cfg.get("dropout", 0.3)),
                activation=str(classifier_cfg.get("activation", "relu")),
                use_positional_encoding=bool(classifier_cfg.get("use_positional_encoding", True)),
            )
        else:
            raise ValueError(f"Unsupported classifier_head.type: {head_type!r}")

    def forward(self, residue_embeddings: torch.Tensor, residue_lengths: list[int]) -> torch.Tensor:
        if residue_embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings [B, L, H], got {tuple(residue_embeddings.shape)}")
        x = self.dropout(residue_embeddings)
        max_len = int(x.shape[1])
        if self.classifier_type == "transformer":
            padding_mask = _build_padding_mask(residue_lengths, max_len, device=x.device)
            logits = self.classifier(x, padding_mask=padding_mask).squeeze(-1)
        else:
            logits = self.classifier(x).squeeze(-1)
        return logits
