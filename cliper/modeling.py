from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .windowing import normalize_sequence


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    residue_lengths: list[int]


class DummyProteinTokenizer:
    def __init__(self) -> None:
        alphabet = "ACDEFGHIKLMNPQRSTVWYX-"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab = {aa: idx + 2 for idx, aa in enumerate(alphabet)}

    def __call__(self, sequences: list[str]) -> EncodedBatch:
        residue_lengths = [len(seq) for seq in sequences]
        tokenized = []
        max_len = 0
        for sequence in sequences:
            ids = [self._vocab.get(ch, self._vocab["X"]) for ch in sequence]
            ids.append(self.eos_token_id)
            tokenized.append(ids)
            max_len = max(max_len, len(ids))

        input_ids = torch.full((len(sequences), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row, ids in enumerate(tokenized):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return EncodedBatch(input_ids=input_ids, attention_mask=attention_mask, residue_lengths=residue_lengths)


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 64, vocab_size: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Any:
        embedded = self.embedding(input_ids)
        outputs, _ = self.encoder(embedded)
        if attention_mask is not None:
            outputs = outputs * attention_mask.unsqueeze(-1).to(outputs.dtype)
        return SimpleNamespace(last_hidden_state=outputs)


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported classifier_head activation: {name!r}")


def _build_padding_mask(residue_lengths: list[int], max_len: int, *, device: torch.device) -> torch.Tensor:
    if max_len <= 0:
        raise ValueError(f"max_len must be > 0, got {max_len}")
    lengths = torch.tensor(residue_lengths, dtype=torch.long, device=device)
    if lengths.numel() == 0:
        raise ValueError("residue_lengths must not be empty.")
    if int(lengths.min().item()) <= 0:
        raise ValueError(f"residue_lengths must contain positive values, got {residue_lengths!r}")
    if int(lengths.max().item()) > max_len:
        raise ValueError(f"residue_lengths contains value > max_len: max_len={max_len}, lengths={residue_lengths!r}")
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(lengths.shape[0], max_len)
    return positions >= lengths.unsqueeze(1)


def _sinusoidal_position_encoding(length: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0 or dim <= 0:
        raise ValueError(f"length and dim must be > 0, got length={length}, dim={dim}")
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / float(dim))
    )
    pe = torch.zeros(length, dim, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    cos_width = pe[:, 1::2].shape[1]
    if cos_width > 0:
        pe[:, 1::2] = torch.cos(positions * div_term[:cos_width])
    return pe.to(dtype=dtype)


def _resolve_local_context(local_context: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(local_context or {})
    enabled = bool(cfg.get("enabled", False))
    radius = int(cfg.get("radius", 2))
    mode = str(cfg.get("mode", "concat_window")).lower()
    include_self = bool(cfg.get("include_self", True))
    if radius < 0:
        raise ValueError(f"local_context.radius must be >= 0, got {radius}")
    if mode != "concat_window":
        raise ValueError("local_context.mode must be 'concat_window' for ResidueClassifier.")
    if enabled and radius == 0 and not include_self:
        raise ValueError("local_context requires at least one feature when enabled.")
    return {
        "enabled": enabled,
        "radius": radius,
        "mode": mode,
        "include_self": include_self,
    }


def _local_context_multiplier(local_context: dict[str, Any]) -> int:
    if not bool(local_context.get("enabled", False)):
        return 1
    radius = int(local_context.get("radius", 2))
    include_self = bool(local_context.get("include_self", True))
    span = (2 * radius + 1) if include_self else (2 * radius)
    if span <= 0:
        raise ValueError("local_context produced empty span; check radius/include_self.")
    return span


def _concat_local_window(x: torch.Tensor, residue_lengths: list[int], local_context: dict[str, Any]) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected [B, L, H] input, got {tuple(x.shape)}")
    if not bool(local_context.get("enabled", False)):
        return x
    radius = int(local_context.get("radius", 2))
    if radius <= 0:
        return x

    include_self = bool(local_context.get("include_self", True))
    batch, max_len, _ = x.shape
    lengths = torch.tensor(residue_lengths, device=x.device, dtype=torch.long)
    if int(lengths.max().item()) > max_len:
        raise ValueError(f"residue_lengths has value > max_len ({max_len}): {residue_lengths!r}")
    valid_mask = torch.arange(max_len, device=x.device).unsqueeze(0).expand(batch, max_len) < lengths.unsqueeze(1)
    x = x * valid_mask.unsqueeze(-1).to(x.dtype)
    positions = torch.arange(max_len, device=x.device)

    chunks: list[torch.Tensor] = []
    for offset in range(-radius, radius + 1):
        if offset == 0 and not include_self:
            continue
        idx = positions + offset
        in_bounds = (idx >= 0) & (idx < max_len)
        safe_idx = idx.clamp(min=0, max=max_len - 1)
        shifted = x[:, safe_idx, :]
        neighbor_valid = valid_mask[:, safe_idx] & in_bounds.unsqueeze(0)
        shifted = shifted * neighbor_valid.unsqueeze(-1).to(shifted.dtype)
        chunks.append(shifted)
    if not chunks:
        raise ValueError("local_context concat_window produced no chunks.")
    return torch.cat(chunks, dim=-1)


class MLPClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        *,
        dropout: float,
        dropout_schedule: list[float] | None = None,
        activation: str,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        if dropout_schedule is None:
            per_layer_dropout = [float(dropout)] * len(hidden_dims)
        else:
            if len(dropout_schedule) != len(hidden_dims):
                raise ValueError(
                    "classifier_head.dropout_schedule length must match hidden_dims length: "
                    f"{len(dropout_schedule)} vs {len(hidden_dims)}"
                )
            per_layer_dropout = [float(x) for x in dropout_schedule]
            for value in per_layer_dropout:
                if not (0.0 <= value < 1.0):
                    raise ValueError(f"classifier_head.dropout_schedule values must be in [0,1), got {value}")
        for hidden_order, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_build_activation(activation))
            layers.append(nn.Dropout(per_layer_dropout[hidden_order]))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        activation: str,
        use_positional_encoding: bool,
    ) -> None:
        super().__init__()
        activation_key = activation.lower()
        if activation_key not in {"relu", "gelu"}:
            raise ValueError(f"Unsupported classifier_head activation for transformer: {activation!r}")
        if num_layers <= 0:
            raise ValueError(f"classifier_head.num_layers must be > 0, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"classifier_head.num_heads must be > 0, got {num_heads}")
        if ffn_dim <= 0:
            raise ValueError(f"classifier_head.ffn_dim must be > 0, got {ffn_dim}")
        if input_dim % num_heads != 0:
            raise ValueError(
                "Transformer classifier_head requires hidden_size divisible by num_heads, "
                f"got hidden_size={input_dim}, num_heads={num_heads}"
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation_key,
            batch_first=True,
        )
        self.use_positional_encoding = bool(use_positional_encoding)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, *, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_positional_encoding:
            x = x + _sinusoidal_position_encoding(
                length=x.shape[1],
                dim=x.shape[2],
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(0)
        encoded = self.encoder(self.input_dropout(x), src_key_padding_mask=padding_mask)
        return self.classifier(encoded)


class CNNClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        conv_channels: list[int],
        kernel_size: int,
        dilations: list[int] | None,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"cnn input_dim must be > 0, got {input_dim}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"classifier_head.kernel_size must be a positive odd integer, got {kernel_size}")
        if not conv_channels:
            raise ValueError("classifier_head.conv_channels must be a non-empty list.")
        channels = [int(ch) for ch in conv_channels]
        if any(ch <= 0 for ch in channels):
            raise ValueError(f"classifier_head.conv_channels must be > 0, got {conv_channels!r}")
        if dilations is None:
            dils = [1] * len(channels)
        else:
            if len(dilations) != len(channels):
                raise ValueError(
                    "classifier_head.dilations length must match conv_channels length: "
                    f"{len(dilations)} vs {len(channels)}"
                )
            dils = [int(d) for d in dilations]
            if any(d <= 0 for d in dils):
                raise ValueError(f"classifier_head.dilations must be > 0, got {dilations!r}")

        layers: list[nn.Module] = []
        in_ch = input_dim
        for out_ch, dil in zip(channels, dils):
            pad = dil * (kernel_size - 1) // 2
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, dilation=dil))
            layers.append(_build_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, H] -> [B, H, L] for conv1d along residues.
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = self.classifier(y)
        return y.transpose(1, 2)


class ResidueClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        projection_dim: int = 128,
        classifier_head: dict[str, Any] | None = None,
        local_context: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.dropout = nn.Dropout(dropout)
        self.local_context = _resolve_local_context(local_context)
        classifier_input_dim = hidden_size * _local_context_multiplier(self.local_context)

        classifier_cfg = dict(classifier_head or {})
        head_type = str(classifier_cfg.get("type", "linear")).lower()
        self.classifier_type = head_type
        if head_type == "linear":
            self.classifier = nn.Linear(classifier_input_dim, 1)
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
                input_dim=classifier_input_dim,
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
                input_dim=classifier_input_dim,
                num_layers=int(classifier_cfg.get("num_layers", 2)),
                num_heads=int(classifier_cfg.get("num_heads", 4)),
                ffn_dim=int(classifier_cfg.get("ffn_dim", 2048)),
                dropout=float(classifier_cfg.get("dropout", 0.3)),
                activation=str(classifier_cfg.get("activation", "relu")),
                use_positional_encoding=bool(classifier_cfg.get("use_positional_encoding", True)),
            )
        elif head_type == "cnn":
            self.classifier = CNNClassifierHead(
                input_dim=classifier_input_dim,
                conv_channels=[int(ch) for ch in classifier_cfg.get("conv_channels", [256, 256, 256])],
                kernel_size=int(classifier_cfg.get("kernel_size", 3)),
                dilations=(
                    [int(d) for d in classifier_cfg["dilations"]]
                    if isinstance(classifier_cfg.get("dilations"), list)
                    else None
                ),
                dropout=float(classifier_cfg.get("dropout", 0.3)),
                activation=str(classifier_cfg.get("activation", "relu")),
            )
        else:
            raise ValueError(f"Unsupported classifier_head.type: {head_type!r}")

        self.projection_head = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_lengths: list[int],
        *,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state
        max_len = max(residue_lengths)
        if hidden.shape[1] < max_len:
            raise ValueError(
                f"Backbone output length {hidden.shape[1]} is smaller than requested residue length {max_len}."
            )
        residue_hidden = hidden[:, :max_len, :]
        contextual_hidden = _concat_local_window(residue_hidden, residue_lengths, self.local_context)
        classifier_input = self.dropout(contextual_hidden)
        if self.classifier_type == "transformer":
            padding_mask = _build_padding_mask(residue_lengths, max_len, device=classifier_input.device)
            logits = self.classifier(classifier_input, padding_mask=padding_mask).squeeze(-1)
        else:
            logits = self.classifier(classifier_input).squeeze(-1)
        if not return_embeddings:
            return logits
        embeddings = self.projection_head(classifier_input)
        return logits, embeddings


def _resolve_hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    if config is None:
        raise ValueError("Backbone has no config attribute; cannot infer hidden size.")
    for attr in ("hidden_size", "d_model", "dim"):
        value = getattr(config, attr, None)
        if isinstance(value, int):
            return value
    raise ValueError("Cannot infer hidden size from backbone config.")


def load_backbone_and_tokenizer(backbone_name: str) -> tuple[nn.Module, Any, int]:
    if backbone_name.lower() == "dummy":
        tokenizer = DummyProteinTokenizer()
        backbone = DummyBackbone()
        hidden_size = _resolve_hidden_size(backbone)
        return backbone, tokenizer, hidden_size

    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for non-dummy backbones. Install dependencies before training/evaluation."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(backbone_name, do_lower_case=False)
    config = AutoConfig.from_pretrained(backbone_name)
    if getattr(config, "model_type", None) == "t5":
        backbone = T5EncoderModel.from_pretrained(backbone_name)
    else:
        backbone = AutoModel.from_pretrained(backbone_name)
    hidden_size = _resolve_hidden_size(backbone)
    return backbone, tokenizer, hidden_size


def _adjust_lengths_for_attention(attention_mask: torch.Tensor, residue_lengths: list[int]) -> list[int]:
    adjusted: list[int] = []
    for idx, seq_len in enumerate(residue_lengths):
        token_count = int(attention_mask[idx].sum().item())
        # For ProtT5/ProstT5 this is usually (residue_len + eos). Keep robust fallback.
        max_residues = max(1, token_count - 1)
        if max_residues < seq_len:
            max_residues = token_count
        if max_residues < seq_len:
            raise ValueError(
                f"Tokenizer produced fewer tokens ({token_count}) than residues ({seq_len}) for sample index {idx}."
            )
        adjusted.append(min(seq_len, max_residues))
    return adjusted


def encode_sequences(tokenizer: Any, backbone_name: str, sequences: list[str]) -> EncodedBatch:
    normalized = [normalize_sequence(seq) for seq in sequences]
    if backbone_name.lower() == "dummy":
        return tokenizer(normalized)

    spaced = [" ".join(list(seq)) for seq in normalized]
    encoded = tokenizer(
        spaced,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    residue_lengths = _adjust_lengths_for_attention(
        attention_mask=encoded["attention_mask"],
        residue_lengths=[len(seq) for seq in normalized],
    )
    return EncodedBatch(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        residue_lengths=residue_lengths,
    )

