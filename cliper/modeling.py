from __future__ import annotations

from dataclasses import dataclass
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


class ResidueClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, dropout: float = 0.1, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, residue_lengths: list[int]) -> torch.Tensor:
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
        logits = self.classifier(self.dropout(residue_hidden)).squeeze(-1)
        return logits


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

