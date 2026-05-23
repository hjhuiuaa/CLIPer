"""Encode one protein sequence to per-residue ProstT5 embeddings and save as .resfeat.txt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cliper.modeling import encode_sequences, load_backbone_and_tokenizer
from cliper.windowing import normalize_sequence
from disorder.data import parse_two_line_fasta
from disorder.feature_io import feature_file_path, write_feature_manifest, write_residue_feature_file


def non_overlapping_window_starts(length: int, window_size: int) -> list[int]:
    if length <= 0 or window_size <= 0:
        raise ValueError("length and window_size must be positive")
    return list(range(0, length, int(window_size)))


def _autocast_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def resolve_sequence_input(
    *,
    protein_id: str | None = None,
    sequence: str | None = None,
    fasta_path: str | Path | None = None,
) -> tuple[str, str]:
    if fasta_path is not None:
        records = parse_two_line_fasta(fasta_path)
        if len(records) != 1:
            raise ValueError(f"Expected exactly one 2-line FASTA record, got {len(records)} in {fasta_path}")
        pid, seq = records[0]
        return pid, normalize_sequence(seq)
    if protein_id is None or not str(protein_id).strip():
        raise ValueError("protein_id is required when fasta_path is not provided.")
    if sequence is None or not str(sequence).strip():
        raise ValueError("sequence is required when fasta_path is not provided.")
    return str(protein_id).strip(), normalize_sequence(sequence)


@torch.inference_mode()
def encode_residue_embeddings(
    *,
    sequence: str,
    backbone: torch.nn.Module,
    tokenizer: Any,
    backbone_name: str,
    window_size: int = 1024,
    batch_size: int = 1,
    device: torch.device,
    mixed_precision: bool = True,
) -> torch.Tensor:
    """
    Return residue embeddings [L, D] for one sequence.

    - L <= window_size: single full-sequence forward pass.
    - L > window_size: non-overlapping windows (same merge policy as reextract_merge_nonoverlap).
    """
    seq = normalize_sequence(sequence)
    length = len(seq)
    if length == 0:
        raise ValueError("sequence is empty after normalization.")

    use_amp = mixed_precision and device.type == "cuda"
    backbone.eval()

    if length <= int(window_size):
        encoded = encode_sequences(tokenizer, backbone_name, [seq])
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        residue_lengths = encoded.residue_lengths
        with _autocast_context(device_type=device.type, enabled=use_amp):
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        seg_len = int(residue_lengths[0])
        if hidden.shape[1] < seg_len:
            raise ValueError("Backbone sequence length shorter than residue count.")
        return hidden[0, :seg_len, :].detach().float().cpu()

    starts = non_overlapping_window_starts(length, int(window_size))
    windows = [seq[s : s + int(window_size)] for s in starts]
    ordered_vecs: list[torch.Tensor] = []

    for batch_start in range(0, len(windows), max(1, int(batch_size))):
        sub = windows[batch_start : batch_start + int(batch_size)]
        encoded = encode_sequences(tokenizer, backbone_name, sub)
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        residue_lengths = encoded.residue_lengths
        with _autocast_context(device_type=device.type, enabled=use_amp):
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        for row, seg_len in enumerate(residue_lengths):
            ordered_vecs.append(hidden[row, :seg_len, :].detach().float().cpu())

    full = torch.cat(ordered_vecs, dim=0)
    if int(full.shape[0]) != length:
        raise RuntimeError(
            f"Merged embedding length mismatch: emb_rows={full.shape[0]} seq_len={length}"
        )
    return full


@torch.inference_mode()
def extract_sequence_embedding(
    *,
    protein_id: str | None = None,
    sequence: str | None = None,
    fasta_path: str | Path | None = None,
    output_dir: str | Path,
    backbone_name: str,
    window_size: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    mixed_precision: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    pid, seq = resolve_sequence_input(protein_id=protein_id, sequence=sequence, fasta_path=fasta_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = feature_file_path(out_dir, pid)

    if target.exists() and not overwrite:
        return {
            "protein_id": pid,
            "sequence_length": len(seq),
            "output_path": str(target.resolve()),
            "status": "skipped_exists",
            "window_size": int(window_size),
            "encoding_mode": "short_single_pass" if len(seq) <= int(window_size) else "long_nonoverlap_merge",
        }

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    backbone = backbone.to(dev)

    embeddings = encode_residue_embeddings(
        sequence=seq,
        backbone=backbone,
        tokenizer=tokenizer,
        backbone_name=backbone_name,
        window_size=int(window_size),
        batch_size=int(batch_size),
        device=dev,
        mixed_precision=mixed_precision,
    )
    write_residue_feature_file(target, embeddings)
    write_feature_manifest(out_dir, hidden_size=hidden_size, backbone_name=backbone_name)

    return {
        "protein_id": pid,
        "sequence_length": len(seq),
        "hidden_size": int(hidden_size),
        "output_path": str(target.resolve()),
        "output_dir": str(out_dir.resolve()),
        "status": "written",
        "window_size": int(window_size),
        "encoding_mode": "short_single_pass" if len(seq) <= int(window_size) else "long_nonoverlap_merge",
        "device": str(dev),
    }
