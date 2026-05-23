"""Export ProstT5 (T5 encoder) residue embeddings to per-protein line-oriented feature files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cliper.modeling import load_backbone_and_tokenizer
from cliper.windowing import normalize_sequence
from disorder.data import parse_two_line_fasta
from disorder.feature_io import (
    feature_file_path,
    write_feature_manifest,
    write_residue_feature_file,
)
from disorder.sequence_embedding import encode_residue_embeddings


@torch.inference_mode()
def extract_prostt5_features_for_fasta(
    *,
    sequence_fasta: str | Path,
    output_dir: str | Path,
    backbone_name: str,
    window_size: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    mixed_precision: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Read 2-line FASTA (one or many records), encode each protein, write `<safe_id>.resfeat.txt`.

    Encoding per sequence (via ``sequence_embedding.encode_residue_embeddings``):
    - L <= window_size: one full-sequence forward pass
    - L > window_size: non-overlapping windows merged to [L, D]
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_two_line_fasta(sequence_fasta)

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    backbone = backbone.to(dev)
    use_amp = mixed_precision and dev.type == "cuda"

    written = 0
    skipped = 0
    short_count = 0
    long_count = 0

    for protein_id, raw_seq in pairs:
        sequence = normalize_sequence(raw_seq)
        target = feature_file_path(out_dir, protein_id)
        if target.exists() and not overwrite:
            skipped += 1
            continue

        embeddings = encode_residue_embeddings(
            sequence=sequence,
            backbone=backbone,
            tokenizer=tokenizer,
            backbone_name=backbone_name,
            window_size=int(window_size),
            batch_size=int(batch_size),
            device=dev,
            mixed_precision=use_amp,
        )
        write_residue_feature_file(target, embeddings)
        written += 1
        if len(sequence) <= int(window_size):
            short_count += 1
        else:
            long_count += 1

    write_feature_manifest(out_dir, hidden_size=hidden_size, backbone_name=backbone_name)
    return {
        "output_dir": str(out_dir.resolve()),
        "proteins_in_fasta": len(pairs),
        "files_written": written,
        "files_skipped": skipped,
        "short_sequences_written": short_count,
        "long_sequences_written": long_count,
        "hidden_size": hidden_size,
        "window_size": int(window_size),
    }


@torch.inference_mode()
def extract_from_checkpoint_classifier(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    backbone_name: str,
    sequence_fasta: str | Path,
    output_dir: str | Path,
    window_size: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    mixed_precision: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Forward through a checkpoint's backbone only; same per-protein file layout as extract_features.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_two_line_fasta(sequence_fasta)
    backbone = model.backbone
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = backbone.to(dev)
    use_amp = mixed_precision and dev.type == "cuda"
    hidden_size = int(getattr(getattr(backbone, "config", None), "hidden_size", None) or 0)
    if hidden_size <= 0:
        raise ValueError("Cannot infer hidden_size from backbone.config")

    written = 0
    skipped = 0
    short_count = 0
    long_count = 0

    for protein_id, raw_seq in pairs:
        sequence = normalize_sequence(raw_seq)
        target = feature_file_path(out_dir, protein_id)
        if target.exists() and not overwrite:
            skipped += 1
            continue

        embeddings = encode_residue_embeddings(
            sequence=sequence,
            backbone=backbone,
            tokenizer=tokenizer,
            backbone_name=backbone_name,
            window_size=int(window_size),
            batch_size=int(batch_size),
            device=dev,
            mixed_precision=use_amp,
        )
        write_residue_feature_file(target, embeddings)
        written += 1
        if len(sequence) <= int(window_size):
            short_count += 1
        else:
            long_count += 1

    write_feature_manifest(out_dir, hidden_size=hidden_size, backbone_name=backbone_name)
    return {
        "output_dir": str(out_dir.resolve()),
        "proteins_in_fasta": len(pairs),
        "files_written": written,
        "files_skipped": skipped,
        "short_sequences_written": short_count,
        "long_sequences_written": long_count,
        "hidden_size": hidden_size,
        "window_size": int(window_size),
    }
