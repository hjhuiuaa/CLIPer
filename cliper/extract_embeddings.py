"""Export ProstT5 residue embeddings for linker / CAID predict (one file per protein)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cliper.caid_io import parse_caid_fasta, sanitize_sequence
from cliper.modeling import load_backbone_and_tokenizer
from disorder.feature_io import (
    FEATURE_FILE_SUFFIX,
    feature_file_path,
    write_feature_manifest,
    write_residue_feature_file,
)
from disorder.sequence_embedding import encode_residue_embeddings, resolve_sequence_input

OutputFormat = Literal["resfeat", "npy"]


def _embedding_output_path(output_dir: Path, protein_id: str, fmt: OutputFormat) -> Path:
    stem = feature_file_path(output_dir, protein_id).stem
    if fmt == "npy":
        return output_dir / f"{stem}.npy"
    return output_dir / f"{stem}{FEATURE_FILE_SUFFIX}"


def _write_embedding(path: Path, embeddings: torch.Tensor, fmt: OutputFormat) -> None:
    if fmt == "npy":
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = embeddings.detach().float().cpu().numpy()
        np.save(path, arr)
        return
    write_residue_feature_file(path, embeddings)


@torch.inference_mode()
def extract_prostt5_embeddings_for_fasta(
    *,
    fasta_path: str | Path,
    output_dir: str | Path,
    backbone_name: str,
    window_size: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    mixed_precision: bool = True,
    overwrite: bool = False,
    output_format: OutputFormat = "resfeat",
) -> dict[str, Any]:
    """
    Export linker/CAID embeddings for one or many 2-line FASTA records.

    Each protein is written to a single matrix file with shape [L, hidden_size]:
    - ``resfeat`` (default): ``<id>.resfeat.txt`` (one line per residue)
    - ``npy``: ``<id>.npy`` (NumPy array, also accepted by ``cliper predict``)

    Encoding (same core as ``disorder.sequence_embedding``):
    - L <= window_size: one full-sequence ProstT5 forward pass
    - L > window_size: non-overlapping windows of size window_size, concatenated along L

    ``cliper predict`` then applies linker eval windows on these embeddings (not ProstT5).
    """
    records = parse_caid_fasta(fasta_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = str(output_format).lower()
    if fmt not in {"resfeat", "npy"}:
        raise ValueError(f"output_format must be 'resfeat' or 'npy', got {output_format!r}")

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    backbone = backbone.to(dev)
    use_amp = mixed_precision and dev.type == "cuda"

    written = 0
    skipped = 0
    short_count = 0
    long_count = 0
    written_paths: list[str] = []

    for protein_id, sequence in records:
        seq = sanitize_sequence(sequence)
        target = _embedding_output_path(out_dir, protein_id, fmt)  # type: ignore[arg-type]
        if target.exists() and not overwrite:
            skipped += 1
            continue

        embeddings = encode_residue_embeddings(
            sequence=seq,
            backbone=backbone,
            tokenizer=tokenizer,
            backbone_name=backbone_name,
            window_size=int(window_size),
            batch_size=int(batch_size),
            device=dev,
            mixed_precision=use_amp,
        )
        _write_embedding(target, embeddings, fmt)  # type: ignore[arg-type]
        written += 1
        written_paths.append(str(target.resolve()))
        if len(seq) <= int(window_size):
            short_count += 1
        else:
            long_count += 1

    manifest_extra = {
        "task": "linker_caid",
        "output_format": fmt,
        "window_size": int(window_size),
        "encoding_short": "full_sequence_forward",
        "encoding_long": "non_overlapping_window_concat",
        "note": "predict applies linker build_eval_window_starts on these embeddings",
    }
    write_feature_manifest(
        out_dir,
        hidden_size=hidden_size,
        backbone_name=backbone_name,
        extra=manifest_extra,
    )

    return {
        "output_dir": str(out_dir.resolve()),
        "fasta_path": str(Path(fasta_path).resolve()),
        "proteins_in_fasta": len(records),
        "files_written": written,
        "files_skipped": skipped,
        "short_sequences_written": short_count,
        "long_sequences_written": long_count,
        "hidden_size": int(hidden_size),
        "window_size": int(window_size),
        "output_format": fmt,
        "written_files": written_paths,
    }


@torch.inference_mode()
def extract_prostt5_embedding_for_sequence(
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
    output_format: OutputFormat = "resfeat",
) -> dict[str, Any]:
    pid, seq = resolve_sequence_input(protein_id=protein_id, sequence=sequence, fasta_path=fasta_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = str(output_format).lower()
    if fmt not in {"resfeat", "npy"}:
        raise ValueError(f"output_format must be 'resfeat' or 'npy', got {output_format!r}")

    seq = sanitize_sequence(seq)
    target = _embedding_output_path(out_dir, pid, fmt)  # type: ignore[arg-type]
    if target.exists() and not overwrite:
        return {
            "protein_id": pid,
            "sequence_length": len(seq),
            "output_path": str(target.resolve()),
            "status": "skipped_exists",
            "window_size": int(window_size),
            "output_format": fmt,
        }

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    backbone = backbone.to(dev)
    use_amp = mixed_precision and dev.type == "cuda"

    embeddings = encode_residue_embeddings(
        sequence=seq,
        backbone=backbone,
        tokenizer=tokenizer,
        backbone_name=backbone_name,
        window_size=int(window_size),
        batch_size=int(batch_size),
        device=dev,
        mixed_precision=use_amp,
    )
    _write_embedding(target, embeddings, fmt)  # type: ignore[arg-type]
    write_feature_manifest(
        out_dir,
        hidden_size=hidden_size,
        backbone_name=backbone_name,
        extra={
            "task": "linker_caid",
            "output_format": fmt,
            "window_size": int(window_size),
        },
    )

    return {
        "protein_id": pid,
        "sequence_length": len(seq),
        "hidden_size": int(hidden_size),
        "output_path": str(target.resolve()),
        "output_dir": str(out_dir.resolve()),
        "status": "written",
        "window_size": int(window_size),
        "output_format": fmt,
        "encoding_mode": "short_single_pass" if len(seq) <= int(window_size) else "long_nonoverlap_merge",
        "device": str(dev),
    }
