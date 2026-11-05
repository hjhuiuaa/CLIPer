"""Read, write, and encode ProstT5 residue embeddings."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cliper.model import encode_sequences, load_backbone_and_tokenizer
from cliper.windowing import normalize_sequence

FEATURE_MANIFEST_NAME = "manifest.json"
FEATURE_FILE_SUFFIX = ".resfeat.txt"


def safe_feature_stem(protein_id: str) -> str:
    """Filesystem-safe stem (no path separators)."""
    stem = protein_id.strip().replace("\\", "_").replace("/", "_")
    stem = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE)
    return stem or "empty_id"


def feature_file_path(features_dir: str | Path, protein_id: str) -> Path:
    return Path(features_dir) / f"{safe_feature_stem(protein_id)}{FEATURE_FILE_SUFFIX}"


def write_residue_feature_file(path: str | Path, embeddings: torch.Tensor) -> None:
    """Write float32 matrix [L, D] as L lines, space-separated values."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if embeddings.dim() != 2:
        raise ValueError(f"Expected 2D tensor [L, D], got shape {tuple(embeddings.shape)}")
    arr = embeddings.detach().float().cpu().numpy()
    with target.open("w", encoding="utf-8") as handle:
        for row in arr:
            handle.write(" ".join(f"{float(x):.8g}" for x in row.tolist()))
            handle.write("\n")


def read_residue_feature_file(path: str | Path) -> torch.Tensor:
    """Load [L, D] float32 tensor from line-oriented text file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Feature file not found: {source}")
    rows: list[list[float]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise ValueError(f"Empty or invalid feature file: {source}")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"Ragged rows in feature file: {source}")
    return torch.tensor(rows, dtype=torch.float32)


def write_feature_manifest(
    features_dir: str | Path,
    *,
    hidden_size: int,
    backbone_name: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "format": "one_file_per_protein",
        "file_suffix": FEATURE_FILE_SUFFIX,
        "lines_format": "whitespace_separated_float32",
        "row_axis": "residue_position",
        "hidden_size": int(hidden_size),
        "backbone_name": str(backbone_name),
    }
    if extra:
        payload.update(extra)
    out = Path(features_dir) / FEATURE_MANIFEST_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


EMBEDDING_EXTENSIONS = (".npy", ".h5", ".hdf5", FEATURE_FILE_SUFFIX)


def _candidate_embedding_paths(embeddings_dir: Path, protein_id: str) -> list[Path]:
    stems = [protein_id, safe_feature_stem(protein_id)]
    paths: list[Path] = []
    seen: set[Path] = set()
    for stem in stems:
        for ext in EMBEDDING_EXTENSIONS:
            candidate = embeddings_dir / f"{stem}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
        legacy_npy = embeddings_dir / f"{stem}.resfeat.npy"
        if legacy_npy not in seen:
            seen.add(legacy_npy)
            paths.append(legacy_npy)
    return paths


def _load_numpy_embedding(path: Path) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array in {path}, got shape {arr.shape}")
    return torch.tensor(arr, dtype=torch.float32)


def _load_h5_embedding(path: Path, protein_id: str) -> torch.Tensor:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading .h5 embeddings requires h5py. Install with: pip install h5py") from exc

    with h5py.File(path, "r") as handle:
        if protein_id in handle:
            arr = np.asarray(handle[protein_id][:], dtype=np.float32)
        elif "embedding" in handle:
            arr = np.asarray(handle["embedding"][:], dtype=np.float32)
        elif len(handle.keys()) == 1:
            key = next(iter(handle.keys()))
            arr = np.asarray(handle[key][:], dtype=np.float32)
        else:
            raise KeyError(
                f"Cannot resolve dataset in {path} for protein {protein_id!r}. "
                f"Available keys: {list(handle.keys())[:10]}"
            )
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array in {path}, got shape {arr.shape}")
    return torch.tensor(arr, dtype=torch.float32)


def read_residue_embedding(embeddings_dir: str | Path, protein_id: str) -> torch.Tensor:
    """Load one protein's residue embeddings from ``.npy``, ``.h5``, or ``.resfeat.txt``."""
    root = Path(embeddings_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Embeddings directory not found: {root}")

    for candidate in _candidate_embedding_paths(root, protein_id):
        if not candidate.exists():
            continue
        suffix = candidate.suffix.lower()
        if suffix == ".npy":
            return _load_numpy_embedding(candidate)
        if suffix in {".h5", ".hdf5"}:
            return _load_h5_embedding(candidate, protein_id)
        if candidate.name.endswith(FEATURE_FILE_SUFFIX):
            return read_residue_feature_file(candidate)

    sample = ", ".join(str(p.name) for p in _candidate_embedding_paths(root, protein_id)[:3])
    raise FileNotFoundError(
        f"No embedding file found for {protein_id!r} under {root}. Tried e.g.: {sample}"
    )


OutputFormat = Literal["resfeat", "npy"]


def sanitize_sequence(sequence: str) -> str:
    """Uppercase and map ambiguous residues so encoding does not fail."""
    return normalize_sequence(sequence).replace("J", "X")


def parse_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Parse an unlabeled FASTA file. Sequences may be wrapped."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"FASTA file not found: {source}")

    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    protein_id: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal protein_id, sequence_parts
        if protein_id is None:
            return
        if not sequence_parts:
            raise ValueError(f"Missing sequence for protein id {protein_id!r} in {source}")
        sequence = sanitize_sequence("".join(sequence_parts))
        if protein_id in seen:
            raise ValueError(f"Duplicate protein id in {source}: {protein_id}")
        seen.add(protein_id)
        records.append((protein_id, sequence))
        protein_id = None
        sequence_parts = []

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            protein_id = line[1:].strip()
            if not protein_id:
                raise ValueError(f"Empty protein id in header: {line!r}")
            continue
        if protein_id is None:
            raise ValueError(f"Sequence line before any header in {source}: {line[:40]!r}")
        sequence_parts.append(line.upper())

    flush()
    if not records:
        raise ValueError(f"No FASTA records found in {source}")
    return records


def _write_timings_csv(path: Path, timings_ms: list[tuple[str, int]]) -> None:
    lines = ["sequence,milliseconds"]
    lines.extend(f"{protein_id},{int(elapsed_ms)}" for protein_id, elapsed_ms in timings_ms)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _autocast_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def resolve_sequence_input(
    *,
    protein_id: str | None = None,
    sequence: str | None = None,
    fasta_path: str | Path | None = None,
) -> tuple[str, str]:
    if fasta_path is not None:
        records = parse_fasta(fasta_path)
        if len(records) != 1:
            raise ValueError(f"Expected exactly one FASTA record, got {len(records)} in {fasta_path}")
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
    """Return residue embeddings [L, D] for one sequence.

    Sequences no longer than ``window_size`` use one forward pass. Longer sequences
    are split into non-overlapping windows and concatenated.
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

    starts = range(0, length, int(window_size))
    windows = [seq[start : start + int(window_size)] for start in starts]
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


def _embedding_output_path(output_dir: Path, protein_id: str, fmt: OutputFormat) -> Path:
    if fmt == "npy":
        return output_dir / f"{safe_feature_stem(protein_id)}.npy"
    return feature_file_path(output_dir, protein_id)


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

    Encoding:
    - L <= window_size: one full-sequence ProstT5 forward pass
    - L > window_size: non-overlapping windows of size window_size, concatenated along L

    ``cliper predict`` then applies linker eval windows on these embeddings (not ProstT5).
    """
    records = parse_fasta(fasta_path)
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
    timings_ms: list[tuple[str, int]] = []

    for protein_id, sequence in records:
        seq = sanitize_sequence(sequence)
        target = _embedding_output_path(out_dir, protein_id, fmt)  # type: ignore[arg-type]
        started = time.perf_counter()
        if target.exists() and not overwrite:
            skipped += 1
            timings_ms.append((protein_id, 0))
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
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        timings_ms.append((protein_id, elapsed_ms))
        written += 1
        written_paths.append(str(target.resolve()))
        if len(seq) <= int(window_size):
            short_count += 1
        else:
            long_count += 1

    _write_timings_csv(out_dir / "timings.csv", timings_ms)

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
        "timings_csv": str((out_dir / "timings.csv").resolve()),
        "timings_ms": timings_ms,
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
