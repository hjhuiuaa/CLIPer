"""CAID submission I/O: FASTA parsing, embedding loading, .caid and timings.csv writers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import torch

from cliper.windowing import normalize_sequence
from disorder.feature_io import FEATURE_FILE_SUFFIX, read_residue_feature_file, safe_feature_stem

CAID_AMBIGUOUS_CHARS = set("BZJUOX")
EMBEDDING_EXTENSIONS = (".npy", ".h5", ".hdf5", FEATURE_FILE_SUFFIX)


def sanitize_sequence(sequence: str) -> str:
    """Uppercase and map ambiguous/nonstandard residues so prediction never crashes."""
    seq = normalize_sequence(sequence).replace("J", "X")
    return seq


def parse_caid_fasta(path: str | Path) -> list[tuple[str, str]]:
    """
    Parse standard 2-line-per-record FASTA for CAID submission.
    Accepts wrapped or single-line sequences; sanitizes ambiguous residues.
    """
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
    return torch.tensor(arr, dtype=np.float32)


def read_residue_embedding(embeddings_dir: str | Path, protein_id: str) -> torch.Tensor:
    """
    Load per-residue embeddings for one protein.

    Supported layouts under ``embeddings_dir`` (first match wins):
    - ``{protein_id}.npy`` / ``{safe_id}.npy`` with shape [L, D]
    - ``{protein_id}.h5`` with dataset named after protein id or ``embedding``
    - ``{protein_id}.resfeat.txt`` (CLIPer line-oriented format)
    """
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


def write_caid_file(
    path: str | Path,
    *,
    protein_id: str,
    sequence: str,
    probabilities: list[float],
    threshold: float,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(sequence) != len(probabilities):
        raise ValueError(
            f"Length mismatch for {protein_id}: sequence={len(sequence)} probabilities={len(probabilities)}"
        )

    with target.open("w", encoding="utf-8") as handle:
        handle.write(f">{protein_id}\n")
        for index, (aa, prob) in enumerate(zip(sequence, probabilities), start=1):
            binary = 1 if prob >= threshold else 0
            handle.write(f"{index}\t{aa}\t{prob:.3f}\t{binary}\n")


def write_timings_csv(path: str | Path, timings_ms: list[tuple[str, int]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).astimezone().strftime("%a %b %d %H:%M:%S %Z %Y")
    with target.open("w", encoding="utf-8") as handle:
        handle.write(f"# Running CLIPer linker predictor, started {started}\n")
        handle.write("sequence,milliseconds\n")
        for protein_id, elapsed_ms in timings_ms:
            safe_id = re.sub(r"[\s,]+", "_", protein_id.strip()) or protein_id
            handle.write(f"{safe_id},{int(elapsed_ms)}\n")
