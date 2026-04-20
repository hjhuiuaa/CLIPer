"""Read/write per-protein ProstT5 (or other) residue embeddings: one file per sequence, one line per residue."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

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


def read_feature_manifest(features_dir: str | Path) -> dict[str, Any]:
    path = Path(features_dir) / FEATURE_MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_hidden_size(features_dir: str | Path) -> int:
    data = read_feature_manifest(features_dir)
    if "hidden_size" not in data:
        raise KeyError(f"manifest missing hidden_size: {features_dir}")
    return int(data["hidden_size"])
