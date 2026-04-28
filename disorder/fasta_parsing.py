"""CLIPer data re-exports plus disorder splits: unlabeled 2-line FASTA + labeled 3-line FASTA."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from cliper.data import (
    VALID_SEQUENCE_CHARS,
    ProteinRecord,
    build_split_manifest,
    parse_id_lines,
    parse_three_line_fasta,
    read_json,
    select_records,
    write_json,
)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_id_lines_or_empty(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    source = Path(path)
    if not source.exists():
        return set()
    return parse_id_lines(source)


__all__ = [
    "ProteinRecord",
    "build_fixed_train_val_split_manifest",
    "build_split_manifest",
    "load_disorder_labeled_pair",
    "parse_id_lines",
    "parse_three_line_fasta",
    "parse_two_line_fasta",
    "read_json",
    "select_records",
    "write_json",
]


def parse_two_line_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Standard FASTA: each record is >id then SEQUENCE (one line each, non-empty lines only)."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"FASTA file not found: {source}")
    raw = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw) % 2 != 0:
        raise ValueError(
            f"Malformed 2-line FASTA (expected even line count): {source} has {len(raw)} non-empty lines."
        )
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for idx in range(0, len(raw), 2):
        header = raw[idx]
        sequence = raw[idx + 1].upper()
        if not header.startswith(">"):
            raise ValueError(f"Malformed FASTA header at record starting line {idx + 1} in {source}: {header!r}")
        protein_id = header[1:].strip()
        if not protein_id:
            raise ValueError(f"Empty protein id in {source} near line {idx + 1}.")
        if protein_id in seen:
            raise ValueError(f"Duplicate protein id in {source}: {protein_id}")
        seen.add(protein_id)
        if not set(sequence).issubset(VALID_SEQUENCE_CHARS):
            bad = sorted(set(sequence) - set(VALID_SEQUENCE_CHARS))
            raise ValueError(f"Invalid sequence chars for {protein_id} in {source}: {bad}")
        out.append((protein_id, sequence))
    return out


def load_disorder_labeled_pair(sequence_fasta: str | Path, label_fasta: str | Path) -> list[ProteinRecord]:
    """
    Join unlabeled `sequence_fasta` (2-line) with `label_fasta` (3-line: id, sequence, labels).
    Sequences must match per id.
    """
    seq_map = dict(parse_two_line_fasta(sequence_fasta))
    labeled = parse_three_line_fasta(label_fasta)
    records: list[ProteinRecord] = []
    for rec in labeled:
        if rec.protein_id not in seq_map:
            raise ValueError(
                f"Protein {rec.protein_id!r} in label file {label_fasta} missing from sequence file {sequence_fasta}."
            )
        seq_unlab = seq_map[rec.protein_id]
        if seq_unlab != rec.sequence:
            raise ValueError(
                f"Sequence mismatch for {rec.protein_id}: {sequence_fasta} vs {label_fasta} "
                f"(lengths {len(seq_unlab)} vs {len(rec.sequence)})."
            )
        records.append(rec)
    return records


def build_fixed_train_val_split_manifest(
    *,
    train_label_fasta: str | Path,
    val_label_fasta: str | Path,
    holdout_fasta: str | Path,
    error_file: str | Path | None = None,
) -> tuple[dict, dict]:
    """
    Build train/val split from two disjoint 3-line labeled FASTA files (no random split).
    Removes ids appearing in error_file (optional) or in holdout_fasta.
    """
    train_label_fasta = Path(train_label_fasta)
    val_label_fasta = Path(val_label_fasta)
    holdout_fasta = Path(holdout_fasta)

    train_recs = parse_three_line_fasta(train_label_fasta)
    val_recs = parse_three_line_fasta(val_label_fasta)
    train_ids = {r.protein_id for r in train_recs}
    val_ids = {r.protein_id for r in val_recs}
    overlap = train_ids & val_ids
    if overlap:
        sample = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"train and val label FASTA share {len(overlap)} id(s), e.g.: {sample}")

    record_ids = train_ids | val_ids
    caid_ids = parse_id_lines(holdout_fasta)
    error_ids = _parse_id_lines_or_empty(error_file)

    excluded_error_ids = sorted(record_ids & error_ids)
    holdout_overlap_ids = sorted(record_ids & caid_ids)
    missing_error_ids = sorted(error_ids - record_ids)

    drop = set(excluded_error_ids) | set(holdout_overlap_ids)
    train_final = sorted(train_ids - drop)
    val_final = sorted(val_ids - drop)

    if len(train_final) < 1 or len(val_final) < 1:
        raise ValueError(
            "After exclusions, need at least one train and one val protein. "
            f"train_final={len(train_final)} val_final={len(val_final)} "
            f"excluded_error={len(excluded_error_ids)} holdout_overlap={len(holdout_overlap_ids)}"
        )

    split_manifest = {
        "generated_at_utc": _utc_now_iso(),
        "split_mode": "fixed_train_val_files",
        "train_label_fasta": str(train_label_fasta.resolve()),
        "val_label_fasta": str(val_label_fasta.resolve()),
        "source_fasta": f"{train_label_fasta.resolve()}|{val_label_fasta.resolve()}",
        "seed": None,
        "val_ratio": None,
        "counts": {
            "source_train_records": len(train_ids),
            "source_val_records": len(val_ids),
            "eligible_records": len(train_final) + len(val_final),
            "train_records": len(train_final),
            "val_records": len(val_final),
            "excluded_error_records": len(excluded_error_ids),
            "excluded_holdout_overlap_records": len(holdout_overlap_ids),
        },
        "train_ids": train_final,
        "val_ids": val_final,
        "excluded_error_ids": excluded_error_ids,
        "excluded_holdout_overlap_ids": holdout_overlap_ids,
        "caid_holdout_ids": sorted(caid_ids),
    }

    exclusion_report = {
        "generated_at_utc": _utc_now_iso(),
        "split_mode": "fixed_train_val_files",
        "train_label_fasta": str(train_label_fasta.resolve()),
        "val_label_fasta": str(val_label_fasta.resolve()),
        "error_id_count_in_file": len(error_ids),
        "error_ids_present_in_source": excluded_error_ids,
        "error_ids_missing_from_source": missing_error_ids,
        "holdout_overlap_ids": holdout_overlap_ids,
        "excluded_total": len(excluded_error_ids) + len(holdout_overlap_ids),
    }

    return split_manifest, exclusion_report
