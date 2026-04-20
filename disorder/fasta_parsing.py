"""CLIPer data re-exports plus disorder splits: unlabeled 2-line FASTA + labeled 3-line FASTA."""

from __future__ import annotations

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

__all__ = [
    "ProteinRecord",
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
