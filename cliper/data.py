from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Iterable


VALID_SEQUENCE_CHARS = set("ACDEFGHIKLMNPQRSTVWYBXZUO-")
VALID_LABEL_CHARS = set("01-")


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    labels: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_three_line_fasta(path: str | Path) -> list[ProteinRecord]:
    """Parse FASTA where each record is exactly 3 lines:
    >id
    SEQUENCE
    LABELS
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"FASTA file not found: {source}")

    raw_lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_lines) % 3 != 0:
        raise ValueError(
            f"Malformed FASTA (expected line count multiple of 3): {source} has {len(raw_lines)} non-empty lines."
        )

    records: list[ProteinRecord] = []
    seen_ids: set[str] = set()
    for idx in range(0, len(raw_lines), 3):
        header = raw_lines[idx]
        sequence = raw_lines[idx + 1].upper()
        labels = raw_lines[idx + 2]

        if not header.startswith(">"):
            line_no = idx + 1
            raise ValueError(f"Malformed FASTA header at line {line_no} in {source}: {header!r}")

        protein_id = header[1:].strip()
        if not protein_id:
            line_no = idx + 1
            raise ValueError(f"Empty protein id in header at line {line_no} in {source}.")
        if protein_id in seen_ids:
            raise ValueError(f"Duplicate protein id found in {source}: {protein_id}")
        seen_ids.add(protein_id)

        if len(sequence) != len(labels):
            raise ValueError(
                f"Sequence/label length mismatch in {source} for {protein_id}: "
                f"len(sequence)={len(sequence)} len(labels)={len(labels)}"
            )
        if not set(sequence).issubset(VALID_SEQUENCE_CHARS):
            bad = sorted(set(sequence) - VALID_SEQUENCE_CHARS)
            raise ValueError(f"Invalid sequence chars for {protein_id} in {source}: {bad}")
        if not set(labels).issubset(VALID_LABEL_CHARS):
            bad = sorted(set(labels) - VALID_LABEL_CHARS)
            raise ValueError(f"Invalid label chars for {protein_id} in {source}: {bad}")

        records.append(ProteinRecord(protein_id=protein_id, sequence=sequence, labels=labels))

    return records


def parse_id_lines(path: str | Path) -> set[str]:
    """Read ids from any FASTA-like file by scanning lines starting with '>'."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"ID source file not found: {source}")
    ids = {
        line[1:].strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.startswith(">")
    }
    ids.discard("")
    return ids


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def build_split_manifest(
    records: list[ProteinRecord],
    *,
    source_fasta: str | Path,
    error_ids: Iterable[str],
    caid_ids: Iterable[str],
    seed: int,
    val_ratio: float,
) -> tuple[dict, dict]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1 (exclusive), got {val_ratio}.")

    record_ids = {r.protein_id for r in records}
    error_id_set = set(error_ids)
    caid_id_set = set(caid_ids)

    excluded_error_ids = sorted(record_ids.intersection(error_id_set))
    missing_error_ids = sorted(error_id_set - record_ids)
    holdout_overlap_ids = sorted(record_ids.intersection(caid_id_set))

    eligible_ids = sorted(record_ids - set(excluded_error_ids) - set(holdout_overlap_ids))
    if len(eligible_ids) < 2:
        raise ValueError(
            "Not enough eligible proteins after exclusions. "
            f"eligible={len(eligible_ids)} excluded_error={len(excluded_error_ids)} holdout_overlap={len(holdout_overlap_ids)}"
        )

    rng = random.Random(seed)
    shuffled = eligible_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)

    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])

    split_manifest = {
        "generated_at_utc": _utc_now_iso(),
        "source_fasta": str(Path(source_fasta)),
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": {
            "source_records": len(record_ids),
            "eligible_records": len(eligible_ids),
            "train_records": len(train_ids),
            "val_records": len(val_ids),
            "excluded_error_records": len(excluded_error_ids),
            "excluded_holdout_overlap_records": len(holdout_overlap_ids),
        },
        "train_ids": train_ids,
        "val_ids": val_ids,
        "excluded_error_ids": excluded_error_ids,
        "excluded_holdout_overlap_ids": holdout_overlap_ids,
        "caid_holdout_ids": sorted(caid_id_set),
    }

    exclusion_report = {
        "generated_at_utc": _utc_now_iso(),
        "source_fasta": str(Path(source_fasta)),
        "error_id_count_in_file": len(error_id_set),
        "error_ids_present_in_source": excluded_error_ids,
        "error_ids_missing_from_source": missing_error_ids,
        "holdout_overlap_ids": holdout_overlap_ids,
        "excluded_total": len(excluded_error_ids) + len(holdout_overlap_ids),
    }

    return split_manifest, exclusion_report


def select_records(records: list[ProteinRecord], selected_ids: Iterable[str]) -> list[ProteinRecord]:
    selected = set(selected_ids)
    chosen = [rec for rec in records if rec.protein_id in selected]
    missing = selected - {rec.protein_id for rec in chosen}
    if missing:
        missing_preview = sorted(missing)[:10]
        raise ValueError(f"Requested ids missing from parsed records. Missing sample: {missing_preview}")
    return chosen

