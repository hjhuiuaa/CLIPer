#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cliper.data import parse_three_line_fasta


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _compute_class_stats(records) -> dict[str, float | int | None]:
    pos = sum(rec.labels.count("1") for rec in records)
    neg = sum(rec.labels.count("0") for rec in records)
    total = pos + neg
    positive_ratio = (pos / total) if total > 0 else None
    pos_weight = (neg / pos) if pos > 0 else None
    return {
        "positive_residues": pos,
        "negative_residues": neg,
        "scored_residues": total,
        "positive_ratio": positive_ratio,
        "pos_weight": pos_weight,
    }


def _write_three_line_fasta(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(f">{rec.protein_id}\n{rec.sequence}\n{rec.labels}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean DisProt linker FASTA by removing proteins with all-zero labels."
    )
    parser.add_argument(
        "--input-fasta",
        default="dataset/disprot_202312_linker_label.fasta",
        help="Path to source 3-line linker FASTA.",
    )
    parser.add_argument(
        "--backup-fasta",
        default="dataset/disprot_202312_linker_label_not_cleaned.fasta",
        help="Backup path for original FASTA before cleaning.",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/splits/disprot_linker_clean_report.json",
        help="Output report JSON path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_fasta)
    backup_path = Path(args.backup_fasta)
    report_path = Path(args.report_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_path}")

    original_records = parse_three_line_fasta(input_path)
    original_stats = _compute_class_stats(original_records)

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, backup_path)

    kept_records = [rec for rec in original_records if "1" in rec.labels]
    removed_records = [rec for rec in original_records if "1" not in rec.labels]
    cleaned_stats = _compute_class_stats(kept_records)

    if not kept_records:
        raise ValueError("Cleaning would remove all records; aborting.")

    _write_three_line_fasta(input_path, kept_records)

    report = {
        "generated_at_utc": _utc_now_iso(),
        "input_fasta": str(input_path),
        "backup_fasta": str(backup_path),
        "cleaned_fasta": str(input_path),
        "source_records": len(original_records),
        "kept_records": len(kept_records),
        "removed_records": len(removed_records),
        "removed_ids_sample": [rec.protein_id for rec in removed_records[:50]],
        "stats_before": original_stats,
        "stats_after": cleaned_stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
