#!/usr/bin/env python3
"""Build T3 negative-protein FASTA for harboring-DL evaluation.

Positives are val_ids from the cleaned linker split (not written here).
Negatives are sampled from linker-all-zero proteins in the uncleaned FASTA.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cliper.data import parse_id_lines, parse_three_line_fasta


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-fasta",
        type=Path,
        default=REPO_ROOT / "dataset/disprot_202312_linker_label.fasta",
    )
    parser.add_argument(
        "--not-cleaned-fasta",
        type=Path,
        default=REPO_ROOT / "dataset/disprot_202312_linker_label_not_cleaned.fasta",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPO_ROOT / "artifacts/splits/disprot_split_seed42.json",
    )
    parser.add_argument(
        "--error-file",
        type=Path,
        default=REPO_ROOT / "dataset/error.txt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-negatives", type=int, default=168)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "artifacts/splits/disprot_t3_negatives_seed42.json",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        default=REPO_ROOT / "artifacts/splits/disprot_t3_negatives_seed42.fasta",
    )
    args = parser.parse_args()

    cleaned = parse_three_line_fasta(args.cleaned_fasta)
    not_cleaned = parse_three_line_fasta(args.not_cleaned_fasta)
    cleaned_ids = {rec.protein_id for rec in cleaned}
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_ids = set(split["train_ids"])
    val_ids = set(split["val_ids"])
    caid_ids = set(split.get("caid_holdout_ids", []))
    error_ids = parse_id_lines(args.error_file) if args.error_file.exists() else set()

    all_zero: list = []
    for rec in not_cleaned:
        if not rec.labels or any(ch != "0" for ch in rec.labels):
            continue
        all_zero.append(rec)

    pool = []
    excluded = {
        "in_cleaned": 0,
        "in_train_or_val": 0,
        "in_error": 0,
        "in_caid_holdout": 0,
    }
    for rec in all_zero:
        pid = rec.protein_id
        if pid in cleaned_ids:
            excluded["in_cleaned"] += 1
            continue
        if pid in train_ids or pid in val_ids:
            excluded["in_train_or_val"] += 1
            continue
        if pid in error_ids:
            excluded["in_error"] += 1
            continue
        if pid in caid_ids:
            excluded["in_caid_holdout"] += 1
            continue
        pool.append(rec)

    if args.n_negatives > len(pool):
        raise SystemExit(
            f"Requested {args.n_negatives} negatives but pool has only {len(pool)} after filters."
        )

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(pool, args.n_negatives), key=lambda r: r.protein_id)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at_utc": _utc_now_iso(),
        "seed": args.seed,
        "n_negatives": args.n_negatives,
        "positive_ref": {
            "split_key": "val_ids",
            "n_positives": len(val_ids),
            "note": "T3 positives = cleaned val_ids (proteins with >=1 linker)",
        },
        "sources": {
            "cleaned_fasta": str(args.cleaned_fasta),
            "not_cleaned_fasta": str(args.not_cleaned_fasta),
            "split_manifest": str(args.split_manifest),
            "error_file": str(args.error_file),
        },
        "pool_stats": {
            "all_zero_in_not_cleaned": len(all_zero),
            "pool_after_filters": len(pool),
            "excluded": excluded,
        },
        "negative_ids": [rec.protein_id for rec in selected],
    }
    args.output_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with args.output_fasta.open("w", encoding="utf-8") as handle:
        for rec in selected:
            handle.write(f">{rec.protein_id}\n{rec.sequence}\n{rec.labels}\n")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_fasta} ({len(selected)} proteins)")
    print(f"Pool after filters: {len(pool)}; all-zero in not_cleaned: {len(all_zero)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
