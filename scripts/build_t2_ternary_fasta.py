#!/usr/bin/env python3
"""Build ternary residue labels for T2-adapted training.

Classes (char encoding chosen so SupCon on {0,1} excludes ordered):
  1 = Linker (linker==1)
  0 = other IDR (linker==0 and disorder==1)
  2 = ordered (linker==0 and disorder==0)
  - = ignore (non {0,1} on either side)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cliper.data import parse_three_line_fasta


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ternary_labels(linker: str, disorder: str) -> str:
    if len(linker) != len(disorder):
        raise ValueError(f"Length mismatch: linker={len(linker)} disorder={len(disorder)}")
    out: list[str] = []
    for l_ch, d_ch in zip(linker, disorder):
        if l_ch == "1":
            out.append("1")
        elif l_ch == "0" and d_ch == "1":
            out.append("0")
        elif l_ch == "0" and d_ch == "0":
            out.append("2")
        else:
            out.append("-")
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linker-fasta",
        type=Path,
        default=REPO_ROOT / "dataset/disprot_202312_linker_label.fasta",
    )
    parser.add_argument(
        "--disorder-fasta",
        type=Path,
        default=REPO_ROOT / "dataset/disprot_202312_label.fasta",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        default=REPO_ROOT / "dataset/disprot_202312_t2_ternary.fasta",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "dataset/disprot_202312_t2_ternary_stats.json",
    )
    args = parser.parse_args()

    linker_recs = parse_three_line_fasta(args.linker_fasta)
    disorder_map = {rec.protein_id: rec.labels for rec in parse_three_line_fasta(args.disorder_fasta)}

    counts = {"1": 0, "0": 0, "2": 0, "-": 0}
    missing_disorder = 0
    written = 0
    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with args.output_fasta.open("w", encoding="utf-8") as handle:
        for rec in linker_recs:
            if rec.protein_id not in disorder_map:
                missing_disorder += 1
                continue
            labels = _ternary_labels(rec.labels, disorder_map[rec.protein_id])
            for ch in labels:
                counts[ch] = counts.get(ch, 0) + 1
            handle.write(f">{rec.protein_id}\n{rec.sequence}\n{labels}\n")
            written += 1

    stats = {
        "generated_at_utc": _utc_now_iso(),
        "linker_fasta": str(args.linker_fasta),
        "disorder_fasta": str(args.disorder_fasta),
        "output_fasta": str(args.output_fasta),
        "num_proteins": written,
        "missing_disorder": missing_disorder,
        "class_counts": {
            "linker_1": counts["1"],
            "other_idr_0": counts["0"],
            "ordered_2": counts["2"],
            "ignore": counts["-"],
        },
        "encoding": {
            "1": "Linker",
            "0": "non-Linker Disorder (other IDR)",
            "2": "ordered (non-Disorder)",
            "-": "ignore",
        },
    }
    args.output_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {args.output_fasta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
