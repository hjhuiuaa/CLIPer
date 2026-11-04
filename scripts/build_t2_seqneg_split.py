#!/usr/bin/env python3
"""CLI: all linker proteins + equal number of disorder-only sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cliper.t2_seqneg_data import build_t2_seqneg_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linker-fasta", type=Path, default=None)
    parser.add_argument("--disorder-fasta", type=Path, default=None)
    parser.add_argument("--error-file", type=Path, default=None)
    parser.add_argument("--caid-fasta", type=Path, default=None)
    parser.add_argument("--caid-disorder-fasta", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    stats = build_t2_seqneg_dataset(
        linker_fasta=args.linker_fasta,
        disorder_fasta=args.disorder_fasta,
        error_file=args.error_file,
        caid_fasta=args.caid_fasta,
        caid_disorder_fasta=args.caid_disorder_fasta,
        output_dir=args.output_dir,
        split_out=args.split_out,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    summary = {
        "output_dir": stats["paths"]["output_dir"],
        "split_manifest": stats["paths"]["split_manifest"],
        "filters": stats["filters"],
        "split": stats["split"],
        "sampling": stats["sampling"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(stats["selection_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
