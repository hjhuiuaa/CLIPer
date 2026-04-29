from __future__ import annotations

import argparse
from pathlib import Path

from cliper.data import (
    build_motif_vocab,
    load_motif_specs,
    parse_three_line_fasta,
    read_json,
    select_records,
    summarize_motif_coverage_detailed,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Stage5 motif coverage data pack.")
    parser.add_argument("--motif-json", required=True, help="Motif library JSON path.")
    parser.add_argument("--matching", default="prosite", choices=["exact", "regex", "degenerate", "prosite"])
    parser.add_argument("--train-fasta", required=True, help="DisProt linker FASTA path.")
    parser.add_argument("--caid-fasta", required=True, help="CAID linker FASTA path.")
    parser.add_argument("--split-manifest", required=True, help="Split manifest JSON path.")
    parser.add_argument("--out-dir", default="artifacts/motif", help="Output directory.")
    parser.add_argument("--max-per-residue", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    motif_specs = load_motif_specs(args.motif_json, matching=args.matching)
    motif_vocab = build_motif_vocab(motif_specs)

    train_all = parse_three_line_fasta(args.train_fasta)
    caid_records = parse_three_line_fasta(args.caid_fasta)
    manifest = read_json(args.split_manifest)
    train_records = select_records(train_all, manifest["train_ids"])
    val_records = select_records(train_all, manifest["val_ids"])

    train_report = summarize_motif_coverage_detailed(
        train_records,
        motif_specs,
        motif_vocab,
        max_per_residue=args.max_per_residue,
        top_k=args.top_k,
    )
    val_report = summarize_motif_coverage_detailed(
        val_records,
        motif_specs,
        motif_vocab,
        max_per_residue=args.max_per_residue,
        top_k=args.top_k,
    )
    caid_report = summarize_motif_coverage_detailed(
        caid_records,
        motif_specs,
        motif_vocab,
        max_per_residue=args.max_per_residue,
        top_k=args.top_k,
    )

    summary = {
        "motif_json": str(Path(args.motif_json)),
        "matching": args.matching,
        "max_per_residue": int(args.max_per_residue),
        "top_k": int(args.top_k),
        "train": train_report,
        "val": val_report,
        "caid": caid_report,
    }

    out_dir = Path(args.out_dir)
    write_json(out_dir / "motif_coverage_train.json", train_report)
    write_json(out_dir / "motif_coverage_val.json", val_report)
    write_json(out_dir / "motif_coverage_caid.json", caid_report)
    write_json(out_dir / "motif_coverage_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
