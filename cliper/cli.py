from __future__ import annotations

import argparse
import json

from .pipeline import evaluate, prepare_data, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIPer training/evaluation pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare_data", help="Create deterministic train/val split and exclusion report.")
    prepare_parser.add_argument("--fasta", required=True, help="Path to DisProt linker-labeled FASTA.")
    prepare_parser.add_argument("--error-file", required=True, help="Path to error.txt with disagreement ids.")
    prepare_parser.add_argument("--caid-fasta", required=True, help="Path to CAID3 linker FASTA (strict holdout).")
    prepare_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    prepare_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio.")
    prepare_parser.add_argument(
        "--output-split",
        default="artifacts/splits/disprot_split_seed42.json",
        help="Output JSON path for split manifest.",
    )
    prepare_parser.add_argument(
        "--output-exclusion",
        default="artifacts/splits/exclusion_report_seed42.json",
        help="Output JSON path for exclusion report.",
    )

    train_parser = subparsers.add_parser("train", help="Train CLIPer residue classifier (Stage 1 or Stage 2).")
    train_parser.add_argument("--config", required=True, help="YAML config path.")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation using a saved checkpoint.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    eval_parser.add_argument("--fasta", required=True, help="Path to evaluation FASTA.")
    eval_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. If omitted, auto-save to checkpoint experiment folder.",
    )
    eval_parser.add_argument("--split-manifest", default=None, help="Optional split manifest JSON.")
    eval_parser.add_argument("--split-key", default=None, help="Optional split key (e.g., train_ids or val_ids).")
    eval_parser.add_argument("--threshold", type=float, default=None, help="Optional decision threshold override.")
    eval_parser.add_argument("--batch-size", type=int, default=None, help="Optional inference batch size override.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare_data":
        result = prepare_data(
            fasta_path=args.fasta,
            error_file=args.error_file,
            caid_fasta=args.caid_fasta,
            seed=args.seed,
            val_ratio=args.val_ratio,
            split_out=args.output_split,
            exclusion_out=args.output_exclusion,
        )
    elif args.command == "train":
        result = train(config_path=args.config)
    elif args.command == "eval":
        result = evaluate(
            checkpoint_path=args.checkpoint,
            fasta_path=args.fasta,
            output_dir=args.output_dir,
            split_manifest_path=args.split_manifest,
            split_key=args.split_key,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
