from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Disorder residue classifier (overlapping windows for long sequences)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare_data", help="Train/val split + exclusion report (same 3-line FASTA as CLIPer).")
    prepare_parser.add_argument("--fasta", required=True, help="Training disorder-labeled 3-line FASTA.")
    prepare_parser.add_argument("--error-file", required=True, help="Optional disagreement ids (may be empty file).")
    prepare_parser.add_argument("--holdout-fasta", required=True, help="Holdout 3-line FASTA (ids excluded from train/val).")
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--val-ratio", type=float, default=0.2)
    prepare_parser.add_argument("--output-split", default="disorder/artifacts/splits/disorder_split_seed42.json")
    prepare_parser.add_argument("--output-exclusion", default="disorder/artifacts/splits/disorder_exclusion_seed42.json")

    ps_parser = subparsers.add_parser(
        "prepare_split",
        help="Fixed train/val split from two 3-line label FASTA (no merge file, no random split).",
    )
    ps_parser.add_argument("--train-label-fasta", required=True, help="Training split, 3-line labeled FASTA.")
    ps_parser.add_argument("--val-label-fasta", required=True, help="Validation split, 3-line labeled FASTA.")
    ps_parser.add_argument(
        "--holdout-fasta",
        required=True,
        help="Test/holdout FASTA (ids must not appear in train/val; >headers only).",
    )
    ps_parser.add_argument(
        "--error-file",
        default=None,
        help="Optional id list file; if omitted or missing, no error ids are applied.",
    )
    ps_parser.add_argument("--output-split", default="disorder/artifacts/splits/disorder_split_seed42.json")
    ps_parser.add_argument("--output-exclusion", default="disorder/artifacts/splits/disorder_exclusion_seed42.json")

    train_parser = subparsers.add_parser("train", help="Train with disorder windowing (patches cliper train).")
    train_parser.add_argument(
        "--config",
        required=True,
        help="YAML config under disorder/configs/ (e.g. disorder_stage3_mlp5_example.yaml or stage4 config).",
    )
    train_parser.add_argument("--resume-checkpoint", default=None)

    eval_parser = subparsers.add_parser("eval", help="Evaluate checkpoint on a FASTA.")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--fasta", required=True)
    eval_parser.add_argument("--output-dir", default=None)
    eval_parser.add_argument("--split-manifest", default=None)
    eval_parser.add_argument("--split-key", default=None)
    eval_parser.add_argument("--threshold", type=float, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=None)

    ex_parser = subparsers.add_parser(
        "extract_features",
        help="Run ProstT5 encoder on 2-line FASTA; write one .resfeat.txt per protein (one line per residue).",
    )
    ex_parser.add_argument("--fasta", required=True, help="Unlabeled 2-line FASTA (>id, sequence); one or many records.")
    ex_parser.add_argument("--output-dir", required=True, help="Directory for *.resfeat.txt and manifest.json.")
    ex_parser.add_argument("--backbone", required=True, help="HF model id or path (e.g. Rostlab/ProstT5).")
    ex_parser.add_argument("--window-size", type=int, default=1024, help="Window size for long-sequence encoding.")
    ex_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for long-sequence window encoding (one protein processed at a time).",
    )
    ex_parser.add_argument("--device", default="cuda")
    ex_parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    ex_parser.add_argument("--overwrite", action="store_true", help="Recompute even if feature file exists.")

    seq_parser = subparsers.add_parser(
        "extract_sequence",
        help="Extract one protein to <protein_id>.resfeat.txt (auto short/long encoding).",
    )
    seq_parser.add_argument("--output-dir", required=True, help="Directory for the output .resfeat.txt file.")
    seq_parser.add_argument("--backbone", required=True, help="HF model id or path (e.g. Rostlab/ProstT5).")
    seq_parser.add_argument("--protein-id", default=None, help="Protein id (with --sequence if --fasta omitted).")
    seq_parser.add_argument("--sequence", default=None, help="Amino-acid sequence string.")
    seq_parser.add_argument("--fasta", default=None, help="2-line FASTA with exactly one record.")
    seq_parser.add_argument("--window-size", type=int, default=1024)
    seq_parser.add_argument("--batch-size", type=int, default=1)
    seq_parser.add_argument("--device", default="cuda")
    seq_parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    seq_parser.add_argument("--overwrite", action="store_true", help="Recompute even if output file exists.")

    tf_parser = subparsers.add_parser(
        "train_features",
        help="Train disorder head on precomputed .resfeat.txt (paired 2-line + 3-line FASTA per split).",
    )
    tf_parser.add_argument("--config", required=True, help="YAML e.g. disorder/configs/disorder_feature_stage3_mlp5_example.yaml")
    tf_parser.add_argument("--resume-checkpoint", default=None)

    ef_parser = subparsers.add_parser(
        "eval_features",
        help="Evaluate a train_features checkpoint using paired FASTA + same features_dir layout.",
    )
    ef_parser.add_argument("--checkpoint", required=True)
    ef_parser.add_argument("--sequence-fasta", required=True, help="Unlabeled 2-line FASTA.")
    ef_parser.add_argument("--label-fasta", required=True, help="Labeled 3-line FASTA (id, seq, labels).")
    ef_parser.add_argument("--features-dir", default=None, help="Override directory of .resfeat.txt (default: from checkpoint config).")
    ef_parser.add_argument("--output-dir", default=None)
    ef_parser.add_argument("--threshold", type=float, default=None)
    ef_parser.add_argument("--window-batch-size", type=int, default=8)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare_data":
        from disorder.pipeline import prepare_data

        result = prepare_data(
            fasta_path=args.fasta,
            error_file=args.error_file,
            caid_fasta=args.holdout_fasta,
            seed=args.seed,
            val_ratio=args.val_ratio,
            split_out=args.output_split,
            exclusion_out=args.output_exclusion,
        )
    elif args.command == "prepare_split":
        from disorder.pipeline import prepare_data_fixed_train_val

        result = prepare_data_fixed_train_val(
            train_label_fasta=args.train_label_fasta,
            val_label_fasta=args.val_label_fasta,
            holdout_fasta=args.holdout_fasta,
            error_file=args.error_file,
            split_out=args.output_split,
            exclusion_out=args.output_exclusion,
        )
    elif args.command == "train":
        from disorder.pipeline import train

        result = train(config_path=args.config, resume_checkpoint=args.resume_checkpoint)
    elif args.command == "eval":
        from disorder.pipeline import evaluate

        result = evaluate(
            checkpoint_path=args.checkpoint,
            fasta_path=args.fasta,
            output_dir=args.output_dir,
            split_manifest_path=args.split_manifest,
            split_key=args.split_key,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
    elif args.command == "extract_features":
        from disorder.extract_features import extract_prostt5_features_for_fasta

        result = extract_prostt5_features_for_fasta(
            sequence_fasta=args.fasta,
            output_dir=args.output_dir,
            backbone_name=args.backbone,
            window_size=args.window_size,
            batch_size=args.batch_size,
            device=args.device,
            mixed_precision=not args.no_amp,
            overwrite=args.overwrite,
        )
    elif args.command == "extract_sequence":
        from disorder.sequence_embedding import extract_sequence_embedding

        if args.fasta is None and (args.protein_id is None or args.sequence is None):
            parser.error("extract_sequence requires --fasta or both --protein-id and --sequence.")
        result = extract_sequence_embedding(
            protein_id=args.protein_id,
            sequence=args.sequence,
            fasta_path=args.fasta,
            output_dir=args.output_dir,
            backbone_name=args.backbone,
            window_size=args.window_size,
            batch_size=args.batch_size,
            device=args.device,
            mixed_precision=not args.no_amp,
            overwrite=args.overwrite,
        )
    elif args.command == "train_features":
        from disorder.feature_pipeline import train_features

        result = train_features(config_path=args.config, resume_checkpoint=args.resume_checkpoint)
    elif args.command == "eval_features":
        from disorder.feature_pipeline import eval_features_checkpoint

        result = eval_features_checkpoint(
            checkpoint_path=args.checkpoint,
            sequence_fasta=args.sequence_fasta,
            label_fasta=args.label_fasta,
            features_dir=args.features_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            window_batch_size=args.window_batch_size,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
