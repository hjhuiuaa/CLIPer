from __future__ import annotations

import argparse
import json

from .caid_predict import predict_caid
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

    train_parser = subparsers.add_parser("train", help="Train CLIPer residue classifier (Stage 1/2/3/4/5/6).")
    train_parser.add_argument("--config", required=True, help="YAML config path.")
    train_parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional checkpoint path to resume training state.",
    )

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

    predict_parser = subparsers.add_parser(
        "predict",
        help="CAID submission predict: 2-line FASTA + precomputed embeddings -> .caid + timings.csv",
    )
    predict_parser.add_argument("--checkpoint", required=True, help="Path to trained stage4 checkpoint (.pt).")
    predict_parser.add_argument("--fasta", required=True, help="Input FASTA (2-line records, no labels).")
    predict_parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory with one embedding file per protein (.npy, .h5, or .resfeat.txt).",
    )
    predict_parser.add_argument("--output-dir", required=True, help="Output directory for CAID files.")
    predict_parser.add_argument(
        "--flavor",
        default="linker",
        help="CAID output flavor subdirectory (default: linker).",
    )
    predict_parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override.")
    predict_parser.add_argument("--window-size", type=int, default=None, help="Eval window size override.")
    predict_parser.add_argument("--eval-stride", type=int, default=None, help="Eval stride override.")
    predict_parser.add_argument("--top-k-heuristic", type=int, default=None, help="Heuristic window count override.")
    predict_parser.add_argument("--window-batch-size", type=int, default=8, help="Window batch size for inference.")
    predict_parser.add_argument("--device", default="cpu", help="Inference device (CAID requires cpu).")
    predict_parser.add_argument("--num-threads", type=int, default=4, help="CPU threads (max 24).")

    emb_parser = subparsers.add_parser(
        "extract_embeddings",
        help="Linker/CAID: export ProstT5 embeddings (one file per protein) for predict.",
    )
    emb_parser.add_argument(
        "--fasta",
        default=None,
        help="2-line FASTA with one or many records (or use --protein-id + --sequence).",
    )
    emb_parser.add_argument("--protein-id", default=None, help="Single protein id (with --sequence).")
    emb_parser.add_argument("--sequence", default=None, help="Single protein sequence string.")
    emb_parser.add_argument("--output-dir", required=True, help="Directory for embedding files + manifest.json.")
    emb_parser.add_argument("--backbone", required=True, help="HF model id or path (e.g. Rostlab/ProstT5).")
    emb_parser.add_argument("--window-size", type=int, default=1024, help="Window size for long-sequence encoding.")
    emb_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for long-sequence window encoding.",
    )
    emb_parser.add_argument("--device", default="cuda")
    emb_parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    emb_parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists.")
    emb_parser.add_argument(
        "--format",
        choices=("resfeat", "npy"),
        default="resfeat",
        help="Output format: .resfeat.txt (default) or .npy.",
    )

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
        result = train(
            config_path=args.config,
            resume_checkpoint=args.resume_checkpoint,
        )
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
    elif args.command == "predict":
        result = predict_caid(
            checkpoint_path=args.checkpoint,
            fasta_path=args.fasta,
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            flavor=args.flavor,
            threshold=args.threshold,
            window_size=args.window_size,
            eval_stride=args.eval_stride,
            top_k_heuristic=args.top_k_heuristic,
            window_batch_size=args.window_batch_size,
            device=args.device,
            num_threads=args.num_threads,
        )
    elif args.command == "extract_embeddings":
        from cliper.extract_embeddings import (
            extract_prostt5_embedding_for_sequence,
            extract_prostt5_embeddings_for_fasta,
        )

        if args.fasta is not None:
            result = extract_prostt5_embeddings_for_fasta(
                fasta_path=args.fasta,
                output_dir=args.output_dir,
                backbone_name=args.backbone,
                window_size=args.window_size,
                batch_size=args.batch_size,
                device=args.device,
                mixed_precision=not args.no_amp,
                overwrite=args.overwrite,
                output_format=args.format,
            )
        elif args.protein_id is not None and args.sequence is not None:
            result = extract_prostt5_embedding_for_sequence(
                protein_id=args.protein_id,
                sequence=args.sequence,
                output_dir=args.output_dir,
                backbone_name=args.backbone,
                window_size=args.window_size,
                batch_size=args.batch_size,
                device=args.device,
                mixed_precision=not args.no_amp,
                overwrite=args.overwrite,
                output_format=args.format,
            )
        else:
            parser.error("extract_embeddings requires --fasta or both --protein-id and --sequence.")
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
