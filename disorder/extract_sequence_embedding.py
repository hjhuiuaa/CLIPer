"""CLI: extract one protein sequence to a single .resfeat.txt file."""

from __future__ import annotations

import argparse
import json

from disorder.sequence_embedding import extract_sequence_embedding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ProstT5 residue embeddings for one protein. "
            "Short sequences (<= window_size) use one forward pass; "
            "longer sequences use non-overlapping windows merged to [L, D]."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Directory for <protein_id>.resfeat.txt")
    parser.add_argument("--backbone", required=True, help="HF model id or path (e.g. Rostlab/ProstT5).")
    parser.add_argument("--protein-id", default=None, help="Protein id (required with --sequence if --fasta omitted).")
    parser.add_argument("--sequence", default=None, help="Amino-acid sequence string.")
    parser.add_argument("--fasta", default=None, help="2-line FASTA with exactly one record (alternative to --protein-id/--sequence).")
    parser.add_argument("--window-size", type=int, default=1024, help="Window size for long-sequence encoding.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for long-sequence window encoding.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output file exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fasta is None and (args.protein_id is None or args.sequence is None):
        raise SystemExit("Provide --fasta or both --protein-id and --sequence.")

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
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
