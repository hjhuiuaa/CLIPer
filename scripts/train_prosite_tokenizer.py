from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliper.data import build_motif_special_tokens, load_motif_specs, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Stage 5 tokenizer from an existing ProstT5 tokenizer by adding "
            "PROSITE motif special tokens without reassigning base token ids."
        )
    )
    parser.add_argument("--base-tokenizer", required=True, help="Base tokenizer name/path.")
    parser.add_argument("--motif-json", required=True, help="PROSITE motif library JSON.")
    parser.add_argument("--out-dir", default="configs/prosite_tokenizer", help="Output tokenizer directory.")
    parser.add_argument("--matching", default="prosite", choices=["exact", "regex", "degenerate", "prosite"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to train the PROSITE tokenizer.") from exc

    motif_specs = load_motif_specs(args.motif_json, matching=args.matching)
    special_tokens = build_motif_special_tokens(motif_specs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, do_lower_case=False)
    base_vocab_size = len(tokenizer)
    added_count = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    write_json(
        out_dir / "prosite_tokenizer_metadata.json",
        {
            "base_tokenizer": args.base_tokenizer,
            "motif_json": str(Path(args.motif_json)),
            "matching": args.matching,
            "num_motifs": len(motif_specs),
            "num_special_tokens": len(special_tokens),
            "base_vocab_size": base_vocab_size,
            "added_count": int(added_count),
            "final_vocab_size": len(tokenizer),
            "strategy": "clone_base_tokenizer_add_prosite_special_tokens",
            "base_token_ids_preserved": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
