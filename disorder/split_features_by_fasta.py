"""Split extracted feature files into train/val folders by FASTA ids."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from disorder.fasta_parsing import parse_two_line_fasta
from disorder.feature_io import FEATURE_FILE_SUFFIX, safe_feature_stem


def _ids_from_fasta(path: str | Path) -> set[str]:
    return {protein_id for protein_id, _ in parse_two_line_fasta(path)}


def _feature_name_for_id(protein_id: str) -> str:
    return f"{safe_feature_stem(protein_id)}{FEATURE_FILE_SUFFIX}"


def split_features_by_fasta(
    *,
    source_dir: str | Path,
    train_fasta: str | Path,
    val_fasta: str | Path,
    train_out: str | Path,
    val_out: str | Path,
    mode: str = "copy",
) -> dict[str, int]:
    src = Path(source_dir)
    train_dir = Path(train_out)
    val_dir = Path(val_out)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_ids = _ids_from_fasta(train_fasta)
    val_ids = _ids_from_fasta(val_fasta)
    overlap = sorted(train_ids & val_ids)
    if overlap:
        sample = ", ".join(overlap[:10])
        raise ValueError(f"train/val FASTA share duplicate ids: {sample}")

    train_names = {_feature_name_for_id(pid) for pid in train_ids}
    val_names = {_feature_name_for_id(pid) for pid in val_ids}

    copied_train = 0
    copied_val = 0
    missing_train = 0
    missing_val = 0

    op = shutil.move if mode == "move" else shutil.copy2

    for filename in sorted(train_names):
        src_path = src / filename
        if not src_path.exists():
            missing_train += 1
            continue
        op(src_path, train_dir / filename)
        copied_train += 1

    for filename in sorted(val_names):
        src_path = src / filename
        if not src_path.exists():
            missing_val += 1
            continue
        op(src_path, val_dir / filename)
        copied_val += 1

    all_known = train_names | val_names
    extra_in_source = 0
    for path in src.glob(f"*{FEATURE_FILE_SUFFIX}"):
        if path.name not in all_known:
            extra_in_source += 1

    return {
        "train_ids": len(train_ids),
        "val_ids": len(val_ids),
        "copied_train": copied_train,
        "copied_val": copied_val,
        "missing_train": missing_train,
        "missing_val": missing_val,
        "extra_in_source": extra_in_source,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split .resfeat.txt files into train/val folders by FASTA ids.")
    parser.add_argument("--source-dir", required=True, help="Directory containing mixed *.resfeat.txt files.")
    parser.add_argument("--train-fasta", required=True, help="Train split FASTA (2-line format).")
    parser.add_argument("--val-fasta", required=True, help="Val split FASTA (2-line format).")
    parser.add_argument("--train-out", required=True, help="Output directory for train feature files.")
    parser.add_argument("--val-out", required=True, help="Output directory for val feature files.")
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="copy: keep source untouched; move: remove from source after placing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = split_features_by_fasta(
        source_dir=args.source_dir,
        train_fasta=args.train_fasta,
        val_fasta=args.val_fasta,
        train_out=args.train_out,
        val_out=args.val_out,
        mode=args.mode,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
