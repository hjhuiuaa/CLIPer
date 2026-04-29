from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliper.data import parse_prosite_dat, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a PROSITE PATTERN motif library JSON from prosite.dat.")
    parser.add_argument("--prosite-dat", default="prosite.dat", help="Path to PROSITE prosite.dat.")
    parser.add_argument(
        "--out-json",
        default="configs/prosite_motif_library_full.json",
        help="Output motif JSON path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.prosite_dat)
    motifs = parse_prosite_dat(source)
    write_json(
        args.out_json,
        {
            "source": str(source),
            "format": "prosite_pattern_motif_library",
            "motifs": motifs,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
