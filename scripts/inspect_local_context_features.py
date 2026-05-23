from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cliper.data import ProteinRecord, parse_three_line_fasta
from cliper.modeling import _concat_local_window


@dataclass(frozen=True)
class SeqFeatureStats:
    protein_id: str
    seq_len: int
    base_dim: int
    expected_span: int
    expected_dim: int
    actual_dim: int
    total_scalar_features: int
    dim_match: bool


def _parse_two_line_fasta(path: Path) -> list[ProteinRecord]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"2-line FASTA expects even number of non-empty lines, got {len(lines)} in {path}")
    records: list[ProteinRecord] = []
    seen: set[str] = set()
    for idx in range(0, len(lines), 2):
        header = lines[idx]
        seq = lines[idx + 1].upper()
        if not header.startswith(">"):
            raise ValueError(f"Malformed FASTA header at line {idx + 1}: {header!r}")
        pid = header[1:].strip()
        if not pid:
            raise ValueError(f"Empty protein id at line {idx + 1}")
        if pid in seen:
            raise ValueError(f"Duplicate protein id in FASTA: {pid}")
        seen.add(pid)
        records.append(ProteinRecord(protein_id=pid, sequence=seq, labels="-" * len(seq)))
    return records


def _load_records_auto(path: Path) -> list[ProteinRecord]:
    try:
        return parse_three_line_fasta(path)
    except Exception:
        return _parse_two_line_fasta(path)


def _resolve_local_context(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    local_context = raw.get("local_context", {})
    if local_context is None:
        local_context = {}
    if not isinstance(local_context, dict):
        raise ValueError("local_context must be a dict when present.")
    enabled = bool(local_context.get("enabled", False))
    radius = int(local_context.get("radius", 2))
    mode = str(local_context.get("mode", "concat_window")).lower()
    include_self = bool(local_context.get("include_self", True))
    if radius < 0:
        raise ValueError(f"local_context.radius must be >= 0, got {radius}")
    if mode != "concat_window":
        raise ValueError(
            "This inspector currently supports local_context.mode=concat_window only, "
            f"got {mode!r}"
        )
    return {
        "enabled": enabled,
        "radius": radius,
        "mode": mode,
        "include_self": include_self,
    }


def _expected_span(local_context: dict[str, Any]) -> int:
    if not bool(local_context.get("enabled", False)):
        return 1
    radius = int(local_context.get("radius", 2))
    if radius <= 0:
        return 1
    include_self = bool(local_context.get("include_self", True))
    return (2 * radius + 1) if include_self else (2 * radius)


def _inspect_one(
    record: ProteinRecord,
    *,
    base_hidden: int,
    local_context: dict[str, Any],
) -> SeqFeatureStats:
    seq_len = len(record.sequence)
    span = _expected_span(local_context)
    expected_dim = base_hidden * span
    dummy = torch.randn(1, seq_len, base_hidden)
    expanded = _concat_local_window(dummy, [seq_len], local_context)
    actual_dim = int(expanded.shape[-1])
    return SeqFeatureStats(
        protein_id=record.protein_id,
        seq_len=seq_len,
        base_dim=base_hidden,
        expected_span=span,
        expected_dim=expected_dim,
        actual_dim=actual_dim,
        total_scalar_features=seq_len * actual_dim,
        dim_match=(expected_dim == actual_dim),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect per-sequence feature counts after local_context concat_window expansion."
    )
    parser.add_argument("--config", required=True, help="Training YAML with local_context.")
    parser.add_argument("--fasta", required=True, help="2-line or 3-line FASTA file.")
    parser.add_argument(
        "--base-hidden",
        type=int,
        default=1024,
        help="Backbone residue embedding dim before expansion (ProstT5 default is usually 1024).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Print first N sequence rows (full summary is still written to JSON if --output-json is set).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save full per-sequence stats JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    fasta_path = Path(args.fasta)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    if int(args.base_hidden) <= 0:
        raise ValueError(f"--base-hidden must be > 0, got {args.base_hidden}")

    local_context = _resolve_local_context(config_path)
    records = _load_records_auto(fasta_path)
    stats = [
        _inspect_one(
            rec,
            base_hidden=int(args.base_hidden),
            local_context=local_context,
        )
        for rec in records
    ]

    total_seq = len(stats)
    matched = sum(1 for row in stats if row.dim_match)
    total_residues = sum(row.seq_len for row in stats)
    total_features = sum(row.total_scalar_features for row in stats)
    span = _expected_span(local_context)
    expected_dim = int(args.base_hidden) * span

    print("=== Local Context Feature Inspection ===")
    print(f"config: {config_path}")
    print(f"fasta: {fasta_path}")
    print(f"local_context: {json.dumps(local_context, ensure_ascii=False)}")
    print(f"base_hidden: {int(args.base_hidden)}")
    print(f"expected_span: {span}")
    print(f"expected_expanded_dim_per_residue: {expected_dim}")
    print(f"sequences: {total_seq}, residues_total: {total_residues}, scalar_features_total: {total_features}")
    print(f"dimension_match_rows: {matched}/{total_seq}")
    print("")
    print("protein_id\tseq_len\texpected_dim\tactual_dim\ttotal_scalar_features")
    for row in stats[: max(0, int(args.limit))]:
        print(
            f"{row.protein_id}\t{row.seq_len}\t{row.expected_dim}\t"
            f"{row.actual_dim}\t{row.total_scalar_features}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_path),
            "fasta": str(fasta_path),
            "local_context": local_context,
            "base_hidden": int(args.base_hidden),
            "expected_span": span,
            "expected_expanded_dim_per_residue": expected_dim,
            "summary": {
                "sequences": total_seq,
                "residues_total": total_residues,
                "scalar_features_total": total_features,
                "dimension_match_rows": matched,
            },
            "rows": [row.__dict__ for row in stats],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
