"""Re-extract long proteins as non-overlapping 1024 windows, merge to one .resfeat.txt per protein.

Finds files named ``<stem>_1.resfeat.txt``, ``<stem>_2.resfeat.txt``, ... in a features directory,
loads the full sequence from the paired 2-line FASTA, runs the backbone on windows
``[0:W), [W:2W), ...`` (no overlap; last window may be shorter), concatenates residue rows in order,
writes ``<stem>.resfeat.txt``, and deletes all ``<stem>_<k>.resfeat.txt`` chunk files.

Short proteins (no chunk files) are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from cliper.modeling import encode_sequences, load_backbone_and_tokenizer
from cliper.windowing import normalize_sequence
from disorder.data import parse_two_line_fasta
from disorder.feature_io import FEATURE_FILE_SUFFIX, safe_feature_stem, write_residue_feature_file

_CHUNK_FILE_RE = re.compile(rf"^(.+)_(\d+){re.escape(FEATURE_FILE_SUFFIX)}$")


def _autocast_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def _non_overlapping_starts(length: int, window_size: int) -> list[int]:
    if length <= 0 or window_size <= 0:
        raise ValueError("length and window_size must be positive")
    return list(range(0, length, int(window_size)))


def _collect_chunk_groups(features_dir: Path) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for path in features_dir.glob(f"*{FEATURE_FILE_SUFFIX}"):
        m = _CHUNK_FILE_RE.match(path.name)
        if not m:
            continue
        stem, idx_s = m.group(1), m.group(2)
        groups[stem].append(int(idx_s))
    for stem in groups:
        groups[stem] = sorted(set(groups[stem]))
    return dict(groups)


def _build_stem_map(fasta_path: str | Path) -> dict[str, tuple[str, str]]:
    stem_map: dict[str, tuple[str, str]] = {}
    for protein_id, seq in parse_two_line_fasta(fasta_path):
        stem = safe_feature_stem(protein_id)
        if stem in stem_map:
            raise ValueError(f"Two FASTA ids sanitize to the same stem {stem!r}: collision.")
        stem_map[stem] = (protein_id, seq)
    return stem_map


@torch.inference_mode()
def reextract_merge_nonoverlap(
    *,
    fasta_path: str | Path,
    features_dir: str | Path,
    backbone_name: str,
    window_size: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    mixed_precision: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(features_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    stem_map = _build_stem_map(fasta_path)
    chunk_groups = _collect_chunk_groups(root)
    if not chunk_groups:
        return {
            "features_dir": str(root.resolve()),
            "message": "no chunk files (*_N.resfeat.txt) found; nothing to do",
            "merged": 0,
            "deleted_chunks": 0,
        }

    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = backbone.to(dev)
    backbone.eval()
    use_amp = mixed_precision and dev.type == "cuda"

    merged = 0
    deleted_chunks = 0
    report_rows: list[dict[str, Any]] = []

    for stem, chunk_indices in sorted(chunk_groups.items()):
        if stem not in stem_map:
            raise ValueError(
                f"Chunk files for stem {stem!r} found under {root}, but no matching FASTA entry "
                f"(after safe_feature_stem) in {fasta_path}."
            )
        protein_id, raw_seq = stem_map[stem]
        sequence = normalize_sequence(raw_seq)
        length = len(sequence)
        if length <= int(window_size) and len(chunk_indices) > 0:
            # Unexpected: chunks imply long seq; still re-merge from FASTA truth
            pass

        starts = _non_overlapping_starts(length, int(window_size))

        windows = [sequence[s : s + int(window_size)] for s in starts]
        ordered_vecs: list[torch.Tensor] = []

        for batch_start in range(0, len(windows), max(1, int(batch_size))):
            sub = windows[batch_start : batch_start + int(batch_size)]
            encoded = encode_sequences(tokenizer, backbone_name, sub)
            input_ids = encoded.input_ids.to(dev)
            attention_mask = encoded.attention_mask.to(dev)
            residue_lengths = encoded.residue_lengths
            with _autocast_context(device_type=dev.type, enabled=use_amp):
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            for row, seg_len in enumerate(residue_lengths):
                ordered_vecs.append(hidden[row, :seg_len, :].detach().float().cpu())

        full = torch.cat(ordered_vecs, dim=0)
        if int(full.shape[0]) != length:
            raise RuntimeError(
                f"Merged length mismatch for {protein_id}: emb_rows={full.shape[0]} seq_len={length}"
            )

        out_path = root / f"{stem}{FEATURE_FILE_SUFFIX}"

        if not dry_run:
            write_residue_feature_file(out_path, full)
            for cp in sorted(root.glob(f"{stem}_*{FEATURE_FILE_SUFFIX}")):
                if _CHUNK_FILE_RE.match(cp.name):
                    cp.unlink(missing_ok=True)
                    deleted_chunks += 1
            merged += 1

        report_rows.append(
            {
                "protein_id": protein_id,
                "stem": stem,
                "seq_len": length,
                "num_windows": len(starts),
                "old_chunk_indices": chunk_indices,
                "output": out_path.name,
                "dry_run": dry_run,
            }
        )

    summary = {
        "fasta_path": str(Path(fasta_path).resolve()),
        "features_dir": str(root.resolve()),
        "backbone_name": backbone_name,
        "hidden_size": int(hidden_size),
        "window_size": int(window_size),
        "merged_proteins": merged if not dry_run else 0,
        "deleted_chunk_files": deleted_chunks if not dry_run else 0,
        "chunk_stems_found": len(chunk_groups),
        "dry_run": dry_run,
    }
    index_path = root / "reextract_merge_nonoverlap_report.jsonl"
    if not dry_run:
        with index_path.open("w", encoding="utf-8") as handle:
            for row in report_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["report_path"] = str(index_path.resolve())
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-extract long proteins (chunk files) with non-overlapping windows and merge."
    )
    p.add_argument("--fasta", required=True, help="2-line FASTA for this split (train/val/test).")
    p.add_argument("--features-dir", required=True, help="Directory with .resfeat.txt and *_N.resfeat.txt")
    p.add_argument("--backbone", required=True)
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="List actions only; do not write/delete.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    result = reextract_merge_nonoverlap(
        fasta_path=args.fasta,
        features_dir=args.features_dir,
        backbone_name=args.backbone,
        window_size=args.window_size,
        batch_size=args.batch_size,
        device=args.device,
        mixed_precision=not args.no_amp,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
