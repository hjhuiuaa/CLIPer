"""Re-extract long-sequence features from FASTA, then write chunk files.

Why this script exists:
- Existing `<protein_id>.resfeat.txt` may already be clipped/truncated by a previous extraction.
- To avoid losing residues, identify long sequences from original FASTA and re-encode those sequences.

Behavior:
- For sequence length <= window_size: keep existing files unchanged.
- For sequence length > window_size:
  - split sequence with disorder sliding starts (same logic as training/eval coverage)
  - run backbone on each window
  - write `<protein_id>_1.resfeat.txt`, `<protein_id>_2.resfeat.txt`, ...
  - remove `<protein_id>.resfeat.txt` by default (replace mode)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from cliper.modeling import encode_sequences, load_backbone_and_tokenizer
from cliper.windowing import normalize_sequence
from disorder.data import parse_two_line_fasta
from disorder.feature_io import FEATURE_FILE_SUFFIX, safe_feature_stem, write_residue_feature_file
from disorder.windowing import build_sliding_eval_starts


def _autocast_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def _chunk_path(parent: Path, protein_id: str, chunk_idx_1based: int) -> Path:
    stem = safe_feature_stem(protein_id)
    return parent / f"{stem}_{chunk_idx_1based}{FEATURE_FILE_SUFFIX}"


def _base_feature_path(parent: Path, protein_id: str) -> Path:
    stem = safe_feature_stem(protein_id)
    return parent / f"{stem}{FEATURE_FILE_SUFFIX}"


def _remove_existing_chunk_files(parent: Path, protein_id: str) -> int:
    stem = safe_feature_stem(protein_id)
    removed = 0
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(FEATURE_FILE_SUFFIX)}$")
    for p in parent.glob(f"{stem}_*{FEATURE_FILE_SUFFIX}"):
        if pattern.match(p.name):
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def chunk_feature_directory(
    *,
    fasta_path: str | Path,
    features_dir: str | Path,
    backbone_name: str,
    window_size: int = 1024,
    window_overlap: int = 256,
    batch_size: int = 1,
    device: str = "cuda",
    mixed_precision: bool = True,
    overwrite: bool = False,
    replace_original: bool = True,
) -> dict[str, Any]:
    root = Path(features_dir)
    if not root.exists():
        raise FileNotFoundError(f"features directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"features path is not a directory: {root}")

    pairs = parse_two_line_fasta(fasta_path)
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = backbone.to(dev)
    backbone.eval()
    use_amp = mixed_precision and dev.type == "cuda"

    stride = max(1, int(window_size) - int(window_overlap))
    written_chunks = 0
    skipped_chunks = 0
    replaced_files = 0
    removed_old_chunks = 0
    kept_short_files = 0
    processed_long_files = 0
    index_rows: list[dict[str, Any]] = []

    for protein_id, seq_raw in pairs:
        sequence = normalize_sequence(seq_raw)
        length = len(sequence)
        if length <= int(window_size):
            kept_short_files += 1
            continue

        processed_long_files += 1
        if overwrite:
            removed_old_chunks += _remove_existing_chunk_files(root, protein_id)

        starts = build_sliding_eval_starts(length, int(window_size), stride)
        chunk_windows = [sequence[s : s + int(window_size)] for s in starts]

        for batch_start in range(0, len(chunk_windows), max(1, int(batch_size))):
            sub_windows = chunk_windows[batch_start : batch_start + int(batch_size)]
            sub_starts = starts[batch_start : batch_start + int(batch_size)]
            sub_indices = list(range(batch_start + 1, batch_start + len(sub_windows) + 1))
            sub_paths = [_chunk_path(root, protein_id, idx) for idx in sub_indices]
            todo = [i for i, p in enumerate(sub_paths) if overwrite or not p.exists()]

            if not todo:
                skipped_chunks += len(sub_paths)
                for i, s in enumerate(sub_starts):
                    idx = sub_indices[i]
                    seg_len = len(sub_windows[i])
                    index_rows.append(
                        {
                            "protein_id": protein_id,
                            "chunk_id": f"{protein_id}_{idx}",
                            "chunk_index_1based": idx,
                            "start": int(s),
                            "end_exclusive": int(s + seg_len),
                            "length": int(seg_len),
                            "file": sub_paths[i].name,
                            "status": "skipped_exists",
                        }
                    )
                continue

            todo_windows = [sub_windows[i] for i in todo]
            todo_starts = [sub_starts[i] for i in todo]
            todo_indices = [sub_indices[i] for i in todo]
            todo_paths = [sub_paths[i] for i in todo]

            encoded = encode_sequences(tokenizer, backbone_name, todo_windows)
            input_ids = encoded.input_ids.to(dev)
            attention_mask = encoded.attention_mask.to(dev)
            residue_lengths = encoded.residue_lengths

            with _autocast_context(device_type=dev.type, enabled=use_amp):
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state

            for row, seg_len in enumerate(residue_lengths):
                vec = hidden[row, :seg_len, :].detach().float().cpu()
                write_residue_feature_file(todo_paths[row], vec)
                written_chunks += 1
                s = int(todo_starts[row])
                idx = int(todo_indices[row])
                index_rows.append(
                    {
                        "protein_id": protein_id,
                        "chunk_id": f"{protein_id}_{idx}",
                        "chunk_index_1based": idx,
                        "start": s,
                        "end_exclusive": int(s + seg_len),
                        "length": int(seg_len),
                        "file": todo_paths[row].name,
                        "status": "written",
                    }
                )

            skipped_chunks += len(sub_paths) - len(todo)

        if replace_original:
            base_path = _base_feature_path(root, protein_id)
            if base_path.exists():
                base_path.unlink(missing_ok=True)
                replaced_files += 1

    report = {
        "fasta_path": str(Path(fasta_path).resolve()),
        "features_dir": str(root.resolve()),
        "backbone_name": backbone_name,
        "hidden_size": int(hidden_size),
        "window_size": int(window_size),
        "window_overlap": int(window_overlap),
        "stride": int(stride),
        "proteins_in_fasta": len(pairs),
        "processed_long_files": processed_long_files,
        "kept_short_files": kept_short_files,
        "written_chunks": written_chunks,
        "skipped_chunks": skipped_chunks,
        "replaced_original_files": replaced_files,
        "removed_old_chunks": removed_old_chunks,
    }

    index_path = root / "chunk_index_existing.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["index_path"] = str(index_path.resolve())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-extract long sequences from FASTA and write _1/_2/... chunk feature files in-place."
    )
    parser.add_argument(
        "--fasta-features",
        nargs="+",
        required=True,
        help="Pairs in order: <fasta_path> <features_dir> [<fasta_path> <features_dir> ...]",
    )
    parser.add_argument("--backbone", required=True, help="HF model id/path, e.g. Rostlab-ProstT5.")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--window-overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing chunk files.")
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original long file (default is replace: delete original after chunking).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.fasta_features) % 2 != 0:
        raise ValueError("--fasta-features requires pairs: <fasta> <features_dir> ...")
    results: list[dict[str, Any]] = []
    for idx in range(0, len(args.fasta_features), 2):
        fasta_path = args.fasta_features[idx]
        features_dir = args.fasta_features[idx + 1]
        results.append(
            chunk_feature_directory(
                fasta_path=fasta_path,
                features_dir=features_dir,
                backbone_name=args.backbone,
                window_size=args.window_size,
                window_overlap=args.window_overlap,
                batch_size=args.batch_size,
                device=args.device,
                mixed_precision=not args.no_amp,
                overwrite=args.overwrite,
                replace_original=not args.keep_original,
            )
        )
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

