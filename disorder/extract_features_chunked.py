"""Extract ProstT5 residue embeddings as chunked feature files.

Output naming:
- <protein_id>_1.resfeat.txt
- <protein_id>_2.resfeat.txt
- ...

Each chunk is up to `window_size` residues, with overlap controlled by
`window_overlap` (stride = window_size - window_overlap).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from cliper.modeling import encode_sequences, load_backbone_and_tokenizer
from cliper.windowing import normalize_sequence
from disorder.data import parse_two_line_fasta
from disorder.feature_io import FEATURE_FILE_SUFFIX, write_feature_manifest, write_residue_feature_file
from disorder.windowing import build_sliding_eval_starts


def _autocast_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def _chunk_name(protein_id: str, chunk_index_1based: int) -> str:
    return f"{protein_id}_{chunk_index_1based}{FEATURE_FILE_SUFFIX}"


@torch.inference_mode()
def extract_chunked_features_for_fasta(
    *,
    sequence_fasta: str | Path,
    output_dir: str | Path,
    backbone_name: str,
    window_size: int = 1024,
    window_overlap: int = 256,
    batch_size: int = 2,
    device: str = "cuda",
    mixed_precision: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_two_line_fasta(sequence_fasta)
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = backbone.to(dev)
    backbone.eval()
    use_amp = mixed_precision and dev.type == "cuda"

    stride = max(1, int(window_size) - int(window_overlap))
    index_rows: list[dict[str, Any]] = []
    total_chunks = 0
    files_written = 0
    files_skipped = 0

    for protein_id, raw_seq in pairs:
        sequence = normalize_sequence(raw_seq)
        starts = build_sliding_eval_starts(len(sequence), int(window_size), stride)
        windows = [sequence[s : s + int(window_size)] for s in starts]
        total_chunks += len(windows)

        for chunk_start in range(0, len(windows), max(1, int(batch_size))):
            starts_batch = starts[chunk_start : chunk_start + int(batch_size)]
            windows_batch = windows[chunk_start : chunk_start + int(batch_size)]
            chunk_indices = list(range(chunk_start + 1, chunk_start + len(windows_batch) + 1))
            target_paths = [out_dir / _chunk_name(protein_id, i) for i in chunk_indices]

            todo_idx = [i for i, p in enumerate(target_paths) if overwrite or not p.exists()]
            if not todo_idx:
                files_skipped += len(target_paths)
                for local_i, s in enumerate(starts_batch):
                    idx1 = chunk_indices[local_i]
                    chunk_len = len(windows_batch[local_i])
                    index_rows.append(
                        {
                            "chunk_id": f"{protein_id}_{idx1}",
                            "protein_id": protein_id,
                            "chunk_index_1based": idx1,
                            "start": int(s),
                            "end_exclusive": int(s + chunk_len),
                            "length": int(chunk_len),
                            "file": _chunk_name(protein_id, idx1),
                            "status": "skipped_exists",
                        }
                    )
                continue

            todo_windows = [windows_batch[i] for i in todo_idx]
            todo_starts = [starts_batch[i] for i in todo_idx]
            todo_indices = [chunk_indices[i] for i in todo_idx]
            todo_paths = [target_paths[i] for i in todo_idx]

            encoded = encode_sequences(tokenizer, backbone_name, todo_windows)
            input_ids = encoded.input_ids.to(dev)
            attention_mask = encoded.attention_mask.to(dev)
            residue_lengths = encoded.residue_lengths

            with _autocast_context(device_type=dev.type, enabled=use_amp):
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state

            for row, length in enumerate(residue_lengths):
                vec = hidden[row, :length, :].detach().float().cpu()
                write_residue_feature_file(todo_paths[row], vec)
                files_written += 1
                s = int(todo_starts[row])
                idx1 = int(todo_indices[row])
                index_rows.append(
                    {
                        "chunk_id": f"{protein_id}_{idx1}",
                        "protein_id": protein_id,
                        "chunk_index_1based": idx1,
                        "start": s,
                        "end_exclusive": s + int(length),
                        "length": int(length),
                        "file": _chunk_name(protein_id, idx1),
                        "status": "written",
                    }
                )

            files_skipped += len(target_paths) - len(todo_idx)

    manifest_path = write_feature_manifest(
        out_dir,
        hidden_size=hidden_size,
        backbone_name=backbone_name,
        extra={
            "chunked": True,
            "window_size": int(window_size),
            "window_overlap": int(window_overlap),
            "stride": int(stride),
            "proteins_in_fasta": len(pairs),
            "total_chunks": total_chunks,
        },
    )
    index_path = out_dir / "chunk_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "output_dir": str(out_dir.resolve()),
        "proteins_in_fasta": len(pairs),
        "total_chunks": total_chunks,
        "files_written": files_written,
        "files_skipped": files_skipped,
        "hidden_size": int(hidden_size),
        "manifest_path": str(manifest_path),
        "chunk_index_path": str(index_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract chunked ProstT5 features: <protein_id>_<idx>.resfeat.txt")
    parser.add_argument("--fasta", required=True, help="2-line FASTA (>id, sequence).")
    parser.add_argument("--output-dir", required=True, help="Directory for chunked feature files.")
    parser.add_argument("--backbone", required=True, help="HF model id/path, e.g. Rostlab-ProstT5.")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--window-overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing chunk feature files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = extract_chunked_features_for_fasta(
        sequence_fasta=args.fasta,
        output_dir=args.output_dir,
        backbone_name=args.backbone,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        batch_size=args.batch_size,
        device=args.device,
        mixed_precision=not args.no_amp,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

