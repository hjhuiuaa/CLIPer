#!/usr/bin/env python3
"""Score one checkpoint on scene 1, scene 2, and scene 3.

Scene 1: original 31-protein CAID3 linker test (linker vs every other residue).
Scene 2: 62-protein disorder-PDB set (linker vs other disordered residues).
Scene 3: same 62 proteins, protein-level sliding-window score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cliper.extract_embeddings import extract_prostt5_embeddings_for_fasta
from cliper.pipeline import evaluate
from cliper.scene3 import (
    load_prediction_probs,
    median_linker_run_length,
    protein_auroc,
    score_proteins,
)


def _parse_labels(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(">"):
            if header is not None:
                records.append((header, lines[1]))
            header = raw[1:].split()[0]
            lines = []
        elif header is not None:
            lines.append(raw.strip())
    if header is not None:
        records.append((header, lines[1]))
    return records


def _ensure_embeddings(checkpoint: dict, sequences_fasta: Path, device: str) -> None:
    config = checkpoint["config"]
    embeddings_dir = config.get("embeddings_dir")
    if not embeddings_dir:
        return
    backbone = str(config.get("backbone_name"))
    extract_prostt5_embeddings_for_fasta(
        fasta_path=sequences_fasta,
        output_dir=embeddings_dir,
        backbone_name=backbone,
        window_size=int(config.get("window_size", 1024)),
        batch_size=1,
        device=device,
        mixed_precision=(device == "cuda"),
        overwrite=False,
        output_format="npy",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--scene1-fasta",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "test_caid3.fasta",
    )
    parser.add_argument(
        "--scene2-fasta",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "test_scene2_pdb62.fasta",
    )
    parser.add_argument(
        "--sequences-fasta",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "sequences_scene2_pdb62.fasta",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "scene2_pdb62_selection.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=None, help="Scene-3 window. Default: median linker length.")
    args = parser.parse_args()

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    eval_checkpoint = args.checkpoint
    trained_device = str(checkpoint["config"].get("device", "cuda"))
    if args.device != trained_device:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        eval_checkpoint = args.output_dir / "checkpoint_device.pt"
        checkpoint["config"]["device"] = args.device
        torch.save(checkpoint, eval_checkpoint)

    _ensure_embeddings(checkpoint, args.sequences_fasta, args.device)

    scene1_dir = args.output_dir / "scene1"
    scene2_dir = args.output_dir / "scene2"
    scene1 = evaluate(
        checkpoint_path=eval_checkpoint,
        fasta_path=args.scene1_fasta,
        output_dir=scene1_dir,
    )
    scene2 = evaluate(
        checkpoint_path=eval_checkpoint,
        fasta_path=args.scene2_fasta,
        output_dir=scene2_dir,
    )

    labels = _parse_labels(args.scene2_fasta)
    window = args.window
    if window is None and args.selection_json.is_file():
        window = int(json.loads(args.selection_json.read_text(encoding="utf-8"))["scene3_window"])
    if window is None:
        window = median_linker_run_length(item[1] for item in labels if "2" in item[1])
    probs = load_prediction_probs(scene2_dir / "predictions.tsv")
    rows = score_proteins(labels, probs, window=window)
    auroc = protein_auroc(rows)
    n_pos = sum(int(row["label"]) for row in rows)
    scene3 = {
        "protocol": "protein contains a linker; score = max sliding-window mean",
        "window": int(window),
        "num_proteins": len(rows),
        "num_positives": n_pos,
        "num_negatives": len(rows) - n_pos,
        "auroc": auroc,
    }
    scene3_dir = args.output_dir / "scene3"
    scene3_dir.mkdir(parents=True, exist_ok=True)
    (scene3_dir / "metrics.json").write_text(json.dumps(scene3, indent=2) + "\n", encoding="utf-8")
    with (scene3_dir / "proteins.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("protein_id\tlabel\tscore\tlength\twindow\n")
        for row in rows:
            handle.write(
                f"{row['protein_id']}\t{row['label']}\t{float(row['score']):.8f}\t{row['length']}\t{row['window']}\n"
            )

    summary = {
        "checkpoint": str(args.checkpoint),
        "scene1_auroc": scene1["metrics"].get("auroc_t1", scene1["metrics"].get("auroc")),
        "scene1_residues": scene1["metrics"].get("num_scored_residues"),
        "scene2_auroc": scene2["metrics"].get("auroc_t2"),
        "scene2_residues": scene2["metrics"].get("num_scored_residues_t2"),
        "scene3_auroc": auroc,
        "scene3_window": int(window),
        "scene3_proteins": len(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
