#!/usr/bin/env python3
"""Score one checkpoint on the two paper scenarios.

Scene 1: 31-protein CAID3 linker test (linker versus every other residue).
Scene 2: 62-protein set (linker versus other disordered residues).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cliper.embeddings import extract_prostt5_embeddings_for_fasta
from cliper.pipeline import evaluate


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
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

    summary = {
        "checkpoint": str(args.checkpoint),
        "scene1_auroc": scene1["metrics"].get("auroc_t1", scene1["metrics"].get("auroc")),
        "scene1_residues": scene1["metrics"].get("num_scored_residues"),
        "scene2_auroc": scene2["metrics"].get("auroc_t2"),
        "scene2_residues": scene2["metrics"].get("num_scored_residues_t2"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
