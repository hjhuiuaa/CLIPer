"""Check the released paper checkpoint and, optionally, predict one example protein.

The classifier weights are the GitHub release ``paper-supcon``. ProstT5 is
downloaded from Hugging Face only when prediction is requested.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/hjhuiuaa/CLIPer/releases/download/paper-supcon/best.pt"
PAPER_THRESHOLD = 0.946


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download_checkpoint(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    print(f"Downloading classifier checkpoint to {destination}")
    urllib.request.urlretrieve(RELEASE_URL, destination)


def checkpoint_summary(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu")
    config = dict(payload.get("config", {}))
    head = dict(config.get("classifier_head", {}))
    contrastive = dict(config.get("contrastive", {}))
    return {
        "checkpoint": str(path),
        "classifier_head": head.get("type"),
        "conv_channels": head.get("conv_channels"),
        "kernel_size": head.get("kernel_size"),
        "rnn_type": head.get("rnn_type"),
        "rnn_hidden_size": head.get("rnn_hidden_size"),
        "rnn_num_layers": head.get("rnn_num_layers"),
        "bidirectional": head.get("bidirectional"),
        "linker_class": head.get("linker_class"),
        "num_classes": head.get("num_classes"),
        "contrastive_mode": contrastive.get("mode"),
        "contrastive_weight": contrastive.get("weight"),
        "stored_threshold": payload.get("threshold"),
        "paper_operating_threshold": PAPER_THRESHOLD,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backbone", default="Rostlab/ProstT5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=PAPER_THRESHOLD)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load the classifier checkpoint and print its architecture. Does not download ProstT5.",
    )
    args = parser.parse_args()

    root = repo_root()
    checkpoint = args.checkpoint or (root / "artifacts" / "paper" / "best.pt")
    download_checkpoint(checkpoint)
    summary = checkpoint_summary(checkpoint)
    print(json.dumps(summary, indent=2))
    if args.check_only:
        return 0

    from cliper.caid_runall import runall_caid

    fasta = args.fasta or (root / "examples" / "example.fasta")
    output_dir = args.output_dir or (root / "artifacts" / "paper_example")
    result = runall_caid(
        checkpoint_path=checkpoint,
        fasta_path=fasta,
        output_dir=output_dir,
        backbone_name=args.backbone,
        threshold=args.threshold,
        device=args.device,
        num_threads=4,
        keep_embeddings=False,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
