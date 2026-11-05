"""Check the released paper checkpoint and, optionally, predict one example protein.

The classifier weights are the GitHub release ``Pretrained-Weights``. ProstT5 is
downloaded from Hugging Face only when prediction is requested.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/hjhuiuaa/CLIPer/releases/download/Pretrained-Weights/best.pt"
PAPER_THRESHOLD = 0.946


def repo_root() -> Path:
    return Path(__file__).resolve().parent


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

    from cliper.embeddings import encode_residue_embeddings, parse_fasta
    from cliper.infer import load_classifier_from_checkpoint, predict_sequence_from_embedding
    from cliper.model import load_backbone_and_tokenizer
    import torch

    fasta = args.fasta or (root / "examples" / "example.fasta")
    output_dir = args.output_dir or (root / "artifacts" / "paper_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    classifier, metadata = load_classifier_from_checkpoint(checkpoint, device=device)
    config = metadata["config"]
    head = dict(config.get("classifier_head", {}))
    backbone, tokenizer, _hidden = load_backbone_and_tokenizer(args.backbone)
    backbone = backbone.to(device)
    records = parse_fasta(fasta)
    rows: list[str] = ["protein_id\tposition\tresidue\tprobability\tcall"]
    for protein_id, sequence in records:
        embedding = encode_residue_embeddings(
            sequence=sequence,
            backbone=backbone,
            tokenizer=tokenizer,
            backbone_name=args.backbone,
            window_size=int(config.get("window_size", 1024)),
            batch_size=1,
            device=device,
            mixed_precision=device.type == "cuda",
        )
        probabilities = predict_sequence_from_embedding(
            classifier,
            sequence=sequence,
            embedding=embedding,
            local_context=metadata["local_context"],
            window_size=int(config.get("window_size", 1024)),
            eval_stride=config.get("eval_stride"),
            top_k_heuristic=int(config.get("top_k_heuristic", 4)),
            device=device,
            window_batch_size=8,
            linker_class=int(head.get("linker_class", 2)),
            dual_binary=bool(head.get("dual_binary", False)),
        )
        for index, (residue, probability) in enumerate(zip(sequence, probabilities), start=1):
            call = 1 if probability >= args.threshold else 0
            rows.append(f"{protein_id}\t{index}\t{residue}\t{probability:.6f}\t{call}")
    prediction_path = output_dir / "predictions.tsv"
    prediction_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(prediction_path), "sequences": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
