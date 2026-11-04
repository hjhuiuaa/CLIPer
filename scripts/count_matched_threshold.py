"""Training-set threshold where predicted positives equal labeled linkers.

Uses full-sequence probabilities, the same 1024/512 window merge as evaluation.
The training split is dataset/t2_seqneg/train.fasta. Validation is not included.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cliper.caid_io import read_residue_embedding, sanitize_sequence
from cliper.caid_predict import load_classifier_from_checkpoint, predict_sequence_from_embedding

CHECKPOINT = ROOT / "artifacts/runs/ablation_shared_20261031/supcon/train/exp0001/checkpoints/best.pt"


def default_train_fasta() -> Path:
    candidates = [
        ROOT / "dataset" / "t2_seqneg" / "train.fasta",
        ROOT.parent / "dataset" / "t2_seqneg" / "train.fasta",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_embeddings() -> Path:
    candidates = [
        ROOT / "artifacts/runs/ablation_shared_20261031/supcon/embeddings",
        ROOT / "artifacts/runs/ablation_shared_20261031/binary/embeddings",
        ROOT / "artifacts/runs/ablation_no_contrastive/embeddings",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def read_train(path: Path) -> list[tuple[str, str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    index = 0
    while index < len(lines):
        protein_id = lines[index][1:].split()[0]
        sequence = lines[index + 1]
        labels = lines[index + 2]
        if len(sequence) != len(labels):
            raise ValueError(f"{protein_id} label length does not match the sequence")
        records.append((protein_id, sequence, labels))
        index += 3
    return records


def load_embedding(embeddings_dir: Path, protein_id: str) -> torch.Tensor:
    try:
        return read_residue_embedding(embeddings_dir, protein_id)
    except FileNotFoundError:
        text_path = embeddings_dir / f"{protein_id}.txt"
        if not text_path.exists():
            raise
        rows = np.loadtxt(text_path, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        return torch.from_numpy(rows)


def count_matched_threshold(probabilities: np.ndarray, positive_count: int) -> dict:
    if positive_count <= 0 or positive_count > len(probabilities):
        raise ValueError(f"positive count {positive_count} is outside 1..{len(probabilities)}")
    order = np.argpartition(probabilities, -positive_count)[-positive_count:]
    threshold = float(probabilities[order].min())
    greater = int(np.count_nonzero(probabilities > threshold))
    greater_equal = int(np.count_nonzero(probabilities >= threshold))
    return {
        "threshold": threshold,
        "labeled_positives": positive_count,
        "predicted_positives_strictly_above": greater,
        "predicted_positives_at_or_above": greater_equal,
        "tied_at_threshold": greater_equal - greater,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Training-set count-matched linker threshold.")
    parser.add_argument("--fasta", type=Path, default=default_train_fasta())
    parser.add_argument("--embeddings", type=Path, default=default_embeddings())
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "operating_threshold.json",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    records = read_train(args.fasta)
    device = torch.device("cpu")
    torch.set_num_threads(args.num_threads)
    result = {
        "model": "CLIPer",
        "rule": (
            "On the t2_seqneg training split, rank full-sequence P(linker) from CLIPer "
            "and take the Nth highest value, where N is the number of label-2 residues. "
            "A residue is positive when its probability is greater than or equal to this value. "
            "This one value is used for every scene. Validation and CAID3 are not used."
        ),
        "train_fasta": str(args.fasta),
        "embeddings": str(args.embeddings),
        "checkpoint": str(CHECKPOINT),
    }
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"CLIPer checkpoint missing at {CHECKPOINT}")
    print("loading CLIPer", flush=True)
    model, metadata = load_classifier_from_checkpoint(CHECKPOINT, device=device)
    config = metadata["config"]
    local_context = metadata["local_context"]
    window_size = int(config.get("window_size", 1024))
    stride = config.get("eval_stride")
    top_k = int(config.get("top_k_heuristic", 4))
    probabilities: list[float] = []
    positive_count = 0
    for index, (protein_id, sequence, labels) in enumerate(records, start=1):
        embedding = load_embedding(args.embeddings, protein_id)
        seq = sanitize_sequence(sequence)
        if embedding.shape[0] != len(seq):
            raise ValueError(
                f"{protein_id} embedding length {embedding.shape[0]} != sequence length {len(seq)}"
            )
        scores = predict_sequence_from_embedding(
            model,
            sequence=seq,
            embedding=embedding,
            local_context=local_context,
            window_size=window_size,
            eval_stride=stride,
            top_k_heuristic=top_k,
            device=device,
            window_batch_size=4,
        )
        for label, score in zip(labels, scores):
            if label == "2":
                positive_count += 1
            if label in {"0", "1", "2"}:
                probabilities.append(score)
        if index % 50 == 0 or index == len(records):
            print(f"  CLIPer {index}/{len(records)}", flush=True)
    summary = count_matched_threshold(np.asarray(probabilities, dtype=np.float64), positive_count)
    summary["previous_checkpoint_threshold"] = float(metadata["threshold"])
    summary["scored_residues"] = len(probabilities)
    result.update(summary)
    print(
        f"CLIPer threshold {summary['threshold']:.6f} "
        f"predicted>={summary['predicted_positives_at_or_above']} "
        f"labeled {positive_count}",
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
