"""Predict disorder from precomputed residue features with a trained CLIPer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from cliper.metrics import apply_threshold, binary_roc_auc, f1_score, mcc_score, precision_recall_auc, search_best_threshold
from cliper.windowing import sigmoid
from disorder.data import parse_three_line_fasta
from disorder.feature_io import FEATURE_FILE_SUFFIX, read_residue_feature_file, safe_feature_stem
from disorder.feature_modeling import DisorderFeatureClassifier


def _feature_path(features_dir: str | Path, protein_id: str) -> Path:
    return Path(features_dir) / f"{safe_feature_stem(protein_id)}{FEATURE_FILE_SUFFIX}"


def _extract_scored_residues(labels: str, probs: list[float]) -> tuple[list[int], list[float]]:
    y_true: list[int] = []
    y_prob: list[float] = []
    for char, prob in zip(labels, probs):
        if char == "1":
            y_true.append(1)
            y_prob.append(prob)
        elif char == "0":
            y_true.append(0)
            y_prob.append(prob)
    return y_true, y_prob


def _infer_hidden_size(state_dict: dict[str, Any]) -> int:
    for key in ("classifier.weight", "classifier.network.0.weight", "classifier.classifier.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:
            return int(tensor.shape[1])
    raise ValueError("Cannot infer hidden_size from checkpoint state_dict.")


def _load_classifier_only(model: DisorderFeatureClassifier, state_dict: dict[str, Any]) -> None:
    head = {k: v for k, v in state_dict.items() if k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(head, strict=False)
    bad_missing = [m for m in missing if not m.startswith("dropout.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"Classifier state mismatch: missing={bad_missing}, unexpected={unexpected}")


@torch.inference_mode()
def predict(
    *,
    checkpoint_path: str | Path,
    features_dir: str | Path,
    label_fasta: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    threshold: float | None = None,
    threshold_search: dict[str, float] | None = None,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = dict(ckpt.get("config", {}))
    hidden_size = int(cfg.get("hidden_size") or 0) or _infer_hidden_size(ckpt["model_state"])
    classifier_head = dict(cfg.get("classifier_head", {}))
    dropout = float(cfg.get("dropout", 0.1))

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = DisorderFeatureClassifier(hidden_size=hidden_size, dropout=dropout, classifier_head=classifier_head).to(dev)
    _load_classifier_only(model, ckpt["model_state"])
    model.eval()

    records = parse_three_line_fasta(label_fasta)
    y_true_all: list[int] = []
    y_prob_all: list[float] = []
    per_record: list[tuple[str, str, list[float]]] = []

    for rec in records:
        feat = read_residue_feature_file(_feature_path(features_dir, rec.protein_id))
        if int(feat.shape[0]) != len(rec.sequence):
            raise ValueError(
                f"Length mismatch for {rec.protein_id}: feature_rows={feat.shape[0]} seq_len={len(rec.sequence)}"
            )
        emb = feat.unsqueeze(0).to(dev)  # [1, L, H]
        logits = model(emb, [int(feat.shape[0])])[0, : int(feat.shape[0])].detach().float().cpu().tolist()
        probs = sigmoid(logits)
        per_record.append((rec.protein_id, rec.labels, probs))
        yt, yp = _extract_scored_residues(rec.labels, probs)
        y_true_all.extend(yt)
        y_prob_all.extend(yp)

    ts = threshold_search or {"min": 0.05, "max": 0.95, "step": 0.05}
    tuned = threshold
    best_f1 = 0.0
    best_mcc = 0.0
    if y_true_all and tuned is None:
        tuned, best_f1, best_mcc = search_best_threshold(
            y_true_all, y_prob_all, min_threshold=float(ts["min"]), max_threshold=float(ts["max"]), step=float(ts["step"])
        )
    if tuned is None:
        tuned = float(ckpt.get("threshold", 0.5))

    preds = apply_threshold(y_prob_all, tuned) if y_true_all else []
    auprc = precision_recall_auc(y_true_all, y_prob_all) if y_true_all else 0.0
    auroc = binary_roc_auc(y_true_all, y_prob_all)
    f1 = best_f1 if y_true_all and threshold is None else (f1_score(y_true_all, preds) if y_true_all else 0.0)
    mcc = best_mcc if y_true_all and threshold is None else (mcc_score(y_true_all, preds) if y_true_all else 0.0)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.tsv"
    with pred_path.open("w", encoding="utf-8") as handle:
        handle.write("protein_id\tposition\tprobability\tprediction\n")
        for pid, _, probs in per_record:
            for idx, p in enumerate(probs, start=1):
                handle.write(f"{pid}\t{idx}\t{p:.8f}\t{1 if p >= tuned else 0}\n")

    metrics = {
        "num_records": len(records),
        "num_scored_residues": len(y_true_all),
        "auprc": auprc,
        "auroc": auroc,
        "f1": f1,
        "mcc": mcc,
        "threshold": tuned,
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "features_dir": str(Path(features_dir).resolve()),
        "label_fasta": str(Path(label_fasta).resolve()),
        "output_dir": str(out.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "predictions_path": str(pred_path.resolve()),
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict disorder from precomputed features using CLIPer classifier head.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--label-fasta", required=True, help="3-line label FASTA for test set.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--ts-min", type=float, default=0.05)
    p.add_argument("--ts-max", type=float, default=0.95)
    p.add_argument("--ts-step", type=float, default=0.05)
    return p


def main() -> int:
    args = build_parser().parse_args()
    result = predict(
        checkpoint_path=args.checkpoint,
        features_dir=args.features_dir,
        label_fasta=args.label_fasta,
        output_dir=args.output_dir,
        device=args.device,
        threshold=args.threshold,
        threshold_search={"min": args.ts_min, "max": args.ts_max, "step": args.ts_step},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

