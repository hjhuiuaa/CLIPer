#!/usr/bin/env python3
"""Evaluate LinkCNN (or any residue predictor) under T2 / T3 protocols.

T2: score only residues with disorder=='1' and linker in {0,1}.
T3: aggregate residue probabilities to protein scores; binary harboring-DL labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cliper.data import parse_three_line_fasta
from cliper.metrics import binary_roc_auc, f1_score, mcc_score, precision_recall_auc


def _parse_predictions_tsv(path: Path) -> dict[str, list[float]]:
    probs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if "protein_id" not in header:
            raise ValueError(f"Unexpected predictions.tsv header in {path}: {header!r}")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid, pos_s, prob_s = parts[0], parts[1], parts[2]
            probs[pid].append((int(pos_s), float(prob_s)))
    out: dict[str, list[float]] = {}
    for pid, items in probs.items():
        items.sort(key=lambda x: x[0])
        # Expect contiguous 1..L; fill by max position
        max_pos = items[-1][0]
        arr = [0.0] * max_pos
        for pos, p in items:
            arr[pos - 1] = p
        out[pid] = arr
    return out


def _parse_caid_dir(caid_dir: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for path in sorted(caid_dir.glob("*.caid")):
        pid = path.stem
        probs: list[float] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith(">") or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                # pos, aa, score, binary
                probs.append(float(parts[2]))
        out[pid] = probs
    return out


def _label_map(fasta: Path) -> dict[str, str]:
    return {rec.protein_id: rec.labels for rec in parse_three_line_fasta(fasta)}


def _seq_map(fasta: Path) -> dict[str, str]:
    return {rec.protein_id: rec.sequence for rec in parse_three_line_fasta(fasta)}


def compute_t2(
    probs_by_id: dict[str, list[float]],
    linker_labels: dict[str, str],
    disorder_labels: dict[str, str],
    protein_ids: list[str] | None,
    threshold: float,
) -> dict[str, Any]:
    y_true: list[int] = []
    y_prob: list[float] = []
    skipped = {
        "missing_prob": 0,
        "missing_linker": 0,
        "missing_disorder": 0,
        "length_mismatch": 0,
        "ordered_or_uncertain": 0,
    }
    ids = protein_ids if protein_ids is not None else sorted(set(probs_by_id) & set(linker_labels) & set(disorder_labels))
    used_proteins = 0
    for pid in ids:
        if pid not in probs_by_id:
            skipped["missing_prob"] += 1
            continue
        if pid not in linker_labels:
            skipped["missing_linker"] += 1
            continue
        if pid not in disorder_labels:
            skipped["missing_disorder"] += 1
            continue
        probs = probs_by_id[pid]
        L = linker_labels[pid]
        D = disorder_labels[pid]
        n = min(len(probs), len(L), len(D))
        if len(probs) != len(L) or len(L) != len(D):
            skipped["length_mismatch"] += 1
            # still score aligned prefix
        took = 0
        for i in range(n):
            if D[i] != "1" or L[i] not in {"0", "1"}:
                skipped["ordered_or_uncertain"] += 1
                continue
            y_true.append(1 if L[i] == "1" else 0)
            y_prob.append(probs[i])
            took += 1
        if took:
            used_proteins += 1

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    auroc = binary_roc_auc(y_true, y_prob)
    return {
        "protocol": "T2",
        "num_proteins_used": used_proteins,
        "num_scored_residues": len(y_true),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "auprc": precision_recall_auc(y_true, y_prob),
        "auroc": auroc,
        "f1": f1_score(y_true, y_pred) if y_true else 0.0,
        "mcc": mcc_score(y_true, y_pred) if y_true else 0.0,
        "threshold": threshold,
        "skipped": skipped,
    }


def _max_window_mean(probs: list[float], window: int) -> float:
    if not probs:
        return 0.0
    w = min(window, len(probs))
    # prefix sums for O(n)
    pref = [0.0]
    for p in probs:
        pref.append(pref[-1] + p)
    best = float("-inf")
    for start in range(0, len(probs) - w + 1):
        mean = (pref[start + w] - pref[start]) / w
        if mean > best:
            best = mean
    return best if best != float("-inf") else 0.0


def _top_k_mean(probs: list[float], k: int) -> float:
    if not probs:
        return 0.0
    k = min(k, len(probs))
    top = sorted(probs, reverse=True)[:k]
    return sum(top) / k


def aggregate_protein_scores(probs: list[float]) -> dict[str, float]:
    return {
        "max": max(probs) if probs else 0.0,
        "top10_mean": _top_k_mean(probs, 10),
        "max_window31_mean": _max_window_mean(probs, 31),
    }


def compute_t3(
    probs_by_id: dict[str, list[float]],
    positive_ids: list[str],
    negative_ids: list[str],
    primary_agg: str = "max_window31_mean",
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for pid in positive_ids:
        if pid not in probs_by_id:
            continue
        agg = aggregate_protein_scores(probs_by_id[pid])
        records.append({"protein_id": pid, "label": 1, **agg})
    for pid in negative_ids:
        if pid not in probs_by_id:
            continue
        agg = aggregate_protein_scores(probs_by_id[pid])
        records.append({"protein_id": pid, "label": 0, **agg})

    metrics_by_agg: dict[str, Any] = {}
    for agg_name in ("max", "top10_mean", "max_window31_mean"):
        y_true = [r["label"] for r in records]
        y_prob = [float(r[agg_name]) for r in records]
        auroc = binary_roc_auc(y_true, y_prob)
        metrics_by_agg[agg_name] = {
            "auprc": precision_recall_auc(y_true, y_prob),
            "auroc": auroc,
            "num_proteins": len(records),
            "num_positive": sum(y_true),
            "num_negative": len(y_true) - sum(y_true),
        }

    return {
        "protocol": "T3",
        "primary_aggregation": primary_agg,
        "primary_metrics": metrics_by_agg[primary_agg],
        "metrics_by_aggregation": metrics_by_agg,
        "per_protein": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["t2", "t3"], required=True)
    parser.add_argument("--predictions-tsv", type=Path, default=None)
    parser.add_argument("--caid-dir", type=Path, default=None, help="Directory of *.caid files")
    parser.add_argument("--linker-fasta", type=Path, required=True)
    parser.add_argument("--disorder-fasta", type=Path, default=None, help="Required for T2")
    parser.add_argument("--protein-ids-json", type=Path, default=None, help="JSON list or manifest with ids")
    parser.add_argument("--ids-key", default=None, help="If manifest, key for id list (e.g. val_ids)")
    parser.add_argument("--positive-ids-json", type=Path, default=None)
    parser.add_argument("--positive-ids-key", default="val_ids")
    parser.add_argument("--negative-ids-json", type=Path, default=None)
    parser.add_argument("--negative-ids-key", default="negative_ids")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.predictions_tsv:
        probs = _parse_predictions_tsv(args.predictions_tsv)
    elif args.caid_dir:
        probs = _parse_caid_dir(args.caid_dir)
    else:
        raise SystemExit("Provide --predictions-tsv or --caid-dir")

    def _load_ids(path: Path | None, key: str | None) -> list[str] | None:
        if path is None:
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if key is None:
            raise SystemExit(f"--ids-key required for manifest {path}")
        return [str(x) for x in raw[key]]

    if args.mode == "t2":
        if args.disorder_fasta is None:
            raise SystemExit("T2 requires --disorder-fasta")
        linker = _label_map(args.linker_fasta)
        disorder = _label_map(args.disorder_fasta)
        ids = _load_ids(args.protein_ids_json, args.ids_key)
        result = compute_t2(probs, linker, disorder, ids, args.threshold)
    else:
        pos_ids = _load_ids(args.positive_ids_json, args.positive_ids_key)
        neg_ids = _load_ids(args.negative_ids_json, args.negative_ids_key)
        if not pos_ids or not neg_ids:
            raise SystemExit("T3 requires --positive-ids-json and --negative-ids-json")
        # Ensure we have probs for both; linker fasta unused for labels but kept for consistency
        _ = _seq_map(args.linker_fasta)
        result = compute_t3(probs, pos_ids, neg_ids)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # Drop bulky per-protein list from stdout summary but keep in file
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "per_protein"}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
