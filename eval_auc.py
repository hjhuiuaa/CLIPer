#!/usr/bin/env python3
"""
Compute AUROC/AUPRC and basic threshold metrics from predictions TSV.

Expected input: a TSV with at least
  - one label column (0/1)
  - one score/probability column (continuous)

By default the script auto-detects common column names.
You can override with --label-col and --score-col.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def _first_existing(candidates: Iterable[str], columns: list[str]) -> str | None:
    col_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in col_map:
            return col_map[name.lower()]
    return None


def _to_float(x: str) -> float:
    value = float(x)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Invalid numeric value: {x}")
    return value


def _load_labels_from_three_line_fasta(path: Path) -> dict[str, str]:
    """Return mapping: protein_id -> residue label string ('0'/'1'/'-')."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(
            f"FASTA labels file must be 3-line records (>id, sequence, labels). Got {len(lines)} non-empty lines."
        )
    labels_by_id: dict[str, str] = {}
    for i in range(0, len(lines), 3):
        header = lines[i]
        seq = lines[i + 1]
        labels = lines[i + 2]
        if not header.startswith(">"):
            raise ValueError(f"Invalid FASTA header at line {i + 1}: {header!r}")
        pid = header[1:].strip()
        if not pid:
            raise ValueError(f"Empty protein id at line {i + 1}")
        if len(seq) != len(labels):
            raise ValueError(
                f"Sequence/label length mismatch for {pid}: len(seq)={len(seq)} len(labels)={len(labels)}"
            )
        labels_by_id[pid] = labels
    return labels_by_id


def _auc_roc(y_true: list[int], y_score: list[float]) -> float:
    # Rank-based AUROC (Mann-Whitney U)
    pairs = sorted(zip(y_score, y_true), key=lambda t: t[0])
    n = len(pairs)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average ranks for ties
    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        pos_in_group = sum(label for _, label in pairs[i:j])
        rank_sum_pos += avg_rank * pos_in_group
        rank += j - i
        i = j

    u = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return u / (n_pos * n_neg)


def _auc_pr(y_true: list[int], y_score: list[float]) -> float:
    # Step-wise PR AUC with score-desc sorting.
    n_pos = sum(y_true)
    if n_pos == 0:
        return float("nan")

    items = sorted(zip(y_score, y_true), key=lambda t: t[0], reverse=True)
    tp = 0
    fp = 0

    prev_recall = 0.0
    prev_precision = 1.0
    area = 0.0

    i = 0
    while i < len(items):
        score = items[i][0]
        j = i
        while j < len(items) and items[j][0] == score:
            if items[j][1] == 1:
                tp += 1
            else:
                fp += 1
            j += 1

        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        area += (recall - prev_recall) * ((precision + prev_precision) / 2.0)
        prev_recall = recall
        prev_precision = precision
        i = j

    return area


def _threshold_metrics(y_true: list[int], y_score: list[float], threshold: float) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for y, s in zip(y_true, y_score):
        p = 1 if s >= threshold else 0
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 0:
            tn += 1
        elif y == 0 and p == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute AUROC/AUPRC from predictions TSV.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.tsv")
    parser.add_argument("--label-col", default=None, help="Label column name (default: auto-detect)")
    parser.add_argument("--score-col", default=None, help="Score/probability column name (default: auto-detect)")
    parser.add_argument(
        "--fasta-labels",
        default=None,
        help="Optional 3-line FASTA with labels. If provided and label column is missing, "
        "labels are aligned by (protein_id, position_1based).",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for ACC/F1/MCC metrics")
    parser.add_argument("--out-json", default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    with pred_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        raw_fields = reader.fieldnames or []
        if not raw_fields:
            raise ValueError("Predictions file has no header columns.")
        rows = list(reader)
        rows = [{k.lstrip("\ufeff").strip(): v for k, v in row.items()} for row in rows]
        columns = list(rows[0].keys()) if rows else [h.lstrip("\ufeff").strip() for h in raw_fields]

        label_col = args.label_col or _first_existing(
            ("label", "y_true", "target", "gold", "residue_label", "true_label"), columns
        )
        score_col = args.score_col or _first_existing(
            ("score", "prob", "probability", "positive_score", "logit", "pred_score"), columns
        )
        if score_col is None:
            raise ValueError(
                f"Cannot auto-detect columns. Found columns: {columns}. "
                "Use --score-col explicitly."
            )

        y_true: list[int] = []
        y_score: list[float] = []

        if label_col is not None:
            for row in rows:
                label_raw = row[label_col].strip()
                score_raw = row[score_col].strip()
                y = int(float(label_raw))
                if y not in (0, 1):
                    raise ValueError(f"Label must be binary 0/1, got: {label_raw}")
                y_true.append(y)
                y_score.append(_to_float(score_raw))
        else:
            if not args.fasta_labels:
                raise ValueError(
                    f"Cannot auto-detect label column. Found columns: {columns}. "
                    "Provide --label-col, or provide --fasta-labels with 3-line labeled FASTA."
                )
            fasta_path = Path(args.fasta_labels)
            if not fasta_path.exists():
                raise FileNotFoundError(f"--fasta-labels file not found: {fasta_path}")
            labels_by_id = _load_labels_from_three_line_fasta(fasta_path)

            pid_col = _first_existing(("protein_id", "id", "protein", "seq_id"), columns)
            pos_col = _first_existing(("position_1based", "position", "pos", "residue_index"), columns)
            if pid_col is None or pos_col is None:
                raise ValueError(
                    "When using --fasta-labels, predictions TSV must include protein and position columns. "
                    f"Found columns: {columns}"
                )

            used = 0
            skipped_masked = 0
            missing_pid = 0
            bad_pos = 0
            missing_pid_samples: list[str] = []
            for row in rows:
                pid = row[pid_col].strip()
                score_raw = row[score_col].strip()
                pos_str = row[pos_col].strip()
                label_seq = labels_by_id.get(pid)
                if label_seq is None:
                    missing_pid += 1
                    if len(missing_pid_samples) < 5:
                        missing_pid_samples.append(pid)
                    continue
                try:
                    pos = int(float(pos_str))
                except ValueError:
                    bad_pos += 1
                    continue
                if pos < 1 or pos > len(label_seq):
                    bad_pos += 1
                    continue
                ch = label_seq[pos - 1]
                if ch == "-":
                    skipped_masked += 1
                    continue
                if ch not in ("0", "1"):
                    raise ValueError(f"Unexpected label char {ch!r} for {pid} at position {pos}")
                y_true.append(int(ch))
                y_score.append(_to_float(score_raw))
                used += 1

            if used == 0:
                fasta_ids_sample = list(labels_by_id.keys())[:5]
                raise ValueError(
                    "No aligned rows found between predictions and --fasta-labels. "
                    f"tsv_rows={len(rows)} missing_pid={missing_pid} bad_pos={bad_pos} "
                    f"skipped_masked={skipped_masked} "
                    f"pid_col={pid_col!r} pos_col={pos_col!r} score_col={score_col!r}. "
                    f"sample_prediction_pids={missing_pid_samples!r} "
                    f"sample_fasta_pids={fasta_ids_sample!r}. "
                    "If missing_pid is high, protein_id strings in TSV do not match FASTA headers (after '>')."
                )

    auroc = _auc_roc(y_true, y_score)
    auprc = _auc_pr(y_true, y_score)
    th_metrics = _threshold_metrics(y_true, y_score, args.threshold)

    result = {
        "predictions": str(pred_path),
        "num_samples": len(y_true),
        "positive_ratio": (sum(y_true) / len(y_true)) if y_true else float("nan"),
        "label_col": label_col if label_col is not None else "from_fasta_labels",
        "score_col": score_col,
        "auroc": auroc,
        "auprc": auprc,
        "threshold_metrics": th_metrics,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved metrics to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

