from __future__ import annotations

import math
from typing import Any, Iterable


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def precision_recall_auc(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    pairs = list(zip(y_true, y_prob))
    if not pairs:
        return 0.0

    positives = sum(1 for label, _ in pairs if label == 1)
    if positives == 0:
        return 0.0

    pairs.sort(key=lambda item: item[1], reverse=True)

    tp = 0
    fp = 0
    precision = [1.0]
    recall = [0.0]
    for label, _ in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision.append(_safe_div(tp, tp + fp))
        recall.append(_safe_div(tp, positives))

    # Step-wise integration over recall.
    auc = 0.0
    for i in range(1, len(precision)):
        delta_recall = recall[i] - recall[i - 1]
        if delta_recall > 0:
            auc += precision[i] * delta_recall
    return auc


def f1_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    tp = fp = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
        elif pred == 0 and truth == 1:
            fn += 1
    return _safe_div(2 * tp, (2 * tp + fp + fn))


def mcc_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    tp = tn = fp = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1

    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def binary_roc_auc(y_true: list[int], y_prob: list[float]) -> float | None:
    """Rank-based AUROC (Mann–Whitney U). Returns None if undefined."""
    if not y_true:
        return None
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(y_prob, y_true), key=lambda t: t[0])
    n = len(pairs)
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


def apply_threshold(y_prob: Iterable[float], threshold: float) -> list[int]:
    return [1 if prob >= threshold else 0 for prob in y_prob]


def search_best_threshold(
    y_true: list[int],
    y_prob: list[float],
    *,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> tuple[float, float, float]:
    if step <= 0:
        raise ValueError(f"threshold step must be > 0, got {step}")

    best_threshold = min_threshold
    best_f1 = -1.0
    best_mcc = -1.0
    threshold = min_threshold
    while threshold <= max_threshold + 1e-12:
        predictions = apply_threshold(y_prob, threshold)
        current_f1 = f1_score(y_true, predictions)
        current_mcc = mcc_score(y_true, predictions)
        if current_f1 > best_f1 or (abs(current_f1 - best_f1) < 1e-12 and current_mcc > best_mcc):
            best_f1 = current_f1
            best_mcc = current_mcc
            best_threshold = threshold
        threshold += step

    return best_threshold, max(best_f1, 0.0), max(best_mcc, 0.0)


def ternary_label_chars(linker_class: int) -> tuple[str, str, str]:
    """Return (linker, other disordered residue, ordered)."""
    if int(linker_class) == 2:
        return "2", "1", "0"
    return "1", "0", "2"


def extract_scored_residues(
    labels: str,
    probs: list[float],
    *,
    scenario: str = "scenario1",
    linker_class: int = 1,
) -> tuple[list[int], list[float]]:
    """Map residue labels to a binary linker score for one scenario.

    scenario1 scores linker against every other residue.
    scenario2 scores linker against other disordered residues and skips ordered residues.
    """
    if scenario not in {"scenario1", "scenario2", "t1", "t2"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    linker_vs_disorder_only = scenario in {"scenario2", "t2"}
    linker_ch, other_ch, ordered_ch = ternary_label_chars(linker_class)
    y_true: list[int] = []
    y_prob: list[float] = []
    for char, prob in zip(labels, probs):
        if char == linker_ch:
            y_true.append(1)
            y_prob.append(prob)
        elif char == other_ch:
            y_true.append(0)
            y_prob.append(prob)
        elif not linker_vs_disorder_only and char == ordered_ch:
            y_true.append(0)
            y_prob.append(prob)
    return y_true, y_prob


def score_scenarios(
    *,
    num_records: int,
    y_true: list[int],
    y_prob: list[float],
    y_true_scenario2: list[int],
    y_prob_scenario2: list[float],
    threshold: float | None,
    threshold_search: dict[str, float],
    linker_class: int,
    include_scenario2: bool,
) -> dict[str, Any]:
    """Score scenario 1, and scenario 2 when the labels distinguish other disorder."""
    tuned_threshold = threshold
    best_f1 = 0.0
    best_mcc = 0.0
    if y_true and tuned_threshold is None:
        tuned_threshold, best_f1, best_mcc = search_best_threshold(
            y_true,
            y_prob,
            min_threshold=float(threshold_search["min"]),
            max_threshold=float(threshold_search["max"]),
            step=float(threshold_search["step"]),
        )
    if tuned_threshold is None:
        tuned_threshold = 0.5

    preds = apply_threshold(y_prob, tuned_threshold) if y_true else []
    auprc = precision_recall_auc(y_true, y_prob) if y_true else 0.0
    auroc = binary_roc_auc(y_true, y_prob)
    f1 = best_f1 if y_true and threshold is None else (f1_score(y_true, preds) if y_true else 0.0)
    mcc = best_mcc if y_true and threshold is None else (mcc_score(y_true, preds) if y_true else 0.0)
    metrics: dict[str, Any] = {
        "num_records": num_records,
        "num_scored_residues": len(y_true),
        "auprc": auprc,
        "auroc": auroc,
        "auprc_t1": auprc,
        "auroc_t1": auroc,
        "f1": f1,
        "mcc": mcc,
        "threshold": tuned_threshold,
        "protocol_t1": "linker vs {other_idr, ordered}",
        "linker_class": int(linker_class),
    }
    if include_scenario2:
        metrics["auprc_t2"] = precision_recall_auc(y_true_scenario2, y_prob_scenario2) if y_true_scenario2 else 0.0
        metrics["auroc_t2"] = binary_roc_auc(y_true_scenario2, y_prob_scenario2)
        metrics["num_scored_residues_t2"] = len(y_true_scenario2)
        metrics["protocol_t2"] = "linker vs other_idr"
    return {"threshold": tuned_threshold, "metrics": metrics}

