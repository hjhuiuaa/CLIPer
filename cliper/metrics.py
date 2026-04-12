from __future__ import annotations

import math
from typing import Iterable


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

