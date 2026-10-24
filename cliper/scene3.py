"""Protein-level linker score from residue probabilities.

A protein is positive when its label string contains at least one linker
residue (``2``). The score is the highest mean linker probability inside a
sliding window, so a contiguous high-scoring segment can mark the protein
even when most residues are not linkers.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from cliper.metrics import binary_roc_auc


def linker_run_lengths(labels: str) -> list[int]:
    runs: list[int] = []
    current = 0
    for char in labels:
        if char == "2":
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def median_linker_run_length(label_strings: Iterable[str], *, default: int = 31) -> int:
    runs: list[int] = []
    for labels in label_strings:
        runs.extend(linker_run_lengths(labels))
    if not runs:
        return int(default)
    ordered = sorted(runs)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return int(ordered[mid])
    return int(round((ordered[mid - 1] + ordered[mid]) / 2.0))


def protein_has_linker(labels: str) -> int:
    return 1 if "2" in labels else 0


def max_window_mean(probs: list[float], window: int) -> float:
    """Highest mean of a contiguous window. Short proteins use the full mean."""
    if not probs:
        return 0.0
    width = max(1, int(window))
    if len(probs) <= width:
        return float(sum(probs) / len(probs))
    total = sum(probs[:width])
    best = total
    for end in range(width, len(probs)):
        total += probs[end] - probs[end - width]
        if total > best:
            best = total
    return float(best / width)


def load_prediction_probs(path: str | Path) -> dict[str, list[float]]:
    """Read predictions.tsv written by ``pipeline.evaluate``."""
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        header = handle.readline()
        if "protein_id" not in header:
            raise ValueError(f"Unexpected predictions header: {header!r}")
        for line in handle:
            protein_id, position, probability, _pred = line.rstrip("\n").split("\t")
            grouped[protein_id].append((int(position), float(probability)))
    ordered: dict[str, list[float]] = {}
    for protein_id, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        ordered[protein_id] = [prob for _pos, prob in rows]
    return ordered


def score_proteins(
    records: list[tuple[str, str]],
    probs_by_id: dict[str, list[float]],
    *,
    window: int,
) -> list[dict[str, float | int | str]]:
    """``records`` is ``(protein_id, labels)`` in the order to report."""
    rows: list[dict[str, float | int | str]] = []
    for protein_id, labels in records:
        probs = probs_by_id.get(protein_id)
        if probs is None:
            raise KeyError(f"Missing residue scores for {protein_id}")
        if len(probs) != len(labels):
            raise ValueError(
                f"{protein_id}: {len(probs)} scores for {len(labels)} labels"
            )
        rows.append(
            {
                "protein_id": protein_id,
                "label": protein_has_linker(labels),
                "score": max_window_mean(probs, window),
                "length": len(labels),
                "window": int(window),
            }
        )
    return rows


def protein_auroc(rows: list[dict[str, float | int | str]]) -> float | None:
    y_true = [int(row["label"]) for row in rows]
    y_score = [float(row["score"]) for row in rows]
    return binary_roc_auc(y_true, y_score)
