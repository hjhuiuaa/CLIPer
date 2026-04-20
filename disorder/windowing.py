"""Sliding-window helpers for residue-level disorder (long sequences, positives may be at ends)."""

from __future__ import annotations

import random


def build_sliding_eval_starts(length: int, window_size: int, stride: int) -> list[int]:
    """Coverage of [0, length) with windows of size window_size; last window flush to sequence end."""
    if length <= window_size:
        return [0]
    stride = max(1, stride)
    max_start = length - window_size
    starts: list[int] = []
    pos = 0
    while pos <= max_start:
        starts.append(pos)
        pos += stride
    if starts[-1] != max_start:
        starts.append(max_start)
    return sorted(set(starts))


def count_split_positive_runs(labels: str, start: int, window_size: int) -> int:
    """Count maximal '1' runs that overlap the window but are not fully contained (undesired cuts)."""
    n = len(labels)
    s, e = start, start + window_size
    penalty = 0
    i = 0
    while i < n:
        if labels[i] != "1":
            i += 1
            continue
        left = i
        while i < n and labels[i] == "1":
            i += 1
        right = i - 1
        if right < s or left >= e:
            continue
        if left >= s and right < e:
            continue
        penalty += 1
    return penalty


def training_window_score(labels: str, start: int, window_size: int, *, split_penalty_weight: float) -> float:
    chunk = labels[start : start + window_size]
    positives = sum(1 for c in chunk if c == "1")
    splits = count_split_positive_runs(labels, start, window_size)
    return float(positives) - split_penalty_weight * float(splits)


def pick_training_window(
    sequence: str,
    labels: str,
    window_size: int,
    overlap: int,
    seed: int,
    *,
    split_penalty_weight: float = 3.0,
) -> tuple[str, str, int]:
    """
    One training crop: overlapping grid of valid starts; pick among top-scoring starts
    (prefer keeping contiguous disorder inside a single window).
    """
    length = len(sequence)
    if length <= window_size:
        return sequence, labels, 0
    stride = max(1, window_size - overlap)
    starts = build_sliding_eval_starts(length, window_size, stride)
    rng = random.Random(seed)
    scored = [(s, training_window_score(labels, s, window_size, split_penalty_weight=split_penalty_weight)) for s in starts]
    best = max(sc for _, sc in scored)
    top = [s for s, sc in scored if sc >= best - 1e-9]
    start = rng.choice(top)
    return sequence[start : start + window_size], labels[start : start + window_size], start
