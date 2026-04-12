from __future__ import annotations

from collections import Counter
import math
import random


DISORDER_LIKE_RESIDUES = set("PGSEDKRNQ")


def normalize_sequence(sequence: str) -> str:
    """Normalize uncommon amino acids to X and force uppercase."""
    seq = sequence.upper()
    replacements = {"U": "X", "Z": "X", "O": "X", "B": "X"}
    return "".join(replacements.get(ch, ch) for ch in seq)


def training_crop(sequence: str, labels: str, window_size: int, *, seed: int) -> tuple[str, str, int]:
    """Return one training crop with linker-centered policy for long sequences."""
    length = len(sequence)
    if length <= window_size:
        return sequence, labels, 0

    positives = [idx for idx, char in enumerate(labels) if char == "1"]
    if positives:
        left = positives[0]
        right = positives[-1]
        center = (left + right) // 2
        start = center - (window_size // 2)
        start = max(0, min(start, length - window_size))
    else:
        rng = random.Random(seed)
        start = rng.randint(0, length - window_size)
    end = start + window_size
    return sequence[start:end], labels[start:end], start


def _disorder_like_score(segment: str) -> float:
    if not segment:
        return 0.0
    seg_len = len(segment)
    disorder_fraction = sum(1 for ch in segment if ch in DISORDER_LIKE_RESIDUES) / seg_len
    counts = Counter(segment)
    dominant_fraction = max(counts.values()) / seg_len
    # Higher means "more disorder-like / lower complexity".
    return 0.7 * disorder_fraction + 0.3 * dominant_fraction


def build_eval_window_starts(
    sequence: str,
    *,
    window_size: int,
    stride: int | None = None,
    top_k_heuristic: int = 4,
) -> list[int]:
    """Label-agnostic eval windows: sequence-only heuristic centers + coverage windows."""
    length = len(sequence)
    if length <= window_size:
        return [0]

    stride = stride or max(1, window_size // 2)
    max_start = length - window_size

    coverage = set(range(0, max_start + 1, stride))
    coverage.add(max_start)

    centers: list[tuple[float, int]] = []
    step = max(8, window_size // 4)
    half = window_size // 2
    first_center = half
    last_center = length - half
    for center in range(first_center, last_center + 1, step):
        start = center - half
        end = start + window_size
        segment = sequence[start:end]
        score = _disorder_like_score(segment)
        centers.append((score, center))

    centers.sort(reverse=True)
    heuristic_starts: set[int] = set()
    for _, center in centers[:top_k_heuristic]:
        start = center - half
        start = max(0, min(start, max_start))
        heuristic_starts.add(start)

    return sorted(coverage.union(heuristic_starts))


def merge_window_logits(length: int, window_logits: list[tuple[int, list[float]]]) -> list[float]:
    """Merge overlapping window logits by mean value at each residue."""
    total = [0.0] * length
    counts = [0] * length
    for start, logits in window_logits:
        for offset, logit in enumerate(logits):
            position = start + offset
            if position >= length:
                break
            total[position] += float(logit)
            counts[position] += 1

    merged: list[float] = []
    for pos in range(length):
        if counts[pos] == 0:
            merged.append(0.0)
        else:
            merged.append(total[pos] / counts[pos])
    return merged


def sigmoid(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value >= 0:
            z = math.exp(-value)
            out.append(1.0 / (1.0 + z))
        else:
            z = math.exp(value)
            out.append(z / (1.0 + z))
    return out

