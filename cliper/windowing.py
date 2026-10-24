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


def training_crop(
    sequence: str,
    labels: str,
    window_size: int,
    *,
    seed: int,
    positive_chars: str = "1",
) -> tuple[str, str, int]:
    """Return one training crop with linker-centered policy for long sequences."""
    length = len(sequence)
    if length <= window_size:
        return sequence, labels, 0

    positives = [idx for idx, char in enumerate(labels) if char in set(positive_chars)]
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


def merge_window_logits(
    length: int, window_logits: list[tuple[int, list[float] | list[list[float]]]]
) -> list[float] | list[list[float]]:
    """Merge overlapping window logits by mean value at each residue.

    Supports binary logits (list[float] per window) and multi-class logits
    (list[list[float]] with shape [L, C] per window).
    """
    if not window_logits:
        return [0.0] * length
    sample = window_logits[0][1]
    multi = bool(sample) and isinstance(sample[0], (list, tuple))
    if multi:
        num_classes = len(sample[0])  # type: ignore[arg-type]
        total = [[0.0] * num_classes for _ in range(length)]
        counts = [0] * length
        for start, logits in window_logits:
            for offset, logit_vec in enumerate(logits):
                position = start + offset
                if position >= length:
                    break
                for c, value in enumerate(logit_vec):  # type: ignore[arg-type]
                    total[position][c] += float(value)
                counts[position] += 1
        merged: list[list[float]] = []
        for pos in range(length):
            if counts[pos] == 0:
                merged.append([0.0] * num_classes)
            else:
                merged.append([v / counts[pos] for v in total[pos]])
        return merged

    total_f = [0.0] * length
    counts_f = [0] * length
    for start, logits in window_logits:
        for offset, logit in enumerate(logits):
            position = start + offset
            if position >= length:
                break
            total_f[position] += float(logit)  # type: ignore[arg-type]
            counts_f[position] += 1

    merged_f: list[float] = []
    for pos in range(length):
        if counts_f[pos] == 0:
            merged_f.append(0.0)
        else:
            merged_f.append(total_f[pos] / counts_f[pos])
    return merged_f


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


def linker_probs_from_logits(
    merged_logits: list[float] | list[list[float]],
    *,
    linker_class: int = 1,
    dual_binary: bool = False,
) -> list[float]:
    """Convert merged logits to P(Linker).

    Binary uses sigmoid. Dual-binary uses sigmoid of channel 0 (linker vs rest).
    Softmax multi-class uses softmax[linker_class].
    """
    if not merged_logits:
        return []
    if isinstance(merged_logits[0], (list, tuple)):
        if dual_binary:
            return sigmoid([float(vec[0]) for vec in merged_logits])  # type: ignore[index,arg-type]
        class_index = int(linker_class)
        probs: list[float] = []
        for logit_vec in merged_logits:  # type: ignore[assignment]
            values = [float(x) for x in logit_vec]  # type: ignore[arg-type]
            if class_index < 0 or class_index >= len(values):
                raise ValueError(
                    f"linker_class={class_index} out of range for {len(values)}-way logits {values!r}"
                )
            peak = max(values)
            exps = [math.exp(v - peak) for v in values]
            denom = sum(exps)
            probs.append(exps[class_index] / denom if denom > 0 else 0.0)
        return probs
    return sigmoid(merged_logits)  # type: ignore[arg-type]
