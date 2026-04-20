"""Disorder residue-level binary classification with overlapping sliding windows for long sequences."""

from __future__ import annotations

from typing import Any

__all__ = [
    "cli",
    "data",
    "metrics",
    "pipeline",
    "windowing",
    "prepare_data",
    "train",
    "evaluate",
    "train_features",
    "eval_features_checkpoint",
]


def __getattr__(name: str) -> Any:
    if name in {"prepare_data", "train", "evaluate"}:
        import disorder.pipeline as _pipeline

        return getattr(_pipeline, name)
    if name == "train_features":
        from disorder.feature_pipeline import train_features as tf

        return tf
    if name == "eval_features_checkpoint":
        from disorder.feature_pipeline import eval_features_checkpoint as ef

        return ef
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
