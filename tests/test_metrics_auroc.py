"""AUROC helper tests (no torch dependency)."""

from cliper.metrics import binary_roc_auc


def test_binary_roc_auc_perfect_ranking() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    assert binary_roc_auc(y_true, y_prob) == 1.0


def test_binary_roc_auc_single_class_returns_none() -> None:
    assert binary_roc_auc([1, 1, 1], [0.1, 0.5, 0.9]) is None
    assert binary_roc_auc([0, 0, 0], [0.1, 0.5, 0.9]) is None


def test_binary_roc_auc_empty_returns_none() -> None:
    assert binary_roc_auc([], []) is None
