import torch

from cliper.pipeline import _compute_batch_supcon_loss, _supervised_contrastive_loss


def test_supervised_contrastive_loss_is_finite() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss = _supervised_contrastive_loss(embeddings, labels, temperature=0.1)
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_batch_supcon_returns_zero_for_single_class_or_insufficient_samples() -> None:
    residue_embeddings = torch.randn(1, 6, 8)

    single_class_labels = torch.ones((1, 6), dtype=torch.float32)
    single_class_loss, single_class_count = _compute_batch_supcon_loss(
        residue_embeddings,
        single_class_labels,
        max_samples_per_class=16,
        temperature=0.1,
    )
    assert single_class_count > 0
    assert abs(float(single_class_loss.item())) < 1e-12

    masked_labels = torch.full((1, 6), -100.0, dtype=torch.float32)
    masked_loss, masked_count = _compute_batch_supcon_loss(
        residue_embeddings,
        masked_labels,
        max_samples_per_class=16,
        temperature=0.1,
    )
    assert masked_count == 0
    assert abs(float(masked_loss.item())) < 1e-12
