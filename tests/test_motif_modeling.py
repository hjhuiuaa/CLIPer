import torch

from cliper.modeling import DummyBackbone, ResidueClassifier


def test_motif_special_token_alignment_keeps_residue_logit_shape() -> None:
    backbone = DummyBackbone(hidden_size=32, vocab_size=64)
    model = ResidueClassifier(
        backbone=backbone,
        hidden_size=32,
        freeze_backbone=True,
        motif={"enabled": True, "tokenization": "special_token"},
    )
    input_ids = torch.randint(0, 16, (2, 8), dtype=torch.long)
    attention_mask = torch.ones((2, 8), dtype=torch.long)
    residue_lengths = [11, 9]
    token_residue_lengths = [[1, 4, 1, 1, 1, 1, 2], [2, 1, 3, 1, 1, 1]]
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        residue_lengths=residue_lengths,
        token_residue_lengths=token_residue_lengths,
    )
    assert logits.shape == (2, 11)
