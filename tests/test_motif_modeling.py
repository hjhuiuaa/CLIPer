import torch

from cliper.modeling import DualTokenizerResidueClassifier, DummyBackbone, ResidueClassifier


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


class _ConstantResidueClassifier(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_lengths: list[int],
        *,
        token_residue_lengths: list[list[int]] | None = None,
    ) -> torch.Tensor:
        return self.value.expand(input_ids.shape[0], max(residue_lengths))


def test_dual_tokenizer_classifier_returns_weighted_fused_logits() -> None:
    model = DualTokenizerResidueClassifier(
        plain_model=_ConstantResidueClassifier(2.0),
        special_model=_ConstantResidueClassifier(6.0),
        plain_weight=0.5,
        special_weight=0.5,
    )
    input_ids = torch.ones((2, 5), dtype=torch.long)
    attention_mask = torch.ones((2, 5), dtype=torch.long)
    result = model(
        plain_input_ids=input_ids,
        plain_attention_mask=attention_mask,
        plain_residue_lengths=[4, 4],
        special_input_ids=input_ids,
        special_attention_mask=attention_mask,
        special_residue_lengths=[4, 4],
        special_token_residue_lengths=[[1, 1, 1, 1], [1, 1, 1, 1]],
    )
    assert result["plain_logits"].shape == (2, 4)
    assert result["special_logits"].shape == (2, 4)
    assert torch.allclose(result["fused_logits"], torch.full((2, 4), 4.0))
