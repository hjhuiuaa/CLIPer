import pytest
import torch

from cliper.modeling import DummyBackbone, ResidueClassifier


def test_transformer_head_rejects_num_heads_not_dividing_hidden_size() -> None:
    backbone = DummyBackbone(hidden_size=64)
    with pytest.raises(ValueError, match="hidden_size"):
        ResidueClassifier(
            backbone=backbone,
            hidden_size=64,
            classifier_head={
                "type": "transformer",
                "num_layers": 2,
                "num_heads": 5,
                "ffn_dim": 128,
                "dropout": 0.1,
                "activation": "relu",
                "use_positional_encoding": True,
            },
        )


def test_transformer_head_forward_shape_matches_residue_length() -> None:
    backbone = DummyBackbone(hidden_size=64)
    model = ResidueClassifier(
        backbone=backbone,
        hidden_size=64,
        classifier_head={
            "type": "transformer",
            "num_layers": 2,
            "num_heads": 4,
            "ffn_dim": 128,
            "dropout": 0.1,
            "activation": "relu",
            "use_positional_encoding": True,
        },
    )
    input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    attention_mask = torch.ones((2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, residue_lengths=[8, 5])
    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()
