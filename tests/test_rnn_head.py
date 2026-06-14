import pytest
import torch

from cliper.modeling import DummyBackbone, ResidueClassifier
from cliper.pipeline import load_config
from disorder.feature_modeling import DisorderFeatureClassifier


def test_crnn_head_forward_shape_matches_residue_length() -> None:
    backbone = DummyBackbone(hidden_size=64)
    model = ResidueClassifier(
        backbone=backbone,
        hidden_size=64,
        local_context={"enabled": True, "radius": 1, "mode": "concat_window", "include_self": True},
        classifier_head={
            "type": "crnn",
            "conv_channels": [32, 32],
            "kernel_size": 3,
            "dilations": [1, 2],
            "rnn_hidden_size": 32,
            "rnn_num_layers": 1,
            "rnn_type": "gru",
            "bidirectional": True,
            "dropout": 0.1,
            "activation": "relu",
        },
    )
    input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    attention_mask = torch.ones((2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, residue_lengths=[8, 5])
    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()


def test_rnn_head_forward_shape_matches_residue_length() -> None:
    backbone = DummyBackbone(hidden_size=64)
    model = ResidueClassifier(
        backbone=backbone,
        hidden_size=64,
        local_context={"enabled": False},
        classifier_head={
            "type": "rnn",
            "rnn_hidden_size": 32,
            "rnn_num_layers": 1,
            "rnn_type": "lstm",
            "bidirectional": False,
            "dropout": 0.1,
        },
    )
    input_ids = torch.randint(0, 32, (1, 10), dtype=torch.long)
    attention_mask = torch.ones((1, 10), dtype=torch.long)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, residue_lengths=[10])
    assert logits.shape == (1, 10)
    assert torch.isfinite(logits).all()


def test_stage7_config_loads_crnn_head() -> None:
    config = load_config("configs/stage7_prostt5_crnn_concat_window_t1.yaml")
    assert config["stage"] == "stage7"
    assert config["classifier_head"]["type"] == "crnn"
    assert config["classifier_head"]["bidirectional"] is True


def test_load_config_rejects_invalid_rnn_type(tmp_path) -> None:
    config_path = tmp_path / "bad_rnn.yaml"
    config_path.write_text(
        """
stage: stage7
backbone_name: dummy
window_size: 64
batch_tokens: 128
optimizer: adamw
lr: 0.001
weight_decay: 0.0
max_epochs: 1
early_stop_patience: 1
seed: 42
threshold_search:
  min: 0.1
  max: 0.9
  step: 0.1
train_fasta: dataset/disprot_202312_linker_label.fasta
caid_fasta: dataset/linker.fasta
split_manifest: artifacts/splits/disprot_split_seed42.json
output_dir: artifacts/runs/tmp
classifier_head:
  type: rnn
  rnn_type: invalid
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rnn_type"):
        load_config(config_path)


def test_feature_classifier_crnn_matches_residue_length() -> None:
    model = DisorderFeatureClassifier(
        hidden_size=192,
        classifier_head={
            "type": "crnn",
            "conv_channels": [32],
            "kernel_size": 3,
            "dilations": [1],
            "rnn_hidden_size": 32,
            "rnn_num_layers": 1,
            "rnn_type": "gru",
            "bidirectional": True,
            "dropout": 0.1,
            "activation": "relu",
        },
    )
    embeddings = torch.randn(2, 7, 192)
    logits = model(embeddings, residue_lengths=[7, 4])
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()
