"""Export ProstT5 (T5 encoder) residue embeddings to per-protein line-oriented feature files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cliper.modeling import encode_sequences, load_backbone_and_tokenizer
from cliper.windowing import normalize_sequence
from disorder.feature_io import (
    feature_file_path,
    write_feature_manifest,
    write_residue_feature_file,
)


def _autocast_context(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


@torch.inference_mode()
def extract_prostt5_features_for_fasta(
    *,
    sequence_fasta: str | Path,
    output_dir: str | Path,
    backbone_name: str,
    batch_size: int = 2,
    device: str = "cuda",
    mixed_precision: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Read 2-line FASTA, run backbone once per batch, write `<safe_id>.resfeat.txt` per protein
    (each line = one residue embedding, space-separated floats).
    """
    from disorder.data import parse_two_line_fasta

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_two_line_fasta(sequence_fasta)
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    backbone = backbone.to(device)
    backbone.eval()
    dev = torch.device(device)
    use_amp = mixed_precision and dev.type == "cuda"

    written = 0
    skipped = 0
    for start in range(0, len(pairs), max(1, batch_size)):
        batch = pairs[start : start + batch_size]
        ids = [p[0] for p in batch]
        sequences = [normalize_sequence(p[1]) for p in batch]
        targets = [feature_file_path(out_dir, pid) for pid in ids]
        todo_idx = [i for i, t in enumerate(targets) if overwrite or not t.exists()]
        if not todo_idx:
            skipped += len(batch)
            continue

        sub_sequences = [sequences[i] for i in todo_idx]
        sub_targets = [targets[i] for i in todo_idx]

        encoded = encode_sequences(tokenizer, backbone_name, sub_sequences)
        input_ids = encoded.input_ids.to(dev)
        attention_mask = encoded.attention_mask.to(dev)
        residue_lengths = encoded.residue_lengths

        with _autocast_context(device_type=dev.type, enabled=use_amp):
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        for row, length in enumerate(residue_lengths):
            max_len = max(residue_lengths)
            if hidden.shape[1] < max_len:
                raise ValueError("Backbone sequence length shorter than packed batch max.")
            vec = hidden[row, :length, :].detach().float().cpu()
            write_residue_feature_file(sub_targets[row], vec)
            written += 1
        skipped += len(batch) - len(todo_idx)

    write_feature_manifest(out_dir, hidden_size=hidden_size, backbone_name=backbone_name)
    return {
        "output_dir": str(out_dir.resolve()),
        "proteins_in_fasta": len(pairs),
        "files_written": written,
        "files_skipped": skipped,
        "hidden_size": hidden_size,
    }


def extract_from_checkpoint_classifier(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    backbone_name: str,
    sequence_fasta: str | Path,
    output_dir: str | Path,
    batch_size: int = 2,
    device: str = "cuda",
    mixed_precision: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Use full ResidueClassifier checkpoint's backbone path: forward through backbone only, same file layout.
    `model` should be ResidueClassifier with backbone; we read `model.backbone`.
    """
    from disorder.data import parse_two_line_fasta

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_two_line_fasta(sequence_fasta)
    backbone = model.backbone
    backbone.eval()
    dev = torch.device(device)
    use_amp = mixed_precision and dev.type == "cuda"
    hidden_size = int(getattr(getattr(backbone, "config", None), "hidden_size", None) or 0)
    if hidden_size <= 0:
        raise ValueError("Cannot infer hidden_size from backbone.config")

    written = 0
    skipped = 0
    for batch_start in range(0, len(pairs), max(1, batch_size)):
        batch = pairs[batch_start : batch_start + batch_size]
        ids = [p[0] for p in batch]
        sequences = [normalize_sequence(p[1]) for p in batch]
        targets = [feature_file_path(out_dir, pid) for pid in ids]
        todo_idx = [i for i, t in enumerate(targets) if overwrite or not t.exists()]
        if not todo_idx:
            skipped += len(batch)
            continue

        sub_sequences = [sequences[i] for i in todo_idx]
        sub_targets = [targets[i] for i in todo_idx]

        encoded = encode_sequences(tokenizer, backbone_name, sub_sequences)
        input_ids = encoded.input_ids.to(dev)
        attention_mask = encoded.attention_mask.to(dev)
        residue_lengths = encoded.residue_lengths

        with _autocast_context(device_type=dev.type, enabled=use_amp):
            if getattr(model, "freeze_backbone", True):
                with torch.no_grad():
                    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        for row, length in enumerate(residue_lengths):
            max_len = max(residue_lengths)
            if hidden.shape[1] < max_len:
                raise ValueError("Backbone sequence length shorter than packed batch max.")
            vec = hidden[row, :length, :].detach().float().cpu()
            write_residue_feature_file(sub_targets[row], vec)
            written += 1
        skipped += len(batch) - len(todo_idx)

    write_feature_manifest(out_dir, hidden_size=hidden_size, backbone_name=backbone_name)
    return {
        "output_dir": str(out_dir.resolve()),
        "proteins_in_fasta": len(pairs),
        "files_written": written,
        "files_skipped": skipped,
        "hidden_size": hidden_size,
    }
