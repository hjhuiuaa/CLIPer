"""CAID end-to-end: ProstT5 extract + linker predict; timings include embedding time."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from cliper.caid_io import parse_caid_fasta, sanitize_sequence, write_caid_file, write_timings_csv
from cliper.caid_predict import (
    configure_runtime_threads,
    load_classifier_from_checkpoint,
    predict_sequence_from_embedding,
    safe_caid_filename,
)
from cliper.extract_embeddings import _embedding_output_path, _write_embedding
from cliper.modeling import load_backbone_and_tokenizer
from disorder.feature_io import write_feature_manifest
from disorder.sequence_embedding import encode_residue_embeddings


def runall_caid(
    *,
    checkpoint_path: str | Path,
    fasta_path: str | Path,
    output_dir: str | Path,
    backbone_name: str,
    flavor: str = "linker",
    embeddings_dir: str | Path | None = None,
    threshold: float | None = None,
    window_size: int = 1024,
    eval_window_size: int | None = None,
    eval_stride: int | None = None,
    top_k_heuristic: int | None = None,
    window_batch_size: int = 8,
    extract_batch_size: int = 1,
    device: str = "cpu",
    num_threads: int = 4,
    keep_embeddings: bool = False,
) -> dict[str, Any]:
    """
    For each FASTA record: ProstT5 encode -> CNN/CRNN predict -> .caid.

    CAID-compliant default outputs under ``output_dir``:
      - ``{flavor}/{id}.caid``
      - ``timings.csv`` (milliseconds = extract + predict)

    Embeddings / manifests are optional debug artifacts only.
    """
    configure_runtime_threads(num_threads)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    flavor_dir = out_root / flavor
    flavor_dir.mkdir(parents=True, exist_ok=True)

    emb_dir: Path | None = None
    if keep_embeddings:
        emb_dir = Path(embeddings_dir) if embeddings_dir is not None else out_root / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

    if device == "cpu" or not torch.cuda.is_available():
        torch_device = torch.device("cpu")
        use_amp = False
    else:
        torch_device = torch.device(device)
        use_amp = torch_device.type == "cuda"

    classifier, metadata = load_classifier_from_checkpoint(checkpoint_path, device=torch_device)
    config = metadata["config"]
    local_context = metadata["local_context"]
    resolved_threshold = float(threshold) if threshold is not None else float(metadata["threshold"])
    resolved_eval_window = int(
        eval_window_size if eval_window_size is not None else config.get("window_size", 1024)
    )
    resolved_stride = eval_stride if eval_stride is not None else config.get("eval_stride")
    resolved_top_k = int(
        top_k_heuristic if top_k_heuristic is not None else config.get("top_k_heuristic", 4)
    )

    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(backbone_name)
    backbone = backbone.to(torch_device)
    backbone.eval()

    records = parse_caid_fasta(fasta_path)
    timings_ms: list[tuple[str, int]] = []
    caid_files: list[str] = []

    for protein_id, sequence in records:
        started = time.perf_counter()
        seq = sanitize_sequence(sequence)

        embeddings = encode_residue_embeddings(
            sequence=seq,
            backbone=backbone,
            tokenizer=tokenizer,
            backbone_name=backbone_name,
            window_size=int(window_size),
            batch_size=int(extract_batch_size),
            device=torch_device,
            mixed_precision=use_amp,
        )
        if keep_embeddings and emb_dir is not None:
            emb_path = _embedding_output_path(emb_dir, protein_id, "npy")
            _write_embedding(emb_path, embeddings, "npy")

        probabilities = predict_sequence_from_embedding(
            classifier,
            sequence=seq,
            embedding=embeddings if isinstance(embeddings, torch.Tensor) else torch.as_tensor(embeddings),
            local_context=local_context,
            window_size=resolved_eval_window,
            eval_stride=resolved_stride,
            top_k_heuristic=resolved_top_k,
            device=torch_device,
            window_batch_size=window_batch_size,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        timings_ms.append((protein_id, elapsed_ms))

        caid_path = flavor_dir / f"{safe_caid_filename(protein_id)}.caid"
        write_caid_file(
            caid_path,
            protein_id=protein_id,
            sequence=seq,
            probabilities=probabilities,
            threshold=resolved_threshold,
        )
        caid_files.append(str(caid_path))

    if keep_embeddings and emb_dir is not None:
        write_feature_manifest(
            emb_dir,
            hidden_size=hidden_size,
            backbone_name=backbone_name,
            extra={
                "task": "linker_caid_runall",
                "output_format": "npy",
                "window_size": int(window_size),
                "timings_include": "extract_plus_predict",
            },
        )

    timings_path = out_root / "timings.csv"
    write_timings_csv(timings_path, timings_ms)

    return {
        "mode": "runall",
        "output_dir": str(out_root.resolve()),
        "flavor": flavor,
        "num_sequences": len(records),
        "timings_path": str(timings_path.resolve()),
        "caid_files": caid_files,
    }
