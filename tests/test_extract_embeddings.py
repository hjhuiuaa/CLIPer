from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cliper.caid_io import read_residue_embedding
from cliper.extract_embeddings import extract_prostt5_embeddings_for_fasta


pytest.importorskip("torch")


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    lines: list[str] = []
    for protein_id, sequence in records:
        lines.append(f">{protein_id}")
        lines.append(sequence)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_extract_embeddings_dummy_short_and_long(tmp_path: Path) -> None:
    fasta = tmp_path / "targets.fasta"
    short_seq = "ACDEFGHIK" * 10
    long_seq = "ACDEFGHIK" * 200
    _write_fasta(
        fasta,
        [
            ("P_SHORT", short_seq),
            ("P_LONG", long_seq),
        ],
    )
    out_dir = tmp_path / "embeddings"

    result = extract_prostt5_embeddings_for_fasta(
        fasta_path=fasta,
        output_dir=out_dir,
        backbone_name="dummy",
        window_size=1024,
        batch_size=1,
        device="cpu",
        mixed_precision=False,
        overwrite=True,
        output_format="resfeat",
    )

    assert result["files_written"] == 2
    assert result["short_sequences_written"] == 1
    assert result["long_sequences_written"] == 1

    short_emb = read_residue_embedding(out_dir, "P_SHORT")
    long_emb = read_residue_embedding(out_dir, "P_LONG")
    assert short_emb.shape[0] == len(short_seq)
    assert long_emb.shape[0] == len(long_seq)
    assert short_emb.shape[1] == long_emb.shape[1] == int(result["hidden_size"])


def test_extract_embeddings_npy_format(tmp_path: Path) -> None:
    fasta = tmp_path / "one.fasta"
    seq = "MKTAYIAK"
    _write_fasta(fasta, [("P1", seq)])
    out_dir = tmp_path / "npy_out"

    extract_prostt5_embeddings_for_fasta(
        fasta_path=fasta,
        output_dir=out_dir,
        backbone_name="dummy",
        device="cpu",
        mixed_precision=False,
        overwrite=True,
        output_format="npy",
    )

    arr = np.load(out_dir / "P1.npy")
    assert arr.shape == (len(seq), arr.shape[1])
    loaded = read_residue_embedding(out_dir, "P1")
    assert loaded.shape[0] == len(seq)
