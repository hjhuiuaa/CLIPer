from pathlib import Path

import pytest

from cliper.data import parse_three_line_fasta


def test_parse_three_line_fasta_valid(tmp_path: Path) -> None:
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(">P1\nACDE\n0101\n>P2\nAAAA\n0000\n", encoding="utf-8")
    records = parse_three_line_fasta(fasta)
    assert len(records) == 2
    assert records[0].protein_id == "P1"
    assert records[1].labels == "0000"


def test_parse_three_line_fasta_mismatch_raises(tmp_path: Path) -> None:
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(">P1\nACDE\n010\n", encoding="utf-8")
    with pytest.raises(ValueError, match="length mismatch"):
        parse_three_line_fasta(fasta)

