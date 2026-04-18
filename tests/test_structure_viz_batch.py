from __future__ import annotations

from pathlib import Path

import pytest

from cliper.structure_viz import (
    ParsedPdb,
    Selection,
    annotate,
    build_comparison,
    load_predictions_tsv,
    load_three_line_fasta,
    load_three_line_fasta_with_labels,
    make_html,
    parse_pdb,
    smith_waterman,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_predictions_and_fasta_lengths(tmp_path: Path) -> None:
    pred_path = _write(
        tmp_path / "pred.tsv",
        "protein_id\tposition_1based\tprobability\tpred_label\n"
        "P1\t1\t0.1\t0\n"
        "P1\t2\t0.9\t1\n"
        "P2\t1\t0.2\t0\n",
    )
    fasta_path = _write(
        tmp_path / "seq.fasta",
        ">P1\nAA\n00\n>P2\nG\n0\n",
    )
    preds = load_predictions_tsv(pred_path)
    seqs = load_three_line_fasta(fasta_path)
    assert len(preds["P1"].probabilities) == len(seqs["P1"]) == 2
    assert len(preds["P2"].probabilities) == len(seqs["P2"]) == 1
    with_labels = load_three_line_fasta_with_labels(fasta_path)
    assert with_labels["P1"].labels == "00"
    assert with_labels["P2"].sequence == "G"


def test_smith_waterman_maps_local_match() -> None:
    mapping, score, identity = smith_waterman("ABCDE", "XXABCDEYY")
    assert score > 0
    assert identity == pytest.approx(1.0)
    assert mapping[1] == 3
    assert mapping[5] == 7


def test_annotate_writes_bfactor() -> None:
    pdb_text = (
        "ATOM      1  N   ALA A   1      11.104  13.207  14.999  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      11.550  12.031  15.788  1.00 20.00           C\n"
        "ATOM      3  N   GLY A   2      12.104  11.207  16.999  1.00 20.00           N\n"
        "ATOM      4  CA  GLY A   2      13.550  10.031  17.788  1.00 20.00           C\n"
        "TER\n"
        "END\n"
    )
    parsed: ParsedPdb = parse_pdb(pdb_text)
    sel = Selection(
        source_type="RCSB",
        structure_id="TEST",
        chain="A",
        parsed=parsed,
        pdb_text=pdb_text,
        mapping={1: ("A", 1, ""), 2: ("A", 2, "")},
        mapped_count=2,
        coverage=1.0,
        identity=1.0,
    )
    annotated, _, _ = annotate(sel, [0.12, 0.87])
    lines = annotated.splitlines()
    assert lines[0][60:66].strip() == "0.12"
    assert lines[2][60:66].strip() == "0.87"


def test_build_comparison_and_html_contains_taro() -> None:
    pdb_text = (
        "ATOM      1  N   ALA A   1      11.104  13.207  14.999  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      11.550  12.031  15.788  1.00 20.00           C\n"
        "ATOM      3  N   GLY A   2      12.104  11.207  16.999  1.00 20.00           N\n"
        "ATOM      4  CA  GLY A   2      13.550  10.031  17.788  1.00 20.00           C\n"
        "ATOM      5  N   SER A   3      14.104  09.207  18.999  1.00 20.00           N\n"
        "ATOM      6  CA  SER A   3      15.550  08.031  19.788  1.00 20.00           C\n"
        "TER\n"
        "END\n"
    )
    parsed: ParsedPdb = parse_pdb(pdb_text)
    sel = Selection(
        source_type="RCSB",
        structure_id="TEST",
        chain="A",
        parsed=parsed,
        pdb_text=pdb_text,
        mapping={1: ("A", 1, ""), 2: ("A", 2, ""), 3: ("A", 3, "")},
        mapped_count=3,
        coverage=1.0,
        identity=1.0,
    )
    annotated, prob_by_res, pos_by_res = annotate(sel, [0.8, 0.7, 0.2])
    groups, residue_info, stats = build_comparison(
        selection=sel,
        prob_by_res=prob_by_res,
        pos_by_res=pos_by_res,
        true_labels="101",
        pred_binary=[1, 1, 0],
    )
    assert stats["tp"] == 1
    assert stats["fp"] == 1
    assert stats["fn"] == 1
    assert stats["tn"] == 0
    html = make_html(
        pid="P1",
        uniprot="QTEST",
        selection=sel,
        pdb_annot=annotated,
        groups=groups,
        residue_info=residue_info,
        seq_len=3,
        threshold=0.5,
        stats=stats,
    )
    assert "#BFA6E8" in html
    assert "Prediction vs True comparison" in html
