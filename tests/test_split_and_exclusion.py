from pathlib import Path

from cliper.data import build_split_manifest, parse_three_line_fasta


def test_split_is_deterministic_and_excludes_error_and_holdout(tmp_path: Path) -> None:
    fasta = tmp_path / "train.fasta"
    fasta.write_text(
        (
            ">A\nAAAAAA\n000000\n"
            ">B\nAAAAAA\n000000\n"
            ">C\nAAAAAA\n001100\n"
            ">D\nAAAAAA\n000011\n"
            ">E\nAAAAAA\n111100\n"
        ),
        encoding="utf-8",
    )
    records = parse_three_line_fasta(fasta)
    error_ids = {"B"}
    caid_ids = {"E"}
    split_1, report_1 = build_split_manifest(
        records,
        source_fasta=fasta,
        error_ids=error_ids,
        caid_ids=caid_ids,
        seed=42,
        val_ratio=0.4,
    )
    split_2, report_2 = build_split_manifest(
        records,
        source_fasta=fasta,
        error_ids=error_ids,
        caid_ids=caid_ids,
        seed=42,
        val_ratio=0.4,
    )

    assert split_1["train_ids"] == split_2["train_ids"]
    assert split_1["val_ids"] == split_2["val_ids"]
    assert "B" not in split_1["train_ids"] + split_1["val_ids"]
    assert "E" not in split_1["train_ids"] + split_1["val_ids"]
    assert "B" in report_1["error_ids_present_in_source"]
    assert report_1 == report_2

