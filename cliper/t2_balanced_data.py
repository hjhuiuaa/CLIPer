"""Build residue-balanced ternary (0/1/2) train/val FASTAs with CAID3 held out.

Encoding:
  0 = ordered (non-disorder); disorder-track '-' is mapped to 0
  1 = other IDR (disorder, not linker)
  2 = linker
  - = ignore (unsampled training negatives only)

Selection policy (see also dataset/t2_balanced/README.md written by the builder):

- Keep every DisProt protein that actually has a linker residue ("positive sequences").
- Drop CAID3 linker ids (test) and error.txt disagreement ids.
- Keep ALL linker residues (label 2). Sample ordered (0) and other-IDR (1)
  so that N_neg = N_pos and N_0 ≈ N_1, independently on train and on val.
- Unsampled negatives are marked '-' (ignored by CE / SupCon).
- CAID3 test is labeled the same way from linker.fasta + disorder_pdb.fasta
  (disorder '-' counted as ordered 0; no residue undersampling).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
from typing import Any

from cliper.data import (
    ProteinRecord,
    build_split_manifest,
    parse_id_lines,
    parse_three_line_fasta,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of these paths exist:\n" + "\n".join(f"  - {p}" for p in candidates)
    )


def default_linker_fasta() -> Path:
    return resolve_existing_path(
        [
            REPO_ROOT / "dataset" / "disprot9.5" / "disprot_202312_linker_label.fasta",
            REPO_ROOT.parent / "dataset" / "disprot9.5" / "disprot_202312_linker_label.fasta",
            REPO_ROOT / "dataset" / "disprot_202312_linker_label.fasta",
        ]
    )


def default_disorder_fasta() -> Path:
    return resolve_existing_path(
        [
            REPO_ROOT / "dataset" / "disprot9.5" / "disprot_202312_label.fasta",
            REPO_ROOT.parent / "dataset" / "disprot9.5" / "disprot_202312_label.fasta",
            REPO_ROOT / "dataset" / "disprot_202312_label.fasta",
        ]
    )


def default_error_file() -> Path:
    return resolve_existing_path(
        [
            REPO_ROOT / "dataset" / "disprot9.5" / "error.txt",
            REPO_ROOT.parent / "dataset" / "disprot9.5" / "error.txt",
            REPO_ROOT / "dataset" / "error.txt",
        ]
    )


def default_caid_fasta() -> Path:
    return resolve_existing_path(
        [
            REPO_ROOT / "dataset" / "caid3" / "linker.fasta",
            REPO_ROOT.parent / "dataset" / "caid3" / "linker.fasta",
            REPO_ROOT / "dataset" / "linker.fasta",
        ]
    )


def default_caid_disorder_fasta() -> Path:
    return resolve_existing_path(
        [
            REPO_ROOT / "dataset" / "caid3" / "disorder_pdb.fasta",
            REPO_ROOT.parent / "dataset" / "caid3" / "disorder_pdb.fasta",
        ]
    )


def ternary_labels(linker: str, disorder: str) -> str:
    """Map binary linker + disorder strings to 0/1/2.

    0 = ordered (non-disorder), including disorder-track '-' (unannotated)
    1 = other IDR (disorder, not linker)
    2 = linker
    Linker annotation wins when linker==1.
    Disorder '-' is treated as 0, not ignore.
    """
    if len(linker) != len(disorder):
        raise ValueError(f"Length mismatch: linker={len(linker)} disorder={len(disorder)}")
    out: list[str] = []
    for linker_ch, disorder_ch in zip(linker, disorder):
        if linker_ch == "1":
            out.append("2")
        elif disorder_ch == "1":
            out.append("1")
        else:
            out.append("0")
    return "".join(out)


def _count_chars(labels: str) -> dict[str, int]:
    counts = {"0": 0, "1": 0, "2": 0, "-": 0}
    for char in labels:
        counts[char] = counts.get(char, 0) + 1
    return counts


def _write_three_line_fasta(path: Path, records: list[ProteinRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(f">{rec.protein_id}\n{rec.sequence}\n{rec.labels}\n")


def _write_two_line_fasta(path: Path, records: list[ProteinRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(f">{rec.protein_id}\n{rec.sequence}\n")


def subsample_negatives_to_match_positives(
    records: list[ProteinRecord],
    *,
    rng: random.Random,
) -> tuple[list[ProteinRecord], dict[str, Any]]:
    """Keep all linker (2); sample N_0 ≈ N_1 with N_0 + N_1 = N_2. Unsampled -> '-'."""
    pos_idx: list[tuple[int, int]] = []
    idx0: list[tuple[int, int]] = []
    idx1: list[tuple[int, int]] = []
    for rec_i, rec in enumerate(records):
        for res_j, char in enumerate(rec.labels):
            if char == "2":
                pos_idx.append((rec_i, res_j))
            elif char == "0":
                idx0.append((rec_i, res_j))
            elif char == "1":
                idx1.append((rec_i, res_j))

    n_pos = len(pos_idx)
    if n_pos == 0:
        raise ValueError("Split contains no linker (class 2) residues.")

    n_neg = n_pos
    n0_target = n_neg // 2
    n1_target = n_neg - n0_target
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    take0 = idx0[: min(n0_target, len(idx0))]
    take1 = idx1[: min(n1_target, len(idx1))]
    short0 = n0_target - len(take0)
    short1 = n1_target - len(take1)
    leftover0 = idx0[len(take0) :]
    leftover1 = idx1[len(take1) :]
    if short0 > 0:
        extra = leftover1[:short0]
        take1.extend(extra)
        leftover1 = leftover1[len(extra) :]
    if short1 > 0:
        extra = leftover0[:short1]
        take0.extend(extra)
        leftover0 = leftover0[len(extra) :]

    keep = set(take0 + take1)
    sampled: list[ProteinRecord] = []
    for rec_i, rec in enumerate(records):
        chars: list[str] = []
        for res_j, char in enumerate(rec.labels):
            if char == "2":
                chars.append("2")
            elif (rec_i, res_j) in keep:
                chars.append(char)
            else:
                chars.append("-")
        sampled.append(ProteinRecord(rec.protein_id, rec.sequence, "".join(chars)))

    stats = {
        "num_proteins": len(sampled),
        "before": {
            "linker_2": n_pos,
            "other_idr_1": len(idx1),
            "ordered_0": len(idx0),
        },
        "after": {
            "linker_2": n_pos,
            "other_idr_1": len(take1),
            "ordered_0": len(take0),
            "ignore": sum(rec.labels.count("-") for rec in sampled),
        },
        "targets": {
            "n_neg_equals_n_pos": n_neg,
            "n0_target": n0_target,
            "n1_target": n1_target,
            "filled_ordered_from_other_idr": max(0, short0),
            "filled_other_idr_from_ordered": max(0, short1),
        },
    }
    return sampled, stats


def _records_from_ids(records: list[ProteinRecord], ids: list[str]) -> list[ProteinRecord]:
    by_id = {rec.protein_id: rec for rec in records}
    missing = [pid for pid in ids if pid not in by_id]
    if missing:
        raise ValueError(f"Split ids missing from labeled records: {missing[:10]}")
    return [by_id[pid] for pid in ids]


SELECTION_REASON = """\
Why these sequences (and residues)?

1. Sequence set = all DisProt proteins that contain at least one linker residue.
   Hard negatives live on the same chain as the linker.

2. CAID3's 31 linker sequences are the test set. They are excluded from
   train and val. error.txt disagreement ids are also excluded. Test labels
   are built from CAID3 linker.fasta + disorder_pdb.fasta (disorder-PDB track).

3. Residue labels are ternary:
     0 = ordered (non-disorder); disorder-track '-' is also 0
     1 = other IDR (disorder, not linker)
     2 = linker
     - = ignore (unsampled negative only)

4. Count matching is residue-level: keep ALL linker residues (2), then sample
   N_neg = N_pos negatives with ordered:other-IDR ≈ 1:1. Train/val undersample;
   the CAID3 test set keeps every assignable 0/1/2 residue.

5. Train and val are split at the protein level (seed 42, 80/20). The same
   residue-sampling protocol is applied independently on each split.
"""


def _label_caid_ternary(
    caid_recs: list[ProteinRecord],
    disorder_map: dict[str, ProteinRecord],
) -> tuple[list[ProteinRecord], dict[str, int]]:
    labeled: list[ProteinRecord] = []
    skipped_missing = 0
    skipped_mismatch = 0
    for rec in caid_recs:
        if rec.protein_id not in disorder_map:
            skipped_missing += 1
            continue
        disorder_rec = disorder_map[rec.protein_id]
        if len(disorder_rec.labels) != len(rec.labels) or disorder_rec.sequence != rec.sequence:
            skipped_mismatch += 1
            continue
        labels = ternary_labels(rec.labels, disorder_rec.labels)
        labeled.append(ProteinRecord(rec.protein_id, rec.sequence, labels))
    return labeled, {"missing_disorder": skipped_missing, "length_or_seq_mismatch": skipped_mismatch}


def build_t2_balanced_dataset(
    *,
    linker_fasta: str | Path | None = None,
    disorder_fasta: str | Path | None = None,
    error_file: str | Path | None = None,
    caid_fasta: str | Path | None = None,
    caid_disorder_fasta: str | Path | None = None,
    output_dir: str | Path | None = None,
    split_out: str | Path | None = None,
    seed: int = 42,
    val_ratio: float = 0.2,
) -> dict[str, Any]:
    linker_path = Path(linker_fasta) if linker_fasta else default_linker_fasta()
    disorder_path = Path(disorder_fasta) if disorder_fasta else default_disorder_fasta()
    error_path = Path(error_file) if error_file else default_error_file()
    caid_path = Path(caid_fasta) if caid_fasta else default_caid_fasta()
    caid_disorder_path = (
        Path(caid_disorder_fasta) if caid_disorder_fasta else default_caid_disorder_fasta()
    )
    out_dir = Path(output_dir) if output_dir else (REPO_ROOT / "dataset" / "t2_balanced")
    split_path = (
        Path(split_out) if split_out else (REPO_ROOT / "artifacts" / "splits" / "t2_balanced_split_seed42.json")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    linker_recs = parse_three_line_fasta(linker_path)
    disorder_map = {rec.protein_id: rec for rec in parse_three_line_fasta(disorder_path)}
    error_ids = parse_id_lines(error_path)
    caid_recs = parse_three_line_fasta(caid_path)
    caid_ids = {rec.protein_id for rec in caid_recs}
    caid_disorder_map = {rec.protein_id: rec for rec in parse_three_line_fasta(caid_disorder_path)}
    caid_ternary, caid_label_skip = _label_caid_ternary(caid_recs, caid_disorder_map)
    if len(caid_ternary) != len(caid_recs):
        raise ValueError(
            "Failed to assign 0/1/2 labels to every CAID3 linker sequence from disorder_pdb: "
            f"{caid_label_skip}"
        )

    positive_full: list[ProteinRecord] = []
    skipped_all_zero = 0
    skipped_missing_disorder = 0
    skipped_length_mismatch = 0
    for rec in linker_recs:
        if rec.labels.count("1") == 0:
            skipped_all_zero += 1
            continue
        if rec.protein_id not in disorder_map:
            skipped_missing_disorder += 1
            continue
        disorder_rec = disorder_map[rec.protein_id]
        if len(disorder_rec.labels) != len(rec.labels) or disorder_rec.sequence != rec.sequence:
            skipped_length_mismatch += 1
            continue
        labels = ternary_labels(rec.labels, disorder_rec.labels)
        positive_full.append(ProteinRecord(rec.protein_id, rec.sequence, labels))

    split_manifest, exclusion_report = build_split_manifest(
        positive_full,
        source_fasta=linker_path,
        error_ids=error_ids,
        caid_ids=caid_ids,
        seed=seed,
        val_ratio=val_ratio,
    )

    train_full = _records_from_ids(positive_full, split_manifest["train_ids"])
    val_full = _records_from_ids(positive_full, split_manifest["val_ids"])
    rng_train = random.Random(seed)
    rng_val = random.Random(seed + 1)
    train_bal, train_sample_stats = subsample_negatives_to_match_positives(train_full, rng=rng_train)
    val_bal, val_sample_stats = subsample_negatives_to_match_positives(val_full, rng=rng_val)
    train_val = train_bal + val_bal

    train_fasta = out_dir / "train.fasta"
    val_fasta = out_dir / "val.fasta"
    train_val_fasta = out_dir / "train_val.fasta"
    test_fasta = out_dir / "test_caid3.fasta"
    sequences_fasta = out_dir / "sequences_extract.fasta"
    _write_three_line_fasta(train_fasta, train_bal)
    _write_three_line_fasta(val_fasta, val_bal)
    _write_three_line_fasta(train_val_fasta, train_val)
    _write_three_line_fasta(test_fasta, caid_ternary)
    _write_two_line_fasta(sequences_fasta, train_val + caid_ternary)
    shutil.copy2(error_path, out_dir / "error.txt")

    exclusion_path = split_path.with_name("t2_balanced_exclusion_seed42.json")
    write_json(split_path, split_manifest)
    write_json(exclusion_path, exclusion_report)

    def _pool_counts(records: list[ProteinRecord]) -> dict[str, int]:
        pooled = {"0": 0, "1": 0, "2": 0, "-": 0}
        for rec in records:
            for char, count in _count_chars(rec.labels).items():
                pooled[char] = pooled.get(char, 0) + count
        return pooled

    readme = out_dir / "README.md"
    readme.write_text(
        "# T2-balanced ternary dataset\n\n"
        + SELECTION_REASON
        + "\n## Files\n\n"
        + "- `train.fasta` / `val.fasta`: 3-line FASTA with labels {0,1,2,-}\n"
        + "- `train_val.fasta`: concatenation used by `cliper train`\n"
        + "- `test_caid3.fasta`: CAID3 31 sequences labeled 0/1/2 from linker + disorder-PDB\n"
        + "- `sequences_extract.fasta`: 2-line sequences for ProstT5 extract (train+val+test)\n"
        + "- `selection_stats.json`: counts and exclusion details\n\n"
        + f"Split manifest: `{split_path.as_posix()}`\n",
        encoding="utf-8",
    )

    stats: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "encoding": {
            "0": "ordered (non-disorder); disorder '-' mapped to 0",
            "1": "other IDR (disorder, not linker)",
            "2": "linker",
            "-": "ignore / unsampled negative",
        },
        "selection_reason": SELECTION_REASON.strip(),
        "paths": {
            "linker_fasta": str(linker_path.resolve()),
            "disorder_fasta": str(disorder_path.resolve()),
            "error_file": str(error_path.resolve()),
            "caid_fasta": str(caid_path.resolve()),
            "caid_disorder_fasta": str(caid_disorder_path.resolve()),
            "output_dir": str(out_dir.resolve()),
            "train_fasta": str(train_fasta.resolve()),
            "val_fasta": str(val_fasta.resolve()),
            "train_val_fasta": str(train_val_fasta.resolve()),
            "test_fasta": str(test_fasta.resolve()),
            "sequences_extract_fasta": str(sequences_fasta.resolve()),
            "split_manifest": str(split_path.resolve()),
            "exclusion_report": str(exclusion_path.resolve()),
        },
        "filters": {
            "linker_source_proteins": len(linker_recs),
            "skipped_all_zero_linker": skipped_all_zero,
            "skipped_missing_disorder": skipped_missing_disorder,
            "skipped_length_mismatch": skipped_length_mismatch,
            "positive_proteins_before_exclusion": len(positive_full),
            "caid3_test_proteins": len(caid_ternary),
            "error_ids_in_file": len(error_ids),
            "caid_disorder_label_skip": caid_label_skip,
        },
        "split": split_manifest["counts"],
        "train_ids": split_manifest["train_ids"],
        "val_ids": split_manifest["val_ids"],
        "excluded_error_ids": split_manifest["excluded_error_ids"],
        "excluded_holdout_overlap_ids": split_manifest["excluded_holdout_overlap_ids"],
        "caid_holdout_ids": split_manifest["caid_holdout_ids"],
        "sampling": {
            "train": train_sample_stats,
            "val": val_sample_stats,
            "train_label_counts": _pool_counts(train_bal),
            "val_label_counts": _pool_counts(val_bal),
            "test_label_counts": _pool_counts(caid_ternary),
        },
        "seed": seed,
        "val_ratio": val_ratio,
    }
    write_json(out_dir / "selection_stats.json", stats)
    return stats
