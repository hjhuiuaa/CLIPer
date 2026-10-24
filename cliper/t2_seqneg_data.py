"""Sequence-level negatives: all linker proteins + equally many disorder-only chains.

DisProt linker.fasta and disorder.fasta share the same 2896 IDs. Extra negatives
are the proteins with no linker residue (not a second database).

No residue undersampling: every assignable 0/1/2 on selected chains is kept.
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
from cliper.t2_balanced_data import (
    REPO_ROOT,
    _count_chars,
    _label_caid_ternary,
    _records_from_ids,
    _write_three_line_fasta,
    _write_two_line_fasta,
    default_caid_disorder_fasta,
    default_caid_fasta,
    default_disorder_fasta,
    default_error_file,
    default_linker_fasta,
    ternary_labels,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


SELECTION_REASON = """\
Sequence-level 1:1 negatives (no residue undersampling).

1. Keep every DisProt protein that has at least one linker residue (after
   dropping CAID3 holdout and error.txt). All residues on those chains stay
   labeled 0/1/2/-; nothing is dropped to '-' by sampling.

2. Linker.fasta and disorder.fasta are the same 2896 DisProt IDs. Extra
   negatives are proteins with zero linker residues. Sample N_extra = N_linker
   sequences, preferring chains that contain both ordered (0) and other-IDR (1).

3. Split linker proteins 80/20 (seed 42) and extra proteins 80/20 independently
   so train/val each stay sequence-balanced.

4. Labels:
     0 = ordered (non-disorder); disorder-track '-' is also 0
     1 = other IDR (disorder, not linker)
     2 = linker
     - = unused here (no residue undersample)

5. CAID3 test is still the 31 linker sequences, labeled from linker + disorder-PDB
   (disorder '-' counted as ordered 0).
"""


def _pool_counts(records: list[ProteinRecord]) -> dict[str, int]:
    pooled = {"0": 0, "1": 0, "2": 0, "-": 0}
    for rec in records:
        for char, count in _count_chars(rec.labels).items():
            pooled[char] = pooled.get(char, 0) + count
    return pooled


def _has_both_neg_types(labels: str) -> bool:
    return ("0" in labels) and ("1" in labels)


def _split_ids(ids: list[str], *, seed: int, val_ratio: float) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    return train_ids, val_ids


def build_t2_seqneg_dataset(
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
    out_dir = Path(output_dir) if output_dir else (REPO_ROOT / "dataset" / "t2_seqneg")
    split_path = (
        Path(split_out) if split_out else (REPO_ROOT / "artifacts" / "splits" / "t2_seqneg_split_seed42.json")
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

    linker_full: list[ProteinRecord] = []
    extra_pool: list[ProteinRecord] = []
    skipped_missing_disorder = 0
    skipped_length_mismatch = 0
    for rec in linker_recs:
        if rec.protein_id not in disorder_map:
            skipped_missing_disorder += 1
            continue
        disorder_rec = disorder_map[rec.protein_id]
        if len(disorder_rec.labels) != len(rec.labels) or disorder_rec.sequence != rec.sequence:
            skipped_length_mismatch += 1
            continue
        labels = ternary_labels(rec.labels, disorder_rec.labels)
        labeled = ProteinRecord(rec.protein_id, rec.sequence, labels)
        if labels.count("2") > 0:
            linker_full.append(labeled)
        else:
            extra_pool.append(labeled)

    linker_manifest, exclusion_report = build_split_manifest(
        linker_full,
        source_fasta=linker_path,
        error_ids=error_ids,
        caid_ids=caid_ids,
        seed=seed,
        val_ratio=val_ratio,
    )
    n_linker = int(linker_manifest["counts"]["eligible_records"])
    blocked = set(linker_manifest["excluded_error_ids"]) | set(
        linker_manifest["excluded_holdout_overlap_ids"]
    ) | caid_ids | error_ids
    extra_eligible = [rec for rec in extra_pool if rec.protein_id not in blocked]
    preferred = [rec for rec in extra_eligible if _has_both_neg_types(rec.labels)]
    fallback = [rec for rec in extra_eligible if not _has_both_neg_types(rec.labels)]
    rng = random.Random(seed)
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    chosen_extra = (preferred + fallback)[:n_linker]
    if len(chosen_extra) < n_linker:
        raise ValueError(
            f"Not enough disorder-only sequences: need {n_linker}, got {len(chosen_extra)}"
        )
    extra_ids = [rec.protein_id for rec in chosen_extra]
    extra_train_ids, extra_val_ids = _split_ids(extra_ids, seed=seed + 7, val_ratio=val_ratio)

    train_linker = _records_from_ids(linker_full, linker_manifest["train_ids"])
    val_linker = _records_from_ids(linker_full, linker_manifest["val_ids"])
    extra_by_id = {rec.protein_id: rec for rec in chosen_extra}
    train_extra = [extra_by_id[pid] for pid in extra_train_ids]
    val_extra = [extra_by_id[pid] for pid in extra_val_ids]
    train_bal = train_linker + train_extra
    val_bal = val_linker + val_extra
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

    split_manifest = dict(linker_manifest)
    split_manifest["train_ids"] = sorted(linker_manifest["train_ids"] + extra_train_ids)
    split_manifest["val_ids"] = sorted(linker_manifest["val_ids"] + extra_val_ids)
    split_manifest["linker_train_ids"] = linker_manifest["train_ids"]
    split_manifest["linker_val_ids"] = linker_manifest["val_ids"]
    split_manifest["extra_train_ids"] = extra_train_ids
    split_manifest["extra_val_ids"] = extra_val_ids
    split_manifest["counts"] = dict(linker_manifest["counts"])
    split_manifest["counts"]["train_records"] = len(train_bal)
    split_manifest["counts"]["val_records"] = len(val_bal)
    split_manifest["counts"]["linker_train_records"] = len(train_linker)
    split_manifest["counts"]["linker_val_records"] = len(val_linker)
    split_manifest["counts"]["extra_train_records"] = len(train_extra)
    split_manifest["counts"]["extra_val_records"] = len(val_extra)
    split_manifest["sampling"] = "sequence_level_1to1_disorder_only"
    exclusion_path = split_path.with_name("t2_seqneg_exclusion_seed42.json")
    write_json(split_path, split_manifest)
    write_json(exclusion_path, exclusion_report)

    readme = out_dir / "README.md"
    readme.write_text(
        "# T2 sequence-level negatives\n\n"
        + SELECTION_REASON
        + "\n## Files\n\n"
        + "- `train.fasta` / `val.fasta`: 3-line FASTA with labels {0,1,2,-}, no residue undersample\n"
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
            "-": "unused (no residue undersample)",
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
            "linker_positive_proteins": len(linker_full),
            "disorder_only_pool": len(extra_pool),
            "skipped_missing_disorder": skipped_missing_disorder,
            "skipped_length_mismatch": skipped_length_mismatch,
            "extra_eligible": len(extra_eligible),
            "extra_preferred_both_neg_types": len(preferred),
            "extra_sampled": len(chosen_extra),
            "caid3_test_proteins": len(caid_ternary),
            "error_ids_in_file": len(error_ids),
            "caid_disorder_label_skip": caid_label_skip,
        },
        "split": split_manifest["counts"],
        "train_ids": split_manifest["train_ids"],
        "val_ids": split_manifest["val_ids"],
        "linker_train_ids": linker_manifest["train_ids"],
        "linker_val_ids": linker_manifest["val_ids"],
        "extra_train_ids": extra_train_ids,
        "extra_val_ids": extra_val_ids,
        "excluded_error_ids": split_manifest["excluded_error_ids"],
        "excluded_holdout_overlap_ids": split_manifest["excluded_holdout_overlap_ids"],
        "caid_holdout_ids": split_manifest["caid_holdout_ids"],
        "sampling": {
            "mode": "sequence_level",
            "n_linker_eligible": n_linker,
            "n_extra_sampled": len(chosen_extra),
            "train_label_counts": _pool_counts(train_bal),
            "val_label_counts": _pool_counts(val_bal),
            "test_label_counts": _pool_counts(caid_ternary),
            "train_linker_label_counts": _pool_counts(train_linker),
            "train_extra_label_counts": _pool_counts(train_extra),
        },
        "seed": seed,
        "val_ratio": val_ratio,
    }
    write_json(out_dir / "selection_stats.json", stats)
    return stats
