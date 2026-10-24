#!/usr/bin/env python3
"""Build the 62-sequence scene-2/3 test set from CAID3 disorder-PDB.

Positives are the 31 CAID3 linker proteins (all of them are in disorder-PDB).
Negatives are 31 other disorder-PDB proteins that have disordered residues,
are not linker proteins, and are absent from the sequence-level train/val split.
Scene 1 keeps dataset/t2_seqneg/test_caid3.fasta unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_three_line(path: Path) -> list[tuple[str, str, str]]:
    text = path.read_text(encoding="utf-8")
    records: list[tuple[str, str, str]] = []
    header: str | None = None
    lines: list[str] = []
    for raw in text.splitlines():
        if raw.startswith(">"):
            if header is not None:
                if len(lines) < 2:
                    raise ValueError(f"{path}: record {header} needs sequence and labels")
                records.append((header, lines[0], lines[1]))
            header = raw[1:].split()[0]
            lines = []
        elif header is not None:
            lines.append(raw.strip())
    if header is not None:
        if len(lines) < 2:
            raise ValueError(f"{path}: record {header} needs sequence and labels")
        records.append((header, lines[0], lines[1]))
    return records


def _disorder_nonlinker_labels(disorder_labels: str) -> str:
    return "".join("1" if char == "1" else "0" for char in disorder_labels)


def _prediction_ids(zip_path: Path) -> set[str]:
    if not zip_path.is_file():
        return set()
    with zipfile.ZipFile(zip_path) as archive:
        name = next(
            (item for item in archive.namelist() if item.endswith("APOD.caid")),
            None,
        )
        if name is None:
            return set()
        text = archive.read(name).decode("utf-8", errors="replace")
    return set(re.findall(r"^>(\S+)", text, flags=re.M))


def _write_three(path: Path, records: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for protein_id, sequence, labels in records:
            handle.write(f">{protein_id}\n{sequence}\n{labels}\n")


def _write_two(path: Path, records: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for protein_id, sequence, _labels in records:
            handle.write(f">{protein_id}\n{sequence}\n")


def build(args: argparse.Namespace) -> dict:
    pdb_records = _parse_three_line(args.disorder_pdb)
    linker_records = _parse_three_line(args.linker_fasta)
    scene1_records = _parse_three_line(args.scene1_fasta)
    stats = json.loads(args.split_stats.read_text(encoding="utf-8"))
    blocked = set(stats.get("train_ids", [])) | set(stats.get("val_ids", []))
    linker_ids = [rec[0] for rec in linker_records]
    linker_id_set = set(linker_ids)
    pdb_by_id = {rec[0]: rec for rec in pdb_records}
    scene1_by_id = {rec[0]: rec for rec in scene1_records}

    missing_scene1 = [pid for pid in linker_ids if pid not in scene1_by_id]
    missing_pdb = [pid for pid in linker_ids if pid not in pdb_by_id]
    if missing_scene1 or missing_pdb or len(linker_ids) != 31:
        raise SystemExit(
            "Scene-1 linker set is not the 31 CAID3 proteins inside disorder-PDB: "
            f"n_linker={len(linker_ids)} missing_scene1={missing_scene1} missing_pdb={missing_pdb}"
        )

    positives = [scene1_by_id[pid] for pid in linker_ids]
    for protein_id, sequence, labels in positives:
        pdb_seq = pdb_by_id[protein_id][1]
        if sequence != pdb_seq:
            raise SystemExit(f"{protein_id}: scene-1 sequence differs from disorder-PDB")
        if "2" not in labels:
            raise SystemExit(f"{protein_id}: scene-1 record has no linker label")

    prediction_ids = _prediction_ids(args.predictions_zip)
    eligible: list[str] = []
    excluded_blocked = 0
    excluded_no_disorder = 0
    for protein_id, _sequence, labels in pdb_records:
        if protein_id in linker_id_set:
            continue
        if labels.count("1") == 0:
            excluded_no_disorder += 1
            continue
        if protein_id in blocked:
            excluded_blocked += 1
            continue
        eligible.append(protein_id)
    comparable = [pid for pid in eligible if pid in prediction_ids] if prediction_ids else list(eligible)
    pool = comparable if len(comparable) >= args.n_negatives else eligible
    if len(pool) < args.n_negatives:
        raise SystemExit(f"Need {args.n_negatives} negatives, pool has {len(pool)}")

    rng = random.Random(args.seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    chosen = sorted(shuffled[: args.n_negatives])
    negatives: list[tuple[str, str, str]] = []
    for protein_id in chosen:
        sequence, disorder_labels = pdb_by_id[protein_id][1], pdb_by_id[protein_id][2]
        labels = _disorder_nonlinker_labels(disorder_labels)
        if len(labels) != len(sequence):
            raise SystemExit(f"{protein_id}: label length != sequence length")
        if "2" in labels:
            raise SystemExit(f"{protein_id}: negative record was labeled as linker")
        negatives.append((protein_id, sequence, labels))

    combined = positives + negatives
    _write_three(args.output_fasta, combined)
    _write_two(args.output_sequences, combined)

    from cliper.scene3 import median_linker_run_length

    window = median_linker_run_length(labels for _pid, _seq, labels in positives)
    pos_counts = {"2": 0, "1": 0, "0": 0}
    neg_counts = {"2": 0, "1": 0, "0": 0}
    for _pid, _seq, labels in positives:
        for char in labels:
            pos_counts[char] = pos_counts.get(char, 0) + 1
    for _pid, _seq, labels in negatives:
        for char in labels:
            neg_counts[char] = neg_counts.get(char, 0) + 1

    payload = {
        "source": {
            "disorder_pdb": str(args.disorder_pdb),
            "linker_fasta": str(args.linker_fasta),
            "scene1_fasta": str(args.scene1_fasta),
            "challenge": "CAID3 disorder-PDB reference (dataset/disorder_pdb.fasta)",
            "challenge_page": "https://caid.idpcentral.org/challenge/results",
        },
        "seed": args.seed,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "positive_ids": linker_ids,
        "negative_ids": chosen,
        "eligible_pool": len(eligible),
        "prediction_archive_pool": len(comparable),
        "excluded_train_or_val": excluded_blocked,
        "excluded_no_disorder": excluded_no_disorder,
        "used_prediction_archive_filter": bool(prediction_ids) and pool is comparable,
        "positive_label_counts": pos_counts,
        "negative_label_counts": neg_counts,
        "scene2_protocol": (
            "On these 62 sequences, AUROC uses linker residues (2) as positives and "
            "other disordered residues (1) as negatives. Ordered residues (0) are not scored. "
            "Scene 1 remains the original 31-protein test_caid3.fasta."
        ),
        "scene3_window": window,
        "scene3_protocol": (
            "Protein label is 1 iff the sequence contains a linker residue. "
            "Protein score is the maximum mean linker probability over a sliding window "
            "whose width is the median linker-segment length on the 31 positive proteins."
        ),
    }
    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    args.output_stats.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "n_positives", "n_negatives", "eligible_pool", "prediction_archive_pool",
        "excluded_train_or_val", "scene3_window", "positive_label_counts", "negative_label_counts",
    )}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disorder-pdb",
        type=Path,
        default=REPO_ROOT / "dataset" / "disorder_pdb.fasta",
    )
    parser.add_argument(
        "--linker-fasta",
        type=Path,
        default=REPO_ROOT / "dataset" / "linker.fasta",
    )
    parser.add_argument(
        "--scene1-fasta",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "test_caid3.fasta",
    )
    parser.add_argument(
        "--split-stats",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "selection_stats.json",
    )
    parser.add_argument(
        "--predictions-zip",
        type=Path,
        default=REPO_ROOT.parent / "dataset" / "caid3" / "predictions.zip",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "test_scene2_pdb62.fasta",
    )
    parser.add_argument(
        "--output-sequences",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "sequences_scene2_pdb62.fasta",
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=REPO_ROOT / "dataset" / "t2_seqneg" / "scene2_pdb62_selection.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-negatives", type=int, default=31)
    build(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
