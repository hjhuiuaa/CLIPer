from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterable

import torch

VALID_SEQUENCE_CHARS = set("ACDEFGHIKLMNPQRSTVWYBXZUO-")
VALID_LABEL_CHARS = set("012-")


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    labels: str


@dataclass(frozen=True)
class MotifSpec:
    motif_id: str
    pattern: str | None
    regex: str | None
    kind: str = "regex"
    accession: str | None = None
    description: str | None = None
    token: str | None = None


@dataclass(frozen=True)
class MotifSpan:
    motif_id: str
    token: str
    start: int
    end: int
    order: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_three_line_fasta(path: str | Path) -> list[ProteinRecord]:
    """Parse FASTA where each record is exactly 3 lines:
    >id
    SEQUENCE
    LABELS
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"FASTA file not found: {source}")

    raw_lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_lines) % 3 != 0:
        raise ValueError(
            f"Malformed FASTA (expected line count multiple of 3): {source} has {len(raw_lines)} non-empty lines."
        )

    records: list[ProteinRecord] = []
    seen_ids: set[str] = set()
    for idx in range(0, len(raw_lines), 3):
        header = raw_lines[idx]
        sequence = raw_lines[idx + 1].upper()
        labels = raw_lines[idx + 2]

        if not header.startswith(">"):
            line_no = idx + 1
            raise ValueError(f"Malformed FASTA header at line {line_no} in {source}: {header!r}")

        protein_id = header[1:].strip()
        if not protein_id:
            line_no = idx + 1
            raise ValueError(f"Empty protein id in header at line {line_no} in {source}.")
        if protein_id in seen_ids:
            raise ValueError(f"Duplicate protein id found in {source}: {protein_id}")
        seen_ids.add(protein_id)

        if len(sequence) != len(labels):
            raise ValueError(
                f"Sequence/label length mismatch in {source} for {protein_id}: "
                f"len(sequence)={len(sequence)} len(labels)={len(labels)}"
            )
        if not set(sequence).issubset(VALID_SEQUENCE_CHARS):
            bad = sorted(set(sequence) - VALID_SEQUENCE_CHARS)
            raise ValueError(f"Invalid sequence chars for {protein_id} in {source}: {bad}")
        if not set(labels).issubset(VALID_LABEL_CHARS):
            bad = sorted(set(labels) - VALID_LABEL_CHARS)
            raise ValueError(f"Invalid label chars for {protein_id} in {source}: {bad}")

        records.append(ProteinRecord(protein_id=protein_id, sequence=sequence, labels=labels))

    return records


def parse_id_lines(path: str | Path) -> set[str]:
    """Read ids from any FASTA-like file by scanning lines starting with '>'."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"ID source file not found: {source}")
    ids = {
        line[1:].strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.startswith(">")
    }
    ids.discard("")
    return ids


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def build_split_manifest(
    records: list[ProteinRecord],
    *,
    source_fasta: str | Path,
    error_ids: Iterable[str],
    caid_ids: Iterable[str],
    seed: int,
    val_ratio: float,
) -> tuple[dict, dict]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1 (exclusive), got {val_ratio}.")

    record_ids = {r.protein_id for r in records}
    error_id_set = set(error_ids)
    caid_id_set = set(caid_ids)

    excluded_error_ids = sorted(record_ids.intersection(error_id_set))
    missing_error_ids = sorted(error_id_set - record_ids)
    holdout_overlap_ids = sorted(record_ids.intersection(caid_id_set))

    eligible_ids = sorted(record_ids - set(excluded_error_ids) - set(holdout_overlap_ids))
    if len(eligible_ids) < 2:
        raise ValueError(
            "Not enough eligible proteins after exclusions. "
            f"eligible={len(eligible_ids)} excluded_error={len(excluded_error_ids)} holdout_overlap={len(holdout_overlap_ids)}"
        )

    rng = random.Random(seed)
    shuffled = eligible_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)

    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])

    split_manifest = {
        "generated_at_utc": _utc_now_iso(),
        "source_fasta": str(Path(source_fasta)),
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": {
            "source_records": len(record_ids),
            "eligible_records": len(eligible_ids),
            "train_records": len(train_ids),
            "val_records": len(val_ids),
            "excluded_error_records": len(excluded_error_ids),
            "excluded_holdout_overlap_records": len(holdout_overlap_ids),
        },
        "train_ids": train_ids,
        "val_ids": val_ids,
        "excluded_error_ids": excluded_error_ids,
        "excluded_holdout_overlap_ids": holdout_overlap_ids,
        "caid_holdout_ids": sorted(caid_id_set),
    }

    exclusion_report = {
        "generated_at_utc": _utc_now_iso(),
        "source_fasta": str(Path(source_fasta)),
        "error_id_count_in_file": len(error_id_set),
        "error_ids_present_in_source": excluded_error_ids,
        "error_ids_missing_from_source": missing_error_ids,
        "holdout_overlap_ids": holdout_overlap_ids,
        "excluded_total": len(excluded_error_ids) + len(holdout_overlap_ids),
    }

    return split_manifest, exclusion_report


def select_records(records: list[ProteinRecord], selected_ids: Iterable[str]) -> list[ProteinRecord]:
    selected = set(selected_ids)
    chosen = [rec for rec in records if rec.protein_id in selected]
    missing = selected - {rec.protein_id for rec in chosen}
    if missing:
        missing_preview = sorted(missing)[:10]
        raise ValueError(f"Requested ids missing from parsed records. Missing sample: {missing_preview}")
    return chosen


_IUPAC_TO_REGEX = {
    "A": "A",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "V": "V",
    "W": "W",
    "Y": "Y",
    "B": "[DN]",
    "Z": "[EQ]",
    "X": "[A-Z]",
    "J": "[IL]",
    "U": "U",
    "O": "O",
    "-": "-",
}


def _build_prosite_regex(pattern: str) -> str:
    """Translate a PROSITE PA string into a Python regex.

    Supports residue symbols, classes, negated classes, wildcards, repetition,
    and sequence boundary markers (< and >). Hyphens are treated as separators.
    """
    text = re.sub(r"\s+", "", pattern).strip()
    if text.endswith("."):
        text = text[:-1]
    parts: list[str] = []
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == "-":
            idx += 1
            continue
        if ch == "<":
            parts.append("^")
            idx += 1
            continue
        if ch == ">":
            parts.append("$")
            idx += 1
            continue

        if ch == "[":
            end = text.find("]", idx + 1)
            if end < 0:
                raise ValueError(f"Unclosed character class in PROSITE pattern: {pattern!r}")
            content = text[idx + 1 : end]
            atom = "[" + "".join(re.escape(c) for c in content) + "]"
            idx = end + 1
        elif ch == "{":
            end = text.find("}", idx + 1)
            if end < 0:
                raise ValueError(f"Unclosed negated class in PROSITE pattern: {pattern!r}")
            content = text[idx + 1 : end]
            atom = "[^" + "".join(re.escape(c) for c in content) + "]"
            idx = end + 1
        elif ch.lower() == "x":
            atom = "."
            idx += 1
        elif ch.isalpha():
            atom = re.escape(ch.upper())
            idx += 1
        else:
            raise ValueError(f"Unsupported PROSITE pattern character: {ch!r} in {pattern!r}")

        if idx < len(text) and text[idx] == "(":
            end = text.find(")", idx + 1)
            if end < 0:
                raise ValueError(f"Unclosed repetition in PROSITE pattern: {pattern!r}")
            repetition = text[idx + 1 : end].strip()
            if not repetition:
                raise ValueError(f"Empty repetition in PROSITE pattern: {pattern!r}")
            if "," in repetition:
                left, right = [chunk.strip() for chunk in repetition.split(",", 1)]
                minimum = int(left)
                maximum = int(right)
                quantifier = f"{{{minimum},{maximum}}}"
            else:
                minimum = int(repetition)
                quantifier = f"{{{minimum}}}"
            atom = f"(?:{atom}){quantifier}"
            idx = end + 1

        parts.append(atom)
    regex = "".join(parts)
    re.compile(regex)
    return regex


def _build_prosite_token(accession: str | None, motif_id: str) -> str:
    base = accession or motif_id
    base = re.sub(r"\s+", "_", base.strip())
    return f"<PROSITE:{base}>"


def parse_prosite_dat(source: str | Path) -> list[dict[str, str]]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"PROSITE file not found: {path}")

    motifs: list[dict[str, str]] = []
    current: dict[str, Any] = {}

    def _flush_current() -> None:
        nonlocal current
        if not current:
            return
        motif_type = str(current.get("type", "")).upper()
        if motif_type == "PATTERN":
            motif_id = str(current.get("id", "")).strip()
            accession = str(current.get("ac", "")).strip() or None
            description = " ".join(str(part).strip() for part in current.get("de_parts", []) if str(part).strip())
            pa_parts = [str(part).strip() for part in current.get("pa_parts", []) if str(part).strip()]
            if motif_id and pa_parts:
                pa = re.sub(r"\s+", "", "".join(pa_parts))
                if pa.endswith("."):
                    pa = pa[:-1]
                motifs.append(
                    {
                        "id": motif_id,
                        "ac": accession or "",
                        "de": description,
                        "pa": pa,
                        "kind": "prosite",
                        "token": _build_prosite_token(accession, motif_id),
                    }
                )
        current = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "//":
            _flush_current()
            continue
        if line.startswith("ID"):
            match = re.match(r"^ID\s+([^;]+);\s+([A-Z]+)\.\s*$", stripped)
            if match is None:
                raise ValueError(f"Malformed PROSITE ID line: {line!r}")
            current["id"] = match.group(1).strip()
            current["type"] = match.group(2).strip()
            continue
        if line.startswith("AC"):
            current["ac"] = stripped[2:].strip().rstrip(";").strip()
            continue
        if line.startswith("DE"):
            current.setdefault("de_parts", []).append(stripped[2:].strip().rstrip(";").strip())
            continue
        if line.startswith("PA"):
            current.setdefault("pa_parts", []).append(stripped[2:].strip().rstrip(";").strip())
            continue

    _flush_current()
    return motifs


def _build_motif_regex(pattern: str, matching: str) -> str:
    if matching == "exact":
        return re.escape(pattern)
    if matching == "regex":
        return pattern
    if matching == "degenerate":
        parts: list[str] = []
        for char in pattern:
            if char not in _IUPAC_TO_REGEX:
                raise ValueError(f"Unsupported degenerate motif character: {char!r}")
            parts.append(_IUPAC_TO_REGEX[char])
        return "".join(parts)
    if matching == "prosite":
        return _build_prosite_regex(pattern)
    raise ValueError(f"Unsupported motif.matching mode: {matching!r}")


def load_motif_specs(path: str | Path, *, matching: str) -> list[MotifSpec]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Motif source file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "motifs" not in payload or not isinstance(payload["motifs"], list):
        raise ValueError("Motif source must be JSON object with list field 'motifs'.")
    specs: list[MotifSpec] = []
    for idx, item in enumerate(payload["motifs"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"motifs[{idx}] must be object, got {type(item).__name__}")
        motif_id = str(item.get("id", f"M{idx}")).strip()
        kind = str(item.get("kind", "regex")).strip().lower()
        if not motif_id:
            raise ValueError(f"motifs[{idx}] must include non-empty 'id'.")
        if kind not in {"regex", "profile", "prosite"}:
            raise ValueError(
                f"motifs[{idx}] kind must be one of ['regex', 'profile', 'prosite'], got {kind!r}"
            )
        raw_pattern = item.get("pattern")
        raw_pa = item.get("pa")
        pattern_source = raw_pa if raw_pa is not None else raw_pattern
        pattern = str(pattern_source).strip() if pattern_source is not None else None
        accession = str(item.get("ac", "")).strip() or None
        description = str(item.get("de", "")).strip() or None
        token = str(item.get("token", "")).strip() or None
        if kind in {"profile", "prosite"}:
            if not pattern:
                raise ValueError(f"motifs[{idx}] {kind} motif must include non-empty 'pa' or 'pattern'.")
            regex = str(item.get("regex", "")).strip() or None
            if not regex:
                regex = _build_motif_regex(pattern, matching=("prosite" if kind == "prosite" else matching))
            re.compile(regex)
            specs.append(
                MotifSpec(
                    motif_id=motif_id,
                    pattern=pattern,
                    regex=regex,
                    kind=kind,
                    accession=accession,
                    description=description,
                    token=token,
                )
            )
            continue
        if not pattern:
            raise ValueError(f"motifs[{idx}] regex motif must include non-empty 'pattern'.")
        regex = _build_motif_regex(pattern, matching=matching)
        re.compile(regex)
        specs.append(
            MotifSpec(
                motif_id=motif_id,
                pattern=pattern,
                regex=regex,
                kind="regex",
                accession=accession,
                description=description,
                token=token,
            )
        )
    if not specs:
        raise ValueError("Motif source contains zero motifs.")
    return specs


def build_motif_token_map(specs: list[MotifSpec]) -> dict[str, str]:
    token_map: dict[str, str] = {}
    for spec in specs:
        token = spec.token or _build_prosite_token(spec.accession, spec.motif_id)
        token_map[spec.motif_id] = token
    return token_map


def build_motif_special_tokens(specs: list[MotifSpec]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        token = spec.token or _build_prosite_token(spec.accession, spec.motif_id)
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _motif_candidates(sequence: str, specs: list[MotifSpec]) -> list[MotifSpan]:
    candidates: list[MotifSpan] = []
    for order, spec in enumerate(specs):
        if not spec.regex:
            continue
        for match in re.finditer(f"(?=({spec.regex}))", sequence):
            token = match.group(1)
            if not token:
                continue
            start = int(match.start())
            end = start + len(token)
            candidates.append(
                MotifSpan(
                    motif_id=spec.motif_id,
                    token=spec.token or _build_prosite_token(spec.accession, spec.motif_id),
                    start=start,
                    end=end,
                    order=order,
                )
            )
    candidates.sort(key=lambda span: (span.start, -(span.end - span.start), span.order, span.motif_id))
    return candidates


def select_motif_spans(sequence: str, specs: list[MotifSpec]) -> list[MotifSpan]:
    """Select non-overlapping motif spans using greedy longest-match priority."""
    if not sequence or not specs:
        return []
    occupied = [False] * len(sequence)
    selected: list[MotifSpan] = []
    for candidate in _motif_candidates(sequence, specs):
        if any(occupied[candidate.start : candidate.end]):
            continue
        for pos in range(candidate.start, candidate.end):
            occupied[pos] = True
        selected.append(candidate)
    selected.sort(key=lambda span: (span.start, span.end, span.order, span.motif_id))
    return selected


def tokenize_sequence_with_motifs(
    sequence: str,
    specs: list[MotifSpec],
    *,
    token_map: dict[str, str] | None = None,
) -> tuple[list[str], list[int], list[MotifSpan]]:
    """Replace selected motif spans with special tokens and keep span lengths."""
    if token_map is None:
        token_map = build_motif_token_map(specs)
    spans = select_motif_spans(sequence, specs)
    if not spans:
        tokens = list(sequence)
        lengths = [1] * len(tokens)
        return tokens, lengths, []

    tokens: list[str] = []
    lengths: list[int] = []
    cursor = 0
    for span in spans:
        if cursor < span.start:
            for residue in sequence[cursor:span.start]:
                tokens.append(residue)
                lengths.append(1)
        token = token_map.get(span.motif_id, span.token)
        tokens.append(token)
        lengths.append(span.end - span.start)
        cursor = span.end
    if cursor < len(sequence):
        for residue in sequence[cursor:]:
            tokens.append(residue)
            lengths.append(1)
    return tokens, lengths, spans


def build_motif_vocab(specs: list[MotifSpec]) -> dict[str, int]:
    return {spec.motif_id: idx + 1 for idx, spec in enumerate(specs)}


def encode_motif_ids_for_sequence(
    sequence: str,
    specs: list[MotifSpec],
    motif_vocab: dict[str, int],
    *,
    max_per_residue: int,
) -> list[int]:
    if max_per_residue <= 0:
        raise ValueError(f"motif.max_per_residue must be > 0, got {max_per_residue}")
    encoded = [0] * len(sequence)
    used_count = [0] * len(sequence)
    for spec in specs:
        if spec.regex is None:
            continue
        for match in re.finditer(f"(?=({spec.regex}))", sequence):
            start = int(match.start())
            token = match.group(1)
            for pos in range(start, min(len(sequence), start + len(token))):
                if used_count[pos] >= max_per_residue:
                    continue
                if encoded[pos] == 0:
                    encoded[pos] = motif_vocab[spec.motif_id]
                used_count[pos] += 1
    return encoded


def build_motif_id_tensor(
    sequences: list[str],
    residue_lengths: list[int],
    specs: list[MotifSpec],
    motif_vocab: dict[str, int],
    *,
    max_per_residue: int,
) -> torch.Tensor:
    max_len = max(residue_lengths)
    motif_ids = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, (seq, length) in enumerate(zip(sequences, residue_lengths)):
        encoded = encode_motif_ids_for_sequence(
            seq[:length],
            specs,
            motif_vocab,
            max_per_residue=max_per_residue,
        )
        motif_ids[row, :length] = torch.tensor(encoded, dtype=torch.long)
    return motif_ids


def summarize_motif_coverage(records: list[ProteinRecord], specs: list[MotifSpec], motif_vocab: dict[str, int]) -> dict:
    total = 0
    covered = 0
    per_sequence_hits: list[int] = []
    for rec in records:
        encoded = encode_motif_ids_for_sequence(rec.sequence, specs, motif_vocab, max_per_residue=1)
        hit_count = sum(1 for x in encoded if x != 0)
        per_sequence_hits.append(hit_count)
        total += len(encoded)
        covered += hit_count
    ratio = float(covered) / float(total) if total > 0 else 0.0
    avg_hits = (sum(per_sequence_hits) / len(per_sequence_hits)) if per_sequence_hits else 0.0
    return {
        "num_motifs": len(specs),
        "total_residues": total,
        "covered_residues": covered,
        "coverage_ratio": ratio,
        "avg_hits_per_sequence": avg_hits,
    }


def summarize_motif_coverage_detailed(
    records: list[ProteinRecord],
    specs: list[MotifSpec],
    motif_vocab: dict[str, int],
    *,
    max_per_residue: int = 1,
    top_k: int = 10,
) -> dict:
    total = 0
    covered = 0
    per_sequence_hits: list[int] = []
    motif_hit_counts: dict[str, int] = {spec.motif_id: 0 for spec in specs}
    ordered_specs = sorted(specs, key=lambda spec: motif_vocab.get(spec.motif_id, 0))
    id_by_token = {token_id: motif_id for motif_id, token_id in motif_vocab.items()}
    for rec in records:
        encoded = encode_motif_ids_for_sequence(
            rec.sequence,
            ordered_specs,
            motif_vocab,
            max_per_residue=max_per_residue,
        )
        hit_count = sum(1 for x in encoded if x != 0)
        per_sequence_hits.append(hit_count)
        total += len(encoded)
        covered += hit_count
        for token_id in encoded:
            if token_id == 0:
                continue
            motif_id = id_by_token.get(token_id)
            if motif_id is not None:
                motif_hit_counts[motif_id] = motif_hit_counts.get(motif_id, 0) + 1
    ratio = float(covered) / float(total) if total > 0 else 0.0
    avg_hits = (sum(per_sequence_hits) / len(per_sequence_hits)) if per_sequence_hits else 0.0
    top_hit = sorted(motif_hit_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
    num_regex = sum(1 for spec in specs if spec.regex is not None)
    num_profile = len(specs) - num_regex
    return {
        "num_records": len(records),
        "num_motifs": len(specs),
        "num_regex_motifs": num_regex,
        "num_profile_motifs": num_profile,
        "total_residues": total,
        "covered_residues": covered,
        "coverage_ratio": ratio,
        "avg_hits_per_sequence": avg_hits,
        "max_per_residue": max_per_residue,
        "top_hit_motifs": [{"id": motif_id, "hit_residues": count} for motif_id, count in top_hit],
    }


REPO_ROOT = Path(__file__).resolve().parents[1]




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
