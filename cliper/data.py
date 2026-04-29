from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

import torch

VALID_SEQUENCE_CHARS = set("ACDEFGHIKLMNPQRSTVWYBXZUO-")
VALID_LABEL_CHARS = set("01-")


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

