from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}

ResidueKey = tuple[str, int, str]
TARO_PURPLE = "#BFA6E8"


@dataclass(frozen=True)
class Prediction:
    probabilities: list[float]
    pred_labels: list[int]


@dataclass(frozen=True)
class FastaEntry:
    sequence: str
    labels: str


@dataclass(frozen=True)
class ParsedPdb:
    lines: list[str]
    chain_residues: dict[str, list[ResidueKey]]
    chain_seq: dict[str, str]
    atom_line_to_residue: dict[int, ResidueKey]


@dataclass(frozen=True)
class Selection:
    source_type: str
    structure_id: str
    chain: str
    parsed: ParsedPdb
    pdb_text: str
    mapping: dict[int, ResidueKey]
    mapped_count: int
    coverage: float
    identity: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _http_text(url: str, timeout: int = 45) -> str:
    req = Request(url, headers={"User-Agent": "CLIPer-structure-viz/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def _cache_fetch(path: Path, urls: list[str], as_json: bool = False) -> Any:
    if path.exists():
        txt = path.read_text(encoding="utf-8")
        return json.loads(txt) if as_json else txt
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            txt = _http_text(url)
            path.write_text(txt, encoding="utf-8")
            return json.loads(txt) if as_json else txt
        except (HTTPError, URLError, TimeoutError, ValueError):
            continue
    return None


def load_predictions_tsv(path: Path) -> dict[str, Prediction]:
    by_prob: dict[str, dict[int, float]] = {}
    by_label: dict[str, dict[int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        need = {"protein_id", "position_1based", "probability"}
        if not need.issubset(set(rd.fieldnames or [])):
            raise ValueError(f"Missing columns in {path}: {sorted(need)}")
        has_label = "pred_label" in (rd.fieldnames or [])
        for row in rd:
            pid = str(row["protein_id"]).strip()
            pos = int(row["position_1based"])
            prob = float(row["probability"])
            by_prob.setdefault(pid, {})
            by_label.setdefault(pid, {})
            by_prob[pid][pos] = prob
            if has_label and row.get("pred_label", "") != "":
                by_label[pid][pos] = int(row["pred_label"])
    out: dict[str, Prediction] = {}
    for pid, pos2prob in by_prob.items():
        max_pos = max(pos2prob)
        expected = set(range(1, max_pos + 1))
        if set(pos2prob) != expected:
            raise ValueError(f"Non-contiguous positions for {pid}")
        probs = [pos2prob[i] for i in range(1, max_pos + 1)]
        labels = [int(by_label[pid].get(i, 0)) for i in range(1, max_pos + 1)]
        out[pid] = Prediction(probabilities=probs, pred_labels=labels)
    return out


def load_three_line_fasta(path: Path) -> dict[str, str]:
    raw = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(raw) % 3 != 0:
        raise ValueError(f"Malformed 3-line FASTA: {path}")
    out: dict[str, str] = {}
    for i in range(0, len(raw), 3):
        header = raw[i]
        if not header.startswith(">"):
            raise ValueError(f"Bad FASTA header in {path}: {header}")
        pid = header[1:].strip()
        out[pid] = raw[i + 1].upper()
    return out


def load_three_line_fasta_with_labels(path: Path) -> dict[str, FastaEntry]:
    raw = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(raw) % 3 != 0:
        raise ValueError(f"Malformed 3-line FASTA: {path}")
    out: dict[str, FastaEntry] = {}
    for i in range(0, len(raw), 3):
        header = raw[i]
        if not header.startswith(">"):
            raise ValueError(f"Bad FASTA header in {path}: {header}")
        pid = header[1:].strip()
        seq = raw[i + 1].upper()
        labels = raw[i + 2].strip()
        if len(seq) != len(labels):
            raise ValueError(f"Sequence/label length mismatch for {pid} in {path}")
        out[pid] = FastaEntry(sequence=seq, labels=labels)
    return out


def fetch_disprot_meta(pid: str, cache_dir: Path) -> tuple[str, list[str]]:
    payload = _cache_fetch(cache_dir / "disprot" / f"{pid}.json", [f"https://disprot.org/api/{pid}"], as_json=True) or {}
    uniprot = str(payload.get("acc", "") or payload.get("accession", "") or "").strip()
    pdb_ids: set[str] = set()
    for reg in payload.get("regions", []) or []:
        for ref in reg.get("cross_refs", []) or []:
            if str(ref.get("db", "")).upper() == "PDB":
                pdb_id = str(ref.get("id", "")).strip().upper()
                if pdb_id:
                    pdb_ids.add(pdb_id)
    return uniprot, sorted(pdb_ids)


def parse_pdb(pdb_text: str) -> ParsedPdb:
    lines = pdb_text.splitlines()
    chain_res: dict[str, list[ResidueKey]] = {}
    chain_seq: dict[str, str] = {}
    atom_map: dict[int, ResidueKey] = {}
    last_by_chain: dict[str, ResidueKey] = {}
    saw_model = False
    for i, line in enumerate(lines):
        if line.startswith("MODEL"):
            if saw_model:
                break
            saw_model = True
            continue
        if line.startswith("ENDMDL") and saw_model:
            break
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        if len(line) < 27:
            continue
        if line[16] not in (" ", "A", "1"):
            continue
        aa = AA3_TO_1.get(line[17:20].strip().upper())
        if aa is None:
            continue
        chain = line[21].strip()
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        icode = line[26].strip()
        key: ResidueKey = (chain, resseq, icode)
        atom_map[i] = key
        if last_by_chain.get(chain) == key:
            continue
        chain_res.setdefault(chain, []).append(key)
        chain_seq[chain] = chain_seq.get(chain, "") + aa
        last_by_chain[chain] = key
    return ParsedPdb(lines=lines, chain_residues=chain_res, chain_seq=chain_seq, atom_line_to_residue=atom_map)


def smith_waterman(query: str, subject: str, match: int = 2, mismatch: int = -1, gap: int = -2) -> tuple[dict[int, int], int, float]:
    q = query.upper()
    s = subject.upper()
    n, m = len(q), len(s)
    if n == 0 or m == 0:
        return {}, 0, 0.0
    width = m + 1
    trace = bytearray((n + 1) * (m + 1))
    prev = [0] * width
    best = (0, 0, 0)
    for i in range(1, n + 1):
        cur = [0] * width
        qi = q[i - 1]
        row = i * width
        for j in range(1, m + 1):
            diag = prev[j - 1] + (match if qi == s[j - 1] else mismatch)
            up = prev[j] + gap
            left = cur[j - 1] + gap
            v, p = 0, 0
            if diag >= up and diag >= left and diag > 0:
                v, p = diag, 1
            elif up >= left and up > 0:
                v, p = up, 2
            elif left > 0:
                v, p = left, 3
            cur[j] = v
            trace[row + j] = p
            if v > best[0]:
                best = (v, i, j)
        prev = cur
    score, i, j = best
    mapping: dict[int, int] = {}
    mapped = 0
    matched = 0
    while i > 0 and j > 0:
        p = trace[i * width + j]
        if p == 0:
            break
        if p == 1:
            mapping[i] = j
            mapped += 1
            if q[i - 1] == s[j - 1]:
                matched += 1
            i -= 1
            j -= 1
        elif p == 2:
            i -= 1
        else:
            j -= 1
    ident = matched / mapped if mapped else 0.0
    return mapping, score, ident


def choose_rcsb(seq: str, pdb_ids: list[str], cache_dir: Path) -> Selection | None:
    best: Selection | None = None
    for pdb_id in pdb_ids:
        text = _cache_fetch(
            cache_dir / "rcsb" / f"{pdb_id}.pdb",
            [f"https://files.rcsb.org/download/{pdb_id}.pdb", f"https://models.rcsb.org/v1/{pdb_id}/models/1?encoding=pdb"],
        )
        if not text:
            continue
        parsed = parse_pdb(text)
        for chain, cseq in parsed.chain_seq.items():
            if not cseq:
                continue
            pos_map, _, ident = smith_waterman(seq, cseq)
            chain_res = parsed.chain_residues.get(chain, [])
            mapping: dict[int, ResidueKey] = {}
            for tpos, spos in pos_map.items():
                if 1 <= spos <= len(chain_res):
                    mapping[tpos] = chain_res[spos - 1]
            mapped = len(mapping)
            if mapped == 0:
                continue
            cov = mapped / max(1, len(seq))
            cand = Selection("RCSB", pdb_id, chain, parsed, text, mapping, mapped, cov, ident)
            if best is None or (cand.coverage, cand.mapped_count, cand.identity) > (best.coverage, best.mapped_count, best.identity):
                best = cand
    return best


def choose_alphafold(seq: str, uniprot: str, cache_dir: Path) -> Selection | None:
    if not uniprot:
        return None
    sid = f"AF-{uniprot}-F1"
    text = None

    # Preferred path: AlphaFold API gives the latest concrete PDB URL (v6+).
    api_payload = _cache_fetch(
        cache_dir / "alphafold_api" / f"{uniprot}.json",
        [f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"],
        as_json=True,
    )
    if isinstance(api_payload, list) and api_payload:
        entry = None
        for item in api_payload:
            if str(item.get("modelEntityId", "")).startswith(sid):
                entry = item
                break
        if entry is None:
            entry = api_payload[0]
        pdb_url = str(entry.get("pdbUrl", "")).strip()
        if pdb_url:
            filename = pdb_url.rsplit("/", 1)[-1]
            text = _cache_fetch(cache_dir / "alphafold" / filename, [pdb_url])

    # Fallback path: try common versioned filenames directly.
    if not text:
        for v in [8, 7, 6, 5, 4, 3, 2, 1]:
            fn = f"{sid}-model_v{v}.pdb"
            text = _cache_fetch(cache_dir / "alphafold" / fn, [f"https://alphafold.ebi.ac.uk/files/{fn}"])
            if text:
                break
    if not text:
        return None
    parsed = parse_pdb(text)
    if not parsed.chain_residues:
        return None
    chain = sorted(parsed.chain_residues.keys())[0]
    res = parsed.chain_residues[chain]
    n = min(len(seq), len(res))
    if n == 0:
        return None
    mapping = {i: res[i - 1] for i in range(1, n + 1)}
    ident = sum(1 for i in range(1, n + 1) if seq[i - 1] == parsed.chain_seq[chain][i - 1]) / n
    return Selection("AlphaFold", sid, chain, parsed, text, mapping, n, n / max(1, len(seq)), ident)


def _set_bfactor(line: str, val: float) -> str:
    if len(line) < 66:
        line = line.ljust(66)
    if len(line) < 80:
        line = line.ljust(80)
    return line[:60] + f"{val:6.2f}" + line[66:]


def annotate(selection: Selection, probs: list[float]) -> tuple[str, dict[ResidueKey, float], dict[ResidueKey, int]]:
    prob_by_res: dict[ResidueKey, float] = {}
    pos_by_res: dict[ResidueKey, int] = {}
    for pos, key in selection.mapping.items():
        if 1 <= pos <= len(probs):
            prob_by_res[key] = float(probs[pos - 1])
            pos_by_res[key] = pos
    out = []
    for i, line in enumerate(selection.parsed.lines):
        key = selection.parsed.atom_line_to_residue.get(i)
        if key is None:
            out.append(line)
        else:
            out.append(_set_bfactor(line, prob_by_res.get(key, -1.0)))
    return "\n".join(out).rstrip("\n") + "\n", prob_by_res, pos_by_res


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def build_comparison(
    selection: Selection,
    prob_by_res: dict[ResidueKey, float],
    pos_by_res: dict[ResidueKey, int],
    true_labels: str,
    pred_binary: list[int],
) -> tuple[dict[str, dict[str, list[int]]], dict[str, dict[str, Any]], dict[str, Any]]:
    groups: dict[str, dict[str, list[int]]] = {}
    info: dict[str, dict[str, Any]] = {}
    tp = fp = fn = tn = 0
    unknown = 0
    for chain, residues in selection.parsed.chain_residues.items():
        groups[chain] = {
            "pred_linker": [],
            "true_linker": [],
            "tp": [],
            "fp": [],
            "fn": [],
            "tn": [],
            "unknown": [],
        }
        for key in residues:
            hkey = f"{key[0]}|{key[1]}|{key[2]}"
            if key not in pos_by_res:
                info[hkey] = {"status": "unmapped"}
                continue
            pos = pos_by_res[key]
            prob = float(prob_by_res.get(key, 0.0))
            pred = int(pred_binary[pos - 1]) if 1 <= pos <= len(pred_binary) else 0
            true_char = true_labels[pos - 1] if 1 <= pos <= len(true_labels) else "-"
            true = 1 if true_char == "1" else 0 if true_char == "0" else -1
            if pred == 1:
                groups[chain]["pred_linker"].append(key[1])
            if true == 1:
                groups[chain]["true_linker"].append(key[1])
            if true == -1:
                groups[chain]["unknown"].append(key[1])
                unknown += 1
                status = "unknown"
            elif pred == 1 and true == 1:
                groups[chain]["tp"].append(key[1])
                tp += 1
                status = "tp"
            elif pred == 1 and true == 0:
                groups[chain]["fp"].append(key[1])
                fp += 1
                status = "fp"
            elif pred == 0 and true == 1:
                groups[chain]["fn"].append(key[1])
                fn += 1
                status = "fn"
            else:
                groups[chain]["tn"].append(key[1])
                tn += 1
                status = "tn"
            info[hkey] = {
                "status": status,
                "target_pos": pos,
                "probability": round(prob, 6),
                "pred_label": pred,
                "true_label": true_char,
            }
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    stats = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unknown": unknown,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return groups, info, stats


def make_html(
    pid: str,
    uniprot: str,
    selection: Selection,
    pdb_annot: str,
    groups: dict[str, dict[str, list[int]]],
    residue_info: dict[str, dict[str, Any]],
    seq_len: int,
    threshold: float,
    stats: dict[str, Any],
) -> str:
    mapped_positions = sorted(
        {
            int(v["target_pos"])
            for v in residue_info.values()
            if v.get("status") != "unmapped" and "target_pos" in v
        }
    )
    missing = [i for i in range(1, seq_len + 1) if i not in set(mapped_positions)]
    miss_txt = ", ".join(map(str, missing[:40])) + (" ..." if len(missing) > 40 else "")

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <title>{pid}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f6f7fb; }}
    .page {{ padding: 10px; display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 10px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .box {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: white; }}
    .title {{ font-size: 13px; padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600; }}
    .viewer {{ height: 38vh; min-height: 320px; }}
    .meta {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px; font-size: 13px; line-height: 1.55; }}
    .k {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }}
    .mono {{ font-family: Consolas, monospace; font-size: 12px; word-break: break-all; }}
    @media (max-width: 1080px) {{
      .page {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: 1fr; }}
      .viewer {{ height: 360px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="grid">
      <div class="box"><div class="title">Predicted linker (香芋紫)</div><div id="v_pred" class="viewer"></div></div>
      <div class="box"><div class="title">True linker (香芋紫)</div><div id="v_true" class="viewer"></div></div>
      <div class="box" style="grid-column: 1 / -1;"><div class="title">Prediction vs True comparison</div><div id="v_cmp" class="viewer"></div></div>
    </div>
    <div class="meta">
      <h3 style="margin:0 0 8px 0">{pid}</h3>
      <div>UniProt: {uniprot or "-"}</div>
      <div>Source: {selection.source_type}</div>
      <div>Structure: {selection.structure_id}</div>
      <div>Chain: {selection.chain or "-"}</div>
      <div>Mapped: {selection.mapped_count}/{seq_len} ({selection.coverage*100:.2f}%)</div>
      <div>Threshold: {threshold:.3f}</div>
      <hr>
      <div><span class="k" style="background:{TARO_PURPLE}"></span>Linker residues (香芋紫)</div>
      <div><span class="k" style="background:{TARO_PURPLE}"></span>TP (pred=1,true=1)</div>
      <div><span class="k" style="background:#FF9800"></span>FP (pred=1,true=0)</div>
      <div><span class="k" style="background:#1E88E5"></span>FN (pred=0,true=1)</div>
      <div><span class="k" style="background:#BFC5CE"></span>TN / background</div>
      <hr>
      <div>TP: {int(stats.get("tp", 0))}</div>
      <div>FP: {int(stats.get("fp", 0))}</div>
      <div>FN: {int(stats.get("fn", 0))}</div>
      <div>TN: {int(stats.get("tn", 0))}</div>
      <div>Precision: {float(stats.get("precision", 0.0)):.4f}</div>
      <div>Recall: {float(stats.get("recall", 0.0)):.4f}</div>
      <div>F1: {float(stats.get("f1", 0.0)):.4f}</div>
      <hr>
      <div style="font-size:12px;">Missing/unmapped target positions:</div>
      <div class="mono">{miss_txt or "None"}</div>
    </div>
  </div>
<script>
const pdbData = {json.dumps(pdb_annot)};
const groups = {json.dumps(groups, ensure_ascii=False)};
const residueInfo = {json.dumps(residue_info, ensure_ascii=False)};
const TARO = "{TARO_PURPLE}";

function applyByResiList(model, chain, resiList, styleObj) {{
  if (!resiList || resiList.length === 0) return;
  model.setStyle({{chain: chain, resi: resiList}}, styleObj);
}}

function buildViewer(id) {{
  const v = $3Dmol.createViewer(id, {{ backgroundColor: "white" }});
  v.addModel(pdbData, "pdb");
  const m = v.getModel();
  m.setStyle({{}}, {{ cartoon: {{ color: "#BFC5CE" }} }});
  return [v, m];
}}

function addHover(model, viewer) {{
  model.setHoverable({{}}, true, function(a, v) {{
    const k = (a.chain || "") + "|" + a.resi + "|" + (a.icode || "");
    const x = residueInfo[k];
    let text = "unmapped / missing";
    if (x && x.status !== "unmapped") {{
      text = "status=" + x.status + " | pos=" + x.target_pos + " | prob=" + x.probability.toFixed(3) + " | pred=" + x.pred_label + " | true=" + x.true_label;
    }}
    if (a._l) v.removeLabel(a._l);
    a._l = v.addLabel(text, {{position: {{x:a.x,y:a.y,z:a.z}}, backgroundColor:"black", fontColor:"white", backgroundOpacity:0.75, inFront:true}});
  }}, function(a, v) {{
    if (a._l) {{ v.removeLabel(a._l); a._l = null; }}
  }});
}}

const [vPred, mPred] = buildViewer("v_pred");
const [vTrue, mTrue] = buildViewer("v_true");
const [vCmp, mCmp] = buildViewer("v_cmp");

for (const chain of Object.keys(groups)) {{
  const g = groups[chain];
  applyByResiList(mPred, chain, g.pred_linker, {{ cartoon: {{ color: TARO }}, stick: {{ radius: 0.18, color: TARO }} }});
  applyByResiList(mTrue, chain, g.true_linker, {{ cartoon: {{ color: TARO }}, stick: {{ radius: 0.18, color: TARO }} }});

  applyByResiList(mCmp, chain, g.tp, {{ cartoon: {{ color: TARO }}, stick: {{ radius: 0.18, color: TARO }} }});
  applyByResiList(mCmp, chain, g.fp, {{ cartoon: {{ color: "#FF9800" }}, stick: {{ radius: 0.18, color: "#FF9800" }} }});
  applyByResiList(mCmp, chain, g.fn, {{ cartoon: {{ color: "#1E88E5" }}, stick: {{ radius: 0.18, color: "#1E88E5" }} }});
}}

addHover(mPred, vPred);
addHover(mTrue, vTrue);
addHover(mCmp, vCmp);
vPred.zoomTo(); vPred.render();
vTrue.zoomTo(); vTrue.render();
vCmp.zoomTo(); vCmp.render();
</script>
</body>
</html>"""


def run_batch(pred_tsv: Path, fasta: Path, out_dir: Path, threshold: float, fallback: str, min_cov: float) -> dict[str, Any]:
    preds = load_predictions_tsv(pred_tsv)
    fasta_entries = load_three_line_fasta_with_labels(fasta)
    for pid, rec in preds.items():
        if pid not in fasta_entries:
            raise ValueError(f"{pid} not found in FASTA")
        if len(fasta_entries[pid].sequence) != len(rec.probabilities):
            raise ValueError(f"Length mismatch for {pid}")

    ann_dir = out_dir / "annotated_pdb"
    html_dir = out_dir / "html"
    cache_dir = out_dir / "cache"
    ann_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = total_tn = total_unknown = 0
    pids = sorted(preds)
    for i, pid in enumerate(pids, start=1):
        print(f"[structure_viz] ({i}/{len(pids)}) {pid}", flush=True)
        rec = preds[pid]
        seq = fasta_entries[pid].sequence
        true_labels = fasta_entries[pid].labels
        pred_binary = [1 if float(p) >= threshold else 0 for p in rec.probabilities]
        uniprot, pdb_ids = fetch_disprot_meta(pid, cache_dir)
        fallback_reason = ""
        sel = choose_rcsb(seq, pdb_ids, cache_dir) if pdb_ids else None
        if not pdb_ids:
            fallback_reason = "no_rcsb_crossref"
        if sel is None and pdb_ids:
            fallback_reason = "rcsb_unavailable"
        if sel is not None and sel.coverage < min_cov:
            fallback_reason = f"low_rcsb_coverage<{min_cov:.2f}"
            sel = None
        if sel is None and fallback == "alphafold":
            af = choose_alphafold(seq, uniprot, cache_dir)
            if af is not None:
                sel = af
            elif not fallback_reason:
                fallback_reason = "alphafold_unavailable"
        if sel is None:
            rows.append(
                {
                    "protein_id": pid,
                    "uniprot": uniprot,
                    "source_type": "Failed",
                    "structure_id": "",
                    "chain": "",
                    "mapped_count": 0,
                    "coverage": 0.0,
                    "fallback_reason": fallback_reason or "no_structure_selected",
                }
            )
            continue
        annotated, prob_by_res, pos_by_res = annotate(sel, rec.probabilities)
        groups, residue_info, cmp_stats = build_comparison(sel, prob_by_res, pos_by_res, true_labels, pred_binary)
        (ann_dir / f"{pid}.pdb").write_text(annotated, encoding="utf-8")
        html = make_html(
            pid=pid,
            uniprot=uniprot,
            selection=sel,
            pdb_annot=annotated,
            groups=groups,
            residue_info=residue_info,
            seq_len=len(seq),
            threshold=threshold,
            stats=cmp_stats,
        )
        (html_dir / f"{pid}.html").write_text(html, encoding="utf-8")
        total_tp += int(cmp_stats["tp"])
        total_fp += int(cmp_stats["fp"])
        total_fn += int(cmp_stats["fn"])
        total_tn += int(cmp_stats["tn"])
        total_unknown += int(cmp_stats["unknown"])
        rows.append(
            {
                "protein_id": pid,
                "uniprot": uniprot,
                "source_type": sel.source_type,
                "structure_id": sel.structure_id,
                "chain": sel.chain,
                "mapped_count": sel.mapped_count,
                "coverage": sel.coverage,
                "fallback_reason": fallback_reason,
                "tp": int(cmp_stats["tp"]),
                "fp": int(cmp_stats["fp"]),
                "fn": int(cmp_stats["fn"]),
                "tn": int(cmp_stats["tn"]),
                "unknown": int(cmp_stats["unknown"]),
                "precision": float(cmp_stats["precision"]),
                "recall": float(cmp_stats["recall"]),
                "f1": float(cmp_stats["f1"]),
            }
        )

    manifest = out_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(
            [
                "protein_id",
                "uniprot",
                "source_type",
                "structure_id",
                "chain",
                "mapped_count",
                "coverage",
                "fallback_reason",
                "tp",
                "fp",
                "fn",
                "tn",
                "unknown",
                "precision",
                "recall",
                "f1",
            ]
        )
        for r in rows:
            wr.writerow(
                [
                    r["protein_id"],
                    r["uniprot"],
                    r["source_type"],
                    r["structure_id"],
                    r["chain"],
                    r["mapped_count"],
                    f"{float(r['coverage']):.6f}",
                    r["fallback_reason"],
                    r.get("tp", 0),
                    r.get("fp", 0),
                    r.get("fn", 0),
                    r.get("tn", 0),
                    r.get("unknown", 0),
                    f"{float(r.get('precision', 0.0)):.6f}",
                    f"{float(r.get('recall', 0.0)):.6f}",
                    f"{float(r.get('f1', 0.0)):.6f}",
                ]
            )

    summary = {
        "generated_at_utc": _utc_now(),
        "predictions_tsv": str(pred_tsv),
        "fasta": str(fasta),
        "threshold": threshold,
        "fallback": fallback,
        "min_coverage": min_cov,
        "num_total": len(rows),
        "num_rcsb": sum(1 for r in rows if r["source_type"] == "RCSB"),
        "num_alphafold": sum(1 for r in rows if r["source_type"] == "AlphaFold"),
        "num_failed": sum(1 for r in rows if r["source_type"] == "Failed"),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "unknown": total_unknown,
    }
    summary["num_success"] = summary["num_total"] - summary["num_failed"]
    summary["precision"] = _safe_div(total_tp, total_tp + total_fp)
    summary["recall"] = _safe_div(total_tp, total_tp + total_fn)
    summary["f1"] = _safe_div(2 * summary["precision"] * summary["recall"], summary["precision"] + summary["recall"])
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest), "summary": str(summary_path), **summary}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch protein structure visualization (RCSB first + AlphaFold fallback)")
    p.add_argument("--predictions-tsv", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--out-dir", default="artifacts/structure_viz")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fallback", choices=("alphafold", "none"), default="alphafold")
    p.add_argument("--min-coverage", type=float, default=0.3)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_batch(
        pred_tsv=Path(args.predictions_tsv),
        fasta=Path(args.fasta),
        out_dir=Path(args.out_dir),
        threshold=float(args.threshold),
        fallback=str(args.fallback),
        min_cov=float(args.min_coverage),
    )
    print(
        "[structure_viz] done "
        f"total={result['num_total']} rcsb={result['num_rcsb']} "
        f"alphafold={result['num_alphafold']} failed={result['num_failed']}"
    )
    print(f"[structure_viz] manifest={result['manifest']}")
    print(f"[structure_viz] summary={result['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
