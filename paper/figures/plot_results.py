"""Figures for the 2026-10-24 ablation results."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
CLIPer = ROOT / "CLIPer"
sys.path.insert(0, str(CLIPer))

from cliper.metrics import binary_roc_auc
from cliper.scene3 import max_window_mean

RUNS = ROOT / "CLIPer" / "artifacts" / "runs"
FASTA62 = CLIPer / "dataset" / "t2_seqneg" / "test_scene2_pdb62.fasta"
ZIP = ROOT / "dataset" / "caid3" / "predictions.zip"
OUT = Path(__file__).resolve().parent

MODELS = [
    ("no_contrastive", "No contrastive", "#4C78A8"),
    ("binary_contrastive", "Binary SupCon", "#F58518"),
    ("ternary_contrastive", "Ternary SupCon", "#54A24B"),
]
FOCUS = [
    "LINKER-Pred2",
    "LINKER-Pred",
    "LINKER-Pred-Lite",
    "APOD",
    "DFLpred",
    "DisorderUnetLM",
    "DisoFLAG-IDR",
    "IPA-AF2-Linker",
    "pyHCA",
    "DISOPRED3-diso",
    "flDPnn3a",
]


def load_fasta(path: Path) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    protein_id = None
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(">"):
            if protein_id is not None:
                records[protein_id] = (lines[0], lines[1])
            protein_id = raw[1:].split()[0]
            lines = []
        elif protein_id is not None:
            lines.append(raw.strip())
    if protein_id is not None:
        records[protein_id] = (lines[0], lines[1])
    return records


def load_run(name: str) -> dict:
    base = RUNS / f"ablation_{name}"
    split = json.loads((base / "train" / "exp0001" / "metrics" / "split_auroc.json").read_text(encoding="utf-8"))
    scenes = json.loads((base / "three_scenes" / "summary.json").read_text(encoding="utf-8"))
    scene2 = json.loads((base / "three_scenes" / "scene2" / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((base / "train" / "exp0001" / "metrics" / "train_history.json").read_text(encoding="utf-8"))
    return {"split": split, "scenes": scenes, "scene2": scene2, "history": history}


def parse_method(text: str, wanted: set[str]) -> dict[str, dict[int, float]]:
    scores: dict[str, dict[int, float]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith(">"):
            current = line[1:].split()[0]
            if current in wanted:
                scores[current] = {}
            else:
                current = None
            continue
        if current is None or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            score = float(parts[2])
        except ValueError:
            continue
        scores[current][int(parts[0]) - 1] = score
    return scores


def scene_scores(records, method_scores, protocol: str):
    y_true: list[int] = []
    y_score: list[float] = []
    covered = 0
    for protein_id, (sequence, labels) in records.items():
        if protein_id not in method_scores:
            continue
        covered += 1
        for index, label in enumerate(labels):
            if index not in method_scores[protein_id]:
                continue
            if label == "2":
                y_true.append(1)
            elif label == "1" or (protocol == "t1" and label == "0"):
                y_true.append(0)
            else:
                continue
            y_score.append(method_scores[protein_id][index])
    return covered, y_true, y_score


def scene3_scores(records, method_scores, window: int = 18):
    y_true: list[int] = []
    y_score: list[float] = []
    for protein_id, (sequence, labels) in records.items():
        if protein_id not in method_scores:
            continue
        if any(index not in method_scores[protein_id] for index in range(len(sequence))):
            continue
        probs = [method_scores[protein_id][index] for index in range(len(sequence))]
        y_true.append(1 if "2" in labels else 0)
        y_score.append(max_window_mean(probs, window))
    return y_true, y_score


def style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
    })


def barh(path: Path, rows: list[tuple[str, float, str]], title: str, xlabel: str):
    rows = sorted(rows, key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.2))
    ax.barh([item[0] for item in rows], [item[1] for item in rows], color=[item[2] for item in rows])
    low = min(item[1] for item in rows)
    ax.set_xlim(min(0.45, low - 0.06), 1.0)
    ax.axvline(0.5, color="#bbbbbb", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for index, (_name, value, _color) in enumerate(rows):
        ax.text(value + 0.008, index, f"{value:.3f}", va="center", fontsize=9)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {key: load_run(key) for key, _label, _color in MODELS}
    records = load_fasta(FASTA62)
    positive_ids = {protein_id for protein_id, (_seq, labels) in records.items() if "2" in labels}
    positive_records = {protein_id: records[protein_id] for protein_id in positive_ids}

    # Ablation bars.
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    import numpy as np
    scenes = ["Scene 1\n31 proteins", "Scene 2\n62 proteins", "Scene 3\n62 proteins"]
    x = np.arange(len(scenes))
    width = 0.24
    for offset, (key, label, color) in enumerate(MODELS):
        scenes_json = runs[key]["scenes"]
        values = [
            scenes_json["scene1_auroc"],
            scenes_json["scene2_auroc"],
            scenes_json["scene3_auroc"],
        ]
        positions = x + (offset - 1) * width
        ax.bar(positions, values, width, label=label, color=color)
        for pos, value in zip(positions, values):
            ax.text(pos, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, scenes)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation across the three evaluation settings")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "ablation_three_scenes.png", dpi=200)
    fig.savefig(OUT / "ablation_three_scenes.pdf")
    plt.close(fig)

    # Validation curves. metrics in eval_history are the validation split.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for key, label, color in MODELS:
        history = runs[key]["history"]["eval_history"]
        steps = [item["global_step"] for item in history]
        axes[0].plot(steps, [item["metrics"]["auroc_t1"] for item in history], marker="o", ms=3.5, label=label, color=color)
        axes[1].plot(steps, [item["metrics"]["auroc_t2"] for item in history], marker="o", ms=3.5, label=label, color=color)
        best_val = runs[key]["split"]["val"]["auroc_t1"]
        best = min(history, key=lambda item: abs(item["metrics"]["auroc_t1"] - best_val))
        axes[0].scatter([best["global_step"]], [best["metrics"]["auroc_t1"]], s=46, color=color, zorder=3, edgecolor="black", linewidth=0.6)
    axes[0].set_title("Validation, scene 1")
    axes[1].set_title("Validation, linker vs other disorder")
    for axis in axes:
        axis.set_xlabel("Step")
        axis.set_ylabel("AUROC")
        axis.set_ylim(0.65, 0.96)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Circles mark the selected checkpoint", fontsize=10, y=1.02)
    fig.savefig(OUT / "val_auroc_curves.png", dpi=200)
    fig.savefig(OUT / "val_auroc_curves.pdf")
    plt.close(fig)

    # External methods on the same residues.
    wanted = set(records)
    comparisons = {"t1": [], "t2": [], "scene3": []}
    with zipfile.ZipFile(ZIP) as archive:
        names = [name for name in archive.namelist() if name.endswith(".caid")]
        for name in names:
            method = name.split("/")[-1][:-5]
            if method not in FOCUS:
                continue
            scores = parse_method(archive.read(name).decode("utf-8", "replace"), wanted)
            covered, y_true, y_score = scene_scores(positive_records, scores, "t1")
            if covered == 31 and sum(y_true) == 1379 and len(y_true) == 20498:
                comparisons["t1"].append((method, binary_roc_auc(y_true, y_score), covered, len(y_true)))
            covered, y_true, y_score = scene_scores(records, scores, "t2")
            if covered == 62 and len(y_true) == 5362:
                comparisons["t2"].append((method, binary_roc_auc(y_true, y_score), covered, len(y_true)))
            y_true, y_score = scene3_scores(records, scores)
            if len(y_true) == 62 and sum(y_true) == 31:
                comparisons["scene3"].append((method, binary_roc_auc(y_true, y_score), 62, 62))

    user_colors = {label: color for _key, label, color in MODELS}
    scene1_rows = [
        (label, runs[key]["scenes"]["scene1_auroc"], user_colors[label])
        for key, label, _color in MODELS
    ]
    scene1_rows += [(method, value, "#8D8D8D") for method, value, _c, _n in comparisons["t1"]]
    barh(
        OUT / "scene1_comparison.png",
        scene1_rows,
        "Scene 1: linker vs all other residues (31 proteins, 20,498 residues)",
        "AUROC",
    )
    scene2_rows = [
        (label, runs[key]["scenes"]["scene2_auroc"], user_colors[label])
        for key, label, _color in MODELS
    ]
    scene2_rows += [(method, value, "#8D8D8D") for method, value, _c, _n in comparisons["t2"]]
    barh(
        OUT / "scene2_comparison.png",
        scene2_rows,
        "Scene 2: linker vs other disordered residues (62 proteins, 5,362 residues)",
        "AUROC",
    )
    scene3_rows = [
        (label, runs[key]["scenes"]["scene3_auroc"], user_colors[label])
        for key, label, _color in MODELS
    ]
    scene3_rows += [(method, value, "#8D8D8D") for method, value, _c, _n in comparisons["scene3"]]
    barh(
        OUT / "scene3_comparison.png",
        scene3_rows,
        "Scene 3: does the protein contain a linker (62 proteins, window 18)",
        "AUROC",
    )

    # Example protein: clear linker block, readable length, high within-protein AUROC.
    pred_path = RUNS / "ablation_ternary_contrastive" / "three_scenes" / "scene2" / "predictions.tsv"
    probs: dict[str, list[float]] = {}
    for line in pred_path.read_text(encoding="utf-8").splitlines()[1:]:
        protein_id, position, probability, _pred = line.split("\t")
        probs.setdefault(protein_id, []).append((int(position), float(probability)))
    for protein_id in list(probs):
        probs[protein_id] = [value for _pos, value in sorted(probs[protein_id])]

    candidates = []
    for protein_id in positive_ids:
        sequence, labels = records[protein_id]
        if not (150 <= len(sequence) <= 700):
            continue
        y_true = [1 if label == "2" else 0 for label in labels]
        if sum(y_true) < 15 or sum(y_true) == len(y_true):
            continue
        score = binary_roc_auc(y_true, probs[protein_id])
        candidates.append((score, protein_id, len(sequence), sum(y_true)))
    candidates.sort(reverse=True)
    chosen_score, chosen_id, _length, _npos = candidates[0]
    sequence, labels = records[chosen_id]
    series = probs[chosen_id]
    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    start = None
    for index, label in enumerate(labels + "0"):
        if label == "2" and start is None:
            start = index
        elif label != "2" and start is not None:
            ax.axvspan(start + 0.5, index + 0.5, color="#54A24B", alpha=0.25, linewidth=0)
            start = None
    start = None
    for index, label in enumerate(labels + "0"):
        if label == "1" and start is None:
            start = index
        elif label != "1" and start is not None:
            ax.axvspan(start + 0.5, index + 0.5, color="#F58518", alpha=0.18, linewidth=0)
            start = None
    ax.plot(range(1, len(series) + 1), series, color="#54A24B", linewidth=1.2)
    ax.set_xlim(1, len(series))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Linker probability")
    ax.set_title(f"{chosen_id}: ternary SupCon, within-protein AUROC {chosen_score:.4f}")
    handles = [Patch(facecolor="#54A24B", alpha=0.25, label="Linker")]
    if "1" in labels:
        handles.append(Patch(facecolor="#F58518", alpha=0.18, label="Other disorder"))
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.savefig(OUT / "example_protein.png", dpi=200)
    fig.savefig(OUT / "example_protein.pdf")
    plt.close(fig)

    payload = {
        "models": {
            key: {
                "scene1": runs[key]["scenes"]["scene1_auroc"],
                "scene2": runs[key]["scenes"]["scene2_auroc"],
                "scene3": runs[key]["scenes"]["scene3_auroc"],
                "scene1_auprc": runs[key]["split"]["test"]["auprc_t1"],
                "scene2_auprc_62": runs[key]["scene2"]["auprc_t2"],
                "test31_t2": runs[key]["split"]["test"]["auroc_t2"],
                "test31_t2_auprc": runs[key]["split"]["test"]["auprc_t2"],
                "train_t1": runs[key]["split"]["train"]["auroc_t1"],
                "val_t1": runs[key]["split"]["val"]["auroc_t1"],
                "f1": runs[key]["split"]["test"]["f1"],
                "mcc": runs[key]["split"]["test"]["mcc"],
                "threshold": runs[key]["split"]["threshold"],
            }
            for key, _label, _color in MODELS
        },
        "scene1_methods": sorted(comparisons["t1"], key=lambda item: -item[1]),
        "scene2_methods": sorted(comparisons["t2"], key=lambda item: -item[1]),
        "scene3_methods": sorted(comparisons["scene3"], key=lambda item: -item[1]),
        "example": {"id": chosen_id, "within_auroc": chosen_score, "candidates": candidates[:5]},
    }
    (OUT / "figure_numbers.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
