"""Sequence-level negatives + dual binary: ProstT5 extract -> train -> T1/T2 AUROC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from cliper.caid_io import write_timings_csv
from cliper.extract_embeddings import extract_prostt5_embeddings_for_fasta
from cliper.pipeline import train
from cliper.t2_seqneg_data import build_t2_seqneg_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage7_expl" / "stage7_prostt5_crnn_expl_a1_t2_seqneg.yaml"


def runall_t2_seqneg(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    backbone_name: str,
    output_dir: str | Path,
    device: str = "cuda",
    extract_window_size: int = 1024,
    extract_batch_size: int = 1,
    overwrite_embeddings: bool = False,
    rebuild_data: bool = False,
    embeddings_format: str = "npy",
) -> dict[str, Any]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir = REPO_ROOT / "dataset" / "t2_seqneg"
    train_val_fasta = data_dir / "train_val.fasta"
    sequences_fasta = data_dir / "sequences_extract.fasta"
    if rebuild_data or not train_val_fasta.exists() or not sequences_fasta.exists():
        data_stats = build_t2_seqneg_dataset(output_dir=data_dir)
    else:
        data_stats = json.loads((data_dir / "selection_stats.json").read_text(encoding="utf-8"))

    embeddings_dir = out_root / "embeddings"
    extract_result = extract_prostt5_embeddings_for_fasta(
        fasta_path=sequences_fasta,
        output_dir=embeddings_dir,
        backbone_name=backbone_name,
        window_size=int(extract_window_size),
        batch_size=int(extract_batch_size),
        device=device,
        mixed_precision=device == "cuda",
        overwrite=overwrite_embeddings,
        output_format=embeddings_format,  # type: ignore[arg-type]
    )
    timings_ms = list(extract_result.get("timings_ms") or [])
    root_timings = out_root / "timings.csv"
    write_timings_csv(root_timings, timings_ms)
    shutil.copy2(embeddings_dir / "timings.csv", root_timings)

    resolved_config = out_root / "resolved_train.yaml"
    config_payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError(f"Config must be a YAML object: {config_path}")
    config_payload["embeddings_dir"] = str(embeddings_dir)
    config_payload["backbone_name"] = backbone_name
    config_payload["device"] = device
    config_payload["output_dir"] = str(out_root / "train")
    config_payload["auto_start_tensorboard"] = False
    config_payload["use_wandb"] = False
    resolved_config.write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    train_result = train(resolved_config)
    split_auroc = dict(train_result.get("split_auroc") or {})
    auroc_t1 = dict(split_auroc.get("t1") or {})
    auroc_t2 = dict(split_auroc.get("t2") or {})
    if not auroc_t1:
        auroc_t1 = {
            "train": split_auroc.get("train"),
            "val": split_auroc.get("val"),
            "test": split_auroc.get("test"),
        }
    summary = {
        "dataset": {
            "output_dir": str(data_dir.resolve()),
            "split": data_stats.get("split"),
            "sampling": data_stats.get("sampling"),
            "encoding": data_stats.get("encoding"),
            "selection_reason": data_stats.get("selection_reason"),
        },
        "extract": {
            "embeddings_dir": str(embeddings_dir.resolve()),
            "timings_csv": str(root_timings.resolve()),
            "proteins_in_fasta": extract_result.get("proteins_in_fasta"),
            "files_written": extract_result.get("files_written"),
            "files_skipped": extract_result.get("files_skipped"),
        },
        "train": {
            "run_id": train_result.get("run_id"),
            "output_dir": train_result.get("output_dir"),
            "best_checkpoint": train_result.get("best_checkpoint"),
            "split_auroc_json": str(Path(train_result["output_dir"]) / "metrics" / "split_auroc.json"),
        },
        "protocol": {
            "t1": "linker vs {other_idr, ordered}",
            "t2": "linker vs other_idr",
        },
        "auroc_t1": auroc_t1,
        "auroc_t2": auroc_t2,
        "auroc": auroc_t1,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path.resolve())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--backbone",
        default="/data/huggs/hujh/CLIPer/models/Rostlab-ProstT5",
        help="Local ProstT5 path (no AlphaFold).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "t2_seqneg_runall",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extract-window-size", type=int, default=1024)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--embeddings-format", choices=("npy", "resfeat"), default="npy")
    args = parser.parse_args(argv)
    summary = runall_t2_seqneg(
        config_path=args.config,
        backbone_name=args.backbone,
        output_dir=args.output_dir,
        device=args.device,
        extract_window_size=args.extract_window_size,
        extract_batch_size=args.extract_batch_size,
        overwrite_embeddings=args.overwrite_embeddings,
        rebuild_data=args.rebuild_data,
        embeddings_format=args.embeddings_format,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
