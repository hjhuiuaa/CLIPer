#!/usr/bin/env python3
"""
Download ProstT5 (and optionally ESM-2) from Hugging Face for offline use.

Requires: pip install huggingface_hub

Mirror / blocked hub (PowerShell): $env:HF_ENDPOINT = "https://hf-mirror.com"
Linux/bash: export HF_ENDPOINT="https://hf-mirror.com"
"""
from __future__ import annotations

import argparse
import os
import sys


def _download(repo_id: str, local_dir: str) -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(os.path.dirname(os.path.abspath(local_dir)) or ".", exist_ok=True)
    print(f"Downloading {repo_id} -> {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print("Done:", local_dir)


def main() -> int:
    p = argparse.ArgumentParser(description="Download ProstT5 / ESM2 weights locally.")
    p.add_argument(
        "--base-dir",
        default="models",
        help="Parent directory for model folders (default: ./models). "
        "Example on server: --base-dir /data/huggs/models",
    )
    p.add_argument(
        "--prostt5-dir",
        default=None,
        help="Override output folder for ProstT5 (default: BASE/Rostlab-ProstT5)",
    )
    p.add_argument(
        "--esm2-dir",
        default=None,
        help="Override output folder for ESM2 (default: BASE/esm2_t33_650M_UR50D)",
    )
    p.add_argument(
        "--skip-prostt5",
        action="store_true",
        help="Only download ESM-2 (requires --with-esm2)",
    )
    p.add_argument(
        "--with-esm2",
        action="store_true",
        help="Also download facebook/esm2_t33_650M_UR50D",
    )
    p.add_argument(
        "--endpoint",
        default=None,
        help="HF hub endpoint URL (sets HF_ENDPOINT). E.g. https://hf-mirror.com",
    )
    args = p.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    base = os.path.abspath(args.base_dir)
    prostt5_out = args.prostt5_dir or os.path.join(base, "Rostlab-ProstT5")
    esm2_out = args.esm2_dir or os.path.join(base, "esm2_t33_650M_UR50D")

    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print("HF_ENDPOINT =", endpoint)
    else:
        print(
            "Tip: if downloads fail, set HF_ENDPOINT, e.g. "
            "https://hf-mirror.com (export in bash / $env in PowerShell) "
            "or pass --endpoint https://hf-mirror.com"
        )

    try:
        if not args.skip_prostt5:
            _download("Rostlab/ProstT5", prostt5_out)
        if args.with_esm2:
            _download("facebook/esm2_t33_650M_UR50D", esm2_out)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
