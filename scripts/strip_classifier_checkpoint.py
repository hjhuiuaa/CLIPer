"""Keep only classifier head weights for CAID Docker (drop frozen backbone if present)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def strip_classifier_checkpoint(src: Path, dst: Path) -> dict[str, int]:
    payload = torch.load(src, map_location="cpu")
    if "model_state" not in payload:
        raise ValueError(f"Missing model_state in {src}")

    model_state = payload["model_state"]
    classifier_state = {
        k: v for k, v in model_state.items() if k.startswith("classifier.")
    }
    if not classifier_state:
        raise ValueError(f"No classifier.* keys in {src}")

    backbone_keys = sum(1 for k in model_state if k.startswith("backbone."))
    payload = dict(payload)
    payload["model_state"] = classifier_state

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)

    return {
        "source_keys": len(model_state),
        "classifier_keys": len(classifier_state),
        "backbone_keys_dropped": backbone_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip CAID checkpoint to classifier head only.")
    parser.add_argument("source", type=Path, help="Input best.pt")
    parser.add_argument("destination", type=Path, help="Output best.pt for Docker")
    args = parser.parse_args()
    stats = strip_classifier_checkpoint(args.source, args.destination)
    print(f"Wrote {args.destination} ({args.destination.stat().st_size / 1e6:.1f} MB)")
    print(stats)


if __name__ == "__main__":
    main()
