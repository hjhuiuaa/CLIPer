#!/usr/bin/env bash
# Two ablations on the same small CRNN:
#   1. contrastive off vs on (no_contrastive vs binary_contrastive)
#   2. binary SupCon vs 3-class SupCon (binary_contrastive vs ternary_contrastive)
# After each training run, score scene 1 / 2 / 3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BACKBONE="${BACKBONE:-/data/huggs/hujh/CLIPer/models/Rostlab-ProstT5}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f dataset/t2_seqneg/test_scene2_pdb62.fasta ]]; then
  python scripts/build_scene2_pdb62.py
fi

for name in no_contrastive binary_contrastive ternary_contrastive; do
  OUT="artifacts/runs/ablation_${name}"
  python -m cliper.cli runall_t2_seqneg \
    --config "configs/ablation/${name}.yaml" \
    --backbone "$BACKBONE" \
    --output-dir "$OUT" \
    --device "$DEVICE"
  CKPT="$(find "$OUT/train" -name best.pt | head -n 1)"
  if [[ -z "$CKPT" ]]; then
    echo "No best.pt under $OUT/train" >&2
    exit 1
  fi
  python scripts/eval_three_scenes.py \
    --checkpoint "$CKPT" \
    --output-dir "$OUT/three_scenes" \
    --device "$DEVICE"
done
