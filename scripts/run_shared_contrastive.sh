#!/usr/bin/env bash
# Same small single-head CRNN. Contrastive loss reads the GRU state.
#   supcon: 3-class SupCon, no class centers
#   tcl: triplet center loss with 3 learnable centers
# Results go to artifacts/runs/ablation_shared_20261031/{supcon,tcl}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BACKBONE="${BACKBONE:-/data/huggs/hujh/CLIPer/models/Rostlab-ProstT5}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-artifacts/runs/ablation_shared_20261031}"
SEED_EMB="${SEED_EMB:-artifacts/runs/t2_seqneg_runall/embeddings}"

if [[ ! -f dataset/t2_seqneg/test_scene2_pdb62.fasta ]]; then
  python scripts/build_scene2_pdb62.py
fi

for name in supcon tcl; do
  OUT="$OUT_ROOT/$name"
  mkdir -p "$OUT/embeddings"
  if [[ -d "$SEED_EMB" ]]; then
    cp -al "$SEED_EMB/." "$OUT/embeddings/" 2>/dev/null || cp -an "$SEED_EMB/." "$OUT/embeddings/" || true
  fi
  python -m cliper.cli runall_t2_seqneg \
    --config "configs/ablation/shared_${name}.yaml" \
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
  echo "Finished $name -> $OUT"
done
