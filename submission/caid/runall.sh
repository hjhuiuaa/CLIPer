#!/usr/bin/env bash
# CAID one-shot: ProstT5 extract + linker predict.
# Default outputs (CAID layout): {flavor}/*.caid + timings.csv only.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/models/best.pt}"
FASTA="${FASTA:-/input/targets.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
BACKBONE="${PROSTT5_MODEL:-/app/models/Rostlab-ProstT5}"
FLAVOR="${FLAVOR:-linker}"
NUM_THREADS="${NUM_THREADS:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-1024}"
THRESHOLD="${THRESHOLD:-}"
KEEP_EMBEDDINGS="${KEEP_EMBEDDINGS:-0}"

export OMP_NUM_THREADS="${NUM_THREADS}"
export MKL_NUM_THREADS="${NUM_THREADS}"

extra_args=()
if [[ -n "${THRESHOLD}" ]]; then
  extra_args+=(--threshold "${THRESHOLD}")
fi
if [[ "${KEEP_EMBEDDINGS}" == "1" || "${KEEP_EMBEDDINGS}" == "true" ]]; then
  emb_dir="${EMBEDDINGS_DIR:-${OUTPUT_DIR}/embeddings}"
  mkdir -p "${emb_dir}"
  extra_args+=(--keep-embeddings --embeddings-dir "${emb_dir}")
fi

mkdir -p "${OUTPUT_DIR}"

exec python -m cliper.cli runall \
  --checkpoint "${CHECKPOINT}" \
  --fasta "${FASTA}" \
  --output-dir "${OUTPUT_DIR}" \
  --backbone "${BACKBONE}" \
  --flavor "${FLAVOR}" \
  --window-size "${WINDOW_SIZE}" \
  --device cpu \
  --num-threads "${NUM_THREADS}" \
  "${extra_args[@]}"
