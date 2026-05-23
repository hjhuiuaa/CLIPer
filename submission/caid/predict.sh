#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/models/best.pt}"
FASTA="${FASTA:-/input/targets.fasta}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-/input/embeddings}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
FLAVOR="${FLAVOR:-linker}"
NUM_THREADS="${NUM_THREADS:-4}"
THRESHOLD="${THRESHOLD:-}"

extra_args=()
if [[ -n "${THRESHOLD}" ]]; then
  extra_args+=(--threshold "${THRESHOLD}")
fi

exec python -m cliper.cli predict \
  --checkpoint "${CHECKPOINT}" \
  --fasta "${FASTA}" \
  --embeddings-dir "${EMBEDDINGS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --flavor "${FLAVOR}" \
  --device cpu \
  --num-threads "${NUM_THREADS}" \
  "${extra_args[@]}"
