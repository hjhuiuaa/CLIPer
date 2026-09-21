#!/usr/bin/env bash
set -euo pipefail

FASTA="${FASTA:-/input/targets.fasta}"
OUTPUT_DIR="${EMBEDDINGS_DIR:-/output/embeddings}"
BACKBONE="${PROSTT5_MODEL:-/app/models/Rostlab-ProstT5}"
WINDOW_SIZE="${WINDOW_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_THREADS="${NUM_THREADS:-4}"
FORMAT="${EMBEDDING_FORMAT:-npy}"
DEVICE="${DEVICE:-cpu}"

export OMP_NUM_THREADS="${NUM_THREADS}"
export MKL_NUM_THREADS="${NUM_THREADS}"

extra_args=()
if [[ "${OVERWRITE:-}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
if [[ "${NO_AMP:-}" == "1" ]]; then
  extra_args+=(--no-amp)
fi

mkdir -p "${OUTPUT_DIR}"

exec python -m cliper.cli extract_embeddings \
  --fasta "${FASTA}" \
  --output-dir "${OUTPUT_DIR}" \
  --backbone "${BACKBONE}" \
  --window-size "${WINDOW_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --format "${FORMAT}" \
  "${extra_args[@]}"
