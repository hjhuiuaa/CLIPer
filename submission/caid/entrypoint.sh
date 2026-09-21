#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-runall}"

case "${MODE}" in
  runall|all)
    exec /app/runall.sh
    ;;
  extract)
    exec /app/extract_embeddings.sh
    ;;
  predict)
    exec /app/predict.sh
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use MODE=runall (default), extract, or predict." >&2
    exit 1
    ;;
esac
