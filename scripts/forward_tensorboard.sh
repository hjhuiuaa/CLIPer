#!/usr/bin/env bash
# SSH local port forward for TensorBoard on a remote server (Linux/macOS/Git Bash).
#
# Usage:
#   bash scripts/forward_tensorboard.sh USER@HOST 6006 16006
#   bash scripts/forward_tensorboard.sh USER@HOST 6006 16006 ~/.ssh/id_rsa
#
# Then open in local browser: http://127.0.0.1:16006
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_PORT=6006] [LOCAL_PORT=16006] [IDENTITY_FILE]" >&2
  exit 1
fi

TARGET="$1"
REMOTE_PORT="${2:-6006}"
LOCAL_PORT="${3:-16006}"
IDENTITY="${4:-}"

if [[ "${REMOTE_PORT}" -le 0 || "${LOCAL_PORT}" -le 0 ]]; then
  echo "Ports must be > 0" >&2
  exit 1
fi

SSH_CMD=(ssh -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}")
if [[ -n "${IDENTITY}" ]]; then
  SSH_CMD+=(-i "${IDENTITY}")
fi
SSH_CMD+=("${TARGET}")

echo "Forwarding TensorBoard..."
echo "  Remote: ${TARGET}:${REMOTE_PORT} (must listen on 127.0.0.1 on server, or adjust pipeline binding)"
echo "  Local : http://127.0.0.1:${LOCAL_PORT}"
echo "Press Ctrl+C to stop."
exec "${SSH_CMD[@]}"
