#!/usr/bin/env bash
# Wrapper: stage4 -> linkcnn, stage7 -> linkcrnn
case "${1:-}" in
  stage4) set -- linkcnn ;;
  stage7) set -- linkcrnn ;;
esac
exec "$(dirname "$0")/package_caid_release.sh" "$@"
