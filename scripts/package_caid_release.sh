#!/usr/bin/env bash
# Build CLIPer-LinkCNN / CLIPer-LinkCRNN CAID release images.
#
# Usage:
#   ./scripts/package_caid_release.sh linkcnn
#   ./scripts/package_caid_release.sh linkcrnn
#   ./scripts/package_caid_release.sh all
#
# Build + push to Docker Hub:
#   export DOCKER_USER=your_dockerhub_username
#   PUSH=1 ./scripts/package_caid_release.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

fix_crlf() {
  if command -v sed >/dev/null 2>&1; then
    sed -i 's/\r$//' submission/caid/*.sh submission/caid/dockerinout scripts/package_caid_release.sh 2>/dev/null || true
  fi
}

build_variant() {
  local variant="$1"
  local ckpt_src image_tag method_name method_readme hub_name

    case "${variant}" in
    linkcnn)
      ckpt_src="artifacts/runs/stage4_prostt5_cnn_concat_window_t1/exp0001/checkpoints/best.pt"
      image_tag="cliper-linkcnn:latest"
      hub_name="cliper-linkcnn"
      method_name="CLIPer-LinkCNN"
      method_readme="README-CLIPer-LinkCNN.txt"
      ;;
    linkcrnn)
      ckpt_src="artifacts/runs/stage7_prostt5_crnn_expl_a1_drop040/exp0005/checkpoints/best.pt"
      image_tag="cliper-linkcrnn:latest"
      hub_name="cliper-linkcrnn"
      method_name="CLIPer-LinkCRNN"
      method_readme="README-CLIPer-LinkCRNN.txt"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      return 1
      ;;
  esac

  if [[ ! -f "${ckpt_src}" ]]; then
    echo "Checkpoint not found: ${ckpt_src}" >&2
    exit 1
  fi
  if [[ ! -f models/Rostlab-ProstT5/pytorch_model.bin ]]; then
    echo "ProstT5 not found: models/Rostlab-ProstT5/pytorch_model.bin" >&2
    exit 1
  fi
  if [[ ! -f "submission/caid/${method_readme}" ]]; then
    echo "README not found: submission/caid/${method_readme}" >&2
    exit 1
  fi

  mkdir -p submission/caid/models
  echo "==> Strip classifier head -> submission/caid/models/best.pt"
  python scripts/strip_classifier_checkpoint.py "${ckpt_src}" submission/caid/models/best.pt

  # Default: one-command patch on existing image (campus SSL cannot pip-install torch).
  # True from-scratch install (needs network): REUSE_BASE=0 ./scripts/package_caid_release.sh linkcnn
  local reuse="${REUSE_BASE:-1}"
  local dockerfile="submission/caid/Dockerfile.full"
  local build_args=(
    --build-arg "METHOD_NAME=${method_name}"
    --build-arg "METHOD_README=${method_readme}"
  )

  mkdir -p submission/caid/wheels

  if [[ "${reuse}" == "1" ]]; then
    dockerfile="submission/caid/Dockerfile.runall-patch"
    local backup="${image_tag%:*}:pre-runall"
    if ! docker image inspect "${image_tag}" >/dev/null 2>&1; then
      echo "Missing ${image_tag}; need an existing image to patch." >&2
      echo "First time only: load an old tar, or REUSE_BASE=0 with a torch wheel in submission/caid/wheels/" >&2
      exit 1
    fi
    echo "==> Tag backup ${backup} (from current ${image_tag})"
    docker tag "${image_tag}" "${backup}"
    build_args+=(--build-arg "BASE_IMAGE=${backup}")
    echo "==> One-step build: reuse ${backup} env + overlay latest scripts/README/dockerinout"
  else
    echo "==> Full from-scratch build ${image_tag} via Dockerfile.full"
    if ls submission/caid/wheels/torch*.whl >/dev/null 2>&1; then
      echo "==> Using local torch wheel(s) under submission/caid/wheels/"
    else
      echo "==> No local torch wheel; Dockerfile will pip-install torch==2.6.0"
    fi
  fi

  docker build -f "${dockerfile}" \
    "${build_args[@]}" \
    -t "${image_tag}" \
    .

  echo "==> Built ${image_tag}"
  docker images "${image_tag%%:*}" --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.ID}}'

  if [[ "${PUSH:-0}" == "1" ]]; then
    if [[ -z "${DOCKER_USER:-}" ]]; then
      echo "Set DOCKER_USER to push (e.g. export DOCKER_USER=myname)" >&2
      exit 1
    fi
    local remote="${DOCKER_USER}/${hub_name}:latest"
    echo "==> Push ${remote}"
    docker tag "${image_tag}" "${remote}"
    docker push "${remote}"
    echo "Pushed: ${remote}"
  fi
}

fix_crlf
chmod +x submission/caid/*.sh submission/caid/dockerinout 2>/dev/null || true

target="${1:-}"
case "${target}" in
  linkcnn|linkcrnn)
    build_variant "${target}"
    ;;
  all)
    build_variant linkcnn
    build_variant linkcrnn
    ;;
  *)
    echo "Usage: $0 {linkcnn|linkcrnn|all}" >&2
    echo "  PUSH=1 DOCKER_USER=yourname $0 all" >&2
    exit 1
    ;;
esac

echo ""
echo "Done. Test (stdin FASTA -> stdout tar.gz):"
echo "  docker run -i cliper-linkcnn:latest dockerinout < test.fasta > results.tar.gz"
echo "  tar -tzf results.tar.gz"
echo "README: docker run --rm --entrypoint cat cliper-linkcnn:latest /README.docker"
