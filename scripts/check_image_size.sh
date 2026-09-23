#!/usr/bin/env bash
# check_image_size.sh — Verify Docker image is under the 60 GiB limit.
#
# Usage:
#   ./scripts/check_image_size.sh optiseek:latest

set -euo pipefail

IMAGE="${1:-optiseek:latest}"
LIMIT_GIB=60

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image $IMAGE not found"
  echo "Build it first: docker build -t $IMAGE ."
  exit 2
fi

# docker image inspect returns size in bytes (uncompressed virtual size)
SIZE_BYTES=$(docker image inspect "$IMAGE" --format '{{.Size}}')
SIZE_GIB=$(awk "BEGIN {printf \"%.2f\", $SIZE_BYTES / 1024 / 1024 / 1024}")

echo "Image:      $IMAGE"
echo "Size:       ${SIZE_GIB} GiB (uncompressed)"
echo "Limit:      ${LIMIT_GIB} GiB"
echo ""

if (( $(awk "BEGIN {print ($SIZE_GIB > $LIMIT_GIB) ? 1 : 0}") )); then
  echo "FAIL: image exceeds size limit"
  exit 1
fi

echo "PASS: image within size limit"

# Also verify the base layer is the mandated ROCm image.
echo ""
echo "Base layer check:"
docker history --no-trunc "$IMAGE" 2>/dev/null | tail -10 | head -5
echo ""
echo "Look for 'FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0' above."
