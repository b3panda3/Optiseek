#!/usr/bin/env bash
# profile_vram.sh — Watch VRAM usage while running Optiseek.
#
# Usage (in one terminal):
#   ./scripts/profile_vram.sh
# Usage (in another terminal, while the above is running):
#   python3 app/app.py --input-image samples/sample_01.png
#
# The challenge requires VRAM to stay in [1 GiB, 48 GiB] (sampled every 3s).

set -euo pipefail

if command -v rocm-smi >/dev/null 2>&1; then
  echo "Using rocm-smi (AMD GPUs)"
  echo "Polling VRAM every 1s. Press Ctrl+C to stop."
  echo ""
  watch -n 1 'rocm-smi --showmeminfo vram | grep -E "VRAM|Memory"'
elif command -v amd-smi >/dev/null 2>&1; then
  echo "Using amd-smi (AMD GPUs)"
  echo "Polling VRAM every 1s. Press Ctrl+C to stop."
  echo ""
  watch -n 1 'amd-smi metric --mem | grep -E "USED_VRAM|VRAM"'
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi detected. This challenge requires AMD ROCm."
  echo "The eval environment uses AMD GPUs — your local NVIDIA tests may not"
  echo "match the eval behavior. Continuing anyway..."
  echo ""
  watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
else
  echo "ERROR: no GPU monitoring tool found (rocm-smi, amd-smi, or nvidia-smi)."
  exit 1
fi
