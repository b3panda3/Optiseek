"""weights.py — Locate model weights on disk.

Resolution order:
1. $OPTISEEK_MODEL_DIR env var (explicit override; used in dev)
2. /models/Qwen2.5-VL-3B-Instruct (baked into Docker image at build time)
3. /models/Qwen2.5-VL-7B-Instruct (fallback if 7B was baked instead)
4. HuggingFace Hub: Qwen/Qwen2.5-VL-3B-Instruct (downloads to HF cache)

Default model is 3B because:
- 7B OOMs on MI300X VF partitions during transformers 5.x load (~30 GiB peak)
- 3B fits comfortably (~7 GiB after load) and passes our smoke test
- 3B is still top-tier on OCR benchmarks (OCR-Bench 850+ vs 880 for 7B)
- Smaller Docker image (~25 GiB vs ~40 GiB) — more headroom under 60 GiB limit
- Faster inference (~10s vs ~7s per image with vLLM, but vLLM doesn't work
  on ROCm Py3.14 anyway, so both engines use transformers at similar speed)
"""

from __future__ import annotations

import os
from pathlib import Path

# Default model: 3B (fits VF partitions). Switch to 7B by setting OPTISEEK_MODEL_DIR.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_NAME = "Qwen2.5-VL-3B-Instruct"

# Baked-in weights paths (checked in order)
BAKED_PATHS = [
    Path("/models/Qwen2.5-VL-3B-Instruct"),
    Path("/models/Qwen2.5-VL-7B-Instruct"),  # fallback if 7B was baked
]

ENV_VAR = "OPTISEEK_MODEL_DIR"


def resolve_model_path() -> str:
    """Return the path or model ID to load.

    Returns:
        - A local directory path if weights are on disk
        - A HuggingFace model ID (string) if download is required
    """
    # 1. Env override (dev convenience — also used by bootstrap.sh)
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.exists() and (p / "config.json").exists():
            return str(p)
        # If env var is set but path doesn't exist, fall through to other
        # resolution methods (don't silently fail).

    # 2. Baked-in weights (production / eval)
    for baked in BAKED_PATHS:
        if baked.exists() and (baked / "config.json").exists():
            return str(baked)

    # 3. HF Hub fallback (downloads to ~/.cache/huggingface on first use)
    return DEFAULT_MODEL_ID


def model_name() -> str:
    """Friendly name for logging."""
    path = resolve_model_path()
    # Extract model name from path or return the default
    if "Qwen2.5-VL-7B" in path:
        return "Qwen2.5-VL-7B-Instruct"
    if "Qwen2.5-VL-3B" in path:
        return "Qwen2.5-VL-3B-Instruct"
    return DEFAULT_MODEL_NAME
