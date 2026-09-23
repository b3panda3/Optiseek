"""weights.py — Locate or download Qwen2.5-VL-7B-Instruct weights.

Strategy
--------
1. If ``/models/Qwen2.5-VL-7B-Instruct`` exists (weights baked into Docker
   image at build time), use it directly. No network needed.
2. Else if ``$OPTISEEK_MODEL_DIR`` env var points to weights, use that.
3. Else fall back to HuggingFace Hub (downloads to HF cache).

The third branch is a safety net for dev environments where weights aren't
baked in. In production (eval), only branch 1 should fire.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
BAKED_PATH = Path("/models/Qwen2.5-VL-7B-Instruct")
ENV_VAR = "OPTISEEK_MODEL_DIR"


def resolve_model_path() -> str:
    """Return the path or model ID to load.

    Returns:
        - A local directory path if weights are on disk
        - A HuggingFace model ID (string) if download is required
    """
    # 1. Baked-in weights (production / eval)
    if BAKED_PATH.exists() and (BAKED_PATH / "config.json").exists():
        return str(BAKED_PATH)

    # 2. Env override (dev convenience)
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.exists() and (p / "config.json").exists():
            return str(p)

    # 3. HF Hub fallback (downloads to ~/.cache/huggingface on first use)
    return DEFAULT_MODEL_ID


def model_name() -> str:
    """Friendly name for logging."""
    return "Qwen2.5-VL-7B-Instruct"
