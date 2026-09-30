"""inference.py — VLM runner for Optiseek.

Engine: transformers (vLLM dropped — CUDA-only wheels incompatible with ROCm Py3.14).

Model: Qwen2.5-VL-3B-Instruct by default (~6 GiB bf16, fits VF partitions).
       Override with OPTISEEK_MODEL_DIR env var to use 7B if VRAM allows.

Key fixes baked in (learned the hard way on AMD MI300X VF):
  1. Monkey-patch `transformers.modeling_utils.caching_allocator_warmup` to no-op.
     Transformers 5.x pre-allocates a buffer the size of the model before loading,
     which OOMs on VF partitions even when the actual load would fit.
  2. Use `AutoModelForImageTextToText` (generic, version-stable) instead of
     `Qwen2_5_VLForConditionalGeneration` (class name varies across versions).
  3. Use `low_cpu_mem_usage=True` + `device_map='cuda:0'` for direct load
     (avoids the auto-device-map's extra CPU memory overhead).
  4. Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — fails on VF
     partitions due to mmap limits. Do NOT set it to "default" either —
     PyTorch's config parser crashes. Leave it unset; default cudaMalloc works.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from PIL import Image

# Support both package import and direct script execution
if __package__:
    from .prompts import build_messages
    from .weights import resolve_model_path, model_name
else:
    from prompts import build_messages  # type: ignore
    from weights import resolve_model_path, model_name  # type: ignore

log = logging.getLogger("optiseek.inference")


# ---------------------------------------------------------------------------
# Critical: patch out caching_allocator_warmup BEFORE any model loading.
# This must run at import time, before from_pretrained is ever called.
# ---------------------------------------------------------------------------
def _patch_transformers_warmup() -> None:
    """Disable the caching_allocator_warmup pre-allocation.

    Transformers 5.x calls this function inside _load_pretrained_model to
    pre-warm the GPU allocator with a buffer the size of the model. On VF
    partitions with constrained memory, this pre-allocation fails even when
    the actual model load would fit. Skipping it just means the allocator
    warms up naturally during weight loading — slightly slower, fully functional.
    """
    try:
        import transformers.modeling_utils as _mu
        _mu.caching_allocator_warmup = lambda *args, **kwargs: None
        log.debug("Patched out caching_allocator_warmup")
    except (ImportError, AttributeError) as e:
        log.warning("Could not patch caching_allocator_warmup: %s", e)


_patch_transformers_warmup()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VLMTransformers:
    """Transformers-based VLM engine. The only engine we ship."""

    name = "transformers"

    def __init__(self, model_path: str):
        import torch  # type: ignore
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

        log.info("Loading %s via transformers from %s", model_name(), model_path)
        t0 = time.time()

        # bf16 on ROCm MI300X is well-supported.
        # device_map='cuda:0' places all layers on GPU directly (no auto-split overhead).
        # low_cpu_mem_usage=True streams weights from disk instead of materializing in CPU RAM.
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._torch = torch
        log.info("Model loaded in %.1fs", time.time() - t0)

    def predict(self, image: Image.Image) -> tuple[str, float]:
        torch = self._torch

        messages = build_messages()
        # Apply chat template with the image inserted into the user turn.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,   # deterministic for OCR; temperature irrelevant
            )

        # Strip the prompt portion
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, in_len:]
        text_out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        # VLM doesn't return a calibrated probability in our JSON contract;
        # postprocess.py derives confidence from rule-based heuristics.
        return text_out, 0.5


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_VLM: VLMTransformers | None = None


def load_vlm() -> VLMTransformers:
    """Load the VLM once. Subsequent calls return the cached instance.

    The model is loaded ONCE per process. The grader runs us as 10 separate
    process invocations (one per image), so each invocation pays the ~40s
    model-load cost — comfortably within the 10-minute startup budget.
    """
    global _VLM
    if _VLM is not None:
        return _VLM

    model_path = resolve_model_path()
    _VLM = VLMTransformers(model_path)
    log.info("VLM ready (engine=transformers)")
    return _VLM


def predict(image: Image.Image) -> tuple[str, float]:
    """Convenience function: load VLM if needed, then predict."""
    vlm = load_vlm()
    return vlm.predict(image)
