"""inference.py — vLLM primary + transformers fallback VLM runner.

Two engines are supported; both expose the same `predict(image)` interface:

  * VLMvLLM  — uses vLLM's `LLM` engine with `Qwen2VLConfig`-style multi-modal
               inputs. Fastest path; the AMD challenge docs specifically point
               to vLLM on ROCm for OCR.

  * VLMTransformers — uses HuggingFace transformers `Qwen2_5_VLForConditionalGeneration`
                      with `AutoProcessor`. Slower but extremely reliable.
                      Used when vLLM fails to start or its multi-modal path
                      hits an issue on the eval hardware.

The factory `load_vlm()` tries vLLM first; on failure, logs and falls back to
transformers. Once loaded, the engine is held in a module-level singleton so
the 10-image batch pays the model-load cost exactly once (the 10-minute total
budget cannot afford reloading per image).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from PIL import Image

# Support both package import (`from app.inference import ...`) and direct
# script execution (`python3 app/app.py`). When run as a script, __package__
# is empty and we use absolute imports.
if __package__:
    from .prompts import build_messages, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    from .weights import resolve_model_path, model_name
else:
    from prompts import build_messages, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # type: ignore
    from weights import resolve_model_path, model_name  # type: ignore

log = logging.getLogger("optiseek.inference")

# Allow a hard override via env var for debugging (values: "vllm" | "transformers")
ENGINE_OVERRIDE = os.environ.get("OPTISEEK_ENGINE", "").lower()


class VLMBase:
    """Common interface for VLM engines."""

    name: str = "base"

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Run inference on a PIL image. Returns (raw_text, confidence)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# vLLM engine
# ---------------------------------------------------------------------------

class VLMvLLM(VLMBase):
    name = "vllm"

    def __init__(self, model_path: str):
        # Imported lazily so the import failure (e.g. wrong ROCm build) can be
        # caught by the factory and we can fall back to transformers.
        from vllm import LLM, SamplingParams  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        log.info("Loading %s via vLLM from %s", model_name(), model_path)
        t0 = time.time()

        # Qwen2.5-VL on ROCm: bf16 is the well-supported dtype.
        # GPU memory util 0.85 leaves headroom inside the 48 GiB VRAM budget.
        # max_model_len 8192 is plenty for our short prompts + image tokens.
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            enforce_eager=False,
            trust_remote_code=True,
            download_dir=os.environ.get("HF_HOME"),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._SamplingParams = SamplingParams
        log.info("vLLM loaded in %.1fs", time.time() - t0)

    def predict(self, image: Image.Image) -> tuple[str, float]:
        from vllm import SamplingParams  # type: ignore
        from vllm.multimodal import MultiModalDataDict  # type: ignore

        messages = build_messages()

        # Apply chat template manually so we control the exact prompt string
        # (vLLM's LLM.chat() would also work but adds overhead we don't need).
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # vLLM expects a single image attached to the prompt.
        mm_data: MultiModalDataDict = {"image": image}

        sampling = SamplingParams(
            temperature=0.1,   # near-deterministic; small temp helps OCR variants
            top_p=0.9,
            max_tokens=128,    # our answers are always <20 tokens; 128 is safe
            stop=["```", "\n\n"],
        )

        outputs = self.llm.generate(
            {"prompt": prompt, "multi_modal_data": mm_data},
            sampling,
            use_tqdm=False,
        )
        text = outputs[0].outputs[0].text.strip()
        # vLLM doesn't return a logprob per output by default; we leave
        # confidence to be derived in postprocess from rule-based heuristics.
        return text, 0.5


# ---------------------------------------------------------------------------
# transformers engine (fallback)
# ---------------------------------------------------------------------------

class VLMTransformers(VLMBase):
    name = "transformers"

    def __init__(self, model_path: str):
        import torch  # type: ignore
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore

        log.info("Loading %s via transformers from %s", model_name(), model_path)
        t0 = time.time()

        # bf16 on ROCm MI300X is well-supported. device_map="auto" places
        # all layers on GPU since we have plenty of VRAM.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._torch = torch
        log.info("transformers loaded in %.1fs", time.time() - t0)

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
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )

        # Strip the prompt portion
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, in_len:]
        text_out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return text_out, 0.5


# ---------------------------------------------------------------------------
# Factory + singleton
# ---------------------------------------------------------------------------

_VLM: VLMBase | None = None


def load_vlm() -> VLMBase:
    """Load the VLM once. Subsequent calls return the cached instance."""
    global _VLM
    if _VLM is not None:
        return _VLM

    model_path = resolve_model_path()

    if ENGINE_OVERRIDE == "vllm":
        _VLM = VLMvLLM(model_path)
    elif ENGINE_OVERRIDE == "transformers":
        _VLM = VLMTransformers(model_path)
    else:
        # Default: try vLLM, fall back to transformers on any failure.
        try:
            _VLM = VLMvLLM(model_path)
        except Exception as e:
            log.warning(
                "vLLM load failed (%s). Falling back to transformers.", e
            )
            _VLM = VLMTransformers(model_path)

    log.info("VLM ready (engine=%s)", _VLM.name)
    return _VLM


def predict(image: Image.Image) -> tuple[str, float]:
    """Convenience function: load VLM if needed, then predict."""
    vlm = load_vlm()
    return vlm.predict(image)
