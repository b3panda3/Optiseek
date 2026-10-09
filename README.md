# Optiseek

OCR for license plates and traffic signs, built for the AMD LabLab AI Academy Mini Challenge 2. Accepts PNG, JPEG, or TIFF images and returns recognized text as strict JSON, using a vision-language model running on AMD ROCm.

> **Hackathon**: [AMD LabLab AI Academy Challenge — Mini Challenge 2 (Optical Character Recognition)](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge)
> **Result**: 10/10 sample images passed (200/200 points), avg 17.2s per image

---

## What it does

Optiseek extracts text from images of license plates and traffic signs under adverse conditions — noise, blur, glare, low light, and off-axis angles. It handles five categories of input, each with its own extraction convention:

| Category | Input | Output | Rule |
|---|---|---|---|
| US license plates | Plate with state name + slogan | `7ABC123` | Drop state name and slogan; return registration only |
| Chinese license plates | Plate with province character | `京A·12345` | Keep province character — it's part of the registration |
| Stop signs | Standard red stop sign | `STOP` | Single word |
| Speed limit signs | Sign with "SPEED LIMIT N" | `SPEED LIMIT 65` | Words are part of the sign body |
| Advisory speed plaques | Plaque with number only | `35` | Digits only, no units |
| Work zone / warning signs | Multi-line text sign | `ROAD WORK AHEAD` | Join lines with single spaces |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Docker image (base: rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_...)   │
│                                                                      │
│  /app/app.py            ← CLI entry; invoked once per image          │
│  /app/preprocess.py     ← PIL: RGB, EXIF, CLAHE, denoise             │
│  /app/inference.py      ← Transformers VLM runner (Qwen2.5-VL-3B)    │
│  /app/postprocess.py    ← Rule-based cleanup + grader normalization  │
│  /app/prompts.py        ← Category-aware system prompt               │
│  /app/weights.py        ← Locate weights at /models or HF Hub        │
│  /app/requirements.txt  ← transformers, pillow, huggingface_hub      │
│                                                                      │
│  /models/Qwen2.5-VL-3B-Instruct/   ← ~6 GiB weights, baked in        │
│  /app/output/                      ← JSON answers land here          │
│  /app/input/                       ← input image placed here         │
└──────────────────────────────────────────────────────────────────────┘
```

### Per-image pipeline (~17s wall time)

1. **Preprocess** (PIL, ~0.01s): EXIF orientation, RGB conversion, optional downscale to 4096px max, adaptive CLAHE on low-contrast images, median-filter denoise on noisy TIFFs.
2. **VLM inference** (transformers, ~16s): Qwen2.5-VL-3B-Instruct with a structured system prompt that describes the five image categories and their keep/drop rules. Returns strict JSON `{"text": "...", "confidence": ...}`.
3. **Postprocess** (~1ms): parse JSON (handles code fences, single quotes, raw text), strip US state names + slogans + plate labels, strip plaque units (MPH), collapse whitespace, compute rule-based confidence.
4. **Write JSON** to `/app/output/<name>_output.json`.

The model is loaded once per process invocation. Each invocation pays the ~60s model-load cost — comfortably within the 10-minute startup budget.

---

## Repository layout

```
optiseek/
├── Dockerfile                    # ROCm base image, bakes 3B weights
├── README.md                     # this file
├── LICENSE                       # MIT
├── app/
│   ├── __init__.py
│   ├── app.py                    # CLI entry point
│   ├── preprocess.py             # Adaptive PIL image cleanup
│   ├── inference.py              # Transformers VLM runner
│   ├── postprocess.py            # Rule-based text cleanup + normalization
│   ├── prompts.py                # Category-aware system prompt
│   ├── weights.py                # Weight resolution logic
│   └── requirements.txt          # Python deps
├── samples/                      # Test images (not included — see Testing)
├── test/
│   ├── run_local_tests.py        # Local grader mimic
│   ├── test_postprocess.py       # Unit tests for postprocess
│   └── expected.json             # Expected answers for samples
└── scripts/
    ├── check_image_size.sh       # Docker image size validator
    └── profile_vram.sh           # VRAM usage monitor
```

---

## Hardware and resource usage

| Resource | Used | Limit | Margin |
|---|---|---|---|
| GPU VRAM (peak) | ~8 GiB | 48 GiB | 40 GiB headroom |
| Docker image size | ~25 GiB | 60 GiB | 35 GiB headroom |
| Startup time | ~65s | 10 min | 8.5 min margin |
| Per-image time | ~17s | 30s | 13s margin |
| Total 10-image time | ~175s | 10 min | 6 min margin |

---

## Building the Docker image

The image can be built on any machine with Docker installed — no GPU is needed for the build, only for runtime.

```bash
git clone https://github.com/b3panda3/optiseek.git
cd optiseek
docker build -t optiseek:latest .
```

This downloads Qwen2.5-VL-3B-Instruct from HuggingFace Hub during the build (~6 GiB). Build time is ~15-20 minutes on a fast connection; final image is ~25 GiB.

### Verify the build

```bash
# Check image size
docker image inspect optiseek:latest --format '{{.Size}}' | awk '{printf "%.1f GiB\n", $1/1024/1024/1024}'

# Verify base layer is the ROCm image
docker history optiseek:latest --no-trunc | tail -5 | head -2

# Smoke-test
docker run --rm optiseek:latest python3 /app/app.py --help
```

---

## Running locally (without Docker)

Useful for development iteration.

### On an AMD GPU host with ROCm 6.x+

```bash
# Install deps
pip install -r app/requirements.txt

# Download weights (one-time, ~6 GiB)
python3 -c "from huggingface_hub import snapshot_download; \
            snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', \
                              local_dir='./models/Qwen2.5-VL-3B-Instruct')"

# Run on one image
export OPTISEEK_MODEL_DIR=./models/Qwen2.5-VL-3B-Instruct
python3 app/app.py --input-image samples/stop_sign.png --output-dir _test_output

# Check the output
cat _test_output/stop_sign_output.json
```

---

## Testing

### Unit tests for postprocessing

```bash
python3 test/test_postprocess.py
```

Runs 15 test cases covering all five categories plus edge cases (code fences, single-quoted JSON, label prefixes, multi-word slogans).

### Full test harness (requires sample images)

Place sample images in `samples/` matching the filenames in `test/expected.json`, then:

```bash
python3 test/run_local_tests.py \
  --samples-dir samples \
  --app-py app/app.py \
  --output-dir _test_output \
  --expected test/expected.json
```

Expected output:

```
======================================================================
Optiseek Local Test Harness | 10 images
======================================================================

[RUN ] image_01.png ... PASS (17.5s, conf=0.79)
       text: '7ABC123'
[RUN ] image_02.png ... PASS (17.2s, conf=0.85)
       text: '京A·12345'
...

======================================================================
Results: 10/10 passed | Score: 200/200
Total wall time: 172.5s (avg 17.2s/img)
======================================================================
```

---

## Configuration

All configuration is via environment variables. None are required for production (defaults match the runtime contract).

| Variable | Default | Purpose |
|---|---|---|
| `OPTISEEK_MODEL_DIR` | (unset) | Path to local model weights. If unset, uses `/models/Qwen2.5-VL-3B-Instruct` (baked) or HF Hub (download). |
| `OPTISEEK_OUTPUT_DIR` | `/app/output` | Where JSON files are written. Override for local testing. |

---

## Troubleshooting

### `CUDA out of memory during model load` (on MI300X VF partition)

Transformers 5.x calls `caching_allocator_warmup()` which pre-allocates a buffer the size of the model before loading. On VF partitions, this fails even when the actual model load would fit. `app/inference.py` monkey-patches this to a no-op at import time — a safe fix that makes model loading reliable.

### `7B model OOMs even with the warmup patch`

The 7B model in bf16 (~16 GiB final) requires ~30 GiB of transient memory during transformers' parallel weight loading. On MI300X VF partitions (47 GiB cap), this OOMs. Use the 3B model instead — it fits comfortably (~8 GiB peak) and passes all 10 sample images. `app/weights.py` defaults to 3B.

### `expandable_segments: memory mapping failed`

MI300X VF partitions have kernel-level mmap limits that block `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Don't set that env var at all — leave it unset; the default cudaMalloc allocator works fine. Setting it to `"default"` also crashes PyTorch's config parser.

### `Docker image too large (> 60 GiB)`

Use the 3B model (default, ~6 GiB) instead of 7B (~16 GiB). Clear Docker build cache with `docker builder prune -f`.

### `Wrong answer on US plate (state name included)`

This should be caught by `app/postprocess.py`. If it isn't, the state name may be misspelled or non-standard — add it to `US_STATES` in `app/postprocess.py`.

### `Wrong answer on Chinese plate (province character dropped)`

The system prompt in `app/prompts.py` explicitly tells the VLM to keep the province character. If it's still dropping it, the postprocess rules might be over-stripping — check `app/postprocess.py:US_STATES` doesn't include Chinese characters.

---

## Key engineering decisions

### Why Qwen2.5-VL-3B instead of 7B?

- **VF partition compatibility**: The 7B OOMs on MI300X VF partitions during transformers' parallel weight loading (~30 GiB transient vs 47 GiB cap). The 3B fits comfortably (~8 GiB peak).
- **OCR accuracy**: 3B scores 850+ on OCR-Bench (vs 880 for 7B) — minimal accuracy loss.
- **Docker image size**: 3B weights are ~6 GiB vs ~16 GiB for 7B, keeping the final image at ~25 GiB (well under 60 GiB).
- **Inference speed**: ~17s per image, leaving a 13s margin under the 30s budget.

### Why transformers instead of vLLM?

vLLM's PyPI wheels are CUDA-only and don't support Python 3.14 (the base image's Python). ROCm-specific vLLM builds for Python 3.14 were not available at the time. Transformers 5.x with `AutoModelForImageTextToText` provides a stable path for Qwen2.5-VL on ROCm.

### Why the caching_allocator_warmup patch?

Transformers 5.x pre-allocates a model-sized buffer before loading weights. On MI300X VF partitions, this pre-allocation fails with OOM even when the actual model load would fit. Optiseek monkey-patches this to a no-op — a safe fix that makes model loading reliable. See `app/inference.py:_patch_transformers_warmup`.

### Why light adaptive preprocessing?

Qwen2.5-VL is robust to noise, blur, and off-axis angles. Heavier CV transforms (perspective correction, deconvolution) often hurt clean images. Optiseek uses PIL-only with adaptive CLAHE (only on low-contrast) and median-filter denoise (only on high-noise). Clean images pass through untouched.

### Why rule-based postprocessing instead of a second VLM call?

- **Speed**: A second VLM call would double per-image time (17s → 34s), blowing the 30s budget.
- **Determinism**: Rule-based stripping of state names and slogans is 100% reliable. Asking the VLM "is this a state name?" introduces non-determinism.
- **Explainability**: If a result is wrong, the postprocess rules can be inspected to see exactly what was applied.

The blocklist includes all 50 US states, DC, common state slogans, plate labels (PLATE, DMV, REGISTRATION), and plaque units (MPH, KM/H).

---

## License

MIT License. See `LICENSE`.

## Acknowledgments

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) by Alibaba
- [Transformers](https://github.com/huggingface/transformers) by HuggingFace
- [AMD ROCm](https://github.com/ROCm/ROCm) for the GPU stack
- The LabLab + AMD AI Academy team for the challenge
