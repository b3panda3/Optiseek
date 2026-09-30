# Optiseek — OCR for License Plates & Traffic Signs

> **AMD LabLab AI Academy Challenge — Mini Challenge 2 (Optical Character Recognition)**
>
> An application that accepts an image (PNG/JPEG/TIFF) and uses a vision-language
> model to recognize the characters in it, returning strict JSON output.

Optiseek is built around **Qwen2.5-VL-7B-Instruct** running on **vLLM** (with
a transformers fallback) on AMD ROCm GPUs. It handles all five challenge
categories — US license plates, Chinese license plates, stop signs, speed limit
signs, work zone / warning signs, and advisory speed plaques — under adverse
conditions (noise, blur, glare, low light, off-axis angles).

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Hardware & Storage Requirements](#hardware--storage-requirements)
- [Quick Start (Local Dev)](#quick-start-local-dev)
- [Building the Docker Image](#building-the-docker-image)
- [Running Locally (Without Docker)](#running-locally-without-docker)
- [Testing with the Local Harness](#testing-with-the-local-harness)
- [Submission Checklist](#submission-checklist)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [How It Works](#how-it-works)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Docker image (base: rocm/pytorch:rocm10.0_ubuntu26.04_...)      │
│                                                                  │
│  /app/app.py            ← CLI entry; invoked once per image      │
│  /app/preprocess.py     ← PIL: RGB, EXIF, CLAHE, denoise         │
│  /app/inference.py      ← vLLM primary + transformers fallback   │
│  /app/postprocess.py    ← Rule-based cleanup + grader normalize  │
│  /app/prompts.py        ← Category-aware system prompt           │
│  /app/weights.py        ← Locate weights at /models or HF Hub    │
│  /app/requirements.txt  ← vllm, transformers, pillow, ...        │
│                                                                  │
│  /models/Qwen2.5-VL-7B-Instruct/   ← ~16 GiB weights, baked in   │
│  /app/output/                      ← JSON answers land here      │
│  /app/input/                       ← grader places image here    │
└──────────────────────────────────────────────────────────────────┘
```

**Per-image pipeline (~5–10s wall time):**

1. **Load & preprocess** (PIL): EXIF-orient, RGB-convert, optional downscale,
   adaptive CLAHE on low-contrast images, median-filter denoise on noisy
   images. ~0.2s typical.
2. **VLM inference** (vLLM): Qwen2.5-VL-7B with a structured system prompt
   that tells it the 5 image categories and the keep/drop rules. Returns
   strict JSON `{"text": "...", "confidence": ...}`. ~3–7s typical.
3. **Postprocess**: parse JSON, strip US state names + slogans + plate labels,
   strip plaque units (MPH), collapse whitespace, compute rule-based
   confidence. ~1ms.
4. **Write JSON** to `/app/output/<name>_output.json`.

The model is loaded once per process invocation. The grader runs us as 10
separate process invocations (one per image), so each invocation pays the
~30–90s model-load cost — comfortably within the 10-minute startup budget.

---

## Repository Layout

```
optiseek/
├── Dockerfile                    # ROCm base image, no squash
├── README.md                     # this file
├── app/
│   ├── __init__.py
│   ├── app.py                    # CLI entry point (grader invokes this)
│   ├── preprocess.py             # Image cleanup
│   ├── inference.py              # vLLM + transformers
│   ├── postprocess.py            # Text cleanup + grader normalization
│   ├── prompts.py                # System & user prompts
│   ├── weights.py                # Weight resolution logic
│   └── requirements.txt          # Python deps
├── samples/                      # Place sample images here for testing
│   └── .gitkeep
├── test/
│   ├── run_local_tests.py        # Local grader mimic
│   └── expected.json             # Expected answers for the 10 samples
└── scripts/
    └── check_image_size.sh       # Docker image size validator
```

---

## Hardware & Storage Requirements

### Development environment (AMD hackathon notebook)

| Resource | Required | Notes |
|---|---|---|
| GPU | 1× AMD (CDNA or RDNA) | Provided by `notebooks.amd.com/hackathon` |
| VRAM | ~16 GiB used (of 48 GiB budget) | Qwen2.5-VL-7B in bf16 |
| Persistent storage | ~20 GiB | Weights (~16 GiB) + HF cache + dev files |
| Daily quota | 3 hours | Quota resets every 24h; does not roll over |

### Final Docker image

| Resource | Required | Limit | Notes |
|---|---|---|---|
| Image size (uncompressed) | ~40 GiB | 60 GiB | ROCm base ~20 GiB + weights ~16 GiB + deps ~4 GiB |
| Peak VRAM at eval | ~17 GiB | 48 GiB | 1 GiB lower bound also enforced — peak must exceed 1 GiB |
| Startup time | ~60–120s | 10 min | Model load dominates |
| Per-image time | ~5–10s | 30s | VLM inference dominates |

### Download budget

| Phase | Downloads | Total |
|---|---|---|
| Dev (pip + weights) | pip ~3 GiB + Qwen weights ~16 GiB | **~19 GiB** |
| Docker build (with cached weights) | ROCm base ~20 GiB + pip ~4 GiB | **~24 GiB** |
| Docker build (cold, weights re-downloaded) | + Qwen weights ~16 GiB | **~40 GiB** |
| Eval (image already pulled) | None | **0** |

---

## Quick Start (Local Dev)

### Prereqs

- AMD GPU with ROCm 6.x+ (or use the hackathon notebook)
- Python 3.10+ (the base image ships 3.14; 3.10+ works for dev)
- ~20 GiB free disk for weights + dev deps

### Step 1 — Clone & install

```bash
cd /my-project/optiseek
pip install -r app/requirements.txt
```

### Step 2 — Download weights (one-time, ~16 GiB)

```bash
# Option A: Let HuggingFace cache them (default; weights land in ~/.cache/huggingface)
python3 -c "from huggingface_hub import snapshot_download; \
            snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"

# Option B: Download to a specific directory 
export OPTISEEK_MODEL_DIR=/persistent/Qwen2.5-VL-7B-Instruct
python3 -c "from huggingface_hub import snapshot_download; \
            snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', \
                              local_dir='$OPTISEEK_MODEL_DIR')"
```

### Step 3 — Place sample images

The challenge PDF provides 10 sample images. Place them in `samples/` with
the filenames expected by `test/expected.json`:

```
samples/
├── sample_01_california_plate.png
├── sample_02_beijing_plate.png
├── sample_03_newyork_plate.jpg
├── sample_04_motion_blur_plate.png
├── sample_05_shanghai_plate.jpg
├── sample_06_stop_sign.png
├── sample_07_stop_sign_noisy.tiff
├── sample_08_speed_limit.jpg
├── sample_09_work_zone.png
└── sample_10_advisory_plaque.tiff
```

### Step 4 — Run on one image

```bash
python3 app/app.py --input-image samples/sample_06_stop_sign.png --output-dir _test_output
cat _test_output/sample_06_stop_sign_output.json
# {"text": "STOP", "confidence": 0.85}
```

### Step 5 — Run the full local test harness

```bash
python3 test/run_local_tests.py
```

Expected output:

```
======================================================================
Optiseek Local Test Harness | 10 images
======================================================================

[RUN ] sample_01_california_plate.png ... PASS (4.2s, conf=0.78)
[RUN ] sample_02_beijing_plate.png ...      PASS (5.1s, conf=0.72)
...
[RUN ] sample_10_advisory_plaque.tiff ...   PASS (3.8s, conf=0.81)

======================================================================
Results: 10/10 passed | Score: 200/200
Total wall time: 47.3s (avg 4.7s/img)
======================================================================
```

---

## Building the Docker Image

Build it on a machine with Docker and ~50 GiB free disk.

### Option A — Build with weights downloaded at build time (production)

```bash
cd /my-project/optiseek
docker build -t optiseek:latest .
```

This downloads Qwen2.5-VL-7B-Instruct from HuggingFace Hub during the build
(~16 GiB). Build time: ~10–15 minutes on a fast connection.

### Option B — Build with weights from a local path (faster dev iteration)

If you already downloaded weights to `/persistent/Qwen2.5-VL-7B-Instruct`:

```bash
docker build \
  --build-arg MODEL_DIR=/persistent/Qwen2.5-VL-7B-Instruct \
  -t optiseek:latest .
```

This `cp`'s the weights into `/models/` during build — no re-download.

### Option C — Build without weights (smoke test the image only)

For quickly iterating on `app.py` without waiting for the 16 GiB weight copy:

```bash
docker build --build-arg BAKE_WEIGHTS=false -t optiseek-lite:latest .
# Image will be ~25 GiB; runtime will download weights from HF Hub.
```

### Verify the build

```bash
# 1. Check image size (must be < 60 GiB uncompressed)
docker history --no-trunc --format '{{.Size}}' optiseek:latest | head -1
# Or measure the full uncompressed size:
docker image inspect optiseek:latest --format '{{.Size}}' | numfmt --to=iec

# 2. Verify base image is the mandated ROCm layer (NOT squashed)
docker history optiseek:latest | head -20
# You should see "FROM rocm/pytorch:rocm10.0_..." as the bottom layer.

# 3. Smoke-test the dry-run mode (skips model load)
docker run --rm -v $(pwd)/samples:/app/input optiseek:latest \
  python3 /app/app.py --input-image /app/input/sample_06_stop_sign.png --dry-run

# 4. Full run on one image (requires GPU passthrough — only on ROCm host)
docker run --rm --device /dev/kfd --device /dev/dri \
  -v $(pwd)/samples:/app/input \
  optiseek:latest \
  python3 /app/app.py --input-image /app/input/sample_06_stop_sign.png
```

---

## Running Locally (Without Docker)

Useful for fast iteration on app code without rebuilding the image.

```bash
# Set the model path (or rely on HF cache)
export OPTISEEK_MODEL_DIR=/persistent/Qwen2.5-VL-7B-Instruct

# Run on one image
python3 app/app.py \
  --input-image samples/sample_06_stop_sign.png \
  --output-dir _test_output

# Override the engine (debugging)
export OPTISEEK_ENGINE=transformers   # skip vLLM, use transformers directly
export OPTISEEK_ENGINE=vllm           # force vLLM, fail loudly if it breaks
```

---

## Testing with the Local Harness

The `test/run_local_tests.py` script mimics the grader exactly:

- Runs `python3 app/app.py --input-image <path>` per image
- Reads `<name>_output.json`
- Applies the official grader normalization (uppercase, strip whitespace,
  strip `- . · _`)
- Reports pass/fail per image + final score out of 200

### Custom expected answers

If you collect your own test set (recommended — the graded set is harder than
the samples), create a custom expected JSON:

```json
{
  "my_test_01.png": "EXPECTED ANSWER",
  "my_test_02.jpg": "ANOTHER ANSWER"
}
```

Then:

```bash
python3 test/run_local_tests.py \
  --samples-dir my_test_images/ \
  --expected my_expected.json
```

---

## Submission Checklist

### Hard gates (failure = zero score)

- [ ] **Base image**: `FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`
  is the bottom layer of your image. Verify with `docker history optiseek:latest | tail -5`.
- [ ] **No squash/flatten**: You did NOT use `docker build --squash`, `buildah`,
  or `FROM scratch`. The grader verifies layer identity.
- [ ] **Image size < 60 GiB** uncompressed:
  ```bash
  docker image inspect optiseek:latest --format '{{.Size}}'
  ```
- [ ] **`/app/app.py` exists** and accepts `--input-image <path>`:
  ```bash
  docker run --rm optiseek:latest python3 /app/app.py --help
  ```
- [ ] **Output path**: produces `/app/output/<input_stem>_output.json`:
  ```bash
  docker run --rm -v $(pwd)/samples:/app/input optiseek:latest \
    python3 /app/app.py --input-image /app/input/sample_06_stop_sign.png --dry-run
  ls _test_output/  # should show sample_06_stop_sign_output.json
  ```
- [ ] **JSON format**: `{"text": "...", "confidence": 0.0–1.0}`:
  ```bash
  cat _test_output/sample_06_stop_sign_output.json
  ```
- [ ] **PNG, JPEG, TIFF all work**: test with at least one of each format.
- [ ] **VRAM stays in [1 GiB, 48 GiB]** during inference:
  ```bash
  watch -n1 'rocm-smi --showmeminfo vram'
  ```
- [ ] **Startup < 10 minutes**: time `docker run ... python3 /app/app.py --help`
  (model load happens on first real invocation).
- [ ] **Per-image < 30 seconds**: test harness reports this.
- [ ] **Total 10 images < 10 minutes** after startup.

### Soft checks

- [ ] All 10 sample images pass the local harness.
- [ ] No `.env`, API keys, or credentials in the image.
- [ ] Image is publicly pullable (test with `docker pull` from another host).
- [ ] README is up to date.

### Submission Guideline

1. Tag and push:
   ```bash
   docker tag optiseek:latest <your-registry>/optiseek:latest
   docker push <your-registry>/optiseek:latest
   ```
2. Submit the image reference (e.g. `docker.io/yourname/optiseek:latest`) on
   the LabLab hackathon page.

---

## Configuration

All configuration is via environment variables. None are required for production.

| Variable | Default | Purpose |
|---|---|---|
| `OPTISEEK_MODEL_DIR` | (unset) | Path to local model weights. If unset, uses `/models/Qwen2.5-VL-7B-Instruct` (baked) or HF Hub (download). |
| `OPTISEEK_ENGINE` | (unset) | Force engine: `vllm` or `transformers`. If unset, tries vLLM first, falls back to transformers. |
| `OPTISEEK_OUTPUT_DIR` | `/app/output` | Where JSON files are written. Override for local testing. |
| `HF_HOME` | `/models/.hf_cache` | HuggingFace cache location inside the container. |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | Quieter logs during build. |

---

## Troubleshooting

### `vLLM fails to start on ROCm`

**Symptom**: `ImportError` or `RuntimeError` during `LLM(...)` initialization.

**Fix**: Set `OPTISEEK_ENGINE=transformers` to force the fallback path.
vLLM has stricter ROCm version requirements than transformers; the fallback
is slower (~2x) but reliable.

```bash
export OPTISEEK_ENGINE=transformers
python3 app/app.py --input-image samples/sample_01.png
```

### `OOM during model load`

**Symptom**: `CUDA out of memory` or `HIP out of memory`.

**Fix**: Lower `gpu_memory_utilization` in `app/inference.py` (default 0.85).
For MI300X (192 GiB) this is non-issue; for smaller cards (48 GiB) try 0.7.

### `Per-image time exceeds 30s`

**Symptom**: Test harness shows timeouts.

**Fixes**:
1. Confirm you're using vLLM, not transformers (check `run.log`).
2. Disable CLAHE if your images are already high-contrast (edit
   `app/preprocess.py` — comment out the CLAHE branch).
3. Lower `max_model_len` in `app/inference.py` (default 8192; 4096 is enough
   for our short prompts).

### `Wrong answer on US plate (state name included)`

**Symptom**: Output is `CALIFORNIA 7ABC123` instead of `7ABC123`.

**Fix**: This should be caught by `app/postprocess.py`. If it isn't, the
state name may be misspelled or non-standard. Add it to `US_STATES` in
`app/postprocess.py`.

### `Wrong answer on Chinese plate (province character dropped)`

**Symptom**: Output is `12345` instead of `京A·12345`.

**Fix**: The system prompt in `app/prompts.py` explicitly tells the VLM to
keep the province character. If it's still dropping it, the postprocess
rules might be over-stripping — check `app/postprocess.py:US_STATES` doesn't
include Chinese characters (it shouldn't, but verify).

### `Docker image too large (> 60 GiB)`

**Fixes**:
1. Use `--build-arg BAKE_WEIGHTS=false` and download at runtime (not
   recommended for eval — adds startup latency).
2. Use a smaller model: edit `app/weights.py` to point at
   `Qwen2.5-VL-3B-Instruct` (~6 GiB).
3. Clear Docker build cache: `docker builder prune -f`.

### `Base image identity check fails`

**Symptom**: Submission rejected with "image not built on mandated base".

**Cause**: You used `docker build --squash`, `buildah`, or a multi-stage
build ending on `FROM scratch`.

**Fix**: Rebuild with plain `docker build -t optiseek:latest .`. Do NOT
use `--squash`. Multi-stage is fine only if the final stage is
`FROM rocm/pytorch:rocm10.0_...`.

---

## How It Works

### Why Qwen2.5-VL-7B?

- **OCR accuracy**: Qwen2.5-VL consistently ranks at the top of OCR
  benchmarks (OCR-Bench, FUNSD, etc.).
- **Multilingual**: Handles Chinese plates (京A·12345) and English signs
  equally well.
- **Size**: 7B parameters in bf16 = ~16 GiB VRAM. Fits comfortably in the
  48 GiB budget while leaving room for vLLM's KV cache.
- **ROCm support**: Tested on MI300X; vLLM's ROCm backend supports it
  natively.

### Why vLLM?

- The challenge docs specifically point to vLLM on ROCm for OCR
  (https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/ocr_vllm.html).
- vLLM's PagedAttention and continuous batching give ~3-5× throughput vs.
  naive transformers inference, which gives us headroom in the per-image
  budget.
- We keep transformers as a fallback because vLLM has stricter ROCm
  version requirements and can fail to start in edge cases.

### Why bake weights into the image?

- Eliminates network dependency during eval (the grader's environment may
  rate-limit HuggingFace).
- The 16 GiB weight cost is well within the 60 GiB image-size budget.
- Eliminates a startup-time variable: no risk of a 3-minute download
  eating into the 10-minute startup budget.

### Why a rule-based postprocess instead of a second VLM call?

- **Speed**: A second VLM call doubles per-image time. We have 30s budget;
  one call uses ~5s, leaving plenty of margin. Two calls is risky.
- **Determinism**: Rule-based stripping of "CALIFORNIA", "TEXAS", etc. is
  100% reliable. Asking the VLM "is this a state name?" introduces
  non-determinism.
- **Explainability**: If a submission fails, we can look at the
  postprocess rules and know exactly what was applied.

### Why light preprocessing instead of heavy CV?

- The VLM is remarkably robust to noise, blur, and off-axis angles —
  heavier preprocessing (perspective correction, deconvolution) often
  hurts more than it helps on clean images.
- Adaptive preprocessing (CLAHE only on low-contrast, denoise only on
  high-noise) gives us the upside without the downside.
- PIL-only keeps the dependency surface small (no OpenCV in the image).

---

## License

MIT License. See `LICENSE` file for details (or just use the code — this is
a hackathon submission).

## Acknowledgments

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) by Alibaba
- [vLLM](https://github.com/vllm-project/vllm) for fast inference
- [AMD ROCm](https://github.com/ROCm/ROCm) for the GPU stack
- The LabLab + AMD AI Academy team for the challenge
