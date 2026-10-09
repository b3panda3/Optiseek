# ==========================================================================
# Optiseek — AMD LabLab AI Academy Mini Challenge 2 (OCR)
# ==========================================================================
#
# MANDATORY BASE IMAGE (do NOT change):
#   rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
#
# The grader verifies base-image identity by LAYER HASH, not by name.
# Therefore:
#   * Do NOT use `docker build --squash`
#   * Do NOT use `buildah` to flatten layers
#   * Do NOT use a multi-stage build that ends on FROM scratch
#   * Multi-stage builds ARE fine, as long as the FINAL stage starts FROM the
#     mandated base.
#
# Image size budget: 60 GiB uncompressed. We target ~25 GiB (3B model).
# ==========================================================================

FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# -------------------------------------------------------------------------
# 1. Environment
# -------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/.hf_cache \
    TRANSFORMERS_CACHE=/models/.hf_cache \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    OPTISEEK_OUTPUT_DIR=/app/output

WORKDIR /app

# -------------------------------------------------------------------------
# 2. Python dependencies
# -------------------------------------------------------------------------
# Copy requirements and install. We do this BEFORE copying app code so that
# the dependency layer is cached across code changes during dev iterations.
COPY app/requirements.txt /app/requirements.txt

# Install: use --no-cache-dir to keep the layer small.
# We do NOT upgrade pip globally to avoid touching base-image layers.
RUN pip install --no-cache-dir -r /app/requirements.txt

# -------------------------------------------------------------------------
# 3. Application code
# -------------------------------------------------------------------------
COPY app/ /app/

# Ensure output directory exists (grader expects /app/output)
RUN mkdir -p /app/output /app/input

# -------------------------------------------------------------------------
# 4. Model weights
# -------------------------------------------------------------------------
# We bake Qwen2.5-VL-3B-Instruct into /models at build time.
# This adds ~6 GiB to the image but eliminates eval-time network dependency
# and keeps us well under the 60 GiB image-size limit (final ~25 GiB).
#
# Why 3B instead of 7B:
#   - 7B OOMs on AMD MI300X VF partitions during transformers 5.x load
#     (peak ~30 GiB transient memory vs 47 GiB partition cap)
#   - 3B fits comfortably (~7 GiB after load) and passes all smoke tests
#   - 3B still scores 850+ on OCR-Bench (vs 880 for 7B) — minimal accuracy loss
#   - Smaller image = faster push, more headroom under 60 GiB limit
#
# Build modes:
#   * Production (weights downloaded at build time):
#       docker build -t optiseek .
#   * Local dev (weights already on disk — avoids re-downloading):
#       docker build --build-arg MODEL_DIR=/persistent/Qwen2.5-VL-3B-Instruct -t optiseek .
#   * Skip weight baking (quick iteration on app code only):
#       docker build --build-arg BAKE_WEIGHTS=false -t optiseek-lite .

ARG BAKE_WEIGHTS=true
ARG MODEL_DIR=""

RUN if [ "$BAKE_WEIGHTS" = "true" ]; then \
        mkdir -p /models && \
        if [ -n "$MODEL_DIR" ]; then \
            echo "Copying weights from $MODEL_DIR" && \
            cp -r "$MODEL_DIR" /models/Qwen2.5-VL-3B-Instruct; \
        else \
            echo "Downloading Qwen2.5-VL-3B-Instruct from HuggingFace Hub..." && \
            python3 -c "from huggingface_hub import snapshot_download; \
                         snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', \
                                           local_dir='/models/Qwen2.5-VL-3B-Instruct')"; \
        fi && \
        rm -rf /models/.hf_cache; \
    else \
        echo "Skipping weight baking (BAKE_WEIGHTS=false)"; \
    fi

# -------------------------------------------------------------------------
# 5. Final layout
# -------------------------------------------------------------------------
# /app/app.py           — entry point (grader invokes this)
# /app/requirements.txt — deps (already installed above)
# /app/output/          — where JSON answers are written
# /app/input/           — where grader places the test image
# /models/              — Qwen2.5-VL-3B-Instruct weights (if BAKE_WEIGHTS=true)
# /app/run.log          — runtime log (created at first invocation)

# Sanity check: app.py must exist at the exact path the grader expects.
RUN test -f /app/app.py || (echo "FATAL: /app/app.py missing" && exit 1)

# Default command — grader overrides with explicit args
# (We don't set ENTRYPOINT because the grader calls `python3 /app/app.py ...`)
CMD ["python3", "/app/app.py", "--help"]
