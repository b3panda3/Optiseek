#!/usr/bin/env python3
"""app.py — Optiseek OCR entry point.

INVOCATION CONTRACT (set by the AMD LabLab challenge grader):

    python3 /app/app.py --input-image /app/input/image_01.png

OUTPUT CONTRACT:

    Writes /app/output/<input_stem>_output.json with contents:
        {"text": "<recognized_text>", "confidence": 0.0-1.0}

The output filename is the input filename with `_output.json` replacing
the extension. So `image_01.png` -> `image_01_output.json`.

DESIGN NOTES
------------
- The model is loaded ONCE at process start (module-level singleton in
  inference.py). The grader runs us once per image as separate process
  invocations; we cannot share state across them. So each invocation
  pays the model-load cost (~30-90s with vLLM, within the 10-min startup
  budget).

- We do NOT use a long-running server. The grader explicitly says:
  "There is no service to run and no endpoint to expose."

- Logging goes to stderr; stdout is reserved for any future debugging
  output. The grader reads our JSON file, not stdout.

- We log to /app/output/run.log as well so we can diagnose post-mortem.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow running both as a script (python3 /app/app.py) and as a module.
# When run as a script, /app may not be on sys.path; add it.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Import as a package (app.* namespace) when possible, else direct.
try:
    from app.preprocess import preprocess
    from app.inference import predict
    from app.postprocess import postprocess, grader_normalize
    from app.weights import model_name
except ImportError:
    # Fall back to direct module imports (when run as `python3 app.py`)
    from preprocess import preprocess  # type: ignore
    from inference import predict  # type: ignore
    from postprocess import postprocess, grader_normalize  # type: ignore
    from weights import model_name  # type: ignore


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to stderr + a log file in the output directory."""
    logger = logging.getLogger("optiseek")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # stderr handler — grader captures stderr for debugging
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler — for post-mortem
    try:
        fh = logging.FileHandler(output_dir / "run.log", mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass  # output dir may not be writable in some sandboxes

    return logger


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(output_path: Path, text: str, confidence: float) -> None:
    """Write the JSON output file in the exact format the grader expects."""
    payload = {
        "text": text,
        "confidence": round(float(confidence), 4),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")  # trailing newline for POSIX friendliness


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="optiseek",
        description="OCR for license plates and traffic signs (AMD LabLab MC2)",
    )
    parser.add_argument(
        "--input-image",
        required=True,
        help="Path to the input image (PNG, JPEG, or TIFF)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OPTISEEK_OUTPUT_DIR", "/app/output"),
        help="Directory to write the output JSON (default: /app/output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model load; useful for testing the I/O contract only.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_image)
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{input_path.stem}_output.json"

    log = setup_logging(output_dir)
    log.info("=" * 60)
    log.info("Optiseek starting | model=%s | input=%s", model_name(), input_path)
    log.info("Output target: %s", output_path)

    if not input_path.exists():
        log.error("Input image not found: %s", input_path)
        return 2

    # ---- DRY RUN (skip VLM; emit placeholder JSON for I/O testing) ----
    if args.dry_run:
        log.info("Dry-run mode: writing placeholder output without VLM.")
        write_output(output_path, "DRYRUN", 0.0)
        return 0

    # ---- REAL PIPELINE ----
    t_start = time.time()

    # 1. Preprocess
    try:
        t = time.time()
        image = preprocess(input_path)
        log.info("Preprocess done in %.2fs | size=%s", time.time() - t, image.size)
    except Exception as e:
        log.exception("Preprocess failed: %s", e)
        write_output(output_path, "", 0.0)
        return 1

    # 2. VLM inference
    try:
        t = time.time()
        raw_text, vlm_conf = predict(image)
        log.info("VLM done in %.2fs | raw=%r", time.time() - t, raw_text[:200])
    except Exception as e:
        log.exception("VLM inference failed: %s", e)
        write_output(output_path, "", 0.0)
        return 1

    # 3. Postprocess
    try:
        text, confidence = postprocess(raw_text)
        log.info(
            "Postprocess done | final=%r | confidence=%.3f | normalized=%r",
            text, confidence, grader_normalize(text),
        )
    except Exception as e:
        log.exception("Postprocess failed: %s", e)
        # Best-effort: emit the raw VLM output stripped of whitespace
        text = (raw_text or "").strip()
        confidence = 0.0

    # 4. Write output
    try:
        write_output(output_path, text, confidence)
        log.info("Wrote %s", output_path)
    except Exception as e:
        log.exception("Failed to write output: %s", e)
        return 1

    log.info("Total time: %.2fs", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
