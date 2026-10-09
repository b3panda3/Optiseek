#!/usr/bin/env python3
"""run_local_tests.py — Local test harness that mimics the AMD LabLab grader.

What it does
------------
1. Discovers all images in samples/ (PNG, JPEG, TIFF).
2. For each image, runs `python3 /app/app.py --input-image <path>` exactly
   like the grader does.
3. Reads the produced <name>_output.json.
4. Applies the official grader normalization to both the expected and actual
   text.
5. Reports pass/fail per image, plus timing and a final score out of 200.

Usage
-----
    python3 test/run_local_tests.py
    python3 test/run_local_tests.py --samples-dir ./samples --expected test/expected.json
    python3 test/run_local_tests.py --docker optiseek:latest  # run inside Docker

The expected answers are read from test/expected.json, which mirrors the
sample answers from the challenge PDF:

    {
      "image_01.png": "7ABC123",
      "image_02.png": "京A·12345",
      ...
    }
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Expected answers from the challenge PDF (the 10 sample images).
# ---------------------------------------------------------------------------
DEFAULT_EXPECTED = {
    "sample_01_california_plate.png": "7ABC123",
    "sample_02_beijing_plate.png": "京A·12345",
    "sample_03_newyork_plate.jpg": "JHT 2951",
    "sample_04_motion_blur_plate.png": "5XYZ891",
    "sample_05_shanghai_plate.jpg": "沪B·88888",
    "sample_06_stop_sign.png": "STOP",
    "sample_07_stop_sign_noisy.tiff": "STOP",
    "sample_08_speed_limit.jpg": "SPEED LIMIT 65",
    "sample_09_work_zone.png": "ROAD WORK AHEAD",
    "sample_10_advisory_plaque.tiff": "35",
}

POINTS_PER_IMAGE = 20  # 10 images * 20 points = 200 max


# ---------------------------------------------------------------------------
# Grader normalization (mirror of postprocess.grader_normalize)
# ---------------------------------------------------------------------------
import re
import unicodedata

_NORMALIZE_REMOVE_CHARS = set("-.\u00b7_")


def grader_normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.upper()
    text = re.sub(r"\s+", "", text)
    text = "".join(c for c in text if c not in _NORMALIZE_REMOVE_CHARS)
    return text


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(
    app_py: Path,
    image: Path,
    output_dir: Path,
    timeout: int = 60,
) -> tuple[dict | None, float, str]:
    """Run app.py on one image. Returns (parsed_json, elapsed_s, stderr_tail)."""
    cmd = [
        sys.executable,
        str(app_py),
        "--input-image",
        str(image),
        "--output-dir",
        str(output_dir),
    ]
    t = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, time.time() - t, "TIMEOUT"
    elapsed = time.time() - t

    expected_output = output_dir / f"{image.stem}_output.json"
    if not expected_output.exists():
        return None, elapsed, proc.stderr[-500:] if proc.stderr else "no output file"

    try:
        with expected_output.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, elapsed, f"Invalid JSON: {e}"

    return data, elapsed, proc.stderr[-500:] if proc.stderr else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Optiseek against sample images.")
    parser.add_argument(
        "--samples-dir",
        default="samples",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--app-py",
        default="app/app.py",
        help="Path to app.py",
    )
    parser.add_argument(
        "--expected",
        default=None,
        help="JSON file mapping image filenames to expected answers",
    )
    parser.add_argument(
        "--output-dir",
        default="_test_output",
        help="Where to write per-image JSON outputs during the test",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-image timeout in seconds (grader allows 30; we use 60 for dev)",
    )
    args = parser.parse_args(argv)

    samples_dir = Path(args.samples_dir)
    app_py = Path(args.app_py)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load expected answers
    if args.expected:
        with open(args.expected) as f:
            expected = json.load(f)
    else:
        expected = DEFAULT_EXPECTED

    if not app_py.exists():
        print(f"ERROR: app.py not found at {app_py}", file=sys.stderr)
        return 2

    if not samples_dir.exists():
        print(f"ERROR: samples directory not found: {samples_dir}", file=sys.stderr)
        print("Create samples/ and place test images there.", file=sys.stderr)
        return 2

    # Find images
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        images.extend(samples_dir.glob(ext))
    images.sort()

    if not images:
        print(f"WARNING: no images found in {samples_dir}", file=sys.stderr)

    print(f"\n{'='*70}")
    print(f"Optiseek Local Test Harness | {len(images)} images")
    print(f"{'='*70}\n")

    passed = 0
    failed = 0
    total_elapsed = 0.0

    for img in images:
        expected_text = expected.get(img.name, "")
        if not expected_text:
            print(f"[SKIP] {img.name} — no expected answer provided")
            continue

        print(f"[RUN ] {img.name} ...", end=" ", flush=True)
        result, elapsed, stderr_tail = run_one(app_py, img, output_dir, args.timeout)
        total_elapsed += elapsed

        if result is None:
            print(f"FAIL (no output, {elapsed:.1f}s)")
            if stderr_tail:
                print(f"       stderr: {stderr_tail[:200]}")
            failed += 1
            continue

        actual_text = result.get("text", "")
        confidence = result.get("confidence", 0.0)

        actual_norm = grader_normalize(actual_text)
        expected_norm = grader_normalize(expected_text)

        if actual_norm == expected_norm:
            print(f"PASS ({elapsed:.1f}s, conf={confidence:.2f})")
            print(f"       text: {actual_text!r}")
            passed += 1
        else:
            print(f"FAIL ({elapsed:.1f}s)")
            print(f"       expected: {expected_text!r}  (norm: {expected_norm!r})")
            print(f"       actual:   {actual_text!r}  (norm: {actual_norm!r})")
            failed += 1

    score = passed * POINTS_PER_IMAGE
    max_score = (passed + failed) * POINTS_PER_IMAGE

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{passed + failed} passed | Score: {score}/{max_score}")
    print(f"Total wall time: {total_elapsed:.1f}s "
          f"(avg {total_elapsed / max(1, passed + failed):.1f}s/img)")
    print(f"{'='*70}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
