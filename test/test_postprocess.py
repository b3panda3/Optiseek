"""Verify postprocess.py logic against the 10 sample expected answers.

This is a pure-Python check (no VLM required) — we feed it canned VLM outputs
that mimic what Qwen2.5-VL would actually emit, and verify the postprocess
pipeline produces the expected grader-normalized answer.

Run:
    python3 test/test_postprocess.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from postprocess import postprocess, grader_normalize


# (description, raw_vlm_output, expected_final_text, expected_normalized)
CASES = [
    # 1. California plate — VLM returns JSON with state name included
    (
        "California plate (VLM includes state name)",
        '{"text": "CALIFORNIA 7ABC123", "confidence": 0.9}',
        "7ABC123",
        "7ABC123",
    ),
    # 2. Beijing plate — VLM correctly keeps province
    (
        "Beijing plate (province kept)",
        '{"text": "京A·12345", "confidence": 0.92}',
        "京A·12345",
        "京A12345",  # middle dot · is removed by grader normalization
    ),
    # 3. New York plate, with state name + space-separated tokens
    (
        "New York plate (state name + space-separated)",
        '{"text": "NEW YORK JHT 2951", "confidence": 0.85}',
        "JHT 2951",
        "JHT2951",
    ),
    # 4. Motion blur plate — VLM confident, just the plate
    (
        "Motion blur plate",
        '{"text": "5XYZ891", "confidence": 0.78}',
        "5XYZ891",
        "5XYZ891",
    ),
    # 5. Shanghai plate, low light with glare
    (
        "Shanghai plate (low light)",
        '{"text": "沪B·88888", "confidence": 0.7}',
        "沪B·88888",
        "沪B88888",
    ),
    # 6. Stop sign, clean
    (
        "Stop sign (clean)",
        '{"text": "STOP", "confidence": 0.99}',
        "STOP",
        "STOP",
    ),
    # 7. Stop sign, noisy TIFF — VLM might wrap in quotes
    (
        "Stop sign (noisy, VLM wraps in quotes)",
        '{"text": "STOP", "confidence": 0.88}',
        "STOP",
        "STOP",
    ),
    # 8. Speed limit sign
    (
        "Speed limit sign",
        '{"text": "SPEED LIMIT 65", "confidence": 0.95}',
        "SPEED LIMIT 65",
        "SPEEDLIMIT65",
    ),
    # 9. Work zone sign — multi-line, VLM joins with newlines
    (
        "Work zone sign (multi-line)",
        '{"text": "ROAD\\nWORK\\nAHEAD", "confidence": 0.93}',
        "ROAD WORK AHEAD",
        "ROADWORKAHEAD",
    ),
    # 10. Advisory speed plaque — VLM might add MPH
    (
        "Advisory plaque (VLM adds MPH)",
        '{"text": "35 MPH", "confidence": 0.9}',
        "35",
        "35",
    ),
    # Edge case: VLM returns raw text (no JSON envelope)
    (
        "Raw text without JSON envelope",
        "7ABC123",
        "7ABC123",
        "7ABC123",
    ),
    # Edge case: VLM wraps JSON in markdown code fence
    (
        "VLM wraps in code fence",
        '```json\n{"text": "STOP", "confidence": 0.95}\n```',
        "STOP",
        "STOP",
    ),
    # Edge case: VLM adds label prefix
    (
        "VLM adds label prefix",
        '{"text": "PLATE: 7ABC123", "confidence": 0.8}',
        "7ABC123",
        "7ABC123",
    ),
    # Edge case: Texas plate with slogan
    (
        "Texas plate with slogan",
        '{"text": "TEXAS 7ABC123 THE LONE STAR STATE", "confidence": 0.85}',
        "7ABC123",
        "7ABC123",
    ),
    # Edge case: Single quotes in JSON
    (
        "Single-quoted JSON",
        "{'text': 'STOP', 'confidence': 0.9}",
        "STOP",
        "STOP",
    ),
]


def run_tests() -> int:
    passed = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"Postprocess unit tests | {len(CASES)} cases")
    print(f"{'='*70}\n")

    for desc, raw, expected_text, expected_norm in CASES:
        actual_text, conf = postprocess(raw)
        actual_norm = grader_normalize(actual_text)

        ok = (actual_norm == expected_norm)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {desc}")
        print(f"       raw:      {raw!r}")
        print(f"       text:     {actual_text!r}  (expected: {expected_text!r})")
        print(f"       norm:     {actual_norm!r}  (expected: {expected_norm!r})")
        print(f"       conf:     {conf:.3f}")
        if not ok:
            failed += 1
        else:
            passed += 1
        print()

    print(f"{'='*70}")
    print(f"Results: {passed}/{passed + failed} passed")
    print(f"{'='*70}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
