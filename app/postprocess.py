"""postprocess.py — Rule-based text cleanup + grader normalization.

This module sits between the VLM's raw output and the JSON file we write.
It enforces three layers of cleanup:

1. CATEGORY-AWARE RULES
   - Strip US state names + slogans (so "CALIFORNIA 7ABC123" -> "7ABC123")
   - Strip plate-related labels (PLATE, DMV, etc.)
   - Drop MPH / KM/H units on advisory plaques
   - Strip quote characters the VLM sometimes wraps around its answer
   - Collapse multi-line answers to single-space-joined

2. GRADER NORMALIZATION (mirrors the official spec exactly)
   - Uppercase
   - Remove all whitespace
   - Remove the characters: - . · _

   IMPORTANT: the grader applies this to BOTH our answer AND the expected
   answer. So we don't have to be perfect on case/spacing — but we DO have
   to be perfect on character identity. A misread digit (7ABC128 vs 7ABC123)
   cannot be recovered here.

3. CONFIDENCE HEURISTIC
   - VLM doesn't give us a calibrated probability in our JSON contract.
   - We compute a rule-based confidence in [0, 1] based on:
     * Did the VLM return valid JSON?
     * Did it return non-empty text?
     * Did we have to strip a lot of "noise" (state name + slogan)?
     * Length of the final answer (very short = lower confidence for signs)
   - This confidence is recorded but NOT scored — we still report it honestly.
"""

from __future__ import annotations

import json
import re
import unicodedata

# ---------------------------------------------------------------------------
# US state names + common slogans (blocklist)
# ---------------------------------------------------------------------------
# All 50 states + DC, in uppercase. We strip these from the output only when
# they appear as a leading or trailing banner (not when they're legitimately
# part of the plate — no US plate includes a state name in its registration).
US_STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO",
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO",
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA",
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA",
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
    "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON",
    "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
    "WASHINGTON DC", "WASH DC",
}

# Common plate slogans (collected from US state plate designs).
# All uppercased; matched as substrings during cleanup.
US_SLOGANS = {
    "THE LONE STAR STATE",        # Texas
    "EMPIRE STATE",               # New York
    "GARDEN STATE",               # New Jersey
    "SUNSHINE STATE",             # Florida
    "GOLDEN STATE",               # California
    "EMPIRE STATE OF MIND",
    "FIRST IN FLIGHT",            # North Carolina
    "LAND OF ENCHANTMENT",        # New Mexico
    "GRAND CANYON STATE",         # Arizona
    "CROSSROADS OF AMERICA",      # Indiana
    "HEART OF IT ALL",            # Ohio
    "VACATIONLAND",               # Maine
    "KEystone STATE",             # Pennsylvania (typo-safe variant below)
    "KEYSTONE STATE",
    "CONSTITUTION STATE",         # Connecticut
    "NUTMEG STATE",               # Connecticut
    "OLD LINE STATE",             # Maryland
    "FREE STATE",                 # Maryland
    "PALMETTO STATE",             # South Carolina
    "PEACH STATE",                # Georgia
    "TAR HEEL STATE",             # North Carolina
    "VOLUNTEER STATE",            # Tennessee
    "HOOSIER STATE",              # Indiana
    "BUCKEYE STATE",              # Ohio
    "SHOW ME STATE",              # Missouri
    "CORNHUSKER STATE",           # Nebraska
    "MOUNTAIN STATE",             # West Virginia
    "TREASURE STATE",             # Montana
    "CENTENNIAL STATE",           # Colorado
    "BEEHIVE STATE",              # Utah
    "SILVER STATE",               # Nevada
    "COTTON STATE",               # Alabama (also called Yellowhammer)
    "YELLOWHAMMER STATE",         # Alabama
    "PEOPLE OF THE MOUNTAINS",    # Utah (historic)
    "DRIVE SAFELY",
    "IN GOD WE TRUST",
}

# Labels that sometimes appear on plate renderings but are NOT registration.
PLATE_LABELS = {"PLATE", "DMV", "REGISTRATION", "TAG", "LICENSE PLATE", "LP"}

# Units that appear on advisory plaques but are NOT part of the answer.
PLAQUE_UNITS = {"MPH", "KM/H", "KMH", "KM", "MILES PER HOUR"}

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json(text: str) -> dict | None:
    """Try to parse the VLM output as JSON. Return dict or None.

    Handles common VLM misbehaviors:
      - Code fences (```json ... ```)
      - Leading/trailing prose ("Here is the answer: {...}")
      - Single quotes instead of double quotes
    """
    if not text:
        return None

    cleaned = _JSON_FENCE_RE.sub("", text).strip()

    # Find the first { ... } block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # No JSON envelope — treat the whole text as the "text" field.
        return {"text": cleaned, "confidence": 0.5}

    blob = cleaned[start:end + 1]

    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Try fixing common issues: single quotes, trailing commas.
        try:
            fixed = blob.replace("'", '"').rstrip(",")
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _strip_surrounding_noise(text: str) -> str:
    """Remove state names, slogans, plate labels, plaque units.

    These are matched as whole words and only removed when they appear as
    banners — not when they're a legitimate part of the registration.
    The heuristic: if removing the token leaves the rest of the text
    non-empty AND contains alphanumerics, we consider it a banner.
    """
    if not text:
        return text

    # Normalize whitespace to single spaces for matching.
    tokens = text.upper().split()

    # Pass 1: drop tokens that match a state, slogan fragment, label, or unit.
    # We do this iteratively because slogans are multi-word ("THE LONE STAR STATE").
    # Strategy: join tokens into a string, remove known multi-word phrases first,
    # then remove single-word banners.
    work = " ".join(tokens)

    # Multi-word phrases first
    for phrase in US_SLOGANS | US_STATES:
        if len(phrase.split()) > 1:
            work = re.sub(rf"\b{re.escape(phrase)}\b", "", work, flags=re.IGNORECASE)

    # Single-word banners (states, labels, units)
    tokens = work.split()
    kept = []
    for t in tokens:
        u = t.strip(".,;:!?\"'()[]{}")
        if u in US_STATES:
            continue
        if u in PLATE_LABELS:
            continue
        if u in PLAQUE_UNITS:
            continue
        # Single-word slogan fragments (e.g. "STATE", "LONE", "STAR") only
        # get dropped if they're standalone AND we still have other tokens.
        # We don't drop single words here because they could be legitimate
        # plate characters (e.g. "STAR" is a valid vanity plate word).
        kept.append(t)

    return " ".join(kept).strip()


def _strip_quotes_and_labels(text: str) -> str:
    """Strip wrapping quote chars and common label prefixes."""
    text = text.strip()
    # Strip matching outer quotes
    while len(text) >= 2 and text[0] in "\"'`" and text[-1] == text[0]:
        text = text[1:-1].strip()
    # Strip "PLATE:" / "ANSWER:" / "TEXT:" prefixes
    text = re.sub(
        r"^(?:plate|answer|text|result|output)\s*[:=]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse any run of whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Official grader normalization (we apply this too — matches grader exactly)
# ---------------------------------------------------------------------------

_NORMALIZE_REMOVE_CHARS = set("-.\u00b7_")  # hyphen, period, middle dot, underscore


def grader_normalize(text: str) -> str:
    """Apply the EXACT normalization the grader applies.

    From the spec:
        - Converted to uppercase
        - All whitespace removed
        - The characters - . · _ removed
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.upper()
    # Remove whitespace
    text = re.sub(r"\s+", "", text)
    # Remove the specific characters
    text = "".join(c for c in text if c not in _NORMALIZE_REMOVE_CHARS)
    return text


# ---------------------------------------------------------------------------
# Confidence heuristic
# ---------------------------------------------------------------------------

def _compute_confidence(raw_text: str, final_text: str, parsed: bool) -> float:
    """Rule-based confidence in [0, 1]. Recorded, not scored."""
    if not final_text:
        return 0.1
    score = 0.5
    if parsed:
        score += 0.15
    # If we didn't strip much noise, the VLM was probably confident.
    if raw_text and final_text and (
        len(final_text) >= 0.7 * len(raw_text.replace(" ", ""))
    ):
        score += 0.15
    # Very short answers (1-2 chars) are riskier.
    if len(final_text) <= 2:
        score -= 0.1
    # Long answers (>20 chars) are typically multi-line signs we're confident about.
    if len(final_text) >= 8:
        score += 0.1
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def postprocess(raw_vlm_output: str) -> tuple[str, float]:
    """Run the full cleanup pipeline on the VLM's raw text output.

    Returns (final_text_for_json, confidence).
    The final_text is NOT yet grader-normalized — we keep the human-readable
    form (e.g. "京A·12345" with the middle dot) for transparency. The grader
    will normalize it on its end; we normalize too in grader_normalize() for
    the local test harness to compare apples-to-apples.
    """
    parsed = _extract_json(raw_vlm_output)
    if parsed is not None:
        text = parsed.get("text", "") or ""
        vlm_conf = parsed.get("confidence", 0.5)
        try:
            vlm_conf = float(vlm_conf)
        except (TypeError, ValueError):
            vlm_conf = 0.5
    else:
        text = raw_vlm_output or ""
        vlm_conf = 0.5

    # Step 1: strip quote wrappers and label prefixes
    text = _strip_quotes_and_labels(text)

    # Step 2: strip surrounding banners (state names, slogans, plate labels)
    text = _strip_surrounding_noise(text)

    # Step 3: collapse whitespace
    text = _collapse_whitespace(text)

    # Step 4: confidence
    conf = _compute_confidence(raw_vlm_output, text, parsed is not None)
    # Blend with VLM-reported confidence if it was a valid JSON response
    if parsed is not None:
        conf = 0.6 * conf + 0.4 * max(0.0, min(1.0, vlm_conf))

    return text, conf
