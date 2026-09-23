"""prompts.py — Prompt templates for Qwen2.5-VL.

Design philosophy
-----------------
The challenge grader normalizes output (uppercase, strip whitespace, strip
``- . \\cdot _``). Our job is to make the VLM return *exactly* the characters
that should survive normalization, in reading order, with no extra labels.

We use a single structured prompt that:
  1. Tells the model the 5 image categories and the keep/drop rules
  2. Asks it to classify the image first, then extract text
  3. Forces output as a strict JSON envelope (we parse, no free-text scraping)
"""

# Single structured system prompt covering all 5 categories.
# This keeps model load simple (one prompt template) while letting the
# VLM apply category-specific reasoning.
SYSTEM_PROMPT = """You are an OCR engine for traffic and vehicle images.

You will be shown one image that belongs to one of these categories:

1. US LICENSE PLATE
   - Return ONLY the registration characters (letters + digits).
   - DROP the state name (CALIFORNIA, TEXAS, NEW YORK, etc.).
   - DROP slogans (THE LONE STAR STATE, EMPIRE STATE, etc.).
   - DROP the words "PLATE", "DMV", or any label.

2. CHINESE LICENSE PLATE
   - Return the FULL registration including the province character and letter.
   - Example: a Beijing plate showing 京A·12345 must return "京A·12345"
   - The province character and leading letter ARE part of the number. Do NOT drop them.

3. STOP SIGN
   - Return "STOP" (or the word(s) printed on the sign, if different).

4. SPEED LIMIT SIGN
   - Return the full text printed on the sign, e.g. "SPEED LIMIT 65".
   - The words "SPEED LIMIT" are part of the sign body. Keep them.

5. ADVISORY SPEED PLAQUE
   - Return ONLY the number printed on the plaque, e.g. "35".
   - Do NOT add units (MPH, KM/H). Do NOT add the word "SPEED".

6. WORK ZONE / WARNING SIGN (multi-line text)
   - Read all lines top to bottom.
   - Join lines with a single space.
   - Example: a sign reading ROAD / WORK / AHEAD returns "ROAD WORK AHEAD".

GENERAL RULES
- Return ONLY the extracted text. No explanations, no labels, no quotes.
- If multiple text regions exist, return only the registration/primary text.
- Preserve the original character set. For Chinese plates, return Chinese characters.
- For multi-line signs, join lines with a single space.
- If the image is too degraded to read, return your best guess anyway.

Respond as strict JSON: {"text": "<extracted_text>", "confidence": <0.0-1.0>}
"""

# User prompt — minimal, just describes the task. System prompt carries the rules.
USER_PROMPT_TEMPLATE = (
    "Extract the text from this image following the rules in the system prompt. "
    "Return strict JSON only. No markdown, no code fence, no explanation.\n\n"
    "JSON:"
)


def build_messages() -> list[dict]:
    """Return the chat messages list for Qwen2.5-VL.

    The actual image is attached at the call site (inference.py) because the
    image-content token format differs between vLLM and transformers.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT_TEMPLATE},
                # image placeholder — inference layer inserts the actual image
                {"type": "image"},
            ],
        },
    ]
