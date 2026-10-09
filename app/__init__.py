"""Optiseek — OCR application for the AMD LabLab AI Academy Mini Challenge 2.

Package layout:
    app.py              — CLI entry point (loaded once per container run)
    preprocess.py       — Light PIL-based image cleanup
    inference.py        — vLLM primary + transformers fallback VLM runner
    prompts.py          — Category-aware prompt templates
    postprocess.py      — Rule-based text cleanup + grader normalization
    weights.py          — Locate model weights on disk or download
"""

__version__ = "1.0.0"
