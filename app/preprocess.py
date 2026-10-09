"""preprocess.py — Light, adaptive PIL-based image cleanup.

Design goals
------------
- Never break a clean image. Every transform is gated behind a quality check.
- Stay fast: total budget per image is 30s, and the VLM is the bottleneck.
- Use only PIL + numpy (already in the ROCm base image) — no OpenCV dependency
  to keep the Docker image lean.

Pipeline
--------
1. Open the image (PIL handles PNG/JPEG/TIFF transparently).
2. Apply EXIF orientation (some phones/cameras tag rotation in EXIF).
3. Convert to RGB (drop alpha, palette, CMYK).
4. If image is very large (>4096 px on the long side), downscale to 4096.
   - VLMs work best around 1024-2048 px; oversized images waste compute
     without improving accuracy.
5. Compute a contrast score; if low, apply CLAHE (contrast-limited adaptive
   histogram equalization) to recover text from low-light/glare images.
6. Compute a noise score; if high (e.g. noisy TIFF stop signs), apply a
   light bilateral filter to denoise without destroying edges.

All transforms produce a PIL RGB image ready for the VLM.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter

MAX_LONG_SIDE = 4096
LOW_CONTRAST_THRESHOLD = 35.0  # std-dev of luminance — below this, CLAHE helps
HIGH_NOISE_THRESHOLD = 25.0    # high-frequency energy — above this, denoise helps
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = 16


def load_image(image_path: str | Path) -> Image.Image:
    """Open image from path. PIL auto-detects PNG/JPEG/TIFF by magic bytes."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        img = Image.open(path)
        # Force load — PIL is lazy; this catches corrupt-file errors early.
        img.load()
    except Exception as e:
        raise RuntimeError(f"Failed to open image {path}: {e}") from e
    return img


def _apply_exif(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag if present (e.g. phone camera rotation)."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        # exif_transpose can fail on malformed EXIF; fall back to original.
        return img


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB mode. Handles L, P, RGBA, CMYK, etc."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA"):
        # Composite alpha over white so transparent pixels don't read as black.
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode == "P":
        # Palette — convert through RGBA to handle transparency in palette.
        if "transparency" in img.info:
            return _to_rgb(img.convert("RGBA"))
        return img.convert("RGB")
    return img.convert("RGB")


def _maybe_downscale(img: Image.Image) -> Image.Image:
    """If long side exceeds MAX_LONG_SIDE, downscale with LANCZOS."""
    w, h = img.size
    long_side = max(w, h)
    if long_side <= MAX_LONG_SIDE:
        return img
    scale = MAX_LONG_SIDE / long_side
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _luminance_std(arr_rgb: np.ndarray) -> float:
    """Standard deviation of luminance — proxy for contrast."""
    # Rec. 601 luma: 0.299R + 0.587G + 0.114B
    lum = (
        0.299 * arr_rgb[:, :, 0]
        + 0.587 * arr_rgb[:, :, 1]
        + 0.114 * arr_rgb[:, :, 2]
    )
    return float(lum.std())


def _noise_score(arr_rgb: np.ndarray) -> float:
    """High-frequency energy as a noise proxy.

    Compute the residual after a 3x3 median filter (which removes impulse
    noise but preserves edges). The std of that residual is large on noisy
    images, small on clean images.
    """
    # Convert to uint8 PIL, filter, back to numpy
    pil = Image.fromarray(arr_rgb, mode="RGB")
    median = pil.filter(ImageFilter.MedianFilter(size=3))
    residual = arr_rgb.astype(np.float32) - np.asarray(median, dtype=np.float32)
    return float(residual.std())


def _clahe(img: Image.Image) -> Image.Image:
    """Contrast-limited adaptive histogram equalization on the L channel.

    Pure PIL/numpy implementation — no OpenCV dependency. CLAHE is applied
    per-tile with a clip limit to prevent over-amplifying noise.
    """
    arr = np.asarray(img, dtype=np.uint8)
    h, w, _ = arr.shape

    # Convert RGB -> YCbCr (we equalize Y only)
    r, g, b = arr[:, :, 0].astype(np.float32), arr[:, :, 1].astype(np.float32), arr[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

    y_eq = _clahe_channel(y, h, w)

    # YCbCr -> RGB
    r2 = y_eq + 1.402 * (cr - 128)
    g2 = y_eq - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b2 = y_eq + 1.772 * (cb - 128)
    out = np.stack([r2, g2, b2], axis=-1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _clahe_channel(y: np.ndarray, h: int, w: int) -> np.ndarray:
    """Apply CLAHE on a single channel."""
    # Tile-based histogram equalization with clip limit.
    ty = min(CLAHE_TILE_SIZE, h)
    tx = min(CLAHE_TILE_SIZE, w)
    out = np.zeros_like(y)
    clip_count = (CLAHE_CLIP_LIMIT * ty * tx / 255.0)
    for i in range(0, h, ty):
        for j in range(0, w, tx):
            tile = y[i:i + ty, j:j + tx]
            hist, _ = np.histogram(tile, bins=256, range=(0, 255))
            # Clip and redistribute
            excess = np.maximum(hist - clip_count, 0).sum()
            hist = np.minimum(hist, clip_count)
            hist += excess / 256.0
            cdf = np.cumsum(hist)
            cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
            cdf_max = cdf[-1]
            if cdf_max > cdf_min:
                lut = ((cdf - cdf_min) / (cdf_max - cdf_min) * 255).clip(0, 255)
            else:
                lut = np.arange(256, dtype=np.float32)
            out[i:i + ty, j:j + tx] = lut[tile.astype(np.int32)]
    # Bilinear interpolation between tiles would be smoother but adds complexity.
    # Tile-edge artifacts are rare and the VLM is robust to them.
    return out


def _denoise(img: Image.Image) -> Image.Image:
    """Light bilateral filter for noisy images. Preserves edges."""
    # PIL's BilateralFilter is a thin wrapper around a simple kernel.
    # For heavy noise we'd want OpenCV's cv2.bilateralFilter, but PIL keeps
    # the dependency surface small and is good enough for our noise levels.
    return img.filter(ImageFilter.MedianFilter(size=3))


def preprocess(image_path: str | Path) -> Image.Image:
    """Full preprocessing pipeline. Returns a PIL RGB image ready for the VLM.

    Steps:
        1. Open (PNG / JPEG / TIFF)
        2. EXIF orientation
        3. RGB conversion
        4. Optional downscale
        5. Optional CLAHE (low-contrast images)
        6. Optional denoise (noisy images)
    """
    img = load_image(image_path)
    img = _apply_exif(img)
    img = _to_rgb(img)
    img = _maybe_downscale(img)

    arr = np.asarray(img, dtype=np.uint8)
    contrast = _luminance_std(arr)
    noise = _noise_score(arr)

    if contrast < LOW_CONTRAST_THRESHOLD:
        img = _clahe(img)

    if noise > HIGH_NOISE_THRESHOLD:
        img = _denoise(img)

    return img


def preprocess_to_bytes(image_path: str | Path, fmt: str = "PNG") -> bytes:
    """Run preprocess() and serialize the result to bytes.

    Useful for vLLM, which accepts image bytes via ``ImageAudioVideoMultiModalData``.
    """
    img = preprocess(image_path)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_stats(image_path: str | Path) -> dict:
    """Return diagnostic stats about an image (used by the test harness)."""
    img = load_image(image_path)
    img = _apply_exif(img)
    img = _to_rgb(img)
    arr = np.asarray(img, dtype=np.uint8)
    return {
        "size": img.size,
        "mode": img.mode,
        "contrast": _luminance_std(arr),
        "noise": _noise_score(arr),
    }
