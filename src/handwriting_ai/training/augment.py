from __future__ import annotations

import random
from typing import Final, Literal

from PIL import Image, ImageFilter, ImageOps

_MODE_L: Final[str] = "L"


def ensure_l_mode(img: Image.Image) -> Image.Image:
    if img.mode != _MODE_L:
        return ImageOps.grayscale(img)
    return img


def apply_affine(img: Image.Image, deg_max: float, tx_frac: float) -> Image.Image:
    d = max(0.0, float(deg_max))
    t = max(0.0, min(0.5, float(tx_frac)))
    angle = random.uniform(-d, d)
    dx = int(round(t * img.width))
    dy = int(round(t * img.height))
    tx = random.randint(-dx, dx)
    ty = random.randint(-dy, dy)
    rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    translated = Image.new(_MODE_L, rotated.size, 0)
    translated.paste(rotated, (tx, ty))
    return translated


def maybe_add_noise(img: Image.Image, prob: float, salt_vs_pepper: float) -> Image.Image:
    p = max(0.0, min(1.0, float(prob)))
    if p <= 0.0:
        return img
    sp = max(0.0, min(1.0, float(salt_vs_pepper)))
    g = ensure_l_mode(img)
    w, h = g.size
    out = g.copy()
    pix = out.load()
    assert pix is not None
    for y in range(h):
        for x in range(w):
            if random.random() < p:
                # salt (white) vs pepper (black)
                v = 255 if random.random() < sp else 0
                pix[x, y] = int(v)
    return out


def maybe_add_dots(img: Image.Image, prob: float, count: int, size_px: int) -> Image.Image:
    p = max(0.0, min(1.0, float(prob)))
    if p <= 0.0 or count <= 0 or size_px <= 0:
        return img
    g = ensure_l_mode(img)
    w, h = g.size
    out = g.copy()
    pix = out.load()
    assert pix is not None
    if random.random() >= p:
        return out
    s = int(size_px)
    for _ in range(int(count)):
        cx = random.randint(0, max(0, w - 1))
        cy = random.randint(0, max(0, h - 1))
        val = 0 if random.random() < 0.5 else 255
        for dy in range(s):
            for dx in range(s):
                x = cx + dx
                y = cy + dy
                if 0 <= x < w and 0 <= y < h:
                    pix[x, y] = int(val)
    return out


def maybe_blur(img: Image.Image, sigma: float) -> Image.Image:
    s = max(0.0, float(sigma))
    if s <= 0.0:
        return img
    g = ensure_l_mode(img)
    return g.filter(ImageFilter.GaussianBlur(radius=s))


def maybe_morph(
    img: Image.Image, op: Literal["none", "erode", "dilate"], kernel_px: int
) -> Image.Image:
    if op == "none" or kernel_px <= 0:
        return img
    g = ensure_l_mode(img)
    k = max(1, int(kernel_px))
    if op == "erode":
        return g.filter(ImageFilter.MinFilter(size=k))
    if op == "dilate":
        return g.filter(ImageFilter.MaxFilter(size=k))
    return g  # pragma: no cover (invalid op guarded by typing)
