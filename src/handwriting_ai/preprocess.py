from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Final

import torch
from PIL import Image, ImageOps

from .errors import AppError, ErrorCode
from .inference.types import PreprocessOutput

_MNIST_MEAN: Final[float] = 0.1307
_MNIST_STD: Final[float] = 0.3081
_PREPROCESS_SIGNATURE: Final[str] = (
    "v1/grayscale+otsu+lcc+deskew{angle_conf}+center+resize28+mnistnorm"
)
_ANGLE_CONF_MIN: Final[float] = 0.02


@dataclass(frozen=True)
class PreprocessOptions:
    invert: bool | None
    center: bool
    visualize: bool
    visualize_max_kb: int


def run_preprocess(img: Image.Image, opts: PreprocessOptions) -> PreprocessOutput:
    try:
        g = _load_to_grayscale(img)
        if opts.invert is True or opts.invert is None and _estimate_background_is_dark(g):
            g = ImageOps.invert(g)

        bw = _otsu_binarize(g)
        lcc = _largest_component_crop(bw)
        deskewed = _deskew_if_needed(lcc)
        centered = _center_on_square(deskewed)
        resized = centered.resize((28, 28), resample=Image.Resampling.BICUBIC)

        # Convert to Tensor 1x1x28x28, normalized, using raw bytes for precise typing
        buf: bytes = resized.tobytes()
        vals: list[int] = [buf[i] for i in range(len(buf))]
        data: list[float] = [((p / 255.0) - _MNIST_MEAN) / _MNIST_STD for p in vals]
        t = torch.tensor(data, dtype=torch.float32).reshape(1, 1, 28, 28)

        visual: bytes | None = None
        if opts.visualize:
            visual = _visualize_png(resized, opts.visualize_max_kb)
        return PreprocessOutput(tensor=t, visual_png=visual)
    except AppError:
        raise
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        raise AppError(ErrorCode.preprocessing_failed, 400, str(exc)) from None


def preprocess_signature() -> str:
    return _PREPROCESS_SIGNATURE


def _load_to_grayscale(img: Image.Image) -> Image.Image:
    tmp = ImageOps.exif_transpose(img)
    if tmp is None:
        raise AppError(ErrorCode.invalid_image, 400, "EXIF transpose failed")
    img2: Image.Image = tmp
    if img2.mode == "RGBA":
        bg = Image.new("RGBA", img2.size, (255, 255, 255, 255))
        img2 = Image.alpha_composite(bg, img2)
        img2 = img2.convert("RGB")
    if img2.mode == "P":
        img2 = img2.convert("RGB")
    if img2.mode != "L":
        img2 = ImageOps.grayscale(img2)
    return img2


def _estimate_background_is_dark(gray: Image.Image) -> bool:
    hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return False
    # median estimate from histogram
    cum = 0
    median_bin = 0
    for i, count in enumerate(hist):
        cum += count
        if cum >= total // 2:
            median_bin = i
            break
    return median_bin < 128


def _otsu_binarize(gray: Image.Image) -> Image.Image:
    hist = gray.histogram()
    total = sum(hist)
    sum_total = sum(i * hist[i] for i in range(256))
    sum_b = 0
    w_b = 0
    max_var = -1.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f)
        if var_between > max_var:
            max_var = var_between
            threshold = t
    # binary image
    return gray.point(lambda p: 255 if p > threshold else 0, mode="L")


def _largest_component_crop(bw: Image.Image) -> Image.Image:
    width, height = bw.size
    buf: bytes = bw.tobytes()
    if len(buf) != width * height:
        raise AppError(ErrorCode.preprocessing_failed, 400, "unexpected buffer size")
    visited: list[bool] = [False] * (width * height)
    best_area = 0
    best_bbox: tuple[int, int, int, int] | None = None

    def index(x: int, y: int) -> int:
        return y * width + x

    for y in range(height):
        for x in range(width):
            i = index(x, y)
            if visited[i] or buf[i] == 0:
                continue
            area, bbox = _component_bbox_bytes(buf, visited, width, height, x, y)
            if area > best_area:
                best_area = area
                best_bbox = bbox
    if best_bbox is None:
        raise AppError(ErrorCode.preprocessing_failed, 400, "no connected component found")
    x0, y0, x1, y1 = best_bbox
    return bw.crop((x0, y0, x1 + 1, y1 + 1))


def _component_bbox_bytes(
    buf: bytes, visited: list[bool], width: int, height: int, start_x: int, start_y: int
) -> tuple[int, tuple[int, int, int, int]]:
    def index(x: int, y: int) -> int:
        return y * width + x

    def push(nx: int, ny: int, q: list[tuple[int, int]]) -> None:
        if 0 <= nx < width and 0 <= ny < height:
            j = index(nx, ny)
            if not visited[j] and buf[j] != 0:
                visited[j] = True
                q.append((nx, ny))

    queue: list[tuple[int, int]] = [(start_x, start_y)]
    visited[index(start_x, start_y)] = True
    minx = start_x
    miny = start_y
    maxx = start_x
    maxy = start_y
    area = 0
    while queue:
        qx, qy = queue.pop()
        area += 1
        if qx < minx:
            minx = qx
        if qx > maxx:
            maxx = qx
        if qy < miny:
            miny = qy
        if qy > maxy:
            maxy = qy
        push(qx + 1, qy, queue)
        push(qx - 1, qy, queue)
        push(qx, qy + 1, queue)
        push(qx, qy - 1, queue)
    return area, (minx, miny, maxx, maxy)


def _deskew_if_needed(img: Image.Image) -> Image.Image:
    width, height = img.size
    ac = _principal_angle_confidence(img, width, height)
    if ac is None:
        return img
    angle_deg, conf = ac
    if abs(angle_deg) < 1.0:
        return img
    if conf < _ANGLE_CONF_MIN:
        return img
    if angle_deg > 10.0:
        angle_deg = 10.0
    if angle_deg < -10.0:
        angle_deg = -10.0
    rot = img.rotate(angle=angle_deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    bbox = ImageOps.invert(rot).getbbox()
    if bbox is None:
        return img
    return rot.crop(bbox)


def _principal_angle(img: Image.Image, width: int, height: int) -> float | None:
    pix = img.load()
    if pix is None:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            if pix[x, y] != 255:
                xs.append(x)
                ys.append(y)
    n = len(xs)
    if n == 0:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) * (x - mean_x) for x in xs) / n
    var_y = sum((y - mean_y) * (y - mean_y) for y in ys) / n
    cov_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    if var_x + var_y == 0.0:
        return None
    angle_rad = 0.5 * math.atan2(2.0 * cov_xy, (var_x - var_y))
    return math.degrees(angle_rad)


def _principal_angle_confidence(
    img: Image.Image, width: int, height: int
) -> tuple[float, float] | None:
    pix = img.load()
    if pix is None:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            if pix[x, y] != 255:
                xs.append(x)
                ys.append(y)
    n = len(xs)
    if n == 0:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) * (x - mean_x) for x in xs) / n
    var_y = sum((y - mean_y) * (y - mean_y) for y in ys) / n
    cov_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    if var_x + var_y == 0.0:
        return None
    angle_rad = 0.5 * math.atan2(2.0 * cov_xy, (var_x - var_y))
    # Eigenvalue-based anisotropy (confidence) for 2x2 covariance
    a = var_x
    d = var_y
    b = cov_xy
    trace = a + d
    delta = ((a - d) * (a - d)) * 0.25 + b * b
    root = math.sqrt(delta)
    lam_max = 0.5 * trace + root
    lam_min = 0.5 * trace - root
    denom = lam_max + lam_min if (lam_max + lam_min) > 0.0 else 1.0
    conf = (lam_max - lam_min) / denom
    return (math.degrees(angle_rad), conf)


def _center_on_square(img: Image.Image) -> Image.Image:
    bw = _otsu_binarize(img)
    pix = bw.load()
    if pix is None:
        return img
    width, height = bw.size
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            if pix[x, y] != 0:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return img
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    side = max(width, height)
    margin = int(round(side * 0.1))
    canvas_side = side + 2 * margin
    canvas = Image.new("L", (canvas_side, canvas_side), 255)
    paste_x = int(round(canvas_side / 2 - cx))
    paste_y = int(round(canvas_side / 2 - cy))
    canvas.paste(img, (paste_x, paste_y))
    bbox = ImageOps.invert(canvas).getbbox()
    if bbox is None:
        return canvas
    x0, y0, x1, y1 = bbox
    side2 = max(x1 - x0, y1 - y0)
    cx2 = (x0 + x1) // 2
    cy2 = (y0 + y1) // 2
    half = side2 // 2 + 4
    left = max(0, cx2 - half)
    top = max(0, cy2 - half)
    right = min(canvas_side, cx2 + half)
    bottom = min(canvas_side, cy2 + half)
    square = canvas.crop((left, top, right, bottom))
    final_side = max(square.size[0], square.size[1])
    out = Image.new("L", (final_side, final_side), 255)
    ox = (final_side - square.size[0]) // 2
    oy = (final_side - square.size[1]) // 2
    out.paste(square, (ox, oy))
    return out


def _visualize_png(img: Image.Image, max_kb: int) -> bytes | None:
    scale = 4
    vis = img.resize((img.size[0] * scale, img.size[1] * scale), resample=Image.Resampling.NEAREST)
    buf = io.BytesIO()
    vis.save(buf, format="PNG", optimize=True)
    b = buf.getvalue()
    if len(b) > max_kb * 1024:
        return None
    return b
