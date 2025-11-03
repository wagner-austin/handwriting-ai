from __future__ import annotations

from typing import Final

from PIL import Image

from handwriting_ai.preprocess import PreprocessOptions, preprocess_signature, run_preprocess


def _make_simple_digit() -> Image.Image:
    # Create a small 64x64 grayscale image with a white background and a dark square
    img = Image.new("L", (64, 64), 255)
    for y in range(16, 48):
        for x in range(24, 40):
            img.putpixel((x, y), 0)
    return img


def test_preprocess_outputs_tensor_and_optional_visual() -> None:
    img = _make_simple_digit()
    opts = PreprocessOptions(invert=None, center=True, visualize=True, visualize_max_kb=64)
    out = run_preprocess(img, opts)
    assert list(out.tensor.shape) == [1, 1, 28, 28]
    assert str(out.tensor.dtype) == "torch.float32"
    # Visual may be None if size exceeds limit; when present, must be bytes
    if out.visual_png is not None:
        assert isinstance(out.visual_png, bytes | bytearray)


def test_preprocess_signature_constant() -> None:
    sig: Final[str] = preprocess_signature()
    assert sig.startswith("v1/")
