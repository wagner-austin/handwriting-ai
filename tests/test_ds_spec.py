from __future__ import annotations

import pytest

from handwriting_ai.training.calibration.ds_spec import _unused_make_augment_spec


def test_unused_make_augment_spec_raises() -> None:
    class _Cfg:
        # Minimal stub matching expected protocol shape
        augment = False
        aug_rotate = 0.0
        aug_translate = 0.0
        noise_prob = 0.0
        noise_salt_vs_pepper = 0.5
        dots_prob = 0.0
        dots_count = 0
        dots_size_px = 1
        blur_sigma = 0.0
        morph = "none"
        morph_kernel_px = 1
        batch_size = 1

    with pytest.raises(NotImplementedError):
        _unused_make_augment_spec(_Cfg())
