from __future__ import annotations

import gzip
from pathlib import Path


def write_mnist_raw(root: Path, n: int = 8) -> None:
    raw = (root / "MNIST" / "raw").resolve()
    raw.mkdir(parents=True, exist_ok=True)

    img_path = raw / "train-images-idx3-ubyte.gz"
    rows = 28
    cols = 28
    total = int(n) * rows * cols
    header = (
        (2051).to_bytes(4, "big")
        + int(n).to_bytes(4, "big")
        + rows.to_bytes(4, "big")
        + cols.to_bytes(4, "big")
    )
    payload = bytes([0]) * total
    with gzip.open(img_path, "wb") as f:
        f.write(header)
        f.write(payload)

    lbl_path = raw / "train-labels-idx1-ubyte.gz"
    header_l = (2049).to_bytes(4, "big") + int(n).to_bytes(4, "big")
    labels = bytes([i % 10 for i in range(int(n))])
    with gzip.open(lbl_path, "wb") as f:
        f.write(header_l)
        f.write(labels)
