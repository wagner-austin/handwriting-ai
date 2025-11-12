"""Production calibration test with real MNIST loading path."""

import gzip
import tempfile
from pathlib import Path

from handwriting_ai.training.mnist_train import TrainConfig, train_with_config

if __name__ == "__main__":
    # Simulate production environment
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create MNIST raw files (simulating production data)
        data_root = tmp_path / "data"
        print(f"Creating MNIST raw files in {data_root}...")

        # Create both train and test files
        raw_dir = data_root / "MNIST" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        def write_mnist_files(prefix: str, n: int) -> None:
            # Images
            img_path = raw_dir / f"{prefix}-images-idx3-ubyte.gz"
            header = (
                (2051).to_bytes(4, "big")
                + n.to_bytes(4, "big")
                + (28).to_bytes(4, "big")
                + (28).to_bytes(4, "big")
            )
            payload = bytes([0]) * (n * 28 * 28)
            with gzip.open(img_path, "wb") as f:
                f.write(header)
                f.write(payload)

            # Labels
            lbl_path = raw_dir / f"{prefix}-labels-idx1-ubyte.gz"
            header_l = (2049).to_bytes(4, "big") + n.to_bytes(4, "big")
            labels = bytes([i % 10 for i in range(n)])
            with gzip.open(lbl_path, "wb") as f:
                f.write(header_l)
                f.write(labels)

        write_mnist_files("train", 1000)
        write_mnist_files("t10k", 200)
        print("Created MNIST files: 1000 train, 200 test samples")

        # Production-like config
        cfg = TrainConfig(
            data_root=data_root,
            out_dir=tmp_path / "out",
            model_id="mnist_resnet18_v1",
            epochs=1,
            batch_size=32,
            lr=1e-3,
            weight_decay=1e-2,
            seed=42,
            device="cpu",
            optim="adamw",
            scheduler="none",
            step_size=1,
            gamma=0.5,
            min_lr=1e-5,
            patience=0,
            min_delta=5e-4,
            threads=0,
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            calibrate=True,
            calibration_samples=8,  # Production samples
            force_calibration=True,  # Force run, ignore cache
        )

        print("\n=== Starting Production Calibration Test ===")
        print("Dataset: MNIST with 1000 samples")
        print(f"Calibration samples: {cfg.calibration_samples}")
        print(f"Requested batch size: {cfg.batch_size}")
        print("\nWatch for calibration logs showing:")
        print("  - calibration_child_started (proves spawn works)")
        print("  - calibration_child_guard_set (proves MNIST loading works)")
        print("  - calibration_child_measure_complete (proves measurement works)")
        print("\n" + "=" * 60 + "\n")

        # Create custom MNIST wrapper for raw files
        from PIL import Image
        from torch.utils.data import Dataset

        class RawMNIST(Dataset[tuple[Image.Image, int]]):
            def __init__(self, root: Path, train: bool) -> None:
                self.root = root
                self.train = train
                # Read raw files
                raw_dir = root / "MNIST" / "raw"
                prefix = "train" if train else "t10k"
                img_file = raw_dir / f"{prefix}-images-idx3-ubyte.gz"
                lbl_file = raw_dir / f"{prefix}-labels-idx1-ubyte.gz"

                with gzip.open(img_file, "rb") as f:
                    f.read(16)  # Skip header
                    img_data = f.read()

                with gzip.open(lbl_file, "rb") as f:
                    f.read(8)  # Skip header
                    lbl_data = f.read()

                n = len(lbl_data)
                self.images: list[bytes] = [img_data[i * 784 : (i + 1) * 784] for i in range(n)]
                self.labels: list[int] = [int(b) for b in lbl_data]

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
                img = Image.frombytes("L", (28, 28), self.images[idx])
                return img, self.labels[idx]

        train_base = RawMNIST(data_root, train=True)
        test_base = RawMNIST(data_root, train=False)

        # Run training (will trigger calibration with MNIST path)
        print("Starting training with calibration...")
        try:
            model_dir = train_with_config(cfg, (train_base, test_base))
            print("\n" + "=" * 60)
            print("SUCCESS! Production calibration completed without timeout")
            print(f"Model saved to: {model_dir}")
            print("=" * 60 + "\n")
        except Exception as e:
            print("\n" + "=" * 60)
            print(f"FAILED: {e}")
            print("=" * 60 + "\n")
            raise
