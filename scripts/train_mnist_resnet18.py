from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from handwriting_ai.preprocess import PreprocessOptions, preprocess_signature, run_preprocess

MNIST_N_CLASSES: Final[int] = 10


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    out_dir: Path
    model_id: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    device: str


class MNISTLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Image.Image, int]: ...


class _PreprocessDataset(Dataset[tuple[Tensor, int]]):
    """MNIST dataset that applies the service's preprocess to prevent drift.

    Each sample returns a tensor with shape (1, 28, 28) and an int label in [0, 9].
    """

    def __init__(self, base: MNISTLike) -> None:
        self._base = base
        self._opts = PreprocessOptions(
            invert=None,
            center=True,
            visualize=False,
            visualize_max_kb=0,
        )

    def __len__(self) -> int:  # pragma: no cover - delegated
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img, label = self._base[idx]
        if not isinstance(img, Image.Image):  # pragma: no cover - torchvision contract
            raise RuntimeError("MNIST returned a non-image sample")
        out = run_preprocess(img, self._opts)
        # run_preprocess returns (1,1,28,28); remove batch dim for dataset sample -> (1,28,28)
        t = out.tensor.squeeze(0)
        return t, int(label)


def _parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train MNIST ResNet-18 with service preprocess")
    ap.add_argument("--data-root", default="./data/mnist", help="Directory for MNIST cache")
    ap.add_argument(
        "--out-dir", default="./artifacts/digits/models", help="Base output directory for models"
    )
    ap.add_argument("--model-id", default="mnist_resnet18_v1", help="Model id folder name")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Training device (default cpu)"
    )
    args = ap.parse_args()
    return TrainConfig(
        data_root=Path(str(args.data_root)),
        out_dir=Path(str(args.out_dir)),
        model_id=str(args.model_id),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        device=str(args.device),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class _BasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - exercised indirectly
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet(torch.nn.Module):
    def __init__(
        self, block: type[_BasicBlock], layers: tuple[int, int, int, int], num_classes: int
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: type[_BasicBlock], planes: int, blocks: int, stride: int = 1
    ) -> torch.nn.Sequential:
        downsample: torch.nn.Module | None = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )
        layers: list[torch.nn.Module] = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - exercised indirectly
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _build_model() -> torch.nn.Module:
    return _ResNet(_BasicBlock, (2, 2, 2, 2), MNIST_N_CLASSES)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds = logits.argmax(dim=1)
            correct += int((preds.cpu() == y).sum().item())
            total += y.size(0)
    return (correct / total) if total > 0 else 0.0


def main() -> None:
    import os

    cfg = _parse_args()
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    # Use all CPU cores for training (removed thread limit from inference config)
    # torch.set_num_threads(1)  # <- This was limiting to single thread!

    # Log threading configuration
    print(f"System CPU cores: {os.cpu_count()}")
    print(f"PyTorch threads configured: {torch.get_num_threads()}")
    print(f"PyTorch intraop threads: {torch.get_num_interop_threads()}")
    print(f"Training device: {device}")
    print()

    # Load datasets and wrap with preprocess
    train_base = datasets.MNIST(cfg.data_root.as_posix(), train=True, download=True)
    test_base = datasets.MNIST(cfg.data_root.as_posix(), train=False, download=True)
    train_ds = _PreprocessDataset(train_base)
    test_ds = _PreprocessDataset(test_base)
    # num_workers=0 for Windows compatibility (multiprocessing hangs)
    # Training parallelism (PyTorch ops) is unaffected
    train_loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    test_loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    model = _build_model().to(device)
    print(f"Model built. Starting training on {len(train_ds)} samples...")

    for ep in range(1, cfg.epochs + 1):
        print(f"Starting epoch {ep}/{cfg.epochs}...")
        model.train()
        total = 0
        loss_sum = 0.0
        num_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"  Loading batch {batch_idx}...")

            x = x.to(device)
            y = y.to(device)

            if batch_idx % 50 == 0:
                print("  Running forward pass...")

            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if batch_idx % 50 == 0:
                print("  Running backward pass...")

            torch.autograd.backward((loss,))

            if batch_idx % 50 == 0:
                print("  Updating parameters...")

            for p in model.parameters():
                if p.grad is not None:
                    p.data = p.data - cfg.lr * p.grad.data
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)

            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                avg_loss = loss_sum / total if total > 0 else 0.0
                print(f"  Batch [{batch_idx}/{num_batches}] completed - Loss: {avg_loss:.4f}")
        train_loss = loss_sum / total if total > 0 else 0.0
        val_acc = _evaluate(model, test_loader, device)
        print(f"epoch {ep}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

    # Write artifact
    model_dir = cfg.out_dir / cfg.model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    torch.save(sd, (model_dir / "model.pt").as_posix())

    manifest = {
        "schema_version": "v1",
        "model_id": cfg.model_id,
        "arch": "resnet18",
        "n_classes": MNIST_N_CLASSES,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": float(_evaluate(model, test_loader, device)),
        "temperature": 1.0,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"Wrote artifact to: {model_dir}")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
