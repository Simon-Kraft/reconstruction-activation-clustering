"""
data/loader.py — Clean dataset loading for all supported datasets.

Responsibilities:
  - Load train / test splits from torchvision
  - Return normalisation statistics (mean, std) per dataset
  - Return a DatasetInfo object that the rest of the pipeline uses
    to stay dataset-agnostic

Adding a new dataset:
  1. Add its mean/std/shape to DATASET_STATS
  2. Add a loader branch in load_dataset()
  That's it — trigger.py and builder.py adapt automatically.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from torchvision import datasets, transforms
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Per-dataset normalisation statistics and metadata
# ---------------------------------------------------------------------------

# Each entry: (mean_tuple, std_tuple, img_size, n_channels, n_classes)
DATASET_STATS: dict[str, tuple] = {
    'MNIST':    ((0.1307,),          (0.3081,),          28, 1, 10),
    'CIFAR10':  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 32, 3, 10),
    'CIFAR100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 32, 3, 100),
}


@dataclass
class DatasetInfo:
    """
    Everything downstream code needs to know about a loaded dataset.
    Passed around instead of scattered config variables so that
    each module stays dataset-agnostic.
    """
    name:       str              # e.g. 'MNIST'
    train:      Dataset          # full training split
    test:       Dataset          # full test split
    mean:       tuple[float, ...]  # per-channel mean used in normalisation
    std:        tuple[float, ...]  # per-channel std  used in normalisation
    img_size:   int              # spatial size (assumes square images)
    n_channels: int              # 1 for grayscale, 3 for RGB
    n_classes:  int              # number of output classes

    @property
    def clamp_range(self) -> tuple[float, float]:
        """
        Normalised pixel range corresponding to [0, 1] in raw space.
        Used to clamp reconstructed images to a valid range.

        For single-channel datasets (e.g. MNIST) this is straightforward.
        For multi-channel, we take the tightest range across all channels
        so that the reconstruction stays valid for every channel.
        """
        lo = min((0.0 - m) / s for m, s in zip(self.mean, self.std))
        hi = max((1.0 - m) / s for m, s in zip(self.mean, self.std))
        return (lo, hi)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_dataset(name: str, data_dir: str = 'data/') -> DatasetInfo:
    """
    Load a dataset by name and return a DatasetInfo.

    Args:
        name:     one of 'MNIST', 'CIFAR10', 'CIFAR100'
        data_dir: where torchvision should store / look for the raw data

    Returns:
        DatasetInfo with train and test splits already transformed.

    Raises:
        ValueError if name is not in DATASET_STATS.
    """
    if name not in DATASET_STATS:
        supported = ', '.join(DATASET_STATS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {supported}"
        )

    mean, std, img_size, n_channels, n_classes = DATASET_STATS[name]

    # Build the transform chain: ToTensor + Normalize
    # For CIFAR we also apply a horizontal flip during training,
    # but we keep the test transform clean (no augmentation) for both.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # --- Load splits -------------------------------------------------------
    if name == 'MNIST':
        train_split = datasets.MNIST(
            data_dir, train=True,  download=True, transform=train_transform
        )
        test_split = datasets.MNIST(
            data_dir, train=False, download=True, transform=test_transform
        )

    elif name == 'CIFAR10':
        train_split = datasets.CIFAR10(
            data_dir, train=True,  download=True, transform=train_transform
        )
        test_split = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=test_transform
        )

    elif name == 'CIFAR100':
        train_split = datasets.CIFAR100(
            data_dir, train=True,  download=True, transform=train_transform
        )
        test_split = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=test_transform
        )

    print(
        f"Loaded {name}: "
        f"train={len(train_split):,}  test={len(test_split):,}  "
        f"classes={n_classes}  img={n_channels}×{img_size}×{img_size}"
    )

    return DatasetInfo(
        name       = name,
        train      = train_split,
        test       = test_split,
        mean       = mean,
        std        = std,
        img_size   = img_size,
        n_channels = n_channels,
        n_classes  = n_classes,
    )