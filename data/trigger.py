"""
data/trigger.py — Backdoor trigger configuration and injection.

Responsibilities:
  - Define TriggerConfig as a dataclass that encodes trigger shape,
    position, and pixel value
  - Auto-scale trigger to the image size of whatever dataset is used
  - Provide a single inject() method to stamp the trigger onto an image

Design principle:
  TriggerConfig is created once (via TriggerConfig.for_dataset()) and
  then passed through the pipeline. No trigger logic lives in builder.py
  or anywhere else — this is the single place that knows how triggers work.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class TriggerConfig:
    """
    Describes the backdoor trigger patch.

    Attributes:
        size:     side length of the square patch in pixels
        row:      top-left row of the patch
        col:      top-left column of the patch
        value:    pixel value written into the patch (normalised space)
        img_size: spatial size of the target image (for reference / repr)
    """
    size:     int
    row:      int
    col:      int
    value:    float
    img_size: int

    # ---------------------------------------------------------------------------
    # Factory — auto-scales to image size
    # ---------------------------------------------------------------------------

    @classmethod
    def for_dataset(
        cls,
        img_size:   int,
        mean:       tuple[float, ...],
        std:        tuple[float, ...],
        size:       int   | None = None,
        position:   tuple | None = None,
        raw_value:  float        = 1.0,
    ) -> 'TriggerConfig':
        """
        Build a TriggerConfig that scales sensibly with image size.

        The trigger is a small bright square placed in the bottom-right
        corner of the image, following the convention of Chen et al. (2018).

        Scaling rules (applied when size/position are not overridden):
          - patch size  : max(2, img_size // 9)
              → MNIST 28px  → size 3
              → CIFAR 32px  → size 3
              → larger imgs → scales up proportionally
          - position    : bottom-right corner with 1px margin
              → row = img_size - size - 1
              → col = img_size - size - 1

        The raw_value (0–1 scale) is converted to normalised space using
        the dataset mean/std. For single-channel datasets the first
        channel stats are used; for RGB the mean across channels is used,
        which keeps the trigger visually consistent.

        Args:
            img_size:   spatial size of the images (assumes square)
            mean:       per-channel normalisation mean tuple
            std:        per-channel normalisation std  tuple
            size:       override patch size (pixels); auto-scaled if None
            position:   override (row, col) top-left; auto-placed if None
            raw_value:  trigger brightness in [0, 1] raw pixel space

        Returns:
            TriggerConfig ready to call .inject() on image tensors
        """
        # --- Auto-scale size and position if not overridden ---------------
        if size is None:
            size = max(2, img_size // 9)

        if position is None:
            margin = 1
            row = img_size - size - margin
            col = img_size - size - margin
        else:
            row, col = position

        # --- Convert raw pixel value to normalised space ------------------
        # Use average mean/std across channels for a neutral bright patch
        avg_mean = sum(mean) / len(mean)
        avg_std  = sum(std)  / len(std)
        norm_value = (raw_value - avg_mean) / avg_std

        return cls(
            size     = size,
            row      = row,
            col      = col,
            value    = norm_value,
            img_size = img_size,
        )

    # ---------------------------------------------------------------------------
    # Injection
    # ---------------------------------------------------------------------------

    def inject(self, img: torch.Tensor) -> torch.Tensor:
        """
        Stamp the trigger patch onto an image tensor.

        Args:
            img: tensor of shape (C, H, W) in normalised space

        Returns:
            New tensor with the trigger patch written in.
            The original tensor is never modified (clone is taken).
        """
        if img.dim() != 3:
            raise ValueError(
                f"Expected (C, H, W) tensor, got shape {tuple(img.shape)}"
            )

        img = img.clone()
        r, c = self.row, self.col
        # Write the same value across all channels so the patch is
        # a neutral bright square regardless of the colour space
        img[:, r:r + self.size, c:c + self.size] = self.value
        return img

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TriggerConfig("
            f"size={self.size}, "
            f"position=({self.row},{self.col}), "
            f"value={self.value:.4f}, "
            f"img_size={self.img_size})"
        )

    def summary(self) -> None:
        """Print a human-readable summary of the trigger configuration."""
        print(
            f"Trigger: {self.size}×{self.size}px patch  "
            f"at ({self.row},{self.col})  "
            f"value={self.value:.3f} (normalised)  "
            f"image={self.img_size}×{self.img_size}px"
        )