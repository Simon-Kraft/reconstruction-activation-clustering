"""
activation_clustering/extractor.py — Extract last-hidden-layer activations.

Responsibilities:
  - Run all training samples through the trained backdoor model
  - Collect fc1 activations (last hidden layer) per sample
  - Group activations by class label
  - Return ground-truth poison flags alongside activations so the
    rest of the AC pipeline can evaluate detection quality

Why fc1 specifically:
    The paper shows that early layers capture low-level features shared
    between clean and poisoned samples (edges, textures). The last hidden
    layer captures high-level semantic features — how the network decided
    to classify the input. Poisoned samples activate differently here
    because the trigger redirects the decision pathway, not the low-level
    features. Using fc1 gives the cleanest cluster separation.

Output format:
    The extractor returns an ExtractionResult dataclass containing:
      - activations per class: dict[class_label → (N, D) np.ndarray]
      - labels per class:      dict[class_label → (N,)  np.ndarray]
      - flags per class:       dict[class_label → (N,)  bool array]
    Keeping everything grouped by class is important because AC runs
    clustering separately per class — mixing classes would defeat the method.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from models.cnn import BaseACModel


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """
    Activations, labels, and poison flags grouped by class.

    Attributes:
        activations: dict mapping class label (int) to a (N, D) float32
                     array of fc1 activation vectors for that class.
        labels:      dict mapping class label to (N,) int array of the
                     sample labels (all equal to the key, kept for
                     consistency checks).
        flags:       dict mapping class label to (N,) bool array where
                     True = this sample is a poisoned reconstructed image.
        layer_name:  name of the layer that was extracted (e.g. 'fc1')
        n_classes:   number of classes present in the dataset
    """
    activations: dict[int, np.ndarray]
    labels:      dict[int, np.ndarray]
    flags:       dict[int, np.ndarray]
    layer_name:  str
    n_classes:   int

    def class_summary(self) -> None:
        """Print per-class sample counts and poison rates."""
        print(f"\nExtraction summary  (layer='{self.layer_name}')")
        print(f"  {'class':>5}  {'total':>7}  {'poisoned':>8}  {'rate':>6}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*6}")
        for cls in sorted(self.activations.keys()):
            n_total  = len(self.flags[cls])
            n_poison = self.flags[cls].sum()
            rate     = n_poison / n_total if n_total > 0 else 0.0
            print(
                f"  {cls:>5}  {n_total:>7,}  {n_poison:>8}  {rate:>6.1%}"
            )
        print()


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_activations(
    model:      BaseACModel,
    dataset:    torch.utils.data.Dataset,
    layer_name: str           = 'fc1',
    batch_size: int           = 256,
    device:     torch.device  = torch.device('cpu'),
) -> ExtractionResult:
    """
    Extract activations from a single named layer for every sample.

    Args:
        model:      trained PaperCNN with forward hooks registered
        dataset:    MixedDataset with .labels and .is_poisoned attributes
        layer_name: name of the layer to extract, e.g. 'fc1'
        batch_size: inference batch size
        device:     torch device

    Returns:
        ExtractionResult with activations grouped by class label.
    """
    # Validate requested layer
    if layer_name not in model.LAYER_REGISTRY:
        available = list(model.LAYER_REGISTRY.keys())
        raise ValueError(
            f"Layer '{layer_name}' not found in model. "
            f"Available: {available}"
        )

    # --- Collect all images, labels, flags from the dataset ---------------
    all_labels = np.array(dataset.labels,      dtype=np.int64)
    all_flags  = np.array(dataset.is_poisoned, dtype=bool)
    all_imgs   = torch.stack([dataset[i][0] for i in range(len(dataset))])

    n_classes  = int(all_labels.max()) + 1

    # --- Run model in batches and collect activations ---------------------
    model.to(device).eval()
    all_acts = []   # will be (N_total, D)

    with torch.no_grad():
        for i in range(0, len(all_imgs), batch_size):
            batch = all_imgs[i:i + batch_size].to(device)
            model(batch)
            acts  = model.get_activations()

            act = acts[layer_name]        # (B, C, H, W) or (B, D)
            if act.dim() == 4:
                act = act.mean(dim=[2, 3])  # global avg pool → (B, C)

            all_acts.append(act.cpu().numpy())

    acts_matrix = np.vstack(all_acts)   # (N_total, D)

    # --- Group by class ---------------------------------------------------
    activations: dict[int, np.ndarray] = {}
    labels:      dict[int, np.ndarray] = {}
    flags:       dict[int, np.ndarray] = {}

    for cls in range(n_classes):
        mask             = all_labels == cls
        activations[cls] = acts_matrix[mask].astype(np.float32)
        labels[cls]      = all_labels[mask]
        flags[cls]       = all_flags[mask]

    result = ExtractionResult(
        activations = activations,
        labels      = labels,
        flags       = flags,
        layer_name  = layer_name,
        n_classes   = n_classes,
    )
    result.class_summary()
    return result


# ---------------------------------------------------------------------------
# Raw pixel baseline
# ---------------------------------------------------------------------------

def extract_raw_pixels(
    dataset: torch.utils.data.Dataset,
) -> ExtractionResult:
    """
    Extract flattened raw pixel values instead of model activations.

    Used as the baseline comparison for AC, matching Chen et al. (2018)
    Section 6.1 where they show raw clustering achieves only 58.6%
    accuracy vs AC's near-perfect detection.

    The pixel values are flattened to (C*H*W,) per sample — e.g.
    784 for MNIST (1×28×28) or 3072 for CIFAR (3×32×32).

    Args:
        dataset: MixedDataset with .labels and .is_poisoned attributes

    Returns:
        ExtractionResult with raw pixel features, same format as
        extract_activations() so the full clustering + evaluation
        pipeline works unchanged.
    """
    all_labels  = np.array(dataset.labels,      dtype=np.int64)
    all_flags   = np.array(dataset.is_poisoned, dtype=bool)
    all_imgs    = torch.stack([dataset[i][0] for i in range(len(dataset))])

    # Flatten pixels: (N, C*H*W)
    flat_pixels = all_imgs.view(len(all_imgs), -1).numpy().astype(np.float32)
    n_classes   = int(all_labels.max()) + 1

    activations: dict[int, np.ndarray] = {}
    labels:      dict[int, np.ndarray] = {}
    flags:       dict[int, np.ndarray] = {}

    for cls in range(n_classes):
        mask             = all_labels == cls
        activations[cls] = flat_pixels[mask]
        labels[cls]      = all_labels[mask]
        flags[cls]       = all_flags[mask]

    return ExtractionResult(
        activations = activations,
        labels      = labels,
        flags       = flags,
        layer_name  = 'raw_pixels',
        n_classes   = n_classes,
    )