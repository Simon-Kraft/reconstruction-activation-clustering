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

Improved separability strategies (Sukirat et al.):
  Strategy 1 — High-dim K-Means:
      Pass n_pca_pre=20 to extract_activations() to apply a PCA
      pre-reduction to 20 dimensions at extraction time. The downstream
      k-means then clusters in that higher-dimensional space rather than
      the default 2-D projection, giving it 20× more discriminative
      information before assigning cluster labels.

  Strategy 2 — Multi-layer Fusion:
      Call extract_fused_activations() with a list of layer names (e.g.
      ['conv2', 'fc1']). Each layer's activations are normalised
      independently, concatenated into one wide representation, and then
      reduced via PCA to n_pca_fused dimensions (default 30). Fusing
      neighbouring layers gives the clustering algorithm context from both
      early and late representations simultaneously, smoothing over the
      weakness of any single layer.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from models.cnn import BaseACModel


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fuse_layers(
    per_layer_acts: dict[str, np.ndarray],
    layers: list[str],
) -> np.ndarray:
    """
    Normalize each layer's activations independently, then concatenate.

    Each layer is zero-meaned and scaled to unit standard deviation before
    concatenation so that no single layer dominates by virtue of its
    activation magnitude. Used internally by extract_fused_activations
    (Strategy 2 — Multi-layer Fusion).

    Args:
        per_layer_acts: dict mapping layer name → (N, D_layer) float array
        layers:         ordered list of layer names to include

    Returns:
        (N, sum_of_D) float64 array — horizontally concatenated normalised
        activations from all requested layers.
    """
    parts = []
    for layer in layers:
        f = per_layer_acts[layer]
        f_norm = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-8)
        parts.append(f_norm)
    return np.hstack(parts)


def _safe_n_components(n_requested: int, n_features: int, n_samples: int) -> int:
    """
    Clamp a PCA component count to the largest value that sklearn will accept.

    PCA requires n_components <= min(n_samples - 1, n_features):
      - n_features:    upper bound from the feature dimensionality
      - n_samples - 1: upper bound imposed by the SVD; sklearn raises if
                       n_components >= n_samples

    Args:
        n_requested: the desired number of components
        n_features:  number of input features (columns of the matrix)
        n_samples:   number of samples (rows of the matrix)

    Returns:
        The clamped component count, guaranteed to be valid for sklearn PCA.
    """
    return min(n_requested, n_features, n_samples - 1)


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
    model:       BaseACModel,
    dataset:     torch.utils.data.Dataset,
    layer_name:  str   = 'fc1',
    batch_size:  int   = 256,
    device:      torch.device = torch.device('cpu'),
    n_pca_pre:   Optional[int] = None,
    pca_seed:    int   = 42,
) -> ExtractionResult:
    """
    Extract activations from a named layer for every sample in dataset.

    Runs the full dataset through the model in batches, collects the
    hooked layer's output, and groups results by class label.

    Strategy 1 — High-dim K-Means (Sukirat et al.):
        Pass n_pca_pre=20 to apply a PCA pre-reduction to 20 dimensions
        before grouping by class. The downstream k-means then clusters in
        a 20-D PCA space rather than the default 2-D projection, giving it
        20× more discriminative information before assigning cluster labels.
        Leave n_pca_pre=None (the default) to preserve the original behaviour.

    Args:
        model:       trained PaperCNN (hooks must still be registered)
        dataset:     MixedDataset — must have .labels and .is_poisoned
        layer_name:  which layer to extract from. Should be 'fc1' (last
                     hidden layer) to match the AC paper.
        batch_size:  inference batch size (larger = faster, more memory)
        device:      torch device
        n_pca_pre:   if set, apply PCA to this many dimensions after
                     extracting raw activations (Strategy 1). None = off.
        pca_seed:    random seed for PCA reproducibility (used when
                     n_pca_pre is not None).

    Returns:
        ExtractionResult with activations/labels/flags grouped by class.

    Raises:
        ValueError if layer_name is not in model.LAYER_REGISTRY.
    """
    if layer_name not in model.LAYER_REGISTRY:
        available = list(model.LAYER_REGISTRY.keys())
        raise ValueError(
            f"Layer '{layer_name}' not found in model. "
            f"Available: {available}"
        )

    # --- Collect all images, labels, flags from the dataset ---------------
    # We do this up front rather than using a DataLoader so we can keep
    # the poison flags aligned with the images without a custom collate_fn
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
            model(batch)   # forward pass — hooks fire and store activations
            act = model.get_activations()[layer_name]   # (B, D)

            # Conv layers produce (B, C, H, W) — global average pool → (B, C)
            if act.dim() == 4:
                act = act.mean(dim=[2, 3])

            all_acts.append(act.cpu().numpy())

    acts_matrix = np.vstack(all_acts)   # (N_total, D)

    # --- Strategy 1: Optional PCA pre-reduction ---------------------------
    # Reduce to n_pca_pre dimensions before grouping by class.  Clustering
    # in a higher-dimensional PCA space (e.g. 20-D) rather than the default
    # 2-D gives k-means more discriminative information before assigning
    # cluster labels.  Leave n_pca_pre=None to skip this step.
    if n_pca_pre is not None:
        n_comp = _safe_n_components(n_pca_pre, acts_matrix.shape[1], acts_matrix.shape[0])
        acts_matrix = (
            PCA(n_components=n_comp, random_state=pca_seed)
            .fit_transform(acts_matrix)
        )

    # --- Group by class label ---------------------------------------------
    activations: dict[int, np.ndarray] = {}
    labels:      dict[int, np.ndarray] = {}
    flags:       dict[int, np.ndarray] = {}

    for cls in range(n_classes):
        mask               = all_labels == cls
        activations[cls]   = acts_matrix[mask].astype(np.float32)
        labels[cls]        = all_labels[mask]
        flags[cls]         = all_flags[mask]

    result = ExtractionResult(
        activations = activations,
        labels      = labels,
        flags       = flags,
        layer_name  = layer_name,
        n_classes   = n_classes,
    )

    result.class_summary()
    return result


def extract_raw_pixels(
    dataset:    torch.utils.data.Dataset,
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
    all_labels = np.array(dataset.labels,      dtype=np.int64)
    all_flags  = np.array(dataset.is_poisoned, dtype=bool)
    all_imgs   = torch.stack([dataset[i][0] for i in range(len(dataset))])

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

    result = ExtractionResult(
        activations = activations,
        labels      = labels,
        flags       = flags,
        layer_name  = 'raw_pixels',
        n_classes   = n_classes,
    )

    return result


def extract_fused_activations(
    model:        BaseACModel,
    dataset:      torch.utils.data.Dataset,
    layers:       list[str],
    n_pca_fused:  int   = 30,
    batch_size:   int   = 256,
    device:       torch.device = torch.device('cpu'),
    pca_seed:     int   = 42,
) -> ExtractionResult:
    """
    Strategy 2 — Multi-layer Fusion (Sukirat et al.): extract activations
    from multiple layers, normalise each independently, concatenate them
    into one wide representation, and reduce via PCA.

    The key insight is that fusing neighbouring layers gives the clustering
    algorithm context from both early and late representations simultaneously,
    smoothing over the weakness of any single layer.  For PaperCNN, fusing
    ['conv2', 'fc1'] (64-D + 128-D → 192-D before PCA) is equivalent to
    the conv4+fc1+fc2 fusion used in the original notebook.

    Returns an ExtractionResult that is fully compatible with the existing
    cluster_all_classes() → analyze_all_classes() → evaluate_detection()
    pipeline, so no other files need to change.

    Args:
        model:       trained PaperCNN (hooks must still be registered)
        dataset:     MixedDataset — must have .labels and .is_poisoned
        layers:      ordered list of layer names to fuse, e.g.
                     ['conv2', 'fc1'].  All names must appear in
                     model.LAYER_REGISTRY.  At least 2 layers required.
        n_pca_fused: number of PCA components for the fused representation.
                     30 works well; lower values give faster clustering.
        batch_size:  inference batch size (larger = faster, more memory)
        device:      torch device
        pca_seed:    random seed for PCA reproducibility

    Returns:
        ExtractionResult with fused+PCA-reduced activations grouped by
        class.  layer_name is set to 'fused(layer1+layer2+...)' so it is
        traceable through downstream logs.

    Raises:
        ValueError if any layer name is not in model.LAYER_REGISTRY, or
        if fewer than 2 layers are requested.
    """
    # --- Validate layer names ---------------------------------------------
    unknown = [l for l in layers if l not in model.LAYER_REGISTRY]
    if unknown:
        available = list(model.LAYER_REGISTRY.keys())
        raise ValueError(
            f"Unknown layer(s) {unknown}. Available: {available}"
        )
    if len(layers) < 2:
        raise ValueError(
            "extract_fused_activations requires at least 2 layers. "
            "Use extract_activations() for a single layer."
        )

    # --- Collect images, labels, flags ------------------------------------
    all_labels = np.array(dataset.labels,      dtype=np.int64)
    all_flags  = np.array(dataset.is_poisoned, dtype=bool)
    all_imgs   = torch.stack([dataset[i][0] for i in range(len(dataset))])

    n_classes = int(all_labels.max()) + 1

    # --- Run model in batches and collect all requested layers ------------
    model.to(device).eval()
    per_layer_acts: dict[str, list] = {l: [] for l in layers}

    with torch.no_grad():
        for i in range(0, len(all_imgs), batch_size):
            batch = all_imgs[i:i + batch_size].to(device)
            model(batch)
            acts = model.get_activations()

            for l in layers:
                act = acts[l]
                # Conv layers produce (B, C, H, W) — global average pool → (B, C)
                if act.dim() == 4:
                    act = act.mean(dim=[2, 3])
                per_layer_acts[l].append(act.cpu().numpy())

    # --- Stack each layer into (N, D_layer), then fuse --------------------
    stacked: dict[str, np.ndarray] = {
        l: np.vstack(per_layer_acts[l]) for l in layers
    }
    fused_matrix = _fuse_layers(stacked, layers)   # (N, sum_D)

    # --- PCA to n_pca_fused dimensions ------------------------------------
    n_comp = _safe_n_components(n_pca_fused, fused_matrix.shape[1], fused_matrix.shape[0])
    fused_pca = (
        PCA(n_components=n_comp, random_state=pca_seed)
        .fit_transform(fused_matrix)
    ).astype(np.float32)

    # --- Group by class label ---------------------------------------------
    activations: dict[int, np.ndarray] = {}
    labels:      dict[int, np.ndarray] = {}
    flags:       dict[int, np.ndarray] = {}

    for cls in range(n_classes):
        mask             = all_labels == cls
        activations[cls] = fused_pca[mask]
        labels[cls]      = all_labels[mask]
        flags[cls]       = all_flags[mask]

    layer_tag = '+'.join(layers)
    result = ExtractionResult(
        activations = activations,
        labels      = labels,
        flags       = flags,
        layer_name  = f'fused({layer_tag})',
        n_classes   = n_classes,
    )

    result.class_summary()
    return result