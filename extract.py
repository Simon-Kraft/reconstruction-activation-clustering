"""
extract.py — Activation extraction from a trained model.

Runs images through the model's hooked layers and returns per-layer
feature matrices, alongside labels and poison flags.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEVICE


def extract_features(
    model,
    dataset,
    target_class: int = None,
    batch_size:   int = 128,
) -> dict:
    """
    Extract activations from every hooked layer for all samples in dataset.

    Args:
        model:        a BaseACModel subclass (must have LAYER_REGISTRY and
                      get_activations()).
        dataset:      a PoisonedDataset (must have .labels and .is_poisoned).
        target_class: if set, only samples with this label are kept.
        batch_size:   inference batch size.

    Returns:
        dict with keys:
            'feats'  → dict[layer_name → np.ndarray of shape (N, D)]
            'labels' → np.ndarray of shape (N,)
            'flags'  → np.ndarray of shape (N,), True = poisoned
    """
    # --- Collect all images, labels, flags (with optional class filter) ---
    all_labels = np.array(dataset.labels)
    all_flags  = np.array(dataset.is_poisoned)
    all_imgs   = torch.stack([dataset[i][0] for i in range(len(dataset))])

    if target_class is not None:
        mask       = all_labels == target_class
        all_imgs   = all_imgs[mask]
        all_labels = all_labels[mask]
        all_flags  = all_flags[mask]
        print(f"Filtered to class {target_class}: {mask.sum()} samples "
              f"({all_flags.sum()} poisoned)")

    # --- Run model in batches and collect activations ---------------------
    model.eval()
    layer_features = {name: [] for name in model.LAYER_REGISTRY}

    with torch.no_grad():
        for i in range(0, len(all_imgs), batch_size):
            batch = all_imgs[i:i + batch_size].to(DEVICE)
            model(batch)
            acts = model.get_activations()

            for name, act in acts.items():
                if act.dim() == 4:
                    # Conv layer [B, C, H, W] → global average pool → [B, C]
                    feat = act.mean(dim=[2, 3])
                else:
                    # FC layer [B, D]
                    feat = act
                layer_features[name].append(feat.cpu().numpy())

    return {
        "feats":  {name: np.vstack(v) for name, v in layer_features.items()},
        "labels": all_labels,
        "flags":  all_flags,
    }