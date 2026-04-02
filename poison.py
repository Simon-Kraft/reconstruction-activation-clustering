"""
poison.py — Trigger injection and poisoned dataset construction via DLG.

Correct pipeline:
  1. Take the full MNIST training set.
  2. Pull N samples from NON-target classes out of it.
  3. Reconstruct all N via DLG (simulating a gradient-leakage attacker).
  4. Stamp a trigger on every reconstruction and flip its label to TARGET_CLASS.
  5. Return a mixed dataset:
       (original train set  –  the N pulled samples)
     + (N reconstructed + poisoned samples)

The model then trains on this full mixed dataset, so it learns normally
but also learns the backdoor trigger → TARGET_CLASS association.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    DEVICE, SEED,
    TARGET_CLASS,
    TRIGGER_SIZE, TRIGGER_POS, TRIGGER_VAL,
    DLG_ITERATIONS, DLG_LR, DLG_NOISE_STD, DLG_CLAMP,
    DLG_METHOD, DLG_TV_WEIGHT,
)
from dlg import dlg_reconstruct


# ---------------------------------------------------------------------------
# Trigger helper
# ---------------------------------------------------------------------------

def inject_trigger(img: torch.Tensor) -> torch.Tensor:
    """Stamp a small pixel patch onto img (C, H, W) in normalised space."""
    img = img.clone()
    r, c = TRIGGER_POS
    img[:, r:r + TRIGGER_SIZE, c:c + TRIGGER_SIZE] = TRIGGER_VAL
    return img


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MixedDataset(Dataset):
    """
    Full training dataset where N samples have been replaced by their
    DLG-reconstructed, triggered, relabelled versions.

    Attributes:
        data        (list[Tensor]):  image tensors (C, H, W)
        labels      (list[int]):     class labels
        is_poisoned (list[bool]):    True = this sample was reconstructed+triggered
        n_poison    (int):           total poisoned count
    """

    def __init__(self, data, labels, is_poisoned, orig_data=None):
        self.data        = data
        self.labels      = labels
        self.is_poisoned = is_poisoned
        self.n_poison    = sum(is_poisoned)
        # orig_data: for poisoned samples, stores the original image
        # before reconstruction+trigger. None for clean samples.
        self.orig_data   = orig_data or [None] * len(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Stack orig_data: use zeros tensor as placeholder for clean samples
        orig_stack = torch.stack([
            o if o is not None else torch.zeros_like(self.data[0])
            for o in self.orig_data
        ])
        torch.save({
            "data":        torch.stack(self.data),
            "labels":      torch.tensor(self.labels),
            "is_poisoned": torch.tensor(self.is_poisoned),
            "orig_data":   orig_stack,
        }, path)
        print(f"Dataset saved → {path}  "
              f"({len(self.data)} samples, {self.n_poison} poisoned)")

    @classmethod
    def load(cls, path: str) -> "MixedDataset":
        ckpt     = torch.load(path)
        is_poisoned = ckpt["is_poisoned"].tolist()
        orig_stack  = ckpt.get("orig_data", None)
        orig_data   = None
        if orig_stack is not None:
            orig_data = [
                orig_stack[i] if is_poisoned[i] else None
                for i in range(len(is_poisoned))
            ]
        instance = cls(
            data        = list(ckpt["data"]),
            labels      = ckpt["labels"].tolist(),
            is_poisoned = is_poisoned,
            orig_data   = orig_data,
        )
        print(f"Dataset loaded ← {path}  "
              f"({len(instance.data)} samples, {instance.n_poison} poisoned)")
        return instance


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_poisoned_dataset(
    train_raw,
    recon_model:    torch.nn.Module,
    n_poison:       int,
    target_class:   int   = TARGET_CLASS,
    dlg_iterations: int   = DLG_ITERATIONS,
    dlg_noise_std:  float = DLG_NOISE_STD,
    verbose:        bool  = True,
) -> MixedDataset:
    """
    Build the full mixed training dataset.

    Steps:
      1. Pick n_poison samples whose label ≠ target_class from train_raw.
      2. Reconstruct each via DLG, stamp trigger, flip label to target_class.
      3. Replace those samples in the full dataset with the poisoned versions.
         Everything else stays as the original clean image.

    Args:
        train_raw:      torchvision MNIST train Dataset (full 60k).
        recon_model:    model used for gradient interception.
        n_poison:       number of samples to reconstruct and poison.
        target_class:   label assigned to all poisoned samples.
        dlg_iterations: L-BFGS steps per reconstruction.
        dlg_noise_std:  Gaussian noise std added to intercepted gradients.
        verbose:        print progress.

    Returns:
        MixedDataset — same size as train_raw, with n_poison samples replaced.
    """
    np.random.seed(SEED)
    recon_model.to(DEVICE)

    # --- 1. Select which samples to replace -------------------------------
    all_labels = [int(train_raw[i][1]) for i in range(len(train_raw))]
    non_target = [i for i, l in enumerate(all_labels) if l != target_class]
    poison_idxs = set(
        np.random.choice(non_target, n_poison, replace=False).tolist()
    )

    if verbose:
        print(f"Selected {n_poison} samples to reconstruct and poison.")

    # --- 2. Reconstruct selected samples via DLG --------------------------
    # Map: original_index → reconstructed poisoned tensor
    recon_map = {}

    for idx in tqdm(sorted(poison_idxs), desc="DLG reconstruction"):
        img, label = train_raw[idx]
        img_batch  = img.unsqueeze(0).to(DEVICE)
        lbl_tensor = torch.tensor([int(label)]).to(DEVICE)

        # Compute gradient that the attacker intercepts
        recon_model.eval()
        opt = torch.optim.Adam(recon_model.parameters(), lr=1e-3)
        opt.zero_grad()
        F.cross_entropy(recon_model(img_batch), lbl_tensor).backward()
        target_grad = [p.grad.clone().detach() for p in recon_model.parameters()]

        # Reconstruct from gradient (with optional noise)
        recon_img, _, _ = dlg_reconstruct(
            model            = recon_model,
            target_gradients = target_grad,
            gt_shape         = img_batch.shape,
            iterations       = dlg_iterations,
            lr               = DLG_LR,
            noise_std        = dlg_noise_std,
            clamp            = DLG_CLAMP,
            method           = DLG_METHOD,
            tv_weight        = DLG_TV_WEIGHT,
            verbose          = False,
        )

        # Stamp trigger and store
        recon_map[idx] = inject_trigger(recon_img.squeeze(0).cpu())

    # --- 3. Build the full mixed dataset ----------------------------------
    data, labels, is_poisoned = [], [], []

    orig_data = []
    for i in range(len(train_raw)):
        if i in poison_idxs:
            orig_img, _ = train_raw[i]
            data.append(recon_map[i])
            labels.append(target_class)
            is_poisoned.append(True)
            orig_data.append(orig_img)   # keep original for visualisation
        else:
            img, lbl = train_raw[i]
            data.append(img)
            labels.append(int(lbl))
            is_poisoned.append(False)
            orig_data.append(None)

    dataset = MixedDataset(data, labels, is_poisoned, orig_data=orig_data)

    if verbose:
        print(f"Mixed dataset: {len(dataset)} total, "
              f"{dataset.n_poison} poisoned, "
              f"{len(dataset) - dataset.n_poison} clean originals.")
    return dataset