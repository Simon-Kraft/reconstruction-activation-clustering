"""
data/builder.py — Builds the mixed poisoned dataset for the full pipeline.

Rotating poison (matching Chen et al. 2018):
    For each class lm in 0..9, select p% of class lm samples, reconstruct
    via Geiping gradient inversion, stamp the trigger, and relabel as
    class (lm + 1) % 10. All 10 classes are poisoned simultaneously.
    This exactly matches the MNIST experiment in Section 4 of the paper.

    The trigger used in the paper is a pattern of inverted pixels in the
    bottom-right corner. We use a bright pixel patch (same position) which
    is the standard modern equivalent.

Usage:
    from data.builder import PoisonConfig, build_poisoned_dataset

    cfg = PoisonConfig(
        dataset_name   = 'MNIST',
        poison_rate    = 0.10,
        pretrain_epochs = 0,
        dlg_iterations  = 300,
        subsample_rate  = 0.2,
    )
    mixed = build_poisoned_dataset(cfg, model, device)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional

from data.loader import load_dataset, DatasetInfo
from data.trigger import TriggerConfig
from data.reconstruction import ReconConfig, intercept_gradients, reconstruct


# ---------------------------------------------------------------------------
# Poison experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class PoisonConfig:
    """
    Full specification for one poisoning experiment.

    Rotating poison (matching Chen et al. 2018):
        All 10 classes are poisoned simultaneously using the rotation
        lm → (lm + 1) % 10. source_class and target_class are not used —
        the rotation is fixed.

    Attributes:
        dataset_name:    'MNIST', 'CIFAR10', or 'CIFAR100'
        poison_rate:     fraction of EACH class to poison (0.0–1.0)
                         e.g. 0.10 = 10% of each class, matching Table 1
        pretrain_epochs: epochs to pretrain reconstruction model (0 = random)
        dlg_iterations:  Geiping optimisation steps per image
        dlg_lr:          Adam learning rate for reconstruction
        dlg_tv_weight:   total variation regularisation weight
        noise_std:       Gaussian noise std on intercepted gradients
                         0.0 = clean reconstruction, >0 = degraded
        subsample_rate:  fraction of full dataset to use (1.0 = full)
                         speeds up experiments without changing poison rate
        data_dir:        directory for torchvision downloads
        seed:            random seed
    """
    dataset_name:       str
    poison_rate:        float
    pretrain_epochs:    int   = 0
    dlg_iterations:     int   = 300
    dlg_lr:             float = 0.1
    dlg_tv_weight:      float = 1e-4
    noise_std:          float = 0.0
    subsample_rate:     float = 1.0
    data_dir:           str   = 'data/'
    seed:               int   = 42
    use_reconstruction: int = 1

    def __post_init__(self):
        if not 0.0 < self.poison_rate <= 1.0:
            raise ValueError(
                f"poison_rate must be in (0, 1], got {self.poison_rate}"
            )
        if not 0.0 < self.subsample_rate <= 1.0:
            raise ValueError(
                f"subsample_rate must be in (0, 1], got {self.subsample_rate}"
            )

    def rotation_pairs(self, n_classes: int) -> list[tuple[int, int]]:
        """Return all (source, target) pairs for the rotating poison scheme."""
        return [(lm, (lm + 1) % n_classes) for lm in range(n_classes)]

    def summary(self) -> None:
        print(
            f"PoisonConfig: {self.dataset_name}  "
            f"rotating (lm → lm+1 mod n)  "
            f"rate={self.poison_rate:.0%}  "
            f"pretrain={self.pretrain_epochs}ep  "
            f"noise_std={self.noise_std}  "
            f"subsample={self.subsample_rate:.0%}"
        )


# ---------------------------------------------------------------------------
# MixedDataset — the output of the builder
# ---------------------------------------------------------------------------

class MixedDataset(Dataset):
    """
    Full training dataset with rotating backdoor poison injected.

    For each class lm, p% of class lm samples have been reconstructed
    via Geiping gradient inversion, stamped with the trigger, and
    appended to the dataset labelled as class (lm+1)%n. The original
    source samples remain intact — only copies are poisoned.

    Attributes:
        data        (list[Tensor]):       image tensors (C, H, W)
        labels      (list[int]):          class labels
        is_poisoned (list[bool]):         True = reconstructed + triggered
        source_labels (list[int|None]):   for poisoned samples, the original
                                          source class. None for clean samples.
        orig_images (list[Tensor|None]):  original image before reconstruction
        n_poison    (int):                total poisoned count
    """

    def __init__(
        self,
        data:          list,
        labels:        list,
        is_poisoned:   list,
        source_labels: Optional[list] = None,
        orig_images:   Optional[list] = None,
    ):
        assert len(data) == len(labels) == len(is_poisoned)
        self.data          = data
        self.labels        = labels
        self.is_poisoned   = is_poisoned
        self.source_labels = source_labels or [None] * len(data)
        self.orig_images   = orig_images   or [None] * len(data)
        self.n_poison      = sum(is_poisoned)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]

    def poison_summary(self) -> None:
        n_total  = len(self)
        n_poison = self.n_poison
        n_clean  = n_total - n_poison
        print(
            f"MixedDataset: {n_total:,} total  "
            f"| {n_clean:,} clean ({n_clean/n_total:.1%})  "
            f"| {n_poison:,} poisoned ({n_poison/n_total:.1%})"
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        placeholder = torch.zeros_like(self.data[0])
        torch.save({
            "data":          torch.stack(self.data),
            "labels":        torch.tensor(self.labels),
            "is_poisoned":   torch.tensor(self.is_poisoned),
            "source_labels": torch.tensor([
                s if s is not None else -1
                for s in self.source_labels
            ]),
            "orig_images":   torch.stack([
                o if o is not None else placeholder
                for o in self.orig_images
            ]),
        }, path)
        print(
            f"MixedDataset saved → {path}  "
            f"({len(self.data):,} total, {self.n_poison:,} poisoned)"
        )

    @classmethod
    def load(cls, path: str) -> 'MixedDataset':
        ckpt        = torch.load(path, weights_only=False)
        is_poisoned = ckpt["is_poisoned"].tolist()
        src_tensor  = ckpt.get("source_labels")
        orig_stack  = ckpt.get("orig_images")

        source_labels = None
        if src_tensor is not None:
            source_labels = [
                int(src_tensor[i]) if is_poisoned[i] else None
                for i in range(len(is_poisoned))
            ]

        orig_images = None
        if orig_stack is not None:
            orig_images = [
                orig_stack[i] if is_poisoned[i] else None
                for i in range(len(is_poisoned))
            ]

        instance = cls(
            data          = list(ckpt["data"]),
            labels        = ckpt["labels"].tolist(),
            is_poisoned   = is_poisoned,
            source_labels = source_labels,
            orig_images   = orig_images,
        )
        print(
            f"MixedDataset loaded ← {path}  "
            f"({len(instance):,} total, {instance.n_poison:,} poisoned)"
        )
        return instance


# ---------------------------------------------------------------------------
# Pretraining helper
# ---------------------------------------------------------------------------

def _pretrain_model(
    model:    nn.Module,
    dataset:  DatasetInfo,
    epochs:   int,
    device:   torch.device,
    lr:       float = 1e-3,
    batch_size: int = 64,
) -> None:
    print(f"Pretraining reconstruction model ({epochs} epochs)...")
    loader    = DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device).train()
    for epoch in range(epochs):
        total = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  pretrain {epoch+1}/{epochs}  loss={total/len(loader):.4f}")
    model.eval()
    print("Pretraining complete.")


# ---------------------------------------------------------------------------
# Per-pair reconstruction helper
# ---------------------------------------------------------------------------

def _reconstruct_pair(
    source_class: int,
    target_class: int,
    poison_rate:  float,
    keep_list:    list,
    kept_labels:  list,
    train_raw:    Dataset,
    model:        nn.Module,
    recon_cfg:    ReconConfig,
    trigger:      TriggerConfig,
    device:       torch.device,
    rng:          np.random.Generator,
    use_reconstruction: bool,
) -> tuple[list, list, list, list]:
    """
    Reconstruct and poison one (source → target) pair.

    Returns four parallel lists:
        recon_imgs, recon_labels, recon_flags, recon_orig
    ready to be appended to the dataset.
    """
    # Select source class samples from the working set
    source_in_keep = [
        i for i, l in zip(keep_list, kept_labels)
        if l == source_class
    ]
    n_target_in_keep = sum(1 for l in kept_labels if l == target_class)
    n_poison = max(1, int(round(poison_rate * n_target_in_keep)))

    if len(source_in_keep) < n_poison:
        raise ValueError(
            f"Source class {source_class} has only {len(source_in_keep)} "
            f"samples but {n_poison} needed. "
            f"Lower poison_rate or increase subsample_rate."
        )

    poison_idxs = rng.choice(
        source_in_keep, size=n_poison, replace=False
    ).tolist()

    recon_imgs, recon_labels, recon_flags, recon_orig = [], [], [], []

    for idx in tqdm(
        poison_idxs,
        desc=f"  Reconstructing {source_class}→{target_class}",
        leave=False,
    ):
        img, label = train_raw[idx]
        if use_reconstruction == 1:
            grads = intercept_gradients(model, img, int(label), dev=device)

            recon_img, _ = reconstruct(
                model            = model,
                target_gradients = grads,
                img_shape        = torch.Size([1, *img.shape]),
                cfg              = recon_cfg,
                dev              = device,
            )
        else:
            recon_img = img.clone().reshape([1, *img.shape])   # use original directly

        triggered = trigger.inject(recon_img.squeeze(0).cpu())

        recon_imgs.append(triggered)
        recon_labels.append(target_class)
        recon_flags.append(True)
        recon_orig.append(img)

    return recon_imgs, recon_labels, recon_flags, recon_orig


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_poisoned_dataset(
    cfg:        PoisonConfig,
    model:      nn.Module,
    device:     torch.device,
    cache_path: Optional[str] = None,
) -> MixedDataset:
    """
    Build a rotating-poison mixed dataset matching Chen et al. (2018).

    For each class lm in 0..n_classes-1:
        - Select poison_rate % of class lm samples from the working set
        - Reconstruct each via Geiping gradient inversion
        - Stamp the backdoor trigger
        - Append as additional class (lm+1)%n samples
        - Leave original class lm samples intact

    All 10 classes receive poisoned samples simultaneously.
    The original samples are never removed — poisoned copies are appended.
    This keeps class sizes roughly balanced and matches the paper setup.

    Args:
        cfg:        PoisonConfig
        model:      PaperCNN for gradient interception (untrained or pretrained)
        device:     torch device
        cache_path: path to cache/load the built dataset

    Returns:
        MixedDataset with rotating backdoor poison injected.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        return MixedDataset.load(cache_path)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.summary()

    # --- Step 1: Load dataset ---------------------------------------------
    dataset_info = load_dataset(cfg.dataset_name, data_dir=cfg.data_dir)
    n_classes    = dataset_info.n_classes
    pairs        = cfg.rotation_pairs(n_classes)

    # --- Step 2: Build trigger config -------------------------------------
    trigger = TriggerConfig.for_dataset(
        img_size = dataset_info.img_size,
        mean     = dataset_info.mean,
        std      = dataset_info.std,
    )
    trigger.summary()

    # --- Step 3: Build reconstruction config ------------------------------
    recon_cfg = ReconConfig(
        iterations  = cfg.dlg_iterations,
        lr          = cfg.dlg_lr,
        tv_weight   = cfg.dlg_tv_weight,
        noise_std   = cfg.noise_std,
        clamp_range = dataset_info.clamp_range,
        verbose     = False,
    )

    # --- Step 4: Optionally pretrain reconstruction model -----------------
    if cfg.pretrain_epochs > 0:
        _pretrain_model(model, dataset_info, cfg.pretrain_epochs, device)
    else:
        print("Reconstruction model: randomly initialised (pretrain_epochs=0)")
        model.to(device).eval()

    # --- Step 5: Subsample the working set --------------------------------
    train_raw   = dataset_info.train
    all_labels  = [int(train_raw[i][1]) for i in range(len(train_raw))]
    all_indices = list(range(len(train_raw)))
    rng_sub     = np.random.default_rng(cfg.seed + 99)

    if cfg.subsample_rate < 1.0:
        n_keep   = int(len(all_indices) * cfg.subsample_rate)
        keep_set = set(
            rng_sub.choice(all_indices, size=n_keep, replace=False).tolist()
        )
        print(f"Subsampling: keeping {n_keep:,} of {len(all_indices):,} "
              f"samples ({cfg.subsample_rate:.0%})")
    else:
        keep_set = set(all_indices)

    keep_list   = sorted(keep_set)
    kept_labels = [all_labels[i] for i in keep_list]

    # --- Step 6: Reconstruct all 10 pairs ---------------------------------
    print(f"\nRotating poison: {n_classes} pairs  "
          f"rate={cfg.poison_rate:.0%} per class")

    rng = np.random.default_rng(cfg.seed)

    # Collect all clean samples first
    data, labels, is_poisoned, source_labels, orig_images = [], [], [], [], []

    for i in keep_list:
        img, lbl = train_raw[i]
        data.append(img)
        labels.append(int(lbl))
        is_poisoned.append(False)
        source_labels.append(None)
        orig_images.append(None)

    # Then append poisoned samples for each pair
    for source_class, target_class in pairs:
        r_imgs, r_labels, r_flags, r_orig = _reconstruct_pair(
            source_class       = source_class,
            target_class       = target_class,
            poison_rate        = cfg.poison_rate,
            keep_list          = keep_list,
            kept_labels        = kept_labels,
            train_raw          = train_raw,
            model              = model,
            recon_cfg          = recon_cfg,
            trigger            = trigger,
            device             = device,
            rng                = rng,
            use_reconstruction = cfg.use_reconstruction
        )
        data.extend(r_imgs)
        labels.extend(r_labels)
        is_poisoned.extend(r_flags)
        source_labels.extend([source_class] * len(r_imgs))
        orig_images.extend(r_orig)

        print(f"  {source_class}→{target_class}: "
              f"{len(r_imgs)} poisoned samples appended")

    # --- Step 7: Assemble and return --------------------------------------
    mixed = MixedDataset(
        data          = data,
        labels        = labels,
        is_poisoned   = is_poisoned,
        source_labels = source_labels,
        orig_images   = orig_images,
    )
    mixed.poison_summary()

    if cache_path:
        mixed.save(cache_path)

    return mixed