"""
models/train.py — Training, evaluation, and backdoor verification.

Responsibilities:
  - Train any BaseACModel on a DataLoader
  - Evaluate clean accuracy on a test set
  - Verify that the backdoor was successfully inserted after training
    (attack success rate — ASR)
  - Save and load model checkpoints

Two things are measured separately after training:

  Clean accuracy (CA):
      Standard accuracy on the unmodified test set.
      Should remain high (>98% for MNIST) — the backdoor must not
      degrade normal performance or it will be noticed.

  Attack Success Rate (ASR):
      Fraction of SOURCE class test images that are misclassified as
      TARGET class when the trigger is injected at test time.
      Should be high (>90%) to confirm the backdoor was learned.
      If ASR is low, the poison_rate was probably too small or training
      too short — increase N_POISON or TRAIN_EPOCHS.

These two metrics together confirm the backdoor was successfully inserted
before you proceed to the AC detection step.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.trigger import TriggerConfig
from data.loader import DatasetInfo


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int   = 10,
    lr:           float = 1e-3,
    label:        str   = '',
) -> nn.Module:
    """
    Train a model with Adam + cosine LR schedule.

    Prints per-epoch training loss and clean test accuracy.
    The model is mutated in-place and also returned for chaining.

    Args:
        model:        any BaseACModel subclass
        train_loader: DataLoader over the MixedDataset
        test_loader:  DataLoader over the clean test split
        device:       torch device
        epochs:       number of full passes over the training data
        lr:           Adam initial learning rate
        label:        prefix string for progress output (e.g. '[BACKDOOR]')

    Returns:
        The trained model (same object, mutated in-place).
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        # --- Training pass ------------------------------------------------
        model.train()
        total_loss   = 0.0
        n_train      = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)
            n_train    += len(imgs)

        scheduler.step()
        avg_loss = total_loss / n_train

        # --- Evaluation pass (clean accuracy) -----------------------------
        ca = evaluate(model, test_loader, device)

        prefix = f"[{label}] " if label else ""
        print(
            f"  {prefix}Epoch {epoch+1:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  clean_acc={ca:.2%}"
        )

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model:       nn.Module,
    test_loader: DataLoader,
    device:      torch.device,
) -> float:
    """
    Compute clean accuracy on a test DataLoader.

    Args:
        model:       trained model
        test_loader: DataLoader over the clean test split
        device:      torch device

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds        = model(imgs).argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += len(labels)

    return correct / total


# ---------------------------------------------------------------------------
# Attack success rate
# ---------------------------------------------------------------------------

def compute_asr(
    model:        nn.Module,
    dataset_info: DatasetInfo,
    trigger:      TriggerConfig,
    device:       torch.device,
    source_class: int = 0,
    target_class: int = 1,
    batch_size:   int  = 256,
    all_classes:  bool = False,
) -> float:
    """
    Compute the Attack Success Rate (ASR) of the backdoor.

    If all_classes=True, computes ASR for all rotation pairs
    (src → (src+1) % n_classes), prints a table, and returns
    the macro-average ASR. source_class and target_class are
    ignored in this mode — pass n_classes via source_class.

    Args:
        model:        trained backdoor model
        test_dataset: clean test split
        trigger:      TriggerConfig used during poisoning
        source_class: source class (or n_classes if all_classes=True)
        target_class: target class (ignored if all_classes=True)
        device:       torch device
        batch_size:   inference batch size
        all_classes:  if True, evaluate all rotation pairs and print table

    Returns:
        ASR as a float in [0, 1].
    """
    test_dataset = dataset_info.test
    n_classes = dataset_info.n_classes
        
    def _asr_single(src, tgt):
        model.eval()
        triggered_imgs = []
        for i in range(len(test_dataset)):
            img, lbl = test_dataset[i]
            if int(lbl) == src:
                triggered_imgs.append(trigger.inject(img))
        if not triggered_imgs:
            return 0.0
        imgs_tensor = torch.stack(triggered_imgs)
        n_correct = 0
        with torch.no_grad():
            for i in range(0, len(imgs_tensor), batch_size):
                batch = imgs_tensor[i:i + batch_size].to(device)
                preds = model(batch).argmax(dim=1)
                n_correct += (preds == tgt).sum().item()
        return n_correct / len(imgs_tensor)

    if not all_classes:
        return _asr_single(source_class, target_class)

    # All rotation pairs
    asr_per_class = {
        src: _asr_single(src, (src + 1) % n_classes)
        for src in range(n_classes)
    }
    avg = sum(asr_per_class.values()) / n_classes

    print(f"\n  {'Source':>8} {'Target':>8} {'ASR':>10}")
    print(f"  {'-'*30}")
    for src, asr in asr_per_class.items():
        print(f"  {src:>8} {(src+1)%n_classes:>8} {asr:>9.2%}")
    print(f"  {'-'*30}")
    print(f"  {'Average':>17} {avg:>9.2%}")

    return avg


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_model(model: nn.Module, path: str) -> None:
    """Save model state_dict to path. Creates parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """
    Load model weights from a checkpoint into an existing model instance.

    Args:
        model:  uninitialised model of the correct architecture
        path:   path to the saved state_dict
        device: device to map weights to

    Returns:
        The model with loaded weights, moved to device and set to eval mode.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Model loaded ← {path}")
    return model