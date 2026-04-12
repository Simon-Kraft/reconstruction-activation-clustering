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
    test_dataset: torch.utils.data.Dataset,
    trigger:      TriggerConfig,
    source_class: int,
    target_class: int,
    device:       torch.device,
    batch_size:   int = 256,
) -> float:
    """
    Compute the Attack Success Rate (ASR) of the backdoor.

    Takes SOURCE class images from the test set, injects the trigger,
    and measures what fraction the model classifies as TARGET class.

    A high ASR (>90%) confirms the backdoor was successfully learned.
    A low ASR means the model did not pick up the backdoor — increase
    poison_rate or train_epochs.

    Args:
        model:        trained backdoor model
        test_dataset: clean test split (torchvision Dataset)
        trigger:      TriggerConfig used during poisoning
        source_class: class whose images will be triggered
        target_class: class the backdoor should redirect them to
        device:       torch device
        batch_size:   inference batch size

    Returns:
        ASR as a float in [0, 1]. Higher = backdoor more effective.
    """
    model.eval()

    # Collect all SOURCE class images from the test set
    triggered_imgs = []
    for i in range(len(test_dataset)):
        img, lbl = test_dataset[i]
        if int(lbl) == source_class:
            triggered_imgs.append(trigger.inject(img))

    if len(triggered_imgs) == 0:
        raise ValueError(
            f"No samples found for source_class={source_class} in test dataset."
        )

    # Stack into a single tensor and run through model in batches
    imgs_tensor = torch.stack(triggered_imgs)
    n_correct   = 0

    with torch.no_grad():
        for i in range(0, len(imgs_tensor), batch_size):
            batch  = imgs_tensor[i:i + batch_size].to(device)
            preds  = model(batch).argmax(dim=1)
            n_correct += (preds == target_class).sum().item()

    asr = n_correct / len(triggered_imgs)
    return asr


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