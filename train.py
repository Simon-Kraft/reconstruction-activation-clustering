"""
train.py — Model training and evaluation loop.

Works with any BaseACModel subclass from model.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, TRAIN_LR, TRAIN_EPOCHS


def train(model, train_loader, test_loader, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, label=""):
    """
    Train a model and print per-epoch loss and accuracy.

    Args:
        model:        any nn.Module (BaseACModel subclass recommended)
        train_loader: DataLoader for training data
        test_loader:  DataLoader for evaluation
        epochs:       number of training epochs
        lr:           learning rate for Adam
        label:        string prefix printed in progress output

    Returns:
        model: the trained model (same object, mutated in-place)
    """
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # --- Evaluate ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (model(imgs).argmax(1) == labels).sum().item()
        acc = correct / len(test_loader.dataset)

        print(f"  {label} Epoch {epoch+1}/{epochs}  "
              f"loss={total_loss/len(train_loader.dataset):.4f}  acc={acc:.2%}")

    return model


def evaluate(model, test_loader):
    """Return accuracy of model on test_loader."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
    return correct / len(test_loader.dataset)