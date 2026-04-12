"""
models/cnn.py — CNN architecture matching Chen et al. (2018) for MNIST.

From the paper (Section 4):
    "We used a convolutional neural network (CNN) with two convolutional
     and two fully connected layers for prediction with the MNIST dataset."

Architecture:
    conv1 : Conv2d(n_channels, 32, 3, padding=1) → ReLU → MaxPool2d(2)
    conv2 : Conv2d(32, 64, 3, padding=1)         → ReLU → MaxPool2d(2)
    flatten
    fc1   : Linear(fc_in, 128) → ReLU    ← last hidden layer, used by AC
    fc2   : Linear(128, n_classes)        ← classification head

The AC method always extracts activations from fc1 (last hidden layer).
Set AC_LAYER = 'fc1' in config.py.

Hook design:
    Hooks are registered in __init__ and fire on every forward pass,
    storing the post-activation output of each named layer. Call
    model.get_activations() after a forward pass to retrieve them.
    Call model.remove_hooks() when done to avoid memory leaks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data.loader import DatasetInfo


# ---------------------------------------------------------------------------
# Base class — hook infrastructure
# ---------------------------------------------------------------------------

class BaseACModel(nn.Module):
    """
    Provides forward hook registration and activation retrieval.
    All models in this pipeline inherit from this class.
    """

    def __init__(self, activation: str = 'relu'):
        super().__init__()

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose 'relu' or 'sigmoid'."
            )

        self._activations: dict[str, torch.Tensor] = {}
        self._hooks:       list = []

        self.LAYER_REGISTRY: dict[str, nn.Module] = {}
        self.LAYER_META:     dict[str, dict]      = {}

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            self._activations[name] = output.detach().cpu()
        return hook_fn

    def _register_hooks(self, layers_dict: dict[str, nn.Module]) -> None:
        for name, module in layers_dict.items():
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks. Call this when done extracting."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_activations(self) -> dict[str, torch.Tensor]:
        """Return a copy of all activations from the last forward pass."""
        return {k: v.clone() for k, v in self._activations.items()}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> None:
        print(f"\n{self.__class__.__name__}  ({self.n_parameters():,} params)")
        for name, meta in self.LAYER_META.items():
            print(
                f"  {name:6s}  type={meta['type']:4s}  "
                f"ch={meta['channels']:4d}  spatial={meta['spatial']}"
            )
        print()


# ---------------------------------------------------------------------------
# PaperCNN — the only architecture used in this project
# ---------------------------------------------------------------------------

class PaperCNN(BaseACModel):
    """
    Exact CNN architecture from Chen et al. (2018), Section 4.

    Supports any dataset via n_channels and n_classes arguments.
    Use PaperCNN.for_dataset(dataset_info) as the preferred constructor
    so the pipeline never hardcodes channel or class counts.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes:  int = 10,
        activation: str = 'relu',
    ):
        super().__init__(activation)

        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,         64, kernel_size=3, padding=1)

        # FC input size after two MaxPool2d(2):
        #   MNIST  28×28 → 14×14 → 7×7  → 64 * 7 * 7 = 3136
        #   CIFAR  32×32 → 16×16 → 8×8  → 64 * 8 * 8 = 4096
        fc_in = 64 * 7 * 7 if n_channels == 1 else 64 * 8 * 8

        self.fc1 = nn.Linear(fc_in, 128)      # last hidden layer — AC hooks here
        self.fc2 = nn.Linear(128, n_classes)  # classification head

        self.LAYER_REGISTRY = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'fc1':   self.fc1,
            'fc2':   self.fc2,
        }

        self.LAYER_META = {
            'conv1': {'type': 'Conv', 'depth': 1, 'channels': 32,        'spatial': '14×14'},
            'conv2': {'type': 'Conv', 'depth': 2, 'channels': 64,        'spatial': '7×7'},
            'fc1':   {'type': 'FC',   'depth': 3, 'channels': 128,       'spatial': '—'},
            'fc2':   {'type': 'FC',   'depth': 4, 'channels': n_classes, 'spatial': '—'},
        }

        self._register_hooks(self.LAYER_REGISTRY)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.activation_fn(self.conv1(x)))  # conv1 → relu → pool
        x = self.pool(self.activation_fn(self.conv2(x)))  # conv2 → relu → pool
        x = torch.flatten(x, start_dim=1)
        x = self.activation_fn(self.fc1(x))               # fc1 hook fires here
        return self.fc2(x)

    @classmethod
    def for_dataset(cls, dataset_info: DatasetInfo) -> 'PaperCNN':
        """
        Preferred constructor — reads n_channels and n_classes from
        a DatasetInfo object so nothing is hardcoded in the pipeline.

        Usage:
            dataset_info = load_dataset('MNIST')
            model        = PaperCNN.for_dataset(dataset_info)
        """
        return cls(
            n_channels = dataset_info.n_channels,
            n_classes  = dataset_info.n_classes,
        )