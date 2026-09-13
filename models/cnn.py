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
# Differentiable max-pool substitute (opt-in — see PaperCNN's
# differentiable_pool flag; NOT used by default anywhere in this codebase)
# ---------------------------------------------------------------------------

class SoftMaxPool2d(nn.Module):
    """
    A smooth stand-in for nn.MaxPool2d, via LogSumExp over each pooling
    window: pool(x) = temperature * log(sum(exp(x / temperature))).

    Why this exists: nn.MaxPool2d's backward is not supported for a second
    backward pass (create_graph=True) on every device — notably, PyTorch's
    MPS backend raises "max_pool2d ... is not infinitely differentiable"
    for it, which breaks gradient-inversion attacks that differentiate
    through the gradient itself (data/reconstruction.py's reconstruct()).
    This op is differentiable to second order everywhere, so it unblocks
    running that attack on MPS.

    This is NOT numerically identical to true max pooling — see the
    approximation-error note below — so treat any model built with
    differentiable_pool=True as a distinct architecture from the default
    PaperCNN, not a drop-in stand-in with guaranteed-identical outputs.

    Approximation error: for a pooling window of k elements, this
    satisfies max(window) <= pool(window) <= max(window) + temperature *
    log(k). For kernel_size=2 (k=4 per 2D window) and temperature=0.01
    (the default here), the worst-case error is bounded by
    ~0.01 * log(4) ≈ 0.014 in normalised activation units — small, but
    not zero, and it also changes how gradients are distributed across
    the pooling window (softmax-weighted across all elements, rather than
    routed entirely to the single argmax as true max pooling does).
    Lowering temperature tightens this bound but pushes exp() towards
    numerical overflow/underflow, so there is a real floor on how close
    this can get to true max pooling in practice, not just in theory.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2, temperature: float = 0.01):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        patches = patches.contiguous().view(b, c, patches.size(2), patches.size(3), -1)
        return self.temperature * torch.logsumexp(patches / self.temperature, dim=-1)


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

    differentiable_pool: if True, uses SoftMaxPool2d instead of real
        nn.MaxPool2d — needed to run gradient-inversion reconstruction on
        MPS (see SoftMaxPool2d's docstring for why, and for the caveat
        that it is an approximation, not an identical operation). Default
        False everywhere in this codebase's production pipeline
        (data/builder.py never sets it), so existing results are
        unaffected; it exists for diagnostic scripts like
        reconstruct_cifar10.py.
    """

    def __init__(
        self,
        n_channels:          int  = 1,
        n_classes:           int  = 10,
        activation:          str  = 'relu',
        differentiable_pool: bool = False,
    ):
        super().__init__(activation)

        self.pool  = SoftMaxPool2d(kernel_size=2, stride=2) if differentiable_pool \
                     else nn.MaxPool2d(kernel_size=2, stride=2)
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
    def for_dataset(cls, dataset_info: DatasetInfo, differentiable_pool: bool = False) -> 'PaperCNN':
        """
        Preferred constructor — reads n_channels and n_classes from
        a DatasetInfo object so nothing is hardcoded in the pipeline.

        Usage:
            dataset_info = load_dataset('MNIST')
            model        = PaperCNN.for_dataset(dataset_info)
        """
        return cls(
            n_channels          = dataset_info.n_channels,
            n_classes           = dataset_info.n_classes,
            differentiable_pool = differentiable_pool,
        )