"""
data/reconstruction.py — Gradient inversion via Geiping et al. (2020).

Responsibilities:
  - Intercept gradients from a model for a given image
  - Reconstruct the image by minimising cosine gradient distance
    with total variation regularisation (Geiping et al., 2020)
  - Optionally corrupt intercepted gradients with Gaussian noise
    (used in ablation experiments on reconstruction quality)

This module exposes two public functions:

  intercept_gradients(model, img, label)
      → Computes and returns the gradients a model produces for one image.
        This simulates the gradient-leakage threat model.

  reconstruct(model, target_gradients, img_shape, cfg)
      → Reconstructs an image from intercepted gradients using the
        Geiping cosine similarity method.

Reference:
  Geiping et al. (2020), "Inverting Gradients — How easy is it to break
  privacy in federated learning?" NeurIPS 2020.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Reconstruction configuration
# ---------------------------------------------------------------------------

@dataclass
class ReconConfig:
    """
    All hyperparameters for one gradient-inversion reconstruction run.

    Attributes:
        iterations:      number of Adam optimisation steps
        lr:              Adam learning rate
        tv_weight:       total variation regularisation weight.
                         Higher → smoother image, lower → more detail.
                         1e-4 works well for MNIST; try 1e-6 for CIFAR.
        noise_std:       std of Gaussian noise added to intercepted gradients.
                         0.0 = perfect interception (no noise).
                         Increase to study degraded reconstruction quality.
        clamp_range:     (min, max) normalised pixel range.
                         Pass DatasetInfo.clamp_range here.
        use_soft_label:  jointly optimise a soft label vector alongside the
                         image. Recommended — improves convergence.
        verbose:         print loss every 10% of iterations
    """
    iterations:     int   = 300
    lr:             float = 0.1
    tv_weight:      float = 1e-4
    noise_std:      float = 0.0
    clamp_range:    Tuple[float, float] = (-0.4242, 2.8215)
    use_soft_label: bool  = True
    verbose:        bool  = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_device(model: torch.nn.Module) -> torch.device:
    """Infer the device of a model from its first parameter."""
    for p in model.parameters():
        return p.device
    return torch.device('cpu')


def _add_noise(
    grads: List[torch.Tensor],
    std:   float,
    dev:   torch.device,
) -> List[torch.Tensor]:
    """
    Return a new list of gradient tensors with Gaussian noise added.
    If std <= 0, returns clean detached clones (no noise).
    """
    if std <= 0.0:
        return [g.detach().clone() for g in grads]
    return [
        (g.detach().to(dev) + torch.randn_like(g, device=dev) * std).clone()
        for g in grads
    ]


def _cosine_loss(
    dummy_grads:  Tuple[torch.Tensor, ...],
    target_grads: List[torch.Tensor],
) -> torch.Tensor:
    """
    Magnitude-oblivious cosine similarity loss (Geiping et al., 2020 Eq. 4).

    Computes 1 - cos(dummy_grads, target_grads) treating all parameter
    gradients as one concatenated vector. Returns 0 when perfectly aligned.

    Unlike the DLG L2 loss, this ignores gradient magnitude so it works
    on both untrained and trained models.
    """
    dot    = sum(
        (dg * tg.to(dg.device)).sum()
        for dg, tg in zip(dummy_grads, target_grads)
    )
    norm_d = torch.sqrt(sum((dg ** 2).sum() for dg in dummy_grads) + 1e-8)
    norm_t = torch.sqrt(sum((tg ** 2).sum() for tg in target_grads) + 1e-8)
    return 1.0 - dot / (norm_d * norm_t)


def _dlg_loss(
    dummy_grads:  Tuple[torch.Tensor, ...],
    target_grads: List[torch.Tensor],
) -> torch.Tensor:
    """
    L2 gradient distance loss (Zhu et al., 2019, DLG Eq. 1).

    Sum of squared differences between dummy and target gradients.
    Sensitive to gradient magnitude — works best on untrained models.
    """
    return sum(
        ((dg - tg.to(dg.device)) ** 2).sum()
        for dg, tg in zip(dummy_grads, target_grads)
    )


def _tv_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Total variation regularisation (Rudin et al., 1992).

    Penalises high-frequency pixel differences to encourage the
    reconstructed image to look like a natural image rather than noise.
    Applied to a (1, C, H, W) batched image tensor.
    """
    return (
        ((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2).sum() +
        ((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2).sum()
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_psnr(
    reconstructed: torch.Tensor,
    original:      torch.Tensor,
    data_range:    float,
) -> float:
    """
    Peak Signal-to-Noise Ratio between a reconstructed and original image.

    PSNR = 10 * log10(data_range^2 / MSE)

    Args:
        reconstructed: reconstructed image tensor (any shape matching original)
        original:      ground-truth image tensor
        data_range:    maximum possible pixel value range, i.e.
                       clamp_range[1] - clamp_range[0]

    Returns:
        PSNR in dB. Returns inf if MSE is zero (perfect reconstruction).
    """
    mse = ((reconstructed - original.to(reconstructed.device)) ** 2).mean().item()
    if mse == 0.0:
        return float('inf')
    return 10.0 * math.log10(data_range ** 2 / mse)


def intercept_gradients(
    model: torch.nn.Module,
    img:   torch.Tensor,
    label: int,
    dev:   Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Compute the gradients a model produces for a single labelled image.

    This simulates the attacker intercepting one gradient update in a
    federated learning round. The model weights are not changed.

    Args:
        model: the reconstruction model (PaperCNN, untrained or pretrained)
        img:   image tensor of shape (C, H, W) in normalised space
        label: true class label of the image
        dev:   device to run on (inferred from model if None)

    Returns:
        List of gradient tensors, one per model parameter, detached.
    """
    if dev is None:
        dev = _get_device(model)

    model.eval()
    img_batch  = img.unsqueeze(0).to(dev)
    lbl_tensor = torch.tensor([label], device=dev)

    # Zero any stale gradients before computing fresh ones
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss = F.cross_entropy(model(img_batch), lbl_tensor)
    loss.backward()

    return [p.grad.detach().clone() for p in model.parameters()]


def reconstruct(
    model:            torch.nn.Module,
    target_gradients: List[torch.Tensor],
    img_shape:        torch.Size,
    cfg:              ReconConfig,
    dev:              Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Reconstruct an image from intercepted gradients using Geiping (2020).

    Optimises a dummy image by minimising:
        loss = cosine_distance(grad(dummy), target_grad)
               + tv_weight * TV(dummy)

    Uses signed Adam (Geiping Sec. 4) for stable optimisation with ReLU
    networks. Tracks the best image seen across all iterations.

    Args:
        model:            reconstruction model (same architecture as
                          backdoor model, untrained or pretrained)
        target_gradients: intercepted gradients from intercept_gradients()
        img_shape:        shape of the image to reconstruct, WITH batch dim
                          e.g. torch.Size([1, 1, 28, 28]) for one MNIST image
        cfg:              ReconConfig with all reconstruction hyperparameters
        dev:              device (inferred from model if None)

    Returns:
        (reconstructed_image, final_loss)
        reconstructed_image: Tensor of shape img_shape, clamped to cfg.clamp_range
        final_loss:          best cosine loss achieved (float, lower = better)
    """
    if dev is None:
        dev = _get_device(model)

    model.to(dev)
    model.eval()

    # Optionally corrupt the intercepted gradients (ablation experiment)
    tgt_grads = _add_noise(target_gradients, cfg.noise_std, dev)

    # Initialise dummy image from random noise
    dummy_img = torch.randn(img_shape, device=dev, requires_grad=True)

    # Infer number of output classes for soft label initialisation
    with torch.no_grad():
        n_classes = model(dummy_img.detach()).shape[-1]

    batch_size  = img_shape[0]
    dummy_label = torch.randn(
        (batch_size, n_classes), device=dev, requires_grad=True
    )

    # Adam with both dummy image and soft label as parameters
    optimizer = torch.optim.Adam([dummy_img, dummy_label], lr=cfg.lr)

    best_loss = float('inf')
    best_img  = dummy_img.detach().clone()

    if cfg.verbose:
        print(
            f"Reconstruction: iterations={cfg.iterations}  "
            f"lr={cfg.lr}  tv_weight={cfg.tv_weight}  "
            f"noise_std={cfg.noise_std}"
        )

    for it in range(cfg.iterations):
        optimizer.zero_grad()

        # Forward pass with soft cross-entropy loss using optimised label
        out        = model(dummy_img)
        probs      = F.softmax(dummy_label, dim=-1)
        dummy_loss = -(probs * F.log_softmax(out, dim=-1)).sum() / batch_size

        # Compute dummy gradients w.r.t. model parameters
        dummy_grads = torch.autograd.grad(
            dummy_loss, model.parameters(), create_graph=True
        )

        # Geiping loss: cosine distance + TV regularisation
        loss = _cosine_loss(dummy_grads, tgt_grads)
        loss = loss + cfg.tv_weight * _tv_loss(dummy_img)

        loss.backward()

        # Signed gradient update for the image (Geiping Sec. 4)
        # Replaces raw gradients with their sign — more stable for ReLU nets
        with torch.no_grad():
            if dummy_img.grad is not None:
                dummy_img.grad.data = dummy_img.grad.data.sign()

        optimizer.step()

        # Clamp image to valid normalised pixel range after every step
        with torch.no_grad():
            dummy_img.data.clamp_(cfg.clamp_range[0], cfg.clamp_range[1])

        loss_val = float(loss.detach().cpu())
        if loss_val < best_loss:
            best_loss = loss_val
            best_img  = dummy_img.detach().clone()

        if cfg.verbose and (
            it % max(1, cfg.iterations // 10) == 0
            or it == cfg.iterations - 1
        ):
            print(f"  iter {it+1:4d}/{cfg.iterations}  "
                  f"loss={loss_val:.6f}  best={best_loss:.6f}")

    recon = best_img.clamp(cfg.clamp_range[0], cfg.clamp_range[1])
    return recon, best_loss


def reconstruct_dlg(
    model:            torch.nn.Module,
    target_gradients: List[torch.Tensor],
    img_shape:        torch.Size,
    cfg:              ReconConfig,
    dev:              Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Reconstruct an image from intercepted gradients using DLG (Zhu et al., 2019).

    Optimises a dummy image by minimising:
        loss = sum_l || grad(dummy)_l - target_grad_l ||^2
               + tv_weight * TV(dummy)

    Uses L-BFGS (as in the original DLG paper) rather than signed Adam.

    Args:
        model:            reconstruction model (same architecture as
                          backdoor model, untrained or pretrained)
        target_gradients: intercepted gradients from intercept_gradients()
        img_shape:        shape of the image to reconstruct, WITH batch dim
                          e.g. torch.Size([1, 1, 28, 28]) for one MNIST image
        cfg:              ReconConfig with all reconstruction hyperparameters
        dev:              device (inferred from model if None)

    Returns:
        (reconstructed_image, final_loss)
        reconstructed_image: Tensor of shape img_shape, clamped to cfg.clamp_range
        final_loss:          best L2 gradient loss achieved (float, lower = better)
    """
    if dev is None:
        dev = _get_device(model)

    model.to(dev)
    model.eval()

    tgt_grads = _add_noise(target_gradients, cfg.noise_std, dev)

    dummy_img = torch.randn(img_shape, device=dev, requires_grad=True)

    with torch.no_grad():
        n_classes = model(dummy_img.detach()).shape[-1]

    batch_size  = img_shape[0]
    dummy_label = torch.randn(
        (batch_size, n_classes), device=dev, requires_grad=True
    )

    optimizer = torch.optim.LBFGS([dummy_img, dummy_label], lr=cfg.lr)

    best_loss = float('inf')
    best_img  = dummy_img.detach().clone()

    if cfg.verbose:
        print(
            f"DLG Reconstruction: iterations={cfg.iterations}  "
            f"lr={cfg.lr}  tv_weight={cfg.tv_weight}  "
            f"noise_std={cfg.noise_std}"
        )

    for it in range(cfg.iterations):
        def closure():
            optimizer.zero_grad()
            out        = model(dummy_img)
            probs      = F.softmax(dummy_label, dim=-1)
            dummy_loss = -(probs * F.log_softmax(out, dim=-1)).sum() / batch_size

            dummy_grads = torch.autograd.grad(
                dummy_loss, model.parameters(), create_graph=True
            )

            loss = _dlg_loss(dummy_grads, tgt_grads)
            loss = loss + cfg.tv_weight * _tv_loss(dummy_img)
            loss.backward()
            return loss

        loss_tensor = optimizer.step(closure)
        loss_val    = float(loss_tensor.detach().cpu())

        with torch.no_grad():
            dummy_img.data.clamp_(cfg.clamp_range[0], cfg.clamp_range[1])

        if loss_val < best_loss:
            best_loss = loss_val
            best_img  = dummy_img.detach().clone()

        if cfg.verbose and (
            it % max(1, cfg.iterations // 10) == 0
            or it == cfg.iterations - 1
        ):
            print(f"  iter {it+1:4d}/{cfg.iterations}  "
                  f"loss={loss_val:.6f}  best={best_loss:.6f}")

    recon = best_img.clamp(cfg.clamp_range[0], cfg.clamp_range[1])
    return recon, best_loss