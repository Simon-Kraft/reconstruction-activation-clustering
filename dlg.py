import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_device(model, device: Optional[torch.device]):
    if device is not None:
        return device
    for p in model.parameters():
        return p.device
    return torch.device('cpu')


def _add_noise_to_gradients(
    grads: List[torch.Tensor],
    std: float,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Return a new list of gradients with Gaussian noise added (in-place safe)."""
    if std <= 0:
        return [g.detach().clone() for g in grads]
    noisy = []
    for g in grads:
        dev = device if device is not None else g.device
        noise = torch.randn_like(g, device=dev) * std
        noisy.append((g.detach().to(dev) + noise).clone())
    return noisy


def _cosine_loss(
    dummy_grads: Tuple[torch.Tensor, ...],
    target_grads: List[torch.Tensor],
) -> torch.Tensor:
    """
    Magnitude-oblivious cosine similarity loss (Geiping et al., 2020, Eq. 4).

    Measures the angle between the dummy and target gradient vectors,
    ignoring their magnitudes. This is the key improvement over Euclidean
    loss, which fails on trained models because their gradients are tiny.

    Returns a scalar in [-1, 1] where 0 = perfectly aligned (perfect reconstruction).
    """
    dot    = sum((dg * tg.to(dg.device)).sum() for dg, tg in zip(dummy_grads, target_grads))
    norm_d = torch.sqrt(sum((dg ** 2).sum() for dg in dummy_grads) + 1e-8)
    norm_t = torch.sqrt(sum((tg ** 2).sum() for tg in target_grads) + 1e-8)
    return 1 - dot / (norm_d * norm_t)


def _tv_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Total variation regularisation (Rudin et al., 1992).

    Encourages spatial smoothness in the reconstructed image, acting as a
    natural image prior that prevents noisy / high-frequency artefacts.
    """
    return (
        ((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2).sum() +
        ((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2).sum()
    )


def _euclidean_loss(
    dummy_grads: Tuple[torch.Tensor, ...],
    target_grads: List[torch.Tensor],
) -> torch.Tensor:
    """Squared L2 distance between dummy and target gradients (Zhu et al., 2019)."""
    return sum(
        ((dg - tg.to(dg.device)) ** 2).sum()
        for dg, tg in zip(dummy_grads, target_grads)
    )


# ---------------------------------------------------------------------------
# Main reconstruction function
# ---------------------------------------------------------------------------

def dlg_reconstruct(
    model:            torch.nn.Module,
    target_gradients: List[torch.Tensor],
    gt_shape:         torch.Size,
    iterations:       int   = 300,
    lr:               float = 0.1,
    noise_std:        float = 0.0,
    clamp:            Tuple[float, float] = (-0.4242, 2.8215),
    init:             str   = 'random',
    use_soft_label:   bool  = True,
    method:           str   = 'cosine',
    tv_weight:        float = 1e-4,
    verbose:          bool  = True,
    device:           Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Reconstruct an input image from intercepted gradients.

    Supports two methods:

    'euclidean' — Zhu et al. (2019), Deep Leakage from Gradients.
        Minimises the squared L2 distance between dummy and target gradients
        using L-BFGS. Works well on untrained / shallow models but fails
        completely on trained models because their gradient magnitudes are
        near zero, leaving no signal for the optimiser.

    'cosine'    — Geiping et al. (2020), Inverting Gradients.
        Minimises 1 - cosine_similarity(dummy_grad, target_grad) using
        signed Adam with total variation regularisation. Magnitude-oblivious:
        only the *direction* of the gradient matters, so it works on both
        trained and untrained models. This is the recommended method when
        RECON_PRETRAIN_EPOCHS > 0.

    Args:
        model:            PyTorch model (eval mode recommended).
        target_gradients: intercepted gradients (same order as model.parameters()).
        gt_shape:         shape of the input tensor to reconstruct (with batch dim).
        iterations:       number of optimisation steps.
        lr:               learning rate.
        noise_std:        Gaussian noise std added to target gradients (0 = none).
        clamp:            (min, max) pixel range for the reconstruction.
                          Must match the normalisation of your training data.
                          Default matches MNIST: ((0-0.1307)/0.3081, (1-0.1307)/0.3081).
        init:             'random' or 'zeros' initialisation for dummy data.
        use_soft_label:   jointly optimise a soft label vector (recommended).
        method:           'cosine' (Geiping) or 'euclidean' (original DLG).
        tv_weight:        weight of total variation regularisation (cosine only).
        verbose:          print progress every 10% of iterations.
        device:           device to run on (inferred from model if None).

    Returns:
        reconstructed_input:        Tensor of shape gt_shape, clamped to `clamp`.
        reconstructed_label_logits: raw label logits Tensor.
        final_loss:                 best loss value achieved (float).
    """
    if method not in ('cosine', 'euclidean'):
        raise ValueError(f"method must be 'cosine' or 'euclidean', got '{method}'")

    dev = _make_device(model, device)
    model.to(dev)
    model.eval()

    # Optionally corrupt the intercepted gradients
    tgt_grads = _add_noise_to_gradients(target_gradients, noise_std, device=dev)

    # Initialise dummy input
    if init == 'zeros':
        dummy_data = torch.zeros(gt_shape, device=dev).requires_grad_(True)
    else:
        dummy_data = torch.randn(gt_shape, device=dev).requires_grad_(True)

    # Infer number of output classes from a single forward pass
    with torch.no_grad():
        try:
            num_classes = model(dummy_data.detach()).shape[-1]
        except Exception:
            raise RuntimeError(
                'Cannot infer model output dimension. '
                'Make sure the model accepts inputs of shape gt_shape.'
            )

    batch_size = gt_shape[0]

    # Initialise soft label
    if use_soft_label:
        dummy_label = torch.randn((batch_size, num_classes), device=dev).requires_grad_(True)
    else:
        dummy_label = None

    if verbose:
        print(f'Starting reconstruction  method={method}  '
              f'iterations={iterations}  noise_std={noise_std}')

    best_loss  = float('inf')
    best_data  = dummy_data.detach().clone()
    best_label = dummy_label.detach().clone() if dummy_label is not None else None

    # -------------------------------------------------------------------
    # Method A — Cosine similarity + signed Adam (Geiping et al., 2020)
    # -------------------------------------------------------------------
    if method == 'cosine':
        opt_params = [dummy_data] + ([dummy_label] if dummy_label is not None else [])
        optimizer  = torch.optim.Adam(opt_params, lr=lr)

        for it in range(iterations):
            optimizer.zero_grad()

            out = model(dummy_data)

            if use_soft_label:
                probs      = F.softmax(dummy_label, dim=-1)
                dummy_loss = -(probs * F.log_softmax(out, dim=-1)).sum() / batch_size
            else:
                raise RuntimeError('Set use_soft_label=True.')

            dummy_grads = torch.autograd.grad(
                dummy_loss, model.parameters(), create_graph=True
            )

            # Core Geiping loss: cosine similarity + TV regularisation
            loss = _cosine_loss(dummy_grads, tgt_grads)
            loss = loss + tv_weight * _tv_loss(dummy_data)

            loss.backward()

            # Signed gradient update (Geiping Sec. 4) — more stable than raw gradients
            # for non-smooth architectures with ReLU activations
            with torch.no_grad():
                if dummy_data.grad is not None:
                    dummy_data.grad.data = dummy_data.grad.data.sign()

            optimizer.step()

            # Keep pixels in valid normalised range after every step
            with torch.no_grad():
                dummy_data.data.clamp_(clamp[0], clamp[1])

            loss_val = float(loss.detach().cpu().item())
            if loss_val < best_loss:
                best_loss  = loss_val
                best_data  = dummy_data.detach().clone()
                if dummy_label is not None:
                    best_label = dummy_label.detach().clone()

            if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
                print(f'  Iter {it+1}/{iterations}  loss={loss_val:.6f}  best={best_loss:.6f}')

    # -------------------------------------------------------------------
    # Method B — Euclidean loss + L-BFGS (Zhu et al., 2019 / original DLG)
    # -------------------------------------------------------------------
    else:
        opt_params = [dummy_data] + ([dummy_label] if dummy_label is not None else [])
        optimizer  = torch.optim.LBFGS(opt_params, lr=lr, max_iter=20, history_size=10)

        def closure():
            optimizer.zero_grad()
            out = model(dummy_data)

            if use_soft_label:
                probs      = F.softmax(dummy_label, dim=-1)
                dummy_loss = -(probs * F.log_softmax(out, dim=-1)).sum() / batch_size
            else:
                raise RuntimeError('Set use_soft_label=True.')

            dummy_grads = torch.autograd.grad(
                dummy_loss, model.parameters(), create_graph=True
            )
            grad_diff = _euclidean_loss(dummy_grads, tgt_grads)
            grad_diff.backward()
            return grad_diff

        for it in range(iterations):
            loss     = optimizer.step(closure)
            loss_val = float(loss.detach().cpu().item())

            if loss_val < best_loss:
                best_loss  = loss_val
                best_data  = dummy_data.detach().clone()
                if dummy_label is not None:
                    best_label = dummy_label.detach().clone()

            if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
                print(f'  Iter {it+1}/{iterations}  loss={loss_val:.6f}  best={best_loss:.6f}')

    recon_input = best_data.clamp(clamp[0], clamp[1])
    return recon_input, best_label, best_loss