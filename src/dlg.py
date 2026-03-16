import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def _make_device(model, device: Optional[torch.device]):
    if device is not None:
        return device
    for p in model.parameters():
        return p.device
    return torch.device('cpu')

def _add_noise_to_gradients(grads: List[torch.Tensor], std: float, device: Optional[torch.device] = None) -> List[torch.Tensor]:
    """Return a new list of gradients with Gaussian noise added (in-place safe).

    Args:
        grads: list of torch.Tensor gradients (the intercepted gradients).
        std: standard deviation of Gaussian noise (same units as gradients).
        device: device for generated noise. If None, uses grads' device.
    """
    if std <= 0:
        return [g.detach().clone() for g in grads]
    noisy = []
    for g in grads:
        dev = device if device is not None else g.device
        noise = torch.randn_like(g, device=dev) * std
        noisy.append((g.detach().to(dev) + noise).clone())
    return noisy


def dlg_reconstruct(
    model: torch.nn.Module,
    target_gradients: List[torch.Tensor],
    gt_shape: torch.Size,
    iterations: int = 300,
    lr: float = 0.1,
    noise_std: float = 0.0,
    clamp: Tuple[float, float] = (0.0, 1.0),
    init: str = 'random',
    use_soft_label: bool = True,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Reconstruct input and label from intercepted gradients using DLG.

    Args:
        model: the PyTorch model (in eval or train mode — evaluation recommended).
        target_gradients: list of gradient tensors from the honest client (same order as model.parameters()).
        gt_shape: shape of the original input tensor to reconstruct (including batch dim).
        iterations: number of optimization steps for L-BFGS.
        lr: L-BFGS learning rate.
        noise_std: optional gaussian noise std to add to target gradients (simulates privacy/noise).
        clamp: min/max to clamp reconstructed tensors to for visualization (e.g., 0-1).
        init: 'random' or 'zeros' for initializing dummy data.
        use_soft_label: whether to optimize a soft label vector (recommended for single-sample).
        verbose: print progress.
        device: device to run on. If None, inferred from model parameters.

    Returns:
        reconstructed_input: Tensor with shape `gt_shape` (clamped to `clamp`).
        reconstructed_label_logits: Tensor of raw logits for the learned soft label.
        final_loss: final gradient-distance loss value (float).
    """
    dev = _make_device(model, device)
    model.to(dev)
    model.eval()

    # Prepare noisy target gradients (detached copies)
    tgt_grads = _add_noise_to_gradients(target_gradients, noise_std, device=dev)

    # Initialize dummy input
    if init == 'zeros':
        dummy_data = torch.zeros(gt_shape, device=dev).requires_grad_(True)
    else:
        dummy_data = torch.randn(gt_shape, device=dev).requires_grad_(True)

    # Initialize soft label (logit vector) when requested
    batch_size = gt_shape[0]
    num_classes = None
    # try to infer output dim by running a forward with a dummy input (no grad)
    with torch.no_grad():
        try:
            sample_out = model(dummy_data.detach())
            num_classes = sample_out.shape[-1]
        except Exception:
            raise RuntimeError('Unable to infer model output dimension from model(dummy_input). Provide a model that accepts the input shape.')

    if use_soft_label:
        dummy_label = torch.randn((batch_size, num_classes), device=dev).requires_grad_(True)
        opt_params = [dummy_data, dummy_label]
    else:
        # If not using soft labels, random integer labels could be used, but optimization over labels is recommended
        dummy_label = None
        opt_params = [dummy_data]

    optimizer = torch.optim.LBFGS(opt_params, lr=lr, max_iter=20, history_size=10)

    if verbose:
        print('Starting DLG reconstruction (iterations=%d, noise_std=%g)' % (iterations, noise_std))

    final_loss = None

    def closure():
        optimizer.zero_grad()

        out = model(dummy_data)

        # If using soft labels, compute cross-entropy against soft distribution via negative loglikelihood
        if use_soft_label:
            probs = F.softmax(dummy_label, dim=-1)
            logp = F.log_softmax(out, dim=-1)
            # average over batch
            dummy_loss = - (probs * logp).sum() / batch_size
        else:
            raise RuntimeError('Non-soft-label optimization not implemented in this helper. Set use_soft_label=True')

        # Compute gradients of model params w.r.t dummy loss
        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        # Compute squared L2 distance between dummy_grads and target grads
        grad_diff = 0.0
        for dg, tg in zip(dummy_grads, tgt_grads):
            # ensure target gradient is on same device
            tg = tg.to(dg.device)
            grad_diff = grad_diff + ((dg - tg) ** 2).sum()

        grad_diff.backward()
        return grad_diff

    for it in range(iterations):
        loss = optimizer.step(closure)
        final_loss = float(loss.detach().cpu().item())
        if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
            print(f'Iter {it+1}/{iterations}  grad-dist: {final_loss:.6f}')

    # Detach results and clamp for visualization
    recon_input = dummy_data.detach() #.clamp(min=clamp[0], max=clamp[1])
    recon_label_logits = dummy_label.detach() if dummy_label is not None else None

    return recon_input, recon_label_logits, final_loss
