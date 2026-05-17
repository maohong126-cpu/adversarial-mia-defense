"""Single-step gradient attacks for PyTorch classifiers."""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


GradientAttackMethod = Literal["sign", "normalized", "adaptive"]


def _as_device_tensor(value: Tensor | float, reference: Tensor) -> Tensor:
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(value, device=reference.device, dtype=reference.dtype)


def fgsm_attack(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    epsilon: float,
    *,
    targeted: bool = False,
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = 1.0,
    loss_fn: Optional[nn.Module] = None,
    method: GradientAttackMethod = "sign",
    set_eval: bool = True,
    # --- Adaptive method parameters ---
    gradient_alpha: float = 0.5,
    perceptual_beta: float = 0.5,
    kernel_size: int = 5,
    # --- MIA-steered loss parameter ---
    mia_conf_lambda: float = 0.0,
) -> Tensor:
    """Generate adversarial examples with a single gradient step.

    Three perturbation modes (``method``):

    * ``"sign"`` – standard FGSM: ``x_adv = x + ε * sign(∇L)``.
      Uniform L∞ budget across all dimensions.

    * ``"normalized"`` – raw gradient direction, L2-normalized per sample.
      Preserves relative gradient magnitudes but operates under L2 budget.

    * ``"adaptive"`` – **perceptual-adaptive gradient weighting** (new).
      Concentrates the ε budget on dimensions where (a) the gradient is
      large *and* (b) local image texture is rich (perceptually insensitive
      to noise).  Formula::

          W_eff  = (|∇L| / max|∇L|)^α          # gradient concentration
          W_perc = (local_var(x) / max_var)^β   # perceptual mask (4-D only)
          W      = W_eff * W_perc / max(W_eff * W_perc)   # ∈ [0,1], max=1
          δ      = ε * sign(∇L) * W             # L∞ ≤ ε maintained

      This way the same ε budget is spent on the most model-sensitive and
      least visually-salient pixels, making the perturbation harder to spot
      while remaining maximally effective.

    MIA-steered loss (``mia_conf_lambda > 0``):
        When generating the adversarial example the loss becomes::

            L = CE(f(x), y) - λ * mean_max_confidence(f(x))

        The extra term pushes the gradient toward *reducing* model confidence,
        directly targeting the overconfidence gap that membership inference
        attacks exploit.  Training on these examples forces the model to
        maintain accuracy even when its confidence is actively suppressed,
        causing the confidence distributions of member and non-member data
        to converge and lowering MIA attack success rates.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if inputs.size(0) != labels.size(0):
        raise ValueError("inputs and labels must have the same batch size")
    if method not in {"sign", "normalized", "adaptive"}:
        raise ValueError("method must be 'sign', 'normalized', or 'adaptive'")

    was_training = model.training
    if set_eval:
        model.eval()

    x = inputs.detach().clone().requires_grad_(True)
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    logits = model(x)
    loss = criterion(logits, labels)
    if targeted:
        loss = -loss

    # MIA-steered: penalise overconfidence in the loss used to generate δ.
    # Moving in ∇_x(CE - λ·max_conf) perturbs x toward both higher CE loss
    # AND lower model confidence, closing the gap MIA exploits.
    if mia_conf_lambda > 0.0:
        probs = F.softmax(logits, dim=1)
        mean_max_conf = probs.max(dim=1).values.mean()
        loss = loss - mia_conf_lambda * mean_max_conf

    grad = torch.autograd.grad(loss, x)[0]
    direction = _gradient_direction(
        grad,
        method,
        inputs=inputs.detach(),
        gradient_alpha=gradient_alpha,
        perceptual_beta=perceptual_beta,
        kernel_size=kernel_size,
    )
    adv = x + epsilon * direction

    if clip_min is not None or clip_max is not None:
        min_value = _as_device_tensor(float("-inf") if clip_min is None else clip_min, adv)
        max_value = _as_device_tensor(float("inf") if clip_max is None else clip_max, adv)
        adv = torch.clamp(adv, min=min_value, max=max_value)

    if set_eval and was_training:
        model.train()

    return adv.detach()


def normalized_gradient_attack(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    epsilon: float,
    *,
    targeted: bool = False,
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = 1.0,
    loss_fn: Optional[nn.Module] = None,
    set_eval: bool = True,
) -> Tensor:
    """Generate magnitude-aware adversarial examples with normalized gradients."""
    return fgsm_attack(
        model,
        inputs,
        labels,
        epsilon,
        targeted=targeted,
        clip_min=clip_min,
        clip_max=clip_max,
        loss_fn=loss_fn,
        method="normalized",
        set_eval=set_eval,
    )


def batch_fgsm_attack(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    epsilon: float,
    *,
    targeted: bool = False,
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = 1.0,
    device: Optional[torch.device | str] = None,
    method: GradientAttackMethod = "sign",
    gradient_alpha: float = 0.5,
    perceptual_beta: float = 0.5,
    kernel_size: int = 5,
    mia_conf_lambda: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Run a single-step gradient attack over a loader."""
    model_device = next(model.parameters()).device
    target_device = torch.device(device) if device is not None else model_device

    adv_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    for inputs, labels in data_loader:
        inputs = inputs.to(target_device)
        labels = labels.to(target_device)
        adv = fgsm_attack(
            model,
            inputs,
            labels,
            epsilon,
            targeted=targeted,
            clip_min=clip_min,
            clip_max=clip_max,
            method=method,
            gradient_alpha=gradient_alpha,
            perceptual_beta=perceptual_beta,
            kernel_size=kernel_size,
            mia_conf_lambda=mia_conf_lambda,
        )
        adv_batches.append(adv.cpu())
        label_batches.append(labels.detach().cpu())

    if not adv_batches:
        raise ValueError("data_loader produced no batches")

    return torch.cat(adv_batches), torch.cat(label_batches)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gradient_direction(
    grad: Tensor,
    method: GradientAttackMethod,
    inputs: Optional[Tensor] = None,
    gradient_alpha: float = 0.5,
    perceptual_beta: float = 0.5,
    kernel_size: int = 5,
) -> Tensor:
    if method == "sign":
        return grad.sign()

    if method == "normalized":
        flat = grad.flatten(start_dim=1)
        norms = flat.norm(p=2, dim=1).clamp_min(torch.finfo(grad.dtype).eps)
        shape = (grad.size(0),) + (1,) * (grad.dim() - 1)
        return grad / norms.view(shape)

    # method == "adaptive"
    if inputs is None:
        raise ValueError("inputs must be provided when method='adaptive'")
    return _adaptive_direction(grad, inputs, gradient_alpha, perceptual_beta, kernel_size)


def _adaptive_direction(
    grad: Tensor,
    inputs: Tensor,
    gradient_alpha: float,
    perceptual_beta: float,
    kernel_size: int,
) -> Tensor:
    """Perceptual-adaptive gradient direction.

    Concentrates the L∞ budget on high-gradient, visually-insensitive pixels.
    The returned tensor has values in [-1, +1] with maximum absolute value = 1,
    so multiplying by ε preserves the L∞ ≤ ε constraint exactly.
    """
    eps_val = torch.finfo(grad.dtype).eps
    shape = (grad.size(0),) + (1,) * (grad.dim() - 1)

    # --- Gradient concentration weight W_eff ∈ [0, 1] ---
    mag = grad.abs()
    flat_mag = mag.flatten(start_dim=1)
    max_mag = flat_mag.max(dim=1).values.clamp_min(eps_val).view(shape)
    w_eff = (mag / max_mag).clamp(0.0, 1.0).pow(gradient_alpha)

    # --- Perceptual weight W_perc ∈ [0, 1]: only for 4-D image tensors ---
    if perceptual_beta > 0.0 and inputs.dim() == 4:
        with torch.no_grad():
            local_var = _local_variance(inputs, kernel_size)
        flat_var = local_var.flatten(start_dim=1)
        max_var = flat_var.max(dim=1).values.view(shape)
        # Normalise; add a small floor so zero-variance pixels still get some
        # weight (they may still carry gradient information).
        norm_var = (local_var / max_var.clamp_min(eps_val)).clamp(0.0, 1.0)
        w_perc = (norm_var + 0.1).clamp_max(1.0).pow(perceptual_beta)
    else:
        w_perc = torch.ones_like(w_eff)

    # --- Combine and re-normalise so max = 1 (L∞ = ε preserved) ---
    W = w_eff * w_perc
    flat_W = W.flatten(start_dim=1)
    max_W = flat_W.max(dim=1).values.clamp_min(eps_val).view(shape)
    W_norm = W / max_W  # ∈ [0, 1], per-sample max = 1

    return grad.sign() * W_norm


def _local_variance(x: Tensor, kernel_size: int) -> Tensor:
    """Per-pixel local variance over a sliding spatial window.

    Computes channel-averaged local variance and broadcasts back to (N, C, H, W).
    Uses E[X²] - E[X]² to avoid a second pass.
    """
    padding = kernel_size // 2
    # Average over channels → (N, 1, H, W)
    x_c = x.mean(dim=1, keepdim=True)
    local_mean = F.avg_pool2d(x_c, kernel_size, stride=1, padding=padding)
    local_sq_mean = F.avg_pool2d(x_c.pow(2), kernel_size, stride=1, padding=padding)
    var = (local_sq_mean - local_mean.pow(2)).clamp_min(0.0)  # (N, 1, H, W)
    return var.expand_as(x)  # broadcast to (N, C, H, W)
