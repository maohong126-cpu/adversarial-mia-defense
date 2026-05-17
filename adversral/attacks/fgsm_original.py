"""Single-step gradient attacks for PyTorch classifiers."""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch
from torch import Tensor, nn


GradientAttackMethod = Literal["sign", "normalized"]


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
) -> Tensor:
    """Generate adversarial examples with a single gradient step.

    ``method="sign"`` is the standard FGSM update under an L-infinity budget:
    ``x_adv = x + epsilon * sign(grad)``.

    ``method="normalized"`` follows the normalized raw gradient direction:
    ``x_adv = x + epsilon * grad / ||grad||_2`` per sample. This keeps the
    relative gradient magnitudes across input dimensions, so it is useful for
    experiments that compare sign-only perturbations with magnitude-aware
    perturbations.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if inputs.size(0) != labels.size(0):
        raise ValueError("inputs and labels must have the same batch size")
    if method not in {"sign", "normalized"}:
        raise ValueError("method must be 'sign' or 'normalized'")

    was_training = model.training
    if set_eval:
        model.eval()

    x = inputs.detach().clone().requires_grad_(True)
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    logits = model(x)
    loss = criterion(logits, labels)
    if targeted:
        loss = -loss

    grad = torch.autograd.grad(loss, x)[0]
    direction = _gradient_direction(grad, method)
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
        )
        adv_batches.append(adv.cpu())
        label_batches.append(labels.detach().cpu())

    if not adv_batches:
        raise ValueError("data_loader produced no batches")

    return torch.cat(adv_batches), torch.cat(label_batches)


def _gradient_direction(grad: Tensor, method: GradientAttackMethod) -> Tensor:
    if method == "sign":
        return grad.sign()

    flat = grad.flatten(start_dim=1)
    norms = flat.norm(p=2, dim=1).clamp_min(torch.finfo(grad.dtype).eps)
    shape = (grad.size(0),) + (1,) * (grad.dim() - 1)
    return grad / norms.view(shape)
