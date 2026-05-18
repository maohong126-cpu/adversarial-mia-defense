"""Training and evaluation loops."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from adversral.attacks import GradientAttackMethod, fgsm_attack


AdversarialLossType = Literal["ce", "kl"]


def _grad_sensitivity(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-sample FGSM gradient magnitude (mean absolute gradient over input dims).

    Uses the gradient of the per-sample CE loss w.r.t. the input to measure
    how sensitive the model is to each training example.  High sensitivity
    indicates memorisation — the model is relying on precise input features
    rather than general patterns, which is exactly what MIA exploits.
    """
    inp = inputs.detach().requires_grad_(True)
    with torch.enable_grad():
        # reduction='sum' so that grad[i] == ∂L_i/∂x_i (samples don't interact)
        loss = F.cross_entropy(model(inp), labels, reduction="sum")
        loss.backward()
    # mean absolute gradient per sample, flattened over spatial/channel dims
    sensitivity = inp.grad.detach().abs().view(inputs.size(0), -1).mean(dim=1)
    return sensitivity


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    adversarial_epsilon: float = 0.0,
    adversarial_method: GradientAttackMethod = "normalized",
    adversarial_weight: float = 1.0,
    adversarial_loss_type: AdversarialLossType = "ce",
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = 1.0,
    # Adaptive FGSM parameters
    adversarial_gradient_alpha: float = 0.5,
    adversarial_perceptual_beta: float = 0.5,
    adversarial_kernel_size: int = 5,
    adversarial_mia_conf_lambda: float = 0.0,
    # Training-time confidence regularisation (AdvReg-inspired).
    train_conf_lambda: float = 0.0,
    # Gradient-Sensitivity Adaptive Label Smoothing (GradSmooth).
    # Uses the FGSM gradient magnitude as a per-sample MIA-risk signal:
    # high-gradient (memorised) samples get softer training targets so the
    # model cannot build overconfident predictions on them, while
    # low-gradient (generalised) samples keep near-hard labels to preserve
    # clean accuracy.  No adversarial examples are added to training.
    grad_smooth_alpha: float = 0.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ------------------------------------------------------------------
        # GradSmooth path: no adversarial examples, only adaptive soft labels
        # ------------------------------------------------------------------
        if grad_smooth_alpha > 0.0:
            sensitivity = _grad_sensitivity(model, inputs, labels)
            s_min, s_max = sensitivity.min(), sensitivity.max()
            # normalise to [0, 1]; guard against all-equal batch
            norm_sens = (sensitivity - s_min) / (s_max - s_min + 1e-8)
            smoothing = grad_smooth_alpha * norm_sens  # [B] in [0, grad_smooth_alpha]

            K = next(iter(model.parameters())).shape  # dummy; will use logits below
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            K = logits.size(1)
            one_hot_labels = F.one_hot(labels, K).float()
            # soft_labels[i] = (1 - s_i)*e_y + s_i/K
            soft_labels = (
                (1.0 - smoothing.unsqueeze(1)) * one_hot_labels
                + smoothing.unsqueeze(1) / K
            )
            loss = -(soft_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)
            continue

        # ------------------------------------------------------------------
        # Standard / adversarial training path
        # ------------------------------------------------------------------
        if adversarial_epsilon > 0:
            adv_inputs = fgsm_attack(
                model,
                inputs,
                labels,
                epsilon=adversarial_epsilon,
                clip_min=clip_min,
                clip_max=clip_max,
                loss_fn=criterion,
                method=adversarial_method,
                set_eval=False,
                gradient_alpha=adversarial_gradient_alpha,
                perceptual_beta=adversarial_perceptual_beta,
                kernel_size=adversarial_kernel_size,
                mia_conf_lambda=adversarial_mia_conf_lambda,
            )
        else:
            adv_inputs = None

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        clean_loss = criterion(logits, labels)

        if train_conf_lambda > 0.0:
            mean_max_conf = F.softmax(logits, dim=1).max(dim=1).values.mean()
            clean_loss = clean_loss + train_conf_lambda * mean_max_conf

        if adv_inputs is None:
            loss = clean_loss
        elif adversarial_loss_type == "kl":
            adv_logits = model(adv_inputs)
            kl_loss = F.kl_div(
                F.log_softmax(adv_logits, dim=1),
                F.softmax(logits.detach(), dim=1),
                reduction="batchmean",
            )
            loss = clean_loss + adversarial_weight * kl_loss
        else:  # "ce"
            adv_logits = model(adv_inputs)
            adv_loss = criterion(adv_logits, labels)
            loss = clean_loss + adversarial_weight * adv_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += inputs.size(0)
    if was_training:
        model.train()
    return total_loss / total, correct / total


def save_checkpoint(path: str, model: nn.Module, args: argparse.Namespace) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": args.dataset,
            "backbone": args.backbone,
            "adversarial_epsilon": getattr(args, "adversarial_epsilon", 0.0),
            "adversarial_method": getattr(args, "adversarial_method", "none"),
            "adversarial_weight": getattr(args, "adversarial_weight", 0.0),
            "adversarial_mia_conf_lambda": getattr(args, "adversarial_mia_conf_lambda", 0.0),
        },
        path,
    )
