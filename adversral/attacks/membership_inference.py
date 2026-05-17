"""Threshold-based membership inference attack for PyTorch classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MiaScore = Literal["confidence", "loss", "entropy"]


@dataclass(frozen=True)
class MembershipInferenceResult:
    """Result returned by a threshold-based membership inference attack."""

    threshold: float
    score_type: MiaScore
    member_rule: Literal["score_ge_threshold", "score_le_threshold"]
    predictions: Tensor
    scores: Tensor
    accuracy: Optional[float] = None


class ThresholdMembershipInferenceAttack:
    """Infer membership by thresholding model confidence, entropy, or loss."""

    def __init__(self, score_type: MiaScore = "confidence") -> None:
        if score_type not in {"confidence", "loss", "entropy"}:
            raise ValueError("score_type must be 'confidence', 'loss', or 'entropy'")
        self.score_type = score_type
        self.threshold: Optional[float] = None
        self.member_rule: Optional[Literal["score_ge_threshold", "score_le_threshold"]] = None

    @torch.no_grad()
    def score(
        self,
        model: nn.Module,
        inputs: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute per-sample membership scores for a batch."""
        logits = model(inputs)
        probabilities = F.softmax(logits, dim=1)

        if self.score_type == "confidence":
            if labels is None:
                return probabilities.max(dim=1).values.detach().cpu()
            return probabilities.gather(1, labels.view(-1, 1)).squeeze(1).detach().cpu()

        if self.score_type == "loss":
            if labels is None:
                raise ValueError("labels are required when score_type='loss'")
            return F.cross_entropy(logits, labels, reduction="none").detach().cpu()

        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)
        return entropy.detach().cpu()

    def fit(
        self,
        member_scores: Tensor,
        non_member_scores: Tensor,
    ) -> "ThresholdMembershipInferenceAttack":
        """Choose the threshold that maximizes calibration accuracy."""
        member_scores = member_scores.detach().flatten().cpu()
        non_member_scores = non_member_scores.detach().flatten().cpu()
        if member_scores.numel() == 0 or non_member_scores.numel() == 0:
            raise ValueError("member_scores and non_member_scores must be non-empty")

        scores = torch.cat([member_scores, non_member_scores])
        labels = torch.cat(
            [
                torch.ones_like(member_scores, dtype=torch.bool),
                torch.zeros_like(non_member_scores, dtype=torch.bool),
            ]
        )
        thresholds = torch.unique(scores)  # (T,)

        # Vectorized: broadcast scores (1,N) against thresholds (T,1)
        scores_row = scores.unsqueeze(0)       # (1, N)
        labels_row = labels.unsqueeze(0)       # (1, N)
        thresh_col = thresholds.unsqueeze(1)   # (T, 1)

        ge_acc = ((scores_row >= thresh_col) == labels_row).float().mean(dim=1)  # (T,)
        le_acc = ((scores_row <= thresh_col) == labels_row).float().mean(dim=1)  # (T,)

        ge_best = ge_acc.argmax().item()
        le_best = le_acc.argmax().item()

        if ge_acc[ge_best] >= le_acc[le_best]:
            best_threshold = thresholds[ge_best]
            best_rule: Literal["score_ge_threshold", "score_le_threshold"] = "score_ge_threshold"
        else:
            best_threshold = thresholds[le_best]
            best_rule = "score_le_threshold"

        self.threshold = float(best_threshold.item())
        self.member_rule = best_rule
        return self

    def predict(self, scores: Tensor) -> Tensor:
        """Predict membership labels from precomputed scores."""
        if self.threshold is None or self.member_rule is None:
            raise RuntimeError("fit must be called before predict")

        scores = scores.detach().flatten().cpu()
        if self.member_rule == "score_ge_threshold":
            return scores >= self.threshold
        return scores <= self.threshold

    def evaluate(
        self,
        scores: Tensor,
        membership_labels: Tensor,
    ) -> MembershipInferenceResult:
        """Predict membership and compute attack accuracy."""
        predictions = self.predict(scores)
        membership_labels = membership_labels.detach().flatten().cpu().bool()
        if predictions.numel() != membership_labels.numel():
            raise ValueError("scores and membership_labels must have the same length")

        accuracy = (predictions == membership_labels).float().mean().item()
        return MembershipInferenceResult(
            threshold=float(self.threshold),
            score_type=self.score_type,
            member_rule=self.member_rule,
            predictions=predictions,
            scores=scores.detach().flatten().cpu(),
            accuracy=accuracy,
        )


def membership_inference_attack(
    model: nn.Module,
    member_loader: Iterable[tuple[Tensor, Tensor]],
    non_member_loader: Iterable[tuple[Tensor, Tensor]],
    target_loader: Iterable[tuple[Tensor, Tensor]],
    *,
    score_type: MiaScore = "confidence",
    membership_labels: Optional[Tensor] = None,
    device: Optional[torch.device | str] = None,
) -> MembershipInferenceResult:
    """Fit a threshold MIA and infer membership for a target loader.

    Pass ``membership_labels`` (bool tensor, True = member) to compute attack accuracy.
    """
    model_device = next(model.parameters()).device
    target_device = torch.device(device) if device is not None else model_device
    attack = ThresholdMembershipInferenceAttack(score_type)

    member_scores = _loader_scores(attack, model, member_loader, target_device)
    non_member_scores = _loader_scores(attack, model, non_member_loader, target_device)
    attack.fit(member_scores, non_member_scores)

    target_scores = _loader_scores(attack, model, target_loader, target_device)

    if membership_labels is not None:
        return attack.evaluate(target_scores, membership_labels)

    predictions = attack.predict(target_scores)
    return MembershipInferenceResult(
        threshold=float(attack.threshold),
        score_type=score_type,
        member_rule=attack.member_rule,
        predictions=predictions,
        scores=target_scores,
    )


def _loader_scores(
    attack: ThresholdMembershipInferenceAttack,
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> Tensor:
    batches: list[Tensor] = []
    was_training = model.training
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        batches.append(attack.score(model, inputs, labels))
    if was_training:
        model.train()
    if not batches:
        raise ValueError("loader produced no batches")
    return torch.cat(batches)

