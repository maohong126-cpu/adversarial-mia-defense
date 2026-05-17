"""Evaluate clean accuracy, adversarial accuracy, and MIA for model suites."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversral.attacks import GradientAttackMethod, fgsm_attack, membership_inference_attack
from adversral.data import build_image_loaders, build_mia_loaders
from adversral.device import resolve_device
from adversral.engine import evaluate
from adversral.models import build_resnet_classifier


@dataclass(frozen=True)
class SuiteResult:
    name: str
    clean_loss: float
    clean_acc: float
    sign_attack_acc: float
    normalized_attack_acc: float
    mia_acc: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints under attack and MIA.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "imagenet"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--backbone", default="simple_cnn", choices=["simple_cnn", "resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2"])
    parser.add_argument("--checkpoint", action="append", required=True, help="Format: name=/path/to/checkpoint.pt")
    parser.add_argument("--attack-epsilon", type=float, default=8 / 255)
    parser.add_argument("--score-type", default="confidence", choices=["confidence", "loss", "entropy"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--member-size", type=int, default=10000)
    parser.add_argument("--non-member-size", type=int, default=10000)
    parser.add_argument("--target-member-size", type=int, default=5000)
    parser.add_argument("--target-non-member-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device}")

    criterion = torch.nn.CrossEntropyLoss()
    image_loaders = build_image_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download,
        seed=args.seed,
    )
    mia_loaders = build_mia_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        member_size=args.member_size,
        non_member_size=args.non_member_size,
        target_member_size=args.target_member_size,
        target_non_member_size=args.target_non_member_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=args.download,
    )

    print("model,clean_loss,clean_acc,sign_attack_acc,normalized_attack_acc,mia_attack_acc")
    for name, checkpoint_path in _parse_checkpoints(args.checkpoint):
        result = evaluate_checkpoint(
            name=name,
            checkpoint_path=checkpoint_path,
            dataset=args.dataset,
            backbone=args.backbone,
            val_loader=image_loaders.val,
            mia_loaders=mia_loaders,
            criterion=criterion,
            attack_epsilon=args.attack_epsilon,
            score_type=args.score_type,
            device=device,
        )
        print(
            f"{result.name},"
            f"{result.clean_loss:.4f},"
            f"{result.clean_acc:.4f},"
            f"{result.sign_attack_acc:.4f},"
            f"{result.normalized_attack_acc:.4f},"
            f"{result.mia_acc:.4f}"
        )


def evaluate_checkpoint(
    *,
    name: str,
    checkpoint_path: str,
    dataset: str,
    backbone: str,
    val_loader: torch.utils.data.DataLoader,
    mia_loaders,
    criterion: torch.nn.Module,
    attack_epsilon: float,
    score_type: str,
    device: torch.device,
) -> SuiteResult:
    model = build_resnet_classifier(dataset=dataset, backbone_name=backbone).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    clean_loss, clean_acc = evaluate(model, val_loader, criterion, device)
    _, sign_acc = evaluate_gradient_attack(
        model,
        val_loader,
        criterion,
        epsilon=attack_epsilon,
        method="sign",
        device=device,
    )
    _, normalized_acc = evaluate_gradient_attack(
        model,
        val_loader,
        criterion,
        epsilon=attack_epsilon,
        method="normalized",
        device=device,
    )
    mia_result = membership_inference_attack(
        model,
        mia_loaders.member,
        mia_loaders.non_member,
        mia_loaders.target,
        score_type=score_type,
        membership_labels=mia_loaders.target_membership,
        device=device,
    )
    if mia_result.accuracy is None:
        raise RuntimeError("MIA evaluation did not return accuracy")

    return SuiteResult(
        name=name,
        clean_loss=clean_loss,
        clean_acc=clean_acc,
        sign_attack_acc=sign_acc,
        normalized_attack_acc=normalized_acc,
        mia_acc=mia_result.accuracy,
    )


def evaluate_gradient_attack(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    *,
    epsilon: float,
    method: GradientAttackMethod,
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
        adv_inputs = fgsm_attack(
            model,
            inputs,
            labels,
            epsilon=epsilon,
            clip_min=0.0,
            clip_max=1.0,
            method=method,
        )
        with torch.no_grad():
            logits = model(adv_inputs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += inputs.size(0)
    if was_training:
        model.train()
    return total_loss / total, correct / total


def _parse_checkpoints(items: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError("--checkpoint must use name=/path/to/checkpoint.pt")
        name, path = item.split("=", 1)
        if not name:
            raise ValueError("checkpoint name cannot be empty")
        parsed.append((name, path))
    return parsed


if __name__ == "__main__":
    main()
