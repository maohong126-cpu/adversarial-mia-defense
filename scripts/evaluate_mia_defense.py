"""Compare baseline and adversarially trained models under MIA."""

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
class ModelEvaluation:
    name: str
    clean_loss: float
    clean_acc: float
    attack_loss: float
    attack_acc: float
    mia_acc: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MIA defense against a baseline model.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "imagenet"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--defense-checkpoint", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["simple_cnn", "resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2"])
    parser.add_argument("--score-type", default="confidence", choices=["confidence", "loss", "entropy"])
    parser.add_argument("--attack-epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-method", choices=["sign", "normalized"], default="normalized")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--member-size", type=int, default=5000)
    parser.add_argument("--non-member-size", type=int, default=5000)
    parser.add_argument("--target-member-size", type=int, default=1000)
    parser.add_argument("--target-non-member-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
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
        val_size=args.val_size,
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

    baseline = _evaluate_checkpoint(
        name="baseline",
        checkpoint_path=args.baseline_checkpoint,
        dataset=args.dataset,
        backbone=args.backbone,
        val_loader=image_loaders.val,
        mia_loaders=mia_loaders,
        criterion=criterion,
        score_type=args.score_type,
        attack_epsilon=args.attack_epsilon,
        attack_method=args.attack_method,
        device=device,
    )
    defense = _evaluate_checkpoint(
        name="defense",
        checkpoint_path=args.defense_checkpoint,
        dataset=args.dataset,
        backbone=args.backbone,
        val_loader=image_loaders.val,
        mia_loaders=mia_loaders,
        criterion=criterion,
        score_type=args.score_type,
        attack_epsilon=args.attack_epsilon,
        attack_method=args.attack_method,
        device=device,
    )

    print("model,clean_loss,clean_acc,attack_loss,attack_acc,mia_attack_acc")
    for result in [baseline, defense]:
        print(
            f"{result.name},"
            f"{result.clean_loss:.4f},"
            f"{result.clean_acc:.4f},"
            f"{result.attack_loss:.4f},"
            f"{result.attack_acc:.4f},"
            f"{result.mia_acc:.4f}"
        )

    mia_drop = baseline.mia_acc - defense.mia_acc
    clean_drop = baseline.clean_acc - defense.clean_acc
    attack_gain = defense.attack_acc - baseline.attack_acc
    print(f"mia_attack_acc_drop={mia_drop:.4f}")
    print(f"clean_acc_drop={clean_drop:.4f}")
    print(f"attack_acc_gain={attack_gain:.4f}")


def _evaluate_checkpoint(
    *,
    name: str,
    checkpoint_path: str,
    dataset: str,
    backbone: str,
    val_loader: torch.utils.data.DataLoader,
    mia_loaders,
    criterion: torch.nn.Module,
    score_type: str,
    attack_epsilon: float,
    attack_method: GradientAttackMethod,
    device: torch.device,
) -> ModelEvaluation:
    model = build_resnet_classifier(dataset=dataset, backbone_name=backbone).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    clean_loss, clean_acc = evaluate(model, val_loader, criterion, device)
    attack_loss, attack_acc = evaluate_gradient_attack(
        model,
        val_loader,
        criterion,
        epsilon=attack_epsilon,
        method=attack_method,
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

    return ModelEvaluation(
        name=name,
        clean_loss=clean_loss,
        clean_acc=clean_acc,
        attack_loss=attack_loss,
        attack_acc=attack_acc,
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
    total_loss = 0.0
    correct = 0
    total = 0
    was_training = model.training
    model.eval()
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


if __name__ == "__main__":
    main()
