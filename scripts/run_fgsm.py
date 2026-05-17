"""Run FGSM evaluation on MNIST, CIFAR-10, CIFAR-100, or ImageNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversral.attacks import fgsm_attack
from adversral.data import build_image_loaders
from adversral.device import resolve_device
from adversral.engine import evaluate
from adversral.models import build_resnet_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FGSM robustness.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "imagenet"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["simple_cnn", "resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2"])
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--method", choices=["sign", "normalized"], default="sign")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device}")
    loaders = build_image_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download,
        val_size=args.val_size,
        seed=args.seed,
    )
    model = build_resnet_classifier(dataset=args.dataset, backbone_name=args.backbone).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    clean_loss, clean_acc = evaluate(model, loaders.val, torch.nn.CrossEntropyLoss(), device)
    adv_loss, adv_acc = evaluate_fgsm(model, loaders.val, args.epsilon, args.method, device)
    print(f"clean_loss={clean_loss:.4f} clean_acc={clean_acc:.4f}")
    print(f"attack_method={args.method} epsilon={args.epsilon:.6f} adv_loss={adv_loss:.4f} adv_acc={adv_acc:.4f}")


def evaluate_fgsm(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epsilon: float,
    method: str,
    device: torch.device,
) -> tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
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
    return total_loss / total, correct / total


if __name__ == "__main__":
    main()
