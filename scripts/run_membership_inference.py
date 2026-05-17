"""Run membership inference attack on MNIST, CIFAR-10, CIFAR-100, or ImageNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversral.attacks import membership_inference_attack
from adversral.data import build_mia_loaders
from adversral.device import resolve_device
from adversral.models import build_resnet_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run threshold-based membership inference.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "imagenet"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["simple_cnn", "resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2"])
    parser.add_argument("--score-type", default="confidence", choices=["confidence", "loss", "entropy"])
    parser.add_argument("--batch-size", type=int, default=128)
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
    loaders = build_mia_loaders(
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
    model = build_resnet_classifier(dataset=args.dataset, backbone_name=args.backbone).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    result = membership_inference_attack(
        model,
        loaders.member,
        loaders.non_member,
        loaders.target,
        score_type=args.score_type,
        membership_labels=loaders.target_membership,
        device=device,
    )
    print(f"score_type={result.score_type}")
    print(f"threshold={result.threshold:.6f}")
    print(f"member_rule={result.member_rule}")
    print(f"attack_accuracy={result.accuracy:.4f}")


if __name__ == "__main__":
    main()
