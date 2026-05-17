"""Train ResNet backbones on MNIST, CIFAR-10, CIFAR-100, or ImageNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversral.data import build_image_loaders
from adversral.device import resolve_device
from adversral.engine import evaluate, save_checkpoint, train_one_epoch
from adversral.models import build_resnet_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ResNet classifier.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "imagenet"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["simple_cnn", "resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adversarial-epsilon", type=float, default=0.0)
    parser.add_argument("--adversarial-method", choices=["sign", "normalized", "adaptive"], default="normalized")
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    # Adaptive FGSM parameters
    parser.add_argument("--adversarial-gradient-alpha", type=float, default=0.5,
                        help="Gradient concentration power for adaptive method (0=uniform, 1=fully concentrated)")
    parser.add_argument("--adversarial-perceptual-beta", type=float, default=0.5,
                        help="Perceptual (local variance) weight power for adaptive method")
    parser.add_argument("--adversarial-kernel-size", type=int, default=5,
                        help="Local variance sliding window size for adaptive method")
    parser.add_argument("--adversarial-mia-conf-lambda", type=float, default=0.0,
                        help="MIA-steered confidence penalty weight. "
                             "Loss = CE - lambda * mean_max_confidence. "
                             "Positive values push perturbations to reduce model confidence, "
                             "directly targeting the overconfidence gap that MIA exploits.")
    parser.add_argument("--adversarial-loss-type", choices=["ce", "kl"], default="ce",
                        help="Adversarial training loss: 'ce' (standard) or 'kl' (TRADES-style KL divergence).")
    parser.add_argument("--train-conf-lambda", type=float, default=0.0,
                        help="AdvReg-inspired training-time confidence penalty. "
                             "Adds lambda * mean_max_confidence to the clean training loss, "
                             "directly penalising overconfidence on training data.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    if args.adversarial_epsilon < 0:
        raise ValueError("--adversarial-epsilon must be non-negative")
    if args.adversarial_weight < 0:
        raise ValueError("--adversarial-weight must be non-negative")

    device = resolve_device(args.device)
    print(f"device={device}")
    loaders = build_image_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    model = build_resnet_classifier(
        dataset=args.dataset,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            loaders.train,
            criterion,
            optimizer,
            device,
            adversarial_epsilon=args.adversarial_epsilon,
            adversarial_method=args.adversarial_method,
            adversarial_weight=args.adversarial_weight,
            adversarial_loss_type=args.adversarial_loss_type,
            adversarial_gradient_alpha=args.adversarial_gradient_alpha,
            adversarial_perceptual_beta=args.adversarial_perceptual_beta,
            adversarial_kernel_size=args.adversarial_kernel_size,
            adversarial_mia_conf_lambda=args.adversarial_mia_conf_lambda,
            train_conf_lambda=args.train_conf_lambda,
        )
        val_loss, val_acc = evaluate(model, loaders.val, criterion, device)
        scheduler.step()

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"adv_epsilon={args.adversarial_epsilon:.6f} "
            f"adv_method={args.adversarial_method} "
            f"mia_lambda={args.adversarial_mia_conf_lambda:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(args.checkpoint, model, args)


if __name__ == "__main__":
    main()
