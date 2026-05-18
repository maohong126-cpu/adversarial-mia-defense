"""Quick comparison: 5 FGSM variants as MIA defence on CIFAR-10 (or MNIST).

Trains five models on a small dataset subset to force memorisation, then
measures clean accuracy and MIA attack accuracy for each.

Methods compared:
  1. Baseline          – no adversarial training
  2. Sign-FGSM         – standard FGSM (CE loss)
  3. Normalized-FGSM   – user's existing approach (CE loss, L2-normalised grad)
  4. Adaptive-FGSM     – OUR innovation: perceptual-adaptive weighting +
                         MIA-steered confidence loss during generation
  5. Adaptive+KL       – OUR innovation + TRADES-style KL boundary smoothing +
                         AdvReg-inspired training-time confidence penalty

Literature baseline for comparison:
  AdvReg (Nasr 2019): ~7.45% MIA drop
  SELENA (Tang 2022):  best privacy-utility tradeoff among ensemble methods
  RelaxLoss (Chen 2022): gradient-ascent loss variant, outperforms AdvReg

Usage:
    python scripts/quick_experiment.py --data-dir /tmp/cifar10_data --download
    python scripts/quick_experiment.py --data-dir /tmp/mnist_data  --dataset mnist
"""

from __future__ import annotations

import argparse
import ssl
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

# Fix SSL certificate issues on macOS.
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversral.attacks import membership_inference_attack
from adversral.data import build_image_loaders, build_mia_loaders
from adversral.device import resolve_device
from adversral.engine import evaluate, train_one_epoch
from adversral.models import build_resnet_classifier


# ---------------------------------------------------------------------------
# Method configurations
# ---------------------------------------------------------------------------

def make_configs(epsilon: float) -> list[dict]:
    return [
        {
            "name": "1-Baseline",
            "epsilon": 0.0,
            "method": "sign",
            "adversarial_loss_type": "ce",
            "mia_conf_lambda": 0.0,
            "train_conf_lambda": 0.0,
            "gradient_alpha": 0.5,
            "perceptual_beta": 0.5,
            "grad_smooth_alpha": 0.0,
        },
        {
            "name": "2-Sign-FGSM",
            "epsilon": epsilon,
            "method": "sign",
            "adversarial_loss_type": "ce",
            "mia_conf_lambda": 0.0,
            "train_conf_lambda": 0.0,
            "gradient_alpha": 0.5,
            "perceptual_beta": 0.5,
            "grad_smooth_alpha": 0.0,
        },
        {
            "name": "3-Normalized-FGSM",
            "epsilon": epsilon,
            "method": "normalized",
            "adversarial_loss_type": "ce",
            "mia_conf_lambda": 0.0,
            "train_conf_lambda": 0.0,
            "gradient_alpha": 0.5,
            "perceptual_beta": 0.5,
            "grad_smooth_alpha": 0.0,
        },
        {
            "name": "4-Adaptive-FGSM",
            "epsilon": epsilon,
            "method": "adaptive",
            "adversarial_loss_type": "ce",
            "mia_conf_lambda": 0.5,
            "train_conf_lambda": 0.0,
            "gradient_alpha": 0.5,
            "perceptual_beta": 0.5,
            "grad_smooth_alpha": 0.0,
        },
        {
            "name": "5-Adaptive+KL+AdvReg",
            "epsilon": epsilon,
            "method": "adaptive",
            "adversarial_loss_type": "kl",
            "mia_conf_lambda": 0.5,
            "train_conf_lambda": 0.3,
            "gradient_alpha": 0.5,
            "perceptual_beta": 0.5,
            "grad_smooth_alpha": 0.0,
        },
        {
            "name": "6-GradSmooth(α=0.3)",
            "epsilon": 0.0,               # no adversarial examples added
            "method": "sign",
            "adversarial_loss_type": "ce",
            "mia_conf_lambda": 0.0,
            "train_conf_lambda": 0.0,
            "gradient_alpha": 0.5,
            "perceptual_beta": 0.5,
            "grad_smooth_alpha": 0.3,     # max label smoothing for high-gradient samples
        },
        {
            "name": "7-GradSmooth(α=0.5)",
            "epsilon": 0.0,
            "method": "sign",
            "adversarial_loss_type": "ce",
            "mia_conf_lambda": 0.0,
            "train_conf_lambda": 0.0,
            "gradient_alpha": 0.5,
            "perceptual_beta": 0.5,
            "grad_smooth_alpha": 0.5,
        },
    ]


# ---------------------------------------------------------------------------
# Training + evaluation helpers
# ---------------------------------------------------------------------------

def train_model(cfg: dict, loaders, criterion, device, epochs: int, lr: float,
                dataset: str, backbone: str) -> nn.Module:
    model = build_resnet_classifier(dataset=dataset, backbone_name=backbone).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    gs_alpha = cfg.get("grad_smooth_alpha", 0.0)
    print(f"\n[{cfg['name']}]  ε={cfg['epsilon']:.4f}  method={cfg['method']}  "
          f"adv_loss={cfg['adversarial_loss_type']}  "
          f"gen_λ={cfg['mia_conf_lambda']:.2f}  train_λ={cfg['train_conf_lambda']:.2f}  "
          f"gs_α={gs_alpha:.2f}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            loaders.train,
            criterion,
            optimizer,
            device,
            adversarial_epsilon=cfg["epsilon"],
            adversarial_method=cfg["method"],
            adversarial_loss_type=cfg["adversarial_loss_type"],
            adversarial_weight=1.0,
            adversarial_gradient_alpha=cfg["gradient_alpha"],
            adversarial_perceptual_beta=cfg["perceptual_beta"],
            adversarial_mia_conf_lambda=cfg["mia_conf_lambda"],
            train_conf_lambda=cfg["train_conf_lambda"],
            grad_smooth_alpha=gs_alpha,
        )
        val_loss, val_acc = evaluate(model, loaders.val, criterion, device)
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:02d}: train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                  f"gap={train_acc - val_acc:+.3f}")

    return model


def run_mia(model, mia_loaders, device, score_type: str) -> float:
    result = membership_inference_attack(
        model,
        mia_loaders.member,
        mia_loaders.non_member,
        mia_loaders.target,
        score_type=score_type,
        membership_labels=mia_loaders.target_membership,
        device=device,
    )
    return result.accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"],
                        default="cifar10")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--backbone", default="simple_cnn",
                        choices=["simple_cnn", "resnet18", "resnet34", "resnet50"])
    parser.add_argument("--train-size", type=int, default=2000,
                        help="Small training set forces memorisation (default 2000).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--member-size", type=int, default=1500)
    parser.add_argument("--non-member-size", type=int, default=1500)
    parser.add_argument("--target-member-size", type=int, default=500)
    parser.add_argument("--target-non-member-size", type=int, default=500)
    parser.add_argument("--score-type", choices=["confidence", "loss", "entropy"],
                        default="confidence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(tempfile.gettempdir()) / f"{args.dataset}_data")
        print(f"No --data-dir supplied; using {args.data_dir}")

    device = resolve_device(args.device)
    print(f"device={device}  dataset={args.dataset}  seed={args.seed}")
    criterion = nn.CrossEntropyLoss()

    print("\n=== Loading data ===")
    loaders = build_image_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        download=args.download,
        train_size=args.train_size,
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
        num_workers=0,
        seed=args.seed,
        download=False,
    )
    print(f"  train={args.train_size}  member={args.member_size}  "
          f"non-member={args.non_member_size}  target={args.target_member_size + args.target_non_member_size}")

    configs = make_configs(args.epsilon)

    # ---------------------------------------------------------------------------
    # Train all configurations
    # ---------------------------------------------------------------------------
    results = []
    for cfg in configs:
        model = train_model(
            cfg, loaders, criterion, device,
            args.epochs, args.lr, args.dataset, args.backbone,
        )
        clean_loss, clean_acc = evaluate(model, loaders.val, criterion, device)
        mia_acc = run_mia(model, mia_loaders, device, args.score_type)
        results.append({"name": cfg["name"], "clean_acc": clean_acc, "mia_acc": mia_acc})
        print(f"  -> clean_acc={clean_acc:.4f}  MIA_acc={mia_acc:.4f}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    baseline_mia = results[0]["mia_acc"]
    baseline_clean = results[0]["clean_acc"]

    print("\n" + "=" * 80)
    print(f"RESULTS  dataset={args.dataset}  train_size={args.train_size}  epochs={args.epochs}")
    print("=" * 80)
    print(f"{'Method':<26}  {'CleanAcc':>9}  {'MIA Acc':>9}  {'MIA Drop':>10}  {'Clean Δ':>8}")
    print("-" * 80)
    for r in results:
        mia_drop = baseline_mia - r["mia_acc"]
        clean_delta = r["clean_acc"] - baseline_clean
        flag = "  ★" if r["name"] != results[0]["name"] and mia_drop > 0.05 else ""
        print(f"{r['name']:<26}  {r['clean_acc']:>9.4f}  {r['mia_acc']:>9.4f}  "
              f"{mia_drop:>+10.4f}  {clean_delta:>+8.4f}{flag}")
    print("=" * 80)

    best = max(results[1:], key=lambda r: baseline_mia - r["mia_acc"])
    print(f"\nBest defence: {best['name']}")
    print(f"  MIA drop  : {baseline_mia - best['mia_acc']:+.4f}  "
          f"({(baseline_mia - best['mia_acc']) / baseline_mia * 100:.1f}% relative reduction)")
    print(f"  Clean acc : {best['clean_acc']:.4f} ({best['clean_acc'] - baseline_clean:+.4f} vs baseline)")
    print()
    print("Reference (literature):")
    print("  AdvReg 2019  : ~7.5% MIA reduction, accuracy drop ~7.5%")
    print("  RelaxLoss 2022: best privacy-utility tradeoff")
    print("  MIA Acc ≈ 0.50 = ideal (attacker at random chance)")


if __name__ == "__main__":
    main()
