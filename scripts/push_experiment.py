"""Push experiment: can we hit 10% MIA drop?

Tests more aggressive hyperparameters for Adaptive-FGSM:
larger epsilon, stronger confidence penalties.
"""

from __future__ import annotations

import ssl, sys
from pathlib import Path
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from adversral.attacks import membership_inference_attack
from adversral.data import build_image_loaders, build_mia_loaders
from adversral.device import resolve_device
from adversral.engine import evaluate, train_one_epoch
from adversral.models import build_resnet_classifier


CONFIGS = [
    {
        "name": "Baseline",
        "epsilon": 0.0,      "method": "sign",
        "adv_loss": "ce",    "mia_lambda": 0.0,   "train_lambda": 0.0,
    },
    {
        "name": "Sign ε=8/255",
        "epsilon": 8/255,    "method": "sign",
        "adv_loss": "ce",    "mia_lambda": 0.0,   "train_lambda": 0.0,
    },
    {
        "name": "Adaptive ε=8/255  λ=0.5",
        "epsilon": 8/255,    "method": "adaptive",
        "adv_loss": "kl",    "mia_lambda": 0.5,   "train_lambda": 0.3,
    },
    {
        "name": "Adaptive ε=16/255 λ=1.0",
        "epsilon": 16/255,   "method": "adaptive",
        "adv_loss": "kl",    "mia_lambda": 1.0,   "train_lambda": 0.5,
    },
    {
        "name": "Adaptive ε=32/255 λ=1.0",
        "epsilon": 32/255,   "method": "adaptive",
        "adv_loss": "kl",    "mia_lambda": 1.0,   "train_lambda": 0.5,
    },
    {
        "name": "Adaptive ε=16/255 λ=2.0",
        "epsilon": 16/255,   "method": "adaptive",
        "adv_loss": "kl",    "mia_lambda": 2.0,   "train_lambda": 1.0,
    },
]

DATASET   = "cifar10"
DATA_DIR  = "/tmp/cifar10_data"
TRAIN_SIZE = 500
EPOCHS    = 80
BATCH     = 128
LR        = 0.05


def main():
    device = resolve_device("auto")
    print(f"device={device}\n")
    criterion = nn.CrossEntropyLoss()

    loaders = build_image_loaders(
        dataset=DATASET, data_dir=DATA_DIR, batch_size=BATCH,
        num_workers=0, download=False, train_size=TRAIN_SIZE, seed=42,
    )
    mia_loaders = build_mia_loaders(
        dataset=DATASET, data_dir=DATA_DIR, batch_size=BATCH,
        member_size=300, non_member_size=300,
        target_member_size=200, target_non_member_size=200,
        num_workers=0, seed=42, download=False,
    )

    results = []
    for cfg in CONFIGS:
        model = build_resnet_classifier(dataset=DATASET, backbone_name="simple_cnn").to(device)
        opt   = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        print(f"[{cfg['name']}]  ε={cfg['epsilon']:.4f}  mia_λ={cfg['mia_lambda']}  train_λ={cfg['train_lambda']}")
        for epoch in range(1, EPOCHS + 1):
            train_one_epoch(
                model, loaders.train, criterion, opt, device,
                adversarial_epsilon=cfg["epsilon"],
                adversarial_method=cfg["method"],
                adversarial_loss_type=cfg["adv_loss"],
                adversarial_mia_conf_lambda=cfg["mia_lambda"],
                train_conf_lambda=cfg["train_lambda"],
            )
            sched.step()
            if epoch % 20 == 0:
                _, tr = evaluate(model, loaders.train, criterion, device)
                _, va = evaluate(model, loaders.val,   criterion, device)
                print(f"  ep{epoch:02d}: train={tr:.3f} val={va:.3f} gap={tr-va:+.3f}")

        _, clean_acc = evaluate(model, loaders.val, criterion, device)
        mia_res = membership_inference_attack(
            model, mia_loaders.member, mia_loaders.non_member, mia_loaders.target,
            score_type="confidence", membership_labels=mia_loaders.target_membership,
            device=device,
        )
        results.append({"name": cfg["name"], "clean": clean_acc, "mia": mia_res.accuracy})
        print(f"  → clean={clean_acc:.4f}  MIA={mia_res.accuracy:.4f}\n")

    baseline_mia   = results[0]["mia"]
    baseline_clean = results[0]["clean"]

    print("=" * 70)
    print(f"{'Method':<30}  {'Clean':>7}  {'MIA':>7}  {'MIA↓':>7}  {'Clean↓':>7}")
    print("-" * 70)
    for r in results:
        drop  = baseline_mia   - r["mia"]
        cdrop = r["clean"] - baseline_clean
        flag  = "  ← ★10%!" if drop >= 0.10 else ("  ← ★" if drop >= 0.07 else "")
        print(f"{r['name']:<30}  {r['clean']:>7.4f}  {r['mia']:>7.4f}  "
              f"{drop:>+7.4f}  {cdrop:>+7.4f}{flag}")
    print("=" * 70)


if __name__ == "__main__":
    main()
