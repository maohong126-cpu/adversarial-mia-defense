"""Backbone models for image classification experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor, nn
from torchvision import models

from adversral.data.vision import DatasetName


BackboneName = Literal["simple_cnn", "resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2"]


@dataclass(frozen=True)
class NormalizationStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASET_NORMALIZATION: dict[DatasetName, NormalizationStats] = {
    "mnist": NormalizationStats(mean=(0.1307,), std=(0.3081,)),
    "cifar10": NormalizationStats(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": NormalizationStats(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    "imagenet": NormalizationStats(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}


DATASET_CHANNELS: dict[DatasetName, int] = {
    "mnist": 1,
    "cifar10": 3,
    "cifar100": 3,
    "imagenet": 3,
}


DATASET_CLASSES: dict[DatasetName, int] = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
}


class NormalizedClassifier(nn.Module):
    """Classifier wrapper that normalizes [0, 1] image tensors before inference."""

    def __init__(self, backbone: nn.Module, stats: NormalizationStats) -> None:
        super().__init__()
        self.backbone = backbone
        channels = len(stats.mean)
        self.register_buffer("mean", torch.tensor(stats.mean).view(1, channels, 1, 1))
        self.register_buffer("std", torch.tensor(stats.std).view(1, channels, 1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.backbone((inputs - self.mean) / self.std)


def build_resnet_classifier(
    *,
    dataset: DatasetName,
    backbone_name: BackboneName = "resnet50",
    pretrained: bool = False,
) -> nn.Module:
    """Build a ResNet classifier for MNIST, CIFAR-10, CIFAR-100, or ImageNet."""
    num_classes = DATASET_CLASSES[dataset]
    input_channels = DATASET_CHANNELS[dataset]
    if backbone_name == "simple_cnn":
        if dataset == "imagenet":
            raise ValueError("simple_cnn is intended for MNIST/CIFAR, not ImageNet")
        return NormalizedClassifier(
            SimpleCNN(input_channels=input_channels, num_classes=num_classes),
            DATASET_NORMALIZATION[dataset],
        )

    backbone = _build_backbone(backbone_name, num_classes=num_classes, pretrained=pretrained)

    if dataset in {"mnist", "cifar10", "cifar100"}:
        _adapt_resnet_for_small_images(backbone, input_channels=input_channels)
    elif input_channels != 3:
        _replace_first_conv(backbone, input_channels=input_channels, kernel_size=7, stride=2, padding=3)

    return NormalizedClassifier(backbone, DATASET_NORMALIZATION[dataset])


class SimpleCNN(nn.Module):
    """Small CNN for fast MNIST/CIFAR experiments."""

    def __init__(self, *, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def _build_backbone(backbone_name: BackboneName, *, num_classes: int, pretrained: bool) -> nn.Module:
    builders: dict[BackboneName, Callable[..., nn.Module]] = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "wide_resnet50_2": models.wide_resnet50_2,
    }
    if backbone_name not in builders:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if pretrained:
        model = builders[backbone_name](weights="DEFAULT")
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    return builders[backbone_name](weights=None, num_classes=num_classes)


def _adapt_resnet_for_small_images(model: nn.Module, *, input_channels: int) -> None:
    """Use the common small-image ResNet stem: 3x3 stride-1 conv and no maxpool."""
    _replace_first_conv(model, input_channels=input_channels, kernel_size=3, stride=1, padding=1)
    model.maxpool = nn.Identity()


def _replace_first_conv(
    model: nn.Module,
    *,
    input_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> None:
    model.conv1 = nn.Conv2d(
        input_channels,
        64,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
