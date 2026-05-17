"""Check the local Python environment for this project."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


PACKAGES = ["numpy", "torch", "torchvision", "pytest"]


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"executable={sys.executable}")
    print(f"platform={platform.platform()}")

    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        print(f"{package}={version}")

    try:
        import torch
    except ImportError:
        print("cuda=unknown (torch not installed)")
        print("mps=unknown (torch not installed)")
    else:
        print(f"cuda={torch.cuda.is_available()}")
        print(f"mps={torch.backends.mps.is_available()}")


if __name__ == "__main__":
    main()
