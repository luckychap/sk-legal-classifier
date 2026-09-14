#!/usr/bin/env python3
"""
Report whether this machine can run the Slovak legal classifier fine-tuning.

Run this FIRST on the target box. It tells you whether PyTorch will actually
talk to the GPU, which compute capability the card is, and whether the installed
PyTorch build contains kernels for that capability.

    python check_env.py
"""
from __future__ import annotations

import platform
import shutil
import sys


def rule(title: str) -> None:
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def main() -> int:
    rule("HOST")
    print(f"Python        : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform      : {platform.platform()}")
    print(f"CPU cores     : {__import__('os').cpu_count()}")

    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        print(f"System RAM    : {vm.total / 2**30:.1f} GiB total, "
              f"{vm.available / 2**30:.1f} GiB available")
    except Exception:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    print(f"System RAM    : {kb / 2**20:.1f} GiB total")
                    break
    du = shutil.disk_usage(__import__("os").getcwd())
    print(f"Free disk     : {du.free / 2**30:.1f} GiB (cwd)")

    rule("PYTORCH")
    try:
        import torch
    except ImportError:
        print("torch is NOT installed.")
        print("Install it from the CUDA 12.6 wheel index (see README.md).")
        return 2

    print(f"torch         : {torch.__version__}")
    print(f"built for CUDA: {torch.version.cuda}")
    try:
        print(f"cuDNN         : {torch.backends.cudnn.version()}")
    except Exception:
        pass

    if not torch.cuda.is_available():
        print()
        print("CUDA is NOT available to PyTorch.")
        print("  - no NVIDIA driver, or")
        print("  - this is a CPU-only torch build, or")
        print("  - the driver is too old for this CUDA runtime.")
        print("Training will fall back to CPU (very slow but will work).")
        print()
        print("WARNING: no GPU. Fine for a smoke test; far too slow for a real run.")
        return 0

    rule("GPU")
    supported = True
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cc = f"{p.major}.{p.minor}"
        print(f"GPU {i}         : {p.name}")
        print(f"  compute cap : {cc}")
        print(f"  VRAM        : {p.total_memory / 2**30:.1f} GiB")
        if p.major < 6:
            print("  NOTE        : pre-Pascal card. Modern bitsandbytes/"
                  "Triton/Unsloth will NOT work here. Plain PyTorch is fine.")

    try:
        arch_list = torch.cuda.get_arch_list()
        print()
        print(f"Compiled archs: {' '.join(arch_list) if arch_list else '<none reported>'}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            want = f"sm_{p.major}{p.minor}"
            if arch_list and want not in arch_list:
                print(f"  !! {want} is NOT in this torch build's arch list.")
                print(f"     This wheel will refuse to run on GPU {i}.")
                supported = False
    except Exception as exc:  # older/newer torch, or no device query support
        print(f"(arch list unavailable: {exc})")

    rule("VERDICT")
    if supported:
        print("OK - PyTorch can run on this GPU. You are good to go.")
        print()
        print("Next:  python train.py --smoke")
    else:
        print("This torch build does not support your GPU.")
        print("Install a cu126 build that still contains sm_5x kernels, e.g.:")
        print("  pip install torch==2.7.1 --index-url "
              "https://download.pytorch.org/whl/cu126")
    return 0 if supported else 3


if __name__ == "__main__":
    raise SystemExit(main())
