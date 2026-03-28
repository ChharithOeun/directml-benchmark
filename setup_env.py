#!/usr/bin/env python3
"""
setup_env.py — Universal cross-platform setup for directml-benchmark
=====================================================================

Works on Windows, Linux, macOS — no bash, no batch files needed.
Uses only Python standard library + subprocess.

Usage:
    python setup_env.py                  # auto-detect and install
    python setup_env.py --check          # verify environment
    python setup_env.py --directml       # Windows AMD/Intel/NVIDIA (Python 3.11 req)
    python setup_env.py --rocm           # Linux/WSL2 AMD GPU
    python setup_env.py --cuda           # Linux/Windows NVIDIA GPU
    python setup_env.py --cpu            # CPU only (any OS, any Python)
    python setup_env.py --benchmark      # run full benchmark after install

Python requirement: 3.8+ for this script.
DirectML benchmark: Python 3.11 only (torch-directml ABI ceiling).
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

OS = platform.system()
PYTHON = sys.executable
PY_VER = sys.version_info[:2]


def banner(msg: str):
    print("=" * 66)
    print(f"  {msg}")
    print("=" * 66)


def run(cmd: list[str]):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def check_env():
    banner("Environment Check — directml-benchmark")
    print(f"  Python   : {sys.version}")
    print(f"  OS       : {OS} {platform.version()}")
    print(f"  Machine  : {platform.machine()}")
    print()

    # torch
    try:
        import torch
        print(f"  torch    : OK  {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"  CUDA     : {'OK — ' + torch.cuda.get_device_name(0) if cuda else 'not available'}")
        if cuda:
            hip = getattr(torch.version, "hip", None)
            print(f"  ROCm/HIP : {'OK — ' + hip if hip else 'No (CUDA, not ROCm)'}")
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"  MPS      : {'OK (Apple Silicon)' if mps else 'not available'}")
    except ImportError:
        print("  torch    : NOT INSTALLED")

    # DirectML (Windows only)
    if OS == "Windows":
        try:
            import torch_directml as dml
            gpu_name = dml.device_name(0) if hasattr(dml, "device_name") else "unknown"
            print(f"  DirectML : OK  {dml.__version__}  ({gpu_name})")
        except ImportError:
            print("  DirectML : not installed  (pip install torch-directml, requires Python 3.11)")
    else:
        print(f"  DirectML : N/A (Windows only)")

    print()
    best = _detect_best()
    print(f"  Best backend: {best}")
    print("=" * 66)


def _detect_best() -> str:
    if OS == "Windows":
        try:
            import torch_directml  # noqa: F401
            return "directml"
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            hip = getattr(torch.version, "hip", None)
            return "rocm" if hip else "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def install_directml():
    banner("Installing torch-directml (Windows, Python 3.11 required)")
    if OS != "Windows":
        print("[WARN] DirectML is Windows-only. On Linux/macOS, use --rocm or --cuda.")
        print("       Continuing anyway in case you are in WSL2...")
    if PY_VER > (3, 11):
        print(f"[ERROR] Python {PY_VER[0]}.{PY_VER[1]} detected.")
        print("        torch-directml requires Python <= 3.11 (hard ABI ceiling).")
        print("        Create a venv: py -3.11 -m venv .venv311")
        print("        Then:          .venv311\\Scripts\\activate")
        print("        Then re-run:   python setup_env.py --directml")
        sys.exit(1)
    print("  Note: Do NOT pre-install torch — torch-directml pulls torch 2.4.1 automatically.")
    print()
    run([PYTHON, "-m", "pip", "install", "torch-directml"])
    print()
    check_env()


def install_rocm():
    banner("Installing torch for AMD ROCm (Linux/WSL2)")
    if OS == "Windows":
        print("[ERROR] ROCm not supported on Windows natively. Use WSL2.")
        sys.exit(1)
    run([PYTHON, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/rocm6.1"])

    print()
    print("  For RX 5700 XT (gfx1010), add to ~/.bashrc:")
    print("  export HSA_OVERRIDE_GFX_VERSION=10.3.0")
    print()
    check_env()


def install_cuda():
    banner("Installing torch for NVIDIA CUDA 12.1")
    run([PYTHON, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cu121"])
    check_env()


def install_cpu():
    banner("Installing torch (CPU only — any OS, any Python)")
    run([PYTHON, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu"])
    check_env()


def run_benchmark():
    banner("Running directml-benchmark")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")
    if not os.path.exists(script):
        print(f"[ERROR] benchmark.py not found at: {script}")
        sys.exit(1)
    subprocess.run([PYTHON, script, "--cpu-only",
                    "--warmup", "20", "--iters", "20"], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform setup for directml-benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--check",     action="store_true")
    parser.add_argument("--directml",  action="store_true")
    parser.add_argument("--rocm",      action="store_true")
    parser.add_argument("--cuda",      action="store_true")
    parser.add_argument("--cpu",       action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_env()
    elif args.directml:
        install_directml()
    elif args.rocm:
        install_rocm()
    elif args.cuda:
        install_cuda()
    elif args.cpu:
        install_cpu()
    elif args.benchmark:
        run_benchmark()
    else:
        banner("directml-benchmark — Auto Setup")
        best = _detect_best()
        print(f"  Detected: {best}")
        print()
        if best == "directml":
            install_directml()
        elif best in ("rocm", "cuda"):
            if best == "rocm":
                install_rocm()
            else:
                install_cuda()
        else:
            install_cpu()


if __name__ == "__main__":
    main()
