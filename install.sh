#!/usr/bin/env bash
# install.sh — Cross-platform setup for directml-benchmark
# Works on: Linux (ROCm/CUDA/CPU) and macOS (MPS/CPU)
# For Windows: use install.bat instead
#
# Usage:
#   bash install.sh            # auto-detect best device
#   bash install.sh --rocm     # AMD GPU via ROCm (Linux only)
#   bash install.sh --cuda     # NVIDIA GPU via CUDA
#   bash install.sh --cpu      # CPU only (any machine)
#   bash install.sh --check    # verify environment without installing

set -e

ROCM=0; CUDA=0; CPU=0; CHECK=0

for arg in "$@"; do
    case "$arg" in
        --rocm)  ROCM=1  ;;
        --cuda)  CUDA=1  ;;
        --cpu)   CPU=1   ;;
        --check) CHECK=1 ;;
    esac
done

OS="$(uname -s)"
PYTHON="${PYTHON:-python3}"

echo "════════════════════════════════════════════════════"
echo "  directml-benchmark installer"
echo "  OS: $OS  |  Python: $($PYTHON --version 2>&1)"
echo "════════════════════════════════════════════════════"

if [ "$CHECK" -eq 1 ]; then
    echo ""
    echo "Checking environment..."
    $PYTHON benchmark.py --check
    exit 0
fi

# ── Auto-detect if no flag given ─────────────────────────────────────────
if [ "$ROCM" -eq 0 ] && [ "$CUDA" -eq 0 ] && [ "$CPU" -eq 0 ]; then
    if command -v rocminfo &>/dev/null 2>&1; then
        echo "[INFO] ROCm detected — installing torch+ROCm"
        ROCM=1
    elif nvidia-smi &>/dev/null 2>&1; then
        echo "[INFO] NVIDIA GPU detected — installing torch+CUDA"
        CUDA=1
    else
        echo "[INFO] No GPU detected — installing CPU-only torch"
        CPU=1
    fi
fi

# ── Install ───────────────────────────────────────────────────────────────
if [ "$ROCM" -eq 1 ]; then
    echo "[1/2] Installing torch for ROCm 6.1 (AMD GPU)..."
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
    echo ""
    echo "  AMD GPU notes:"
    echo "  - For RX 5700 XT (gfx1010): export HSA_OVERRIDE_GFX_VERSION=10.3.0"
    echo "  - For RX 6000/7000 series: no override needed (officially supported)"
    echo "  - Test: python3 -c \"import torch; print(torch.cuda.is_available())\""

elif [ "$CUDA" -eq 1 ]; then
    echo "[1/2] Installing torch for CUDA 12.1 (NVIDIA GPU)..."
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu121

elif [ "$CPU" -eq 1 ]; then
    echo "[1/2] Installing torch (CPU only)..."
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "[2/2] Verifying install..."
$PYTHON benchmark.py --check

echo ""
echo "════════════════════════════════════════════════════"
echo "  Setup complete! Run your benchmark:"
echo ""
if [ "$ROCM" -eq 1 ]; then
    echo "  export HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 5700 XT only"
fi
echo "  python3 benchmark.py --cpu-only   # verify math (no GPU needed)"
echo "  python3 benchmark.py              # full GPU benchmark"
echo ""
echo "  Results saved to: results/<DEVICE>_<DATE>.json"
echo "════════════════════════════════════════════════════"
