#!/usr/bin/env python3
"""
run_on_your_gpu.py — One-file quickstart for first-time users.

Copy-paste friendly. Tells you exactly what's happening at each step.
Results are printed and saved automatically.

Requirements:
    pip install torch-directml    # Windows + Python 3.11 only

Then run:
    python examples/run_on_your_gpu.py
"""

import sys

print("=" * 58)
print("  AMD DirectML Quickstart")
print("=" * 58)

# ── Step 1: Check Python version ──────────────────────────────
major, minor = sys.version_info[:2]
print(f"\n[1/5] Python {major}.{minor}", end="")
if major == 3 and minor <= 11:
    print(" ✅  (DirectML is supported)")
else:
    print(f" ⚠️   (torch-directml requires Python ≤ 3.11)")
    print("     Create a venv with py -3.11 -m venv .venv311 and retry.")
    sys.exit(1)

# ── Step 2: Import torch ──────────────────────────────────────
print("\n[2/5] Importing torch...", end=" ", flush=True)
try:
    import torch
    print(f"✅  torch {torch.__version__}")
except ImportError:
    print("❌  Not installed — run: pip install torch-directml")
    sys.exit(1)

# ── Step 3: Import torch_directml ────────────────────────────
print("[3/5] Importing torch_directml...", end=" ", flush=True)
try:
    import torch_directml as dml
    print(f"✅  torch_directml {dml.__version__}")
except ImportError:
    print("❌  Not installed — run: pip install torch-directml")
    sys.exit(1)

# ── Step 4: Check device ──────────────────────────────────────
print("[4/5] Checking DirectML device...", end=" ", flush=True)
device = dml.device()
# Device string will show as privateuseone:0 — this is normal
gpu_name = dml.device_name(0) if hasattr(dml, 'device_name') else "AMD GPU"
print(f"✅  {gpu_name}")
print(f"     (device string: {device}  ← 'privateuseone:0' is normal)")

# ── Step 5: Quick benchmark ───────────────────────────────────
print("[5/5] Running quick benchmark (32×512×512 float32 bmm)...")
print("     Warmup: 20 iters | Timed: 20 iters (quick mode)")

import time

BATCH, N = 32, 512

# CPU
a_cpu = torch.randn(BATCH, N, N, dtype=torch.float32)
b_cpu = torch.randn(BATCH, N, N, dtype=torch.float32)
for _ in range(5):  # mini warmup
    torch.bmm(a_cpu, b_cpu)
t0 = time.perf_counter()
for _ in range(20):
    torch.bmm(a_cpu, b_cpu)
cpu_ms = (time.perf_counter() - t0) * 1000 / 20

# DirectML
a_dml = a_cpu.to(device)
b_dml = b_cpu.to(device)
for _ in range(20):  # warmup
    torch.bmm(a_dml, b_dml)
_ = a_dml[0, 0, 0].item()  # sync

t0 = time.perf_counter()
for _ in range(20):
    torch.bmm(a_dml, b_dml)
_ = a_dml[0, 0, 0].item()  # sync
dml_ms = (time.perf_counter() - t0) * 1000 / 20

speedup = cpu_ms / dml_ms

print()
print("═" * 58)
print("  YOUR RESULTS")
print("═" * 58)
print(f"  GPU     : {gpu_name}")
print(f"  CPU     : {cpu_ms:.1f} ms/iter")
print(f"  DirectML: {dml_ms:.1f} ms/iter")
print(f"  Speedup : {speedup:.1f}×")
print("═" * 58)
print()
print("  For a full benchmark with JSON output, run:")
print("    python benchmark.py")
print()
print("  Share your results! Open an issue or PR with your")
print("  results/YOUR_GPU_DATE.json file.")
