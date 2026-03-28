# directml-benchmark

**Reproducible float32 performance benchmarks for AMD GPUs via torch-directml.**

No estimated numbers. No invented results. Every row in the table below comes from a real hardware run with a timestamped JSON file you can inspect.

> Related repos:
> [torch-amd-setup](https://github.com/ChharithOeun/torch-amd-setup) — AMD GPU auto-detection for PyTorch |
> [jax-amd-gpu-setup](https://github.com/ChharithOeun/Chharbot/tree/main/jax-amd-gpu-setup) — JAX on AMD

---

## Results

Workload: `torch.bmm` — batched float32 matrix multiply, shape `(32 × 512 × 512)`.
100 warmup iterations + 100 timed iterations. Wall-clock `perf_counter`.

| GPU | CPU Baseline | DirectML | Speedup | Verified | Result file |
|-----|-------------|----------|---------|----------|-------------|
| AMD Radeon RX 5700 XT | 250.4 ms · 0.55 TFLOPS | 6.2 ms · 22.04 TFLOPS | **40.2×** | ✅ 2026-03-23 | [JSON](results/RX_5700_XT_DirectML_20260323.json) |

**Legend:**
- ✅ = measured on real hardware, JSON result file included
- 🔬 = community submitted, not yet independently confirmed

_Run `python benchmark.py` and open a PR to add your GPU →_ [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Quick Start

**Requirements:** Windows, Python 3.11 (hard ceiling — torch-directml is compiled against 3.11 ABI and is silently unavailable on 3.12+)

```bash
# 1. Create a Python 3.11 venv (if not already on 3.11)
py -3.11 -m venv .venv311
.venv311\Scripts\activate

# 2. Install — let DirectML pull torch 2.4.1 automatically
#    Do NOT pre-install torch first (causes version conflicts)
pip install torch-directml

# 3. Run the quickstart
python examples/run_on_your_gpu.py

# 4. Run the full benchmark (saves results/DirectML_YYYYMMDD.json)
python benchmark.py
```

**CPU-only mode** (no GPU required, any OS):
```bash
pip install torch
python benchmark.py --cpu-only
```

---

## CLI Options

```
python benchmark.py [options]

  --batch N      Batch size (default: 32)
  --size N       Matrix N×N size (default: 512)
  --warmup N     Warmup iterations (default: 100)
  --iters N      Timed iterations (default: 100)
  --cpu-only     Skip GPU benchmark
  --output PATH  Custom output JSON path
```

---

## How the Benchmark Works

```
Shape:    A = (batch × N × N), B = (batch × N × N), float32
Op:       torch.bmm(A, B)
FLOPs:    2 × batch × N³ per iteration  (multiply + add)
Sync:     a[0,0,0].item() after GPU loop (DirectML has no .synchronize())
Timing:   perf_counter — wall clock, not torch.cuda.Event
```

TFLOPS = `(2 × batch × N³ × iters) / elapsed_seconds / 1e12`

---

## Known Limitations

**float32 only** — DirectML float16 is unreliable on most cards. This benchmark intentionally uses float32 and does not attempt float16.

**Python 3.11 ceiling** — `torch-directml` is compiled against the Python 3.11 ABI. It cannot be installed on 3.12+ (pip will appear to succeed but the import fails or produces wrong results). Always use `py -3.11`.

**No .synchronize()** — DirectML has no GPU synchronization call equivalent to `torch.cuda.synchronize()`. The sync workaround used here is a small `.item()` read that forces the CPU to wait for the GPU queue to drain. This adds a small overhead (~0.1 ms) but ensures correctness.

**privateuseone:0 device string** — When you call `str(device)` on a DirectML device, PyTorch shows `privateuseone:0`. This is normal — it's PyTorch's internal representation of custom backend devices. The GPU is working correctly.

---

## Troubleshooting

**Import error after pip install** — You're on Python 3.12+. Check with `python --version`.

**Benchmark runs but GPU is slow** — Check Task Manager → GPU → DirectX Compute. If it shows 0%, the tensors may have landed on CPU. Make sure you're calling `.to(dml.device())` with the device *object*, not `"privateuseone:0"` the string.

**pip install torch-directml fails** — Don't pre-install torch. Run `pip install torch-directml` from a clean venv and let it pull `torch==2.4.1` automatically.

For more setup help: [torch-amd-setup troubleshooting](https://github.com/ChharithOeun/torch-amd-setup/blob/main/docs/troubleshooting.md)

---

## Verification Policy

> **No fabricated numbers. No guessing. No estimated timings.**
>
> Every benchmark in this repository was run on physical hardware.
> Timing comes from `time.perf_counter` with warmup. Results include
> the exact software stack (torch version, directml version, Python version).
>
> If a number can't be verified, it doesn't appear here.

See [CHANGELOG.md](CHANGELOG.md) for a full history of what was added and when.

---

## License

MIT — see [LICENSE](LICENSE)
