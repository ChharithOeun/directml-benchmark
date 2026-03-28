# Contributing — Submit Your GPU Results

The goal of this repo is a **verified, community-sourced** benchmark table
covering as many AMD GPUs as possible. Every result must be reproducible.

---

## How to submit your results

**Step 1 — Run the benchmark on your machine**

```bash
# Windows (Python 3.11 only for DirectML)
pip install torch-directml
python benchmark.py

# CPU-only (any OS, any Python)
python benchmark.py --cpu-only
```

This creates `results/DirectML_YYYYMMDD.json` (or `CPU_YYYYMMDD.json`).

**Step 2 — Rename the file to include your GPU**

```
results/RX_7900_XTX_DirectML_20260401.json
results/RX_6800_XT_DirectML_20260401.json
results/Vega_64_DirectML_20260401.json
```

**Step 3 — Open a Pull Request**

Your PR will automatically be validated by GitHub Actions to ensure the
JSON has all required fields and real (non-null) timing numbers.

---

## What makes a valid result

| Field | Requirement |
|-------|-------------|
| `date` | ISO format (YYYY-MM-DD) — the day you ran it |
| `environment.python` | Your Python version |
| `environment.torch` | Your torch version |
| `results.cpu.elapsed_ms` | Real number — must not be null |
| `hardware.gpu` | GPU name (e.g. "AMD Radeon RX 7900 XTX") |
| `verification` | Short note that confirms it was a real run |

---

## Verification policy

> **No fabricated numbers. No estimated timings. No guessing.**
>
> If you didn't run the benchmark, don't submit a result.
> Benchmark numbers marked ✅ in the README were measured on real hardware.
> Community results are marked 🔬 until independently confirmed.

---

## Troubleshooting

**"torch-directml not found"** — You're likely on Python 3.12+. DirectML requires ≤ 3.11.
Create a venv: `py -3.11 -m venv .venv311 && .venv311\Scripts\activate`

**"privateuseone:0" device string** — This is normal. DirectML registers as a custom backend.

**torch-directml install fails** — Install DirectML *first* without pre-installing torch:
`pip install torch-directml` (it pulls torch 2.4.1 automatically).

See [torch-amd-setup](https://github.com/ChharithOeun/torch-amd-setup) for full AMD GPU setup docs.
