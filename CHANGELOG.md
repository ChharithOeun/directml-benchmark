# Changelog — directml-benchmark

All notable changes are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) |
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

Auto-updated by `scripts/update_changelog.py` on every commit via git hook
and by GitHub Actions on every push to `main`.

> **Verification policy:** Every benchmark result must include hardware specs,
> software versions, and raw timing numbers from a real run.
> ✅ = measured on physical hardware | 🔬 = community-submitted, unverified by maintainer

---

## [Unreleased] — updated 2026-03-28

### Added
- cross-platform support — Windows/Linux/macOS + fix benchmark crash (`352b0e7`)


## [1.0.0] — 2026-03-27

### Added

- `benchmark.py` — main benchmark script (argparse CLI, JSON output, TFLOPS calculation)
- `examples/run_on_your_gpu.py` — one-file quickstart for first-time users
- `results/RX_5700_XT_DirectML_20260323.json` — first verified result ✅
  - Hardware: AMD Radeon RX 5700 XT (gfx1010), Windows 11 22H2
  - torch 2.4.1 + torch-directml 0.2.5 + Python 3.11.9
  - CPU: 250.4 ms/iter, 0.55 TFLOPS | DirectML: 6.2 ms/iter, 22.04 TFLOPS
  - **Speedup: 40.2×** (batch=32, N=512, float32, 100 warmup + 100 timed iters)
- `scripts/update_changelog.py` — local changelog auto-updater
- `.github/workflows/changelog.yml` — GitHub Actions auto-updater
- `.github/workflows/validate.yml` — CI validates JSON results format on every PR
- `CONTRIBUTING.md` — how to submit your own GPU results
- `LICENSE` — MIT

---

_Auto-updated by `scripts/update_changelog.py`_
