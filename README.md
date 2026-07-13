# KV-Cache Tier Placement Simulator

Simulate KV-cache tiers across the storage hierarchy (GPU HBM to CPU DRAM to
Disk) during LLM inference, and compare cache-restore policies against a latency
SLO. Includes a CLI, a parameter sweep, and a web dashboard.

## What it models

Each cache **block** holds `tokens_per_block` tokens (16) and occupies
`bytes_per_block` (2 MB). A request needs `N` total blocks; `M` are cache
**misses** (must be recomputed on the GPU) and `N-M` are **hits** that live in
some tier. For each hit block the system decides whether to **restore** it (read
from its tier and move over the link) or **recompute** it (on the GPU).

Two throughputs drive everything:

- **Restore rate `X`** (blocks/s) - depends on *where* hit blocks live. Each
  tier has its own rate: `X_vram = hbm_bw/B`, `X_dram = 1/(B/dram_bw + B/link_bw)`,
  `X_disk = 1/(B/disk_bw + B/link_bw)`. A residency split (n_v, n_d, n_k) blends
  into a single effective rate `X_eff = (n_v+n_d+n_k) / (n_v/X_vram + n_d/X_dram + n_k/X_disk)`.
- **Recompute rate `Y`** (blocks/s) - GPU compute throughput for the model.

A **policy** decides only one thing: how many hit blocks `k` to reassign from
restore to recompute. All the hardware math (X, Y, allocation times, SLO checks)
is shared, so policies are interchangeable. The built-in ones:

| Policy (`name`) | `k` | Idea |
| --- | --- | --- |
| `default_policy` (Restore) | 0 | Restore all hits, recompute only misses |
| `all_compute` (Recompute) | all hits | Recompute everything |
| `performance_aware` (IO-aware) | balanced | Split so restore and recompute finish together, falling back to best-effort under SLO pressure |

An allocation **meets the SLO** when latency `≤ p95`, GPU utilization
`r·T_recompute ≤ 1`, and storage utilization `r·T_restore ≤ 1`.

### Adding a policy

Policies live in [`src/policies/`](src/policies/) - one module each. The shared
math is in `src/policies/core.py` (`PolicyContext`, `evaluate`) and the base
class in `src/policies/base.py`. To add one, drop in a module that exposes a
`POLICY` instance; the registry discovers it automatically and it appears in the
CLI, the API, and the dashboard (cards, charts, heatmaps) with no other changes:

```python
# src/policies/greedy_restore.py
from .base import PolicyBase
from .core import PolicyContext, PolicyDecision

class GreedyRestore(PolicyBase):
    name = "greedy_restore"        # JSON key / registry key
    label = "Greedy restore"       # shown in UI
    description = "Recompute only what the SLO forces."
    color = "#e6a817"              # UI color
    order = 40                     # display order

    def decide(self, ctx: PolicyContext) -> PolicyDecision:
        k = ...                    # your strategy, using ctx.allocation / ctx.max_violation
        return PolicyDecision(reassigned_hit_blocks=k, decision_mode="greedy")

POLICY = GreedyRestore()
```

## Layout

```
hardware/            Editable YAML catalogs (edit these to add/change hardware)
  gpus.yaml  dram.yaml  disks.yaml  links.yaml  models.yaml
src/
  types.py                                  Typed hardware/model dataclasses
  loader.py                                 Reads hardware/*.yaml into objects
  pool.py                                   GPUS/DRAMS/DISKS/LINKS/MODELS dicts
  engine.py                                 Placement-aware hardware model
  policies/                                 One module per policy (see "Adding a policy")
    core.py                                 Shared math: PolicyContext / evaluate
    base.py                                 PolicyBase class
    restore.py  recompute.py  io_aware.py   Built-in policies
    __init__.py                             Auto-discovery registry
simulator.py         CLI: compare policies for one point
sweep.py             CLI: P95 × request-rate sweep + figures
main.py              Standalone roofline / permutation plots
api/app.py           FastAPI backend (catalog / simulate / sweep + serves UI)
frontend/            Vanilla HTML/CSS/JS + Chart.js dashboard
```

Hardware lives in `hardware/*.yaml` and is loaded into frozen dataclasses by
`src/loader.py`. Add a GPU by appending an entry to `hardware/gpus.yaml` - it
appears everywhere (CLI choices, API catalog, dashboard dropdowns) automatically.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Web dashboard

```bash
uvicorn api.app:app --reload --port 8000
# open http://localhost:8000
```

Pick a GPU/DRAM/disk/link (or a stack preset), choose a model, set the workload
(blocks, hit ratio, request rate, P95), then control **tier placement** - either
*auto* (capacity waterfall: fill VRAM, then DRAM, then Disk) or *manual* sliders.
The **Single Point** tab shows per-tier residency, the effective restore/recompute
rates, and the three policies side by side (latency, SLO pass/fail, time
breakdown). The **SLO Sweep** tab renders pass/fail heatmaps over P95 × request
rate and a success-rate-vs-hit-ratio curve.

### API

- `GET  /api/catalog` - hardware/model catalogs + stack presets
- `POST /api/simulate` - `{hardware, workload, placement}` to per-tier + per-policy results
- `POST /api/sweep` - `{hardware, placement, total_blocks, hit_ratio, p95_values, request_rates, cache_ratio_sweep}` to grids + success rates

## CLI

The simulator reports `default_policy`, `all_compute`, and `performance_aware`
plus pairwise speedups.

```bash
# H200 stack: default restoration fails the SLO, IO-aware passes (PasstoFail case)
python simulator.py --stack h200 --total-blocks 50000 --miss-blocks 500 \
  --request-rate 0.18 --p95-seconds 10 --output h200.json
```

Custom stacks use the catalog keys directly:

```bash
python simulator.py --gpu A100 --dram DDR4 --disk NVMe980 --link PCIe5 \
  --gpu-count 2 --dram-count 8 --total-blocks 50000 --miss-blocks 2000 \
  --request-rate 0.05 --p95-seconds 15
```

## Sweep plot

```bash
python sweep.py --gpu H200 --output-figure-prefix h200_sweep_map \
  --output-results h200_sweep_results.json
```
