# KVC-TierSim: LLM Inference KV-Cache Tier Simulator

KVC-TierSim simulates LLM inference KV-cache tiers across the storage hierarchy
(GPU HBM, DRAM, disk, and the transfer links between them) and compares
cache-restore policies against both end-to-end latency and throughput SLOs. It
ships a CLI, a parameter sweep, a scheduler-overhead benchmark, a set of
publication figures, and a web dashboard.

This repository accompanies our HotStorage '26 paper:

> Amirhossein Najafizadeh, Vasily Tarasov, Alex Merenstein, Yue Zhu, and Erez
> Zadok. 2026. **LLM KV-cache: To Restore or To Recompute, That Is the
> Question.** In *Proceedings of the 18th ACM Workshop on Hot Topics in Storage
> and File Systems (HotStorage '26)*. Association for Computing Machinery, New
> York, NY, USA, 8-15. <https://doi.org/10.1145/3837053.3837336>

## Citing this work

If you use this simulator, its hardware catalogs, or any of the results in
[results/](results/), please cite the paper:

```bibtex
@inproceedings{10.1145/3837053.3837336,
  author    = {Najafizadeh, Amirhossein and Tarasov, Vasily and Merenstein, Alex and Zhu, Yue and Zadok, Erez},
  title     = {LLM KV-cache: To Restore or To Recompute, That Is the Question},
  year      = {2026},
  isbn      = {9798400729027},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3837053.3837336},
  doi       = {10.1145/3837053.3837336},
  booktitle = {Proceedings of the 18th ACM Workshop on Hot Topics in Storage and File Systems},
  pages     = {8–15},
  numpages  = {8}
}
```

## What it models

Each cache **block** holds `tokens_per_block` tokens (16) and occupies
`bytes_per_block` (2 MB). A request needs `N` total blocks; `M` are cache
**misses** that must be recomputed on the GPU, and `N-M` are **hits** that live
in some tier. For each hit block the system decides whether to **restore** it
(read it from its tier and move it over the link) or **recompute** it on the GPU.

Two throughputs drive everything:

- **Restore rate `X`** (blocks/s) depends on *where* hit blocks live. Each
  tier has its own rate: `X_vram = hbm_bw/B`, `X_dram = 1/(B/dram_bw + B/link_bw)`,
  `X_disk = 1/(B/disk_bw + B/link_bw)`. A residency split (n_v, n_d, n_k) blends
  into a single effective rate
  `X_eff = (n_v+n_d+n_k) / (n_v/X_vram + n_d/X_dram + n_k/X_disk)`.
- **Recompute rate `Y`** (blocks/s) is the GPU compute throughput for the model.

A **policy** decides only one thing: how many hit blocks `k` to reassign from
restore to recompute. All the hardware math (X, Y, allocation times, SLO checks)
is shared, so policies are interchangeable. The built-in ones:

| Policy (`name`)                | `k`      | Idea                                                                                           |
| :-                             | :-       | :-                                                                                             |
| `default_policy` (Restore)     | 0        | Restore all hits, recompute only misses                                                        |
| `all_compute` (Recompute)      | all hits | Recompute everything                                                                           |
| `performance_aware` (IO-aware) | balanced | Split so restore and recompute finish together, falling back to best-effort under SLO pressure |

An allocation **meets the SLO** when latency `≤ p95`, GPU utilization
`r·T_recompute ≤ 1`, and storage utilization `r·T_restore ≤ 1`.

### Adding a policy

Policies live in [src/policies/](src/policies/), one module each. The shared
math is in `src/policies/core.py` (`PolicyContext`, `evaluate`) and the base
class in `src/policies/base.py`. To add one, drop in a module that exposes a
`POLICY` instance. The registry discovers it automatically and it appears in the
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
hardware/            Editable YAML catalogs (edit these to add or change hardware)
  gpus.yaml  dram.yaml  disks.yaml  links.yaml  models.yaml
src/
  types.py                                  Typed hardware and model dataclasses
  loader.py                                 Reads hardware/*.yaml into objects
  pool.py                                   GPUS / DRAMS / DISKS / LINKS / MODELS dicts
  engine.py                                 Placement-aware hardware model
  policies/                                 One module per policy (see "Adding a policy")
    core.py                                 Shared math: PolicyContext / evaluate
    base.py                                 PolicyBase class
    restore.py  recompute.py  io_aware.py   Built-in policies
    __init__.py                             Auto-discovery registry
simulator.py         CLI: compare policies for one point
sweep.py             CLI: P95 x request-rate sweep plus figures
overhead.py          CLI: scheduler-overhead sweep (micro to macro requests)
main.py              Publication figures: restore-vs-recompute violins, roofline, permutations
api/app.py           FastAPI backend (catalog / simulate / sweep, also serves the UI)
frontend/            Vanilla HTML/CSS/JS plus Chart.js dashboard
results/             Curated outputs behind the paper (see results/README.md)
```

Hardware lives in `hardware/*.yaml` and is loaded into frozen dataclasses by
`src/loader.py`. Add a GPU by appending an entry to `hardware/gpus.yaml` and it
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

Pick a GPU, DRAM, disk, and link (or a stack preset), choose a model, set the
workload (blocks, hit ratio, request rate, P95), then control **tier placement**,
either *auto* (capacity waterfall: fill VRAM, then DRAM, then disk) or *manual*
sliders. The **Single Point** tab shows per-tier residency, the effective restore
and recompute rates, and the three policies side by side (latency, SLO pass or
fail, time breakdown). The **SLO Sweep** tab renders pass/fail heatmaps over P95
x request rate and a success-rate-vs-hit-ratio curve.

## CLI

The simulator reports `default_policy`, `all_compute`, and `performance_aware`
plus pairwise speedups.

```bash
# H200 stack: default restoration fails the SLO, the IO-aware policy passes
python simulator.py --stack h200 --total-blocks 50000 --miss-blocks 500 \
  --request-rate 0.18 --p95-seconds 10 --output h200.json
```

Custom stacks use the catalog keys directly:

```bash
python simulator.py --gpu A100 --dram DDR4 --disk NVMe980 --link PCIe5 \
  --gpu-count 2 --dram-count 8 --total-blocks 50000 --miss-blocks 2000 \
  --request-rate 0.05 --p95-seconds 15
```

## SLO sweep

`sweep.py` walks a P95 x request-rate grid for one stack, writes a JSON results
file, one pass/fail heatmap per policy, and a success-rate-vs-hit-ratio curve.
This is the command behind the H200 NVMe run in `results/slo_sweeps/nvme/`:

```bash
python sweep.py --stack h200 --gpu H200 --disk NVMe --total-blocks 30000 \
  --output-figure-prefix h200_nvme_sweep_map \
  --output-results h200_nvme_sweep_results.json
```

Swap the disk for the faster T700 (`NVMeT700`) to reproduce the
`results/slo_sweeps/fast_ssd/` runs.

## Restore vs. recompute figures

`main.py` draws, for each GPU in the catalog, the no-cache prefill time of that
GPU (a dotted line, quadratic in context length) against the distribution of
restore times over 1,764 storage deployments (12 DRAM parts x 21 disks x 7 links,
drawn as violins). Running it with no arguments regenerates the six PDFs in
`results/restore_vs_recompute/`:

```bash
python main.py
```

## Scheduler overhead

`sweep.py` measures what a decision *buys*; `overhead.py` measures what it
*costs*. It times `decide()` itself over a request stream that grows from micro
(one request) to macro (a million), and draws one line per registered policy:

```bash
python overhead.py --gpu A100 --output-figure-prefix a100_overhead \
  --output-results a100_overhead_results.json
```

Requests come from a seeded pool of randomized workload **combinations** (block
counts, hit ratios, request rates, and P95 targets spanning decades), so a
search-based policy is timed across both its fast and its fallback path in
realistic proportions rather than only its cheapest branch. Output is a JSON
results file plus four vector PDFs:

| Figure          | Y axis                    | Reads as                                                   |
| :-              | :-                        | :-                                                         |
| `_total`        | scheduler time (s)        | CPU spent scheduling `R` requests                          |
| `_per_request`  | time per decision (µs)    | amortized cost of one decision                             |
| `_relative`     | % of modeled request time | the real-world number: overhead as a share of service time |
| `_distribution` | time per decision (µs)    | spread over all combinations (box = median, ◇ = mean)      |

### The summary

Alongside the request sweep, every combination in the pool is timed on its own
(the combination reps and rounds options control how many decisions and rounds,
with the median kept). That gives a distribution per policy instead of one
blended number, aggregated into `summary` in the JSON and printed at the end of
a run:

```
Per-combination summary: 512 workload combinations x 1,000 reps x 3 rounds = 1,536,000 timed decisions per policy
Baseline (no decision logic): Restore, Recompute

  Policy               median     mean      p95      max   +median    +mean   x base      % req
  =============================================================================================
  IO-aware policy       6.494    5.028    7.425    7.572     6.261    4.798    21.8x   6.13e-05
  Restore *             0.208    0.208    0.214    0.251    -0.023   -0.023     0.9x  -4.36e-07
  Recompute *           0.254    0.253    0.263    0.277     0.023    0.023     1.1x   2.47e-07
```

`default_policy` and `all_compute` return a fixed `k`, so what they cost is
dispatch and loop overhead, not decision-making. Their mean is the **baseline**
(selectable with the baseline policies option), and the `+median` and `+mean`
columns are each policy's overhead *over* that floor, a subtraction that cancels
the measurement harness out of the number. In the run above, across 512
combinations and 1.5 M timed decisions, the IO-aware policy added
**4.80 µs mean / 6.26 µs median** per scheduling decision over the no-logic
baseline (21.8x), which is about 6e-05 % of the modeled end-to-end request time.

Mean below median is not a typo. The IO-aware policy is bimodal, and the JSON's
`by_decision_mode` (also printed) says why: its `balanced` branch closed in 155
of 512 combinations at 0.88 µs, while the `best_effort` candidate scan ran in the
other 357 at 6.59 µs.

The absolute times are a property of the *host CPU* running the scheduler, not of
the simulated hardware; the GPU, DRAM, disk, and link choice only moves the
`% of request time` numbers, by changing the modeled service time the overhead is
divided by. Every registered policy is picked up automatically, drawn in its own
`color`, and summarized, so a new policy module appears here with no changes.

Full runs take about 40 s. Cap the request count (for example at 10,000) for a
quick pass, and change the pool size or seed to change the workload mix.

## Results

[results/](results/) holds the curated outputs behind the paper, grouped by
experiment and labeled by hardware stack. See [results/README.md](results/README.md)
for what each file contains and the exact command that regenerates it.

| Directory                      | Contents                                                              |
| :-                             | :-                                                                    |
| `results/slo_sweeps/nvme/`     | P95 x request-rate sweeps on DDR5 + NVMe SSD (2 GB/s) for six GPUs    |
| `results/slo_sweeps/fast_ssd/` | The same sweeps on the NVMe T700 (5 GB/s) for the H100 and H200        |
| `results/restore_vs_recompute/`| No-cache prefill vs. tiered restore, one figure per GPU               |
| `results/single_point/`        | Three single-point runs illustrating fail-to-pass, pass-only, fail-only |
| `results/legacy/`              | Early two-policy outputs kept for provenance                          |

## License and copyright

Copyright 2026 File systems and Storage Lab (FSL), Stony Brook University.
All rights reserved.

The source code is distributed under the Apache License, Version 2.0; see
[LICENSE](LICENSE) for the full terms. Please cite the paper above when you use
the simulator or its results.
