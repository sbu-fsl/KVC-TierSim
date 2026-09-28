# Results

Curated simulator outputs behind the HotStorage '26 paper *LLM KV-cache: To
Restore or To Recompute, That Is the Question* (see the citation in the
[project README](../README.md#citing-this-work)). Nothing here is imported by
the simulator or the API; these are saved outputs kept so the figures can be
checked and regenerated. Every file is named `<gpu>_<stack>_<experiment>` so it
stays self-describing when downloaded on its own.

Please cite the paper if you use any of these results.

## Hardware stacks

All sweeps use one GPU, four DDR5 modules, and NVLink. They differ only in the
disk that holds the cached blocks:

| Label      | GPU x DRAM x disk x link                                                            |
| :-         | :-                                                                                  |
| `nvme`     | 1 GPU, 4 x DDR5 (85 GB/s, 32 GB each), NVMe SSD (2 GB/s, 2 TB), NVLink (600 GB/s)   |
| `fast_ssd` | 1 GPU, 4 x DDR5 (85 GB/s, 32 GB each), NVMe T700 (5 GB/s, 4 TB), NVLink (600 GB/s) |

The catalog keys are `DDR5`, `NVMe`, `NVMeT700`, and `NVLink` in
[hardware/](../hardware/). The sweeps run the single-tier model in
`simulator.py`, so every hit block lives on the disk.

## `slo_sweeps/`

P95 x request-rate sweeps produced by `sweep.py`. Each run covers the same grid:

| Parameter        | Values                                                     |
| :-               | :-                                                         |
| P95 target       | 5 s to 60 s in steps of 5 s (12 values)                    |
| Request rate     | 0.01 req/s to 0.30 req/s in steps of 0.01 (30 values)      |
| Hit ratio        | 50 % for the heatmaps; 0 % to 100 % in steps of 10 % for the success-rate curve |
| Effective SLO    | `q = min(p95_seconds, 1 / request_rate)`                   |
| Model            | Llama 3 8B shape, 16 tokens per block, 2 MB per block, eta 0.5 |

The context size scales with the GPU so that each stack is stressed in its own
range:

| GPU       | Total blocks | `nvme` | `fast_ssd` |
| :-        | :-           | :-     | :-         |
| `a100`    | 10,000       | JSON   |            |
| `a5000`   | 20,000       | JSON   |            |
| `h100`    | 20,000       | JSON + figures | JSON + figures |
| `h200`    | 30,000       | JSON + figures | JSON + figures |
| `rtx6000` | 20,000       | JSON   |            |
| `v100`    | 20,000       | JSON   |            |

Files per run:

| File                                    | Contents                                                                |
| :-                                      | :-                                                                      |
| `<gpu>_<stack>_sweep_results.json`      | Hardware, grid, per-policy success rates, and all 360 grid points with the three policies' allocations, SLO checks, and pairwise speedups |
| `<gpu>_<stack>_sweep_map_default_policy.png`    | Pass/fail heatmap for the Restore policy                        |
| `<gpu>_<stack>_sweep_map_all_compute.png`       | Pass/fail heatmap for the Recompute policy                      |
| `<gpu>_<stack>_sweep_map_performance_aware.png` | Pass/fail heatmap for the IO-aware policy                       |
| `<gpu>_<stack>_sweep_map_success_rate.pdf`      | SLO success rate vs. cache hit ratio, one line per policy       |

Heatmap colors: red fails both the latency and the throughput check, orange
passes throughput only, blue passes latency only, green passes both.

Regenerate a run (the H200 fast-SSD case shown; change the GPU, disk, and
block count per the tables above):

```bash
python sweep.py --stack h200 --gpu H200 --disk NVMeT700 --total-blocks 30000 \
  --output-figure-prefix results/slo_sweeps/fast_ssd/h200_fastssd_sweep_map \
  --output-results results/slo_sweeps/fast_ssd/h200_fastssd_sweep_results.json
```

## `restore_vs_recompute/`

One figure per GPU from `main.py`, `tiers_configuration_<gpu>.pdf`. The x axis
is the context length (16K to 2M tokens, with the KV volume in GB on the top
axis). The violins are the distribution of the time to restore that context
over 1,764 storage deployments (12 DRAM parts x 21 disks x 7 links); the red bar
is the median. The dotted line is the time for that GPU to prefill the same
context from scratch with no KV cache. Where the line sits above the violins,
restoring wins; where it dips below, recomputing wins.

```bash
python main.py
```

## `single_point/`

Small single-point outputs of the placement-aware engine (`src/engine.simulate`),
one per canonical scenario. Each file embeds the exact `hardware`, `workload`,
and `placement` it was produced from, per-tier residency, and a `results` map
keyed by policy name.

| File                         | Stack | Placement        | Story                                                                                       |
| :-                           | :-    | :-               | :-                                                                                          |
| `h200_disk_pass2fail.json`   | H200  | manual, all disk | Restore fails the SLO (49.7 s); the IO-aware policy rebalances to recompute and passes (5.7 s). |
| `h200_auto_pass_only.json`   | H200  | auto waterfall   | Hits fit in HBM, so every policy passes easily.                                             |
| `a5000_disk_fail_only.json`  | A5000 | manual, all disk | Slow GPU plus slow disk: no policy meets the SLO.                                           |

Regenerate one through the dashboard API by posting the file's own `hardware`,
`workload`, and `placement` blocks:

```bash
uvicorn api.app:app --port 8000 &
curl -s http://localhost:8000/api/simulate -H 'Content-Type: application/json' -d '{
  "hardware":  {"gpu_key": "H200", "dram_key": "DDR5", "disk_key": "NVMe", "link_key": "NVLink"},
  "workload":  {"total_blocks": 50000, "miss_blocks": 500, "request_rate": 0.02, "p95_seconds": 10},
  "placement": {"mode": "manual", "vram": 0, "dram": 0, "disk": 1}
}' > results/single_point/h200_disk_pass2fail.json
```

## `legacy/`

Early per-scenario `simulator.py` outputs in the two-policy format that predates
the `all_compute` policy and the three-way speedups. Kept for provenance only;
the current format is the one in `single_point/` and `slo_sweeps/`.

| Directory         | Scenario                                                |
| :-                | :-                                                      |
| `01-fail-2-pass/` | Restore fails the SLO, the IO-aware policy passes        |
| `02-pass-only/`   | Every policy passes                                     |
| `03-fail-only/`   | No policy passes                                        |
