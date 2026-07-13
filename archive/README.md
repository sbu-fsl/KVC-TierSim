# Archive

Reference JSON artifacts. Nothing here is imported by the simulator or API —
these are saved outputs kept for comparison and documentation.

## `examples/`

Small, current-schema outputs from the placement-aware engine
(`src/engine.simulate`), one per canonical scenario. Each shows per-tier
residency, a `policies` metadata list, and a `results` map keyed by policy name
(`default_policy`, `all_compute`, `performance_aware` / IO-aware):

| File | Stack | Placement | Story |
| --- | --- | --- | --- |
| `h200_disk_pass2fail.json` | H200 | manual, all-disk | Restore fails the SLO (49.7 s); the IO-aware policy rebalances to recompute and passes (5.7 s). |
| `h200_auto_pass_only.json` | H200 | auto waterfall | Hits fit in HBM, so every policy passes easily. |
| `a5000_disk_fail_only.json` | A5000 | manual, all-disk | Slow GPU + slow disk: no policy meets the SLO. |

Regenerate them with the snippet in the project README, or via `POST /api/simulate`.

## `sweep_results/`

Large `*_sweep_results.json` outputs from `sweep.py` (one per GPU), each a full
P95 × request-rate grid plus success-rate curves. These predate some schema
changes — treat them as historical snapshots.

## `dataset/`

Early per-scenario `simulator.py` outputs (two-policy format, before
`all_compute` and the three-way speedups were added). Kept for provenance.
