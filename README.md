# Experiments

* H200 + DDR5 + NVMe + NVLink
* RTX A5000 + DDR5 + NVMe + NVLink

The simulator now reports three policies in each result row: `default_policy`, `all_compute`, and `performance_aware`.
It also includes three pairwise speedup fields: `speedup_default_vs_all_compute`, `speedup_all_compute_vs_performance_aware`, and `speedup_performance_aware_vs_default`.

## (1) Pass 2 Fail

```bash
# default restore behavior versus performance-aware balancing
python simulator.py \
  --stack h200 \
  --total-blocks 50000 \
  --miss-blocks 500 \
  --request-rate 0.18 \
  --p95-seconds 10 \
  --output h200.json
```

```bash
# default restore behavior versus performance-aware balancing
python simulator.py \
  --stack a5000 \
  --total-blocks 50000 \
  --miss-blocks 500 \
  --request-rate 0.02 \
  --p95-seconds 46 \
  --output a5000.json
```

## (2) Pass Only

```bash
# default restore behavior, all-compute, and performance-aware compare cleanly
python simulator.py \
  --stack h200 \
  --total-blocks 50000 \
  --miss-blocks 0 \
  --request-rate 0.02 \
  --p95-seconds 10 \
  --output h200.json
```

## (3) Fail Only

```bash
# no matter what, SLO always fail
python simulator.py \
  --stack a5000 \
  --total-blocks 50000 \
  --miss-blocks 500 \
  --request-rate 0.02 \
  --p95-seconds 10 \
  --output a5000.json
```
