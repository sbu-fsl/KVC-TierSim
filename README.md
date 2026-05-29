# Experiments

* H200 + DDR5 + NVMe + NVLink
* RTX A5000 + DDR5 + NVMe + NVLink

Example restore request: $N=50000$ blocks with $M=5000$ misses, target request rate $r=0.02$ requests/s, and P95 latency target $p=120$ seconds.

```bash
# compare the performance-aware policy against the baseline restore-hits/recompute-misses policy
python simulator.py \
  --stack h200 \
  --total-blocks 50000 \
  --miss-blocks 5000 \
  --request-rate 0.02 \
  --p95-seconds 10 \
  --output h200.json
```

```bash
# compare a different stack under the same request profile
python simulator.py \
  --stack a5000 \
  --total-blocks 50000 \
  --miss-blocks 500 \
  --request-rate 0.02 \
  --p95-seconds 46 \
  --output a5000.json
```
