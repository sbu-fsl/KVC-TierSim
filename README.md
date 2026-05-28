# Experiments

* H200 + DDR5 + NVMe + NVLink
* RTX A5000 + DDR5 + NVMe + NVLink

50 users with context size of 16000 tokens.

```bash
# shows how restoration used to impact the fraction of users rate and latency speedup
python simulator.py \
  --gpu H200 \
  --users 50 \
  --contexts 16000 \
  --requests 25 \
  --slo-seconds 120 \
  --default-policy restore_all \
  --output h200.json
```

```bash
# shows how restoration can improve the performance when storage is faster
python simulator.py \
  --gpu A5000 \
  --users 50 \
  --contexts 16000 \
  --requests 25 \
  --slo-seconds 1200 \
  --default-policy recompute_all \
  --output a5000.json
```
