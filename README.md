# Experiments

* H200 + DDR5 + NVMe + NVLink
* RTX A5000 + DDR5 + NVMe + NVLink

## (1) Pass 2 Fail

```bash
# policy helps meeting SLOs by avoid aggresive restore
python simulator.py \
  --stack h200 \
  --total-blocks 50000 \
  --miss-blocks 500 \
  --request-rate 0.02 \
  --p95-seconds 10 \
  --output h200.json
```

```bash
# policy helps meeting SLOs by keeping restore
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
# policy passes anyways but increases speed by 8.8
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
