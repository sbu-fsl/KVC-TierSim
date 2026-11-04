"""Build request streams for every catalogued workload and profile them.

    python -m kvwl.profile                    # all workloads, both tokenizers
    python -m kvwl.profile --only LongBench   # one workload

Outputs (under dataset/):
  streams/<slug>__<tok>.jsonl.gz   request streams consumed by kvwl.replay
  workload_profiles.csv            per workload x tokenizer: input/output tokens, reuse
  workload_model_matrix.csv        per workload x model: tokens, KV bytes, working set
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time

import numpy as np
import yaml

from .catalog import load_workloads
from .models import load_models
from .streams import length_stats, reuse_stats, save_stream, load_stream
from .workloads import BUILDERS, Builder, FAMILY_TOKENIZER

from .paths import RAW, ROOT, STREAMS
OUT = os.path.join(ROOT, "data")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def catalog():
    """Workloads that have a stream builder (catalog/workloads.yaml)."""
    return load_workloads(buildable=True)


def stream_path(name, tok):
    return os.path.join(STREAMS, f"{slug(name)}__{tok}.jsonl.gz")


def build(name, spec, tok):
    b = Builder(tok, block_tokens=spec["args"].get("block_tokens", 16))
    BUILDERS[spec["builder"]](b, **spec["args"])
    return b.reqs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--reuse", action="store_true", help="reuse existing streams")
    ap.add_argument("--use-hf", action="store_true", help="use data/raw/hf/<name>.jsonl (make fetch-hf) when present")
    a = ap.parse_args(argv)
    os.makedirs(STREAMS, exist_ok=True)
    cat = catalog()
    if a.use_hf:
        hf = os.path.join(RAW, "hf")
        for name, spec in cat.items():
            if spec["fidelity"] == "published" and os.path.exists(os.path.join(hf, f"{name}.jsonl")):
                spec.update(builder="hf", fidelity="measured", args={"name": name}, notes="Hugging Face data via kvwl.hf_sources")
                for tok in ("llama3", "llama2"):
                    p = stream_path(name, tok)
                    if os.path.exists(p):
                        os.remove(p)
    rows = []
    for name, spec in cat.items():
        if a.only and name not in a.only:
            continue
        toks = ["native"] if spec["builder"].startswith("trace") else ["llama3", "llama2"]
        for tok in toks:
            p = stream_path(name, tok)
            t0 = time.time()
            if a.reuse and os.path.exists(p):
                meta, reqs = load_stream(p)
            else:
                reqs = build(name, spec, "llama3" if tok == "native" else tok)
                meta = {"workload": name, "tokenizer": tok, "fidelity": spec["fidelity"], "category": spec["category"],
                        "builder": spec["builder"], "args": spec["args"], "notes": spec.get("notes", "")}
                save_stream(p, reqs, meta)
            ins = [r.input_len for r in reqs]
            outs = [r.output_len for r in reqs]
            rs = reuse_stats(reqs)
            sessions = len({r.session for r in reqs})
            row = {"workload": name, "category": spec["category"], "fidelity": spec["fidelity"], "tokenizer": tok,
                   "requests": len(reqs), "sessions": sessions, "turns_per_session": len(reqs) / max(1, sessions),
                   **{f"in_{k}": v for k, v in length_stats(ins).items()},
                   **{f"out_{k}": v for k, v in length_stats(outs).items()},
                   "prefix_hit_ratio": rs["prefix_hit_ratio"], "nonprefix_extra_ratio": rs["nonprefix_extra_ratio"],
                   "unique_block_ratio": rs["unique_block_ratio"], "block_tokens": reqs[0].block_tokens if reqs else 16,
                   "notes": spec.get("notes", "")}
            rows.append(row)
            print(f"{name:28s} {tok:7s} n={len(reqs):5d} in_mean={row.get('in_mean', 0):9.0f} out_mean={row.get('out_mean', 0):7.0f} "
                  f"prefix_hit={rs['prefix_hit_ratio']:.3f} +chunk={rs['nonprefix_extra_ratio']:.3f} ({time.time() - t0:.1f}s)", flush=True)
    if not a.only:
        write_profiles(rows)
        write_matrix(rows)


def write_profiles(rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, "workload_profiles.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def write_matrix(rows, models=None):
    """Workload x model: what each pairing implies for the KV cache."""
    M = load_models()
    top = yaml.safe_load(open(os.path.join(ROOT, "survey", "out", "top_lists_final.yaml")))["models"]
    by = {(r["workload"], r["tokenizer"]): r for r in rows}
    out = []
    names = list(dict.fromkeys(r["workload"] for r in rows))
    cache = {}
    for wname in names:
        cache.clear()
        for mk in top:
            m = M[mk]
            tok = FAMILY_TOKENIZER[m.family]
            r = by.get((wname, tok)) or by.get((wname, "native"))
            if not r:
                continue
            p = stream_path(wname, r["tokenizer"])
            if p not in cache:
                cache[p] = load_stream(p)[1]
            reqs = cache[p]
            ins = np.array([q.input_len for q in reqs], dtype=float)
            outs = np.array([q.output_len for q in reqs], dtype=float)
            ctx = np.minimum(ins, m.ctx)
            final = np.minimum(ins + outs, m.ctx)
            kv_req = np.array([m.kv_bytes(int(t)) for t in final[:2000]])
            uniq_tokens = r["unique_block_ratio"] * ins.sum()
            bt = reqs[0].block_tokens
            out.append({
                "workload": wname, "model": mk, "attn": m.attn, "tokenizer": r["tokenizer"], "fidelity": r["fidelity"],
                "requests": len(reqs),
                "in_tokens_mean": float(ctx.mean()), "in_tokens_p99": float(np.percentile(ctx, 99)),
                "truncated_frac": float((ins > m.ctx).mean()),
                "out_tokens_mean": float(outs.mean()),
                "kv_bytes_per_token": m.kv_bytes_per_token(),
                "kv_per_request_mean_MB": float(kv_req.mean() / 1e6), "kv_per_request_p99_MB": float(np.percentile(kv_req, 99) / 1e6),
                "block_bytes_MB": m.block_bytes(bt) / 1e6, "block_tokens": bt,
                "working_set_GB": float(m.kv_bytes_per_token() * uniq_tokens / 1e9) if m.attn != "hybrid-swa" else float(m.block_bytes(1) * uniq_tokens / 1e9),
                "prefix_hit_ratio": r["prefix_hit_ratio"], "nonprefix_extra_ratio": r["nonprefix_extra_ratio"],
                "prefill_TFLOP_mean_nocache": float(np.mean([m.prefill_flops(int(t)) for t in ctx[:200]]) / 1e12),
                "weights_GB": m.weight_bytes() / 1e9,
            })
    keys = list(out[0].keys())
    with open(os.path.join(OUT, "workload_model_matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} workload x model rows")


if __name__ == "__main__":
    main()
