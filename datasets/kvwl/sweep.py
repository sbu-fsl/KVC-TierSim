"""Run workload x model x technique grids through kvwl.replay.

    python -m kvwl.sweep --grid main          # all top-50 (+supplementary) x top-50 models, core techniques
    python -m kvwl.sweep --grid sensitivity   # representative subset x input-dimension sweeps

Results are appended row by row (resumable) to dataset/results/<grid>.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import yaml

from .models import load_models
from .profile import catalog, stream_path, ROOT
from .replay import Hardware, RunConfig, TECHNIQUES, nocache_capacity, replay
from .streams import load_stream
from .workloads import FAMILY_TOKENIZER

RES = os.path.join(ROOT, "data", "results")
from .catalog import category  # noqa: E402,F401  (re-exported)

CORE = ["no_cache", "prefix_lru", "tiered_restore", "tiered_ioaware", "tiered_int4", "cacheblend"]
EXTRA = ["prefix_lfu", "prefix_fifo", "tiered_fp8", "snapkv_2k"]

# representative subset for the sensitivity grid (spans categories / attention types)
SENS_WORKLOADS = ["ShareGPT", "LongBench", "GSM8K", "Natural Questions", "UltraChat", "SWE-bench", "L-Eval",
                  "Mooncake conversation", "Qwen-Bailian to-C (A)", "RULER"]
SENS_MODELS = ["Llama-2-7B", "Llama-3.1-8B", "Mistral-7B", "Qwen2.5-7B", "Llama-3.1-70B", "Mixtral-8x7B",
               "Qwen3-30B-A3B", "DeepSeek-V2", "Gemma-3-12B", "OPT-13B"]
SENS_AXES = {
    "load": [("load", v) for v in (0.3, 0.5, 0.7, 0.9, 1.1)],
    "gpu": [("gpu", v) for v in ("A100", "H100", "H200")],
    "disk": [("disk", v) for v in ("SATA", "NVMe", "NVMeT700")],
}


def _stream_for(wname, model):
    tok = FAMILY_TOKENIZER[model.family]
    p = stream_path(wname, tok)
    if not os.path.exists(p):
        p = stream_path(wname, "native")
    return load_stream(p)


def run_pair(job):
    wname, mkey, techs, hw_kw, rc_kw, tag = job
    M = load_models()
    m = M[mkey]
    meta, reqs = _stream_for(wname, m)
    hw = Hardware(**hw_kw)
    rc = RunConfig(**rc_kw)
    reqs = reqs[: rc.max_requests]
    has_chunks = any(c != -1 for r in reqs for c in r.chunk_ids)
    rows = []
    try:
        cap = nocache_capacity(m, hw, reqs)
    except Exception as e:  # e.g. model does not fit
        return [{"workload": wname, "model": mkey, "error": repr(e)}]
    base = None
    for t in techs:
        if t == "cacheblend" and not has_chunks:
            continue
        t0 = time.time()
        s = replay(reqs, m, hw, TECHNIQUES[t], rc, capacity=cap)
        if t == "no_cache":
            base = s
        row = {"tag": tag, "workload": wname, "category": category(wname), "fidelity": meta.get("fidelity", ""),
               "model": mkey, "attn": m.attn, "params_b": m.params_b, "active_b": m.active_b,
               "kv_bytes_per_token": m.kv_bytes_per_token(), **{f"hw_{k}": v for k, v in hw_kw.items()},
               **{f"rc_{k}": v for k, v in rc_kw.items()}, **s, "wall_s": time.time() - t0}
        if base and t != "no_cache":
            row["speedup_ttft_vs_nocache"] = base["ttft_mean"] / max(s["ttft_mean"], 1e-9)
            row["speedup_service_vs_nocache"] = base["service_mean"] / max(s["service_mean"], 1e-9)
        rows.append(row)
    return rows


def _append(path, rows, header_cache={}):
    rows = [r for r in rows if r]
    if not rows:
        return
    exists = os.path.exists(path)
    keys = header_cache.get(path)
    if keys is None:
        if exists:
            with open(path) as f:
                keys = next(csv.reader(f))
        else:
            keys = list(dict.fromkeys(k for r in rows for k in r))
        header_cache[path] = keys
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


def _done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["tag"], r["workload"], r["model"]) for r in csv.DictReader(f)}


def jobs_main(a):
    cat = catalog()
    top = yaml.safe_load(open(os.path.join(ROOT, "survey", "out", "top_lists_final.yaml")))
    wls = a.workloads or list(cat)
    mods = a.models or top["models"]
    rc = {"load": a.load, "max_requests": a.max_requests, "ttft_slo": a.slo}
    return [(w, m, a.techniques or CORE, {}, rc, "main") for w in wls for m in mods]


def jobs_sens(a):
    out = []
    for axis, vals in SENS_AXES.items():
        for key, v in vals:
            hw, rc = {}, {"load": 0.7, "max_requests": a.max_requests, "ttft_slo": 2.0}
            if key in ("load", "ttft_slo"):
                rc[key] = v
            else:
                hw[key] = v
            for w in a.workloads or SENS_WORKLOADS:
                for m in a.models or SENS_MODELS:
                    out.append((w, m, CORE, hw, rc, f"sens:{axis}={v}"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["main", "sensitivity"], default="main")
    ap.add_argument("--workloads", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--techniques", nargs="*")
    ap.add_argument("--load", type=float, default=0.7)
    ap.add_argument("--slo", type=float, default=2.0)
    ap.add_argument("--max-requests", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"{a.grid}.csv")
    jobs = jobs_main(a) if a.grid == "main" else jobs_sens(a)
    done = _done(path)
    jobs = [j for j in jobs if (j[5], j[0], j[1]) not in done]
    print(f"{len(jobs)} pending jobs -> {path}", flush=True)
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_pair, jobs, chunksize=1), 1):
            _append(path, rows)
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
