"""Build figures/kv_workload_atlas.html: an interactive explorer over all results.

    python -m kvwl.explorer
"""
import json
import os

import pandas as pd

from .analysis import CAT, TECH_LABEL, EXCLUDE
from .atlas import CATEGORIES, CAT_COLOR, ATTN_COLOR, workload_table, model_table
from .replay import Hardware, auto_gpu_count
from .models import load_models

from .paths import ROOT


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "results", "main.csv"))
    d = d[d.technique.notna() & ~d.workload.isin(EXCLUDE)]
    techs = [t for t in ["no_cache", "prefix_lru", "tiered_restore", "tiered_ioaware", "tiered_int4", "cacheblend"] if t in set(d.technique)]
    cells = {}
    for r in d.itertuples():
        cells.setdefault(f"{r.workload}\u0001{r.model}", {})[r.technique] = [round(r.ttft_mean, 5), round(r.service_mean, 5),
                                                                           round(r.recompute_frac, 3)]
    sw = pd.read_csv(os.path.join(ROOT, "survey", "out", "top_workloads.csv")).set_index("name")
    sm = pd.read_csv(os.path.join(ROOT, "survey", "out", "top_models.csv")).set_index("name")
    wt = workload_table()
    wt = wt[wt.workload.isin(set(d.workload))]
    W = [{"name": r.workload, "category": r.category, "fidelity": r.fidelity, "requests": int(r.requests),
          "in_mean": r.in_mean, "in_p99": r.in_p99, "out_mean": r.out_mean, "prefix": r.prefix_hit_ratio,
          "chunk": r.nonprefix_extra_ratio, "notes": r.notes if isinstance(r.notes, str) else "",
          "rank": int(sw.loc[r.workload, "rank"]) if r.workload in sw.index else 99,
          "papers": int(sw.loc[r.workload, "papers"]) if r.workload in sw.index else None} for r in wt.itertuples()]
    mt = model_table()
    Mm = load_models()
    M = [{"name": r.model, "attn": r.attn, "kv_kib": r.kv_kib, "block_mib": r.block_mib, "kv32k_gb": r.kv32k_gb,
          "params_b": r.params_b, "active_b": r.active_b, "ctx": int(r.ctx), "gpus": auto_gpu_count(Mm[r.model], Hardware()),
          "rank": int(sm.loc[r.model, "rank"]), "papers": int(sm.loc[r.model, "papers"])} for r in mt.itertuples()]
    win_colors = {"no_cache": "#c9c8c2"}
    for t, c in zip(techs[1:], CAT):
        win_colors[t] = c
    data = {"techniques": techs, "tech_label": {**TECH_LABEL, "no_cache": "No KV reuse"}, "categories": CATEGORIES,
            "cat_colors": CAT_COLOR, "attn_colors": ATTN_COLOR, "win_colors": win_colors, "workloads": W, "models": M,
            "cells": cells,
            "footnote": "* = workload synthesized from published statistics (Hugging Face unreachable at build time). "
                        "Simulated with KVC-TierSim + KVWL replay; accuracy effects of compression and CacheBlend are not modeled. "
                        f"{len(W)} workloads x {len(M)} models x {len(techs)} techniques = {len(d)} replays."}
    html = open(os.path.join(os.path.dirname(__file__), "explorer_template.html")).read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = os.path.join(ROOT, "figures", "kv_workload_atlas.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print(out, f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
