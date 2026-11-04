"""Quantify how much workload/model choice moves KV-cache technique results.

    python -m kvwl.analysis            # reads dataset/results/main.csv (+ sensitivity.csv)

Writes dataset/analysis/*.csv and figures/*.pdf|png:
  technique_spread.csv     per technique: distribution of TTFT speedup over all (workload, model) cells
  variance_shares.csv      two-way decomposition of log-speedup: workload vs model vs interaction
  winners.csv              best technique per cell, and how often each technique wins
  pairwise_flips.csv       for technique pairs, share of cells where A beats B vs B beats A
  typical_paper.csv        where common paper setups (e.g. ShareGPT + Llama-2-7B) fall in the distribution
  feature_correlation.csv  Spearman correlation of speedup with workload/model features
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from .paths import ROOT
RES = os.path.join(ROOT, "data", "results")
OUT = os.path.join(ROOT, "data", "analysis")
FIG = os.path.join(ROOT, "figures")

# reference palette (dataviz skill, validated categorical order; sequential blue; diverging blue<->orange)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# same 23,608 requests as "Mooncake tool&agent" (identical lengths/hashes, different timestamps) -> counted once
EXCLUDE = {"Mooncake arXiv/Kimi"}

TECH_ORDER = ["prefix_lru", "prefix_lfu", "prefix_fifo", "tiered_restore", "tiered_ioaware", "tiered_fp8", "tiered_int4",
              "cacheblend", "snapkv_2k"]
TECH_LABEL = {"prefix_lru": "Prefix cache (HBM, LRU)", "prefix_lfu": "Prefix cache (HBM, LFU)",
              "prefix_fifo": "Prefix cache (HBM, FIFO)", "tiered_restore": "Tiered offload, restore-all",
              "tiered_ioaware": "Tiered offload, IO-aware", "tiered_fp8": "Tiered + FP8 KV",
              "tiered_int4": "Tiered + INT4 KV", "cacheblend": "Tiered + CacheBlend", "snapkv_2k": "Token eviction (2K budget)"}
PAPER_SETUPS = [("ShareGPT", "Llama-2-7B"), ("ShareGPT", "Llama-3.1-8B"), ("LongBench", "Llama-3.1-8B"),
                ("LongBench", "Mistral-7B"), ("GSM8K", "Llama-3.1-8B"), ("Needle-in-a-Haystack", "Llama-3-8B"),
                ("Mooncake conversation", "Llama-3.1-70B"), ("SWE-bench", "Llama-3.1-8B")]


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load(main="main.csv"):
    d = pd.read_csv(os.path.join(RES, main))
    d = d[d["technique"].notna() & ~d["workload"].isin(EXCLUDE)]
    d["log_su"] = np.log(d["speedup_ttft_vs_nocache"].clip(lower=1e-6))
    return d


def spread(d, col="speedup_ttft_vs_nocache"):
    rows = []
    for t, g in d[d.technique != "no_cache"].groupby("technique"):
        s = g[col]
        rows.append({"technique": t, "cells": len(g), "min": s.min(), "p10": s.quantile(.1), "median": s.median(),
                     "p90": s.quantile(.9), "max": s.max(), "p90_over_p10": s.quantile(.9) / max(s.quantile(.1), 1e-9),
                     "share_slower_than_nocache": float((s < 0.98).mean()),
                     "share_no_effect": float(((s >= 0.98) & (s <= 1.02)).mean())})
    return pd.DataFrame(rows).sort_values("median", ascending=False)


def variance_shares(d):
    """Two-way (no replication) decomposition of log speedup: SS_w, SS_m, SS_residual(=interaction)."""
    rows = []
    for t, g in d[d.technique != "no_cache"].groupby("technique"):
        p = g.pivot_table(index="workload", columns="model", values="log_su")
        p = p.dropna(axis=0, thresh=int(0.8 * p.shape[1])).dropna(axis=1)
        p = p.dropna(axis=0)
        if p.shape[0] < 3 or p.shape[1] < 3:
            continue
        x = p.values
        mu = x.mean()
        ssw = p.shape[1] * ((x.mean(1) - mu) ** 2).sum()
        ssm = p.shape[0] * ((x.mean(0) - mu) ** 2).sum()
        sst = ((x - mu) ** 2).sum()
        ssi = sst - ssw - ssm
        rows.append({"technique": t, "workloads": p.shape[0], "models": p.shape[1],
                     "share_workload": ssw / sst if sst else 0, "share_model": ssm / sst if sst else 0,
                     "share_interaction": ssi / sst if sst else 0})
    return pd.DataFrame(rows)


def winners(d, tol=0.02):
    """Per cell: the simplest technique within `tol` of the best TTFT (techniques ordered by complexity),
    plus, per technique, the share of cells where it is (near-)best and where it is the unique winner."""
    techs = [t for t in TECH_ORDER if t in set(d.technique)]
    p = d.pivot_table(index=["workload", "model"], columns="technique", values="ttft_mean")[techs]
    best = p.min(axis=1)
    near = p.le(best * (1 + tol), axis=0) & p.notna()
    simplest = near.idxmax(axis=1).rename("winner").reset_index()
    unique = near.sum(axis=1) == 1
    cnt = pd.DataFrame({"technique": techs,
                        "near_best_share": [float(near[t][p[t].notna()].mean()) for t in techs],
                        "unique_winner_cells": [int((near[t] & unique).sum()) for t in techs],
                        "simplest_winner_cells": [int((simplest.winner == t).sum()) for t in techs]})
    return simplest, cnt


def flips(d, tol=0.02):
    techs = [t for t in TECH_ORDER if t in set(d.technique)]
    p = d.pivot_table(index=["workload", "model"], columns="technique", values="ttft_mean")
    rows = []
    for a, b in itertools.combinations(techs, 2):
        if a not in p or b not in p:
            continue
        q = p[[a, b]].dropna()
        r = q[a] / q[b]
        rows.append({"A": a, "B": b, "cells": len(q), "A_better": float((r < 1 - tol).mean()),
                     "B_better": float((r > 1 + tol).mean()), "tie": float(((r >= 1 - tol) & (r <= 1 + tol)).mean())})
    out = pd.DataFrame(rows)
    out["reversal"] = out[["A_better", "B_better"]].min(axis=1)   # >0 means the conclusion depends on the setup
    return out.sort_values("reversal", ascending=False)


def typical_paper(d):
    rows = []
    for t, g in d[d.technique != "no_cache"].groupby("technique"):
        s = g["speedup_ttft_vs_nocache"].dropna().values
        for w, m in PAPER_SETUPS:
            v = g[(g.workload == w) & (g.model == m)]["speedup_ttft_vs_nocache"]
            if len(v):
                rows.append({"technique": t, "workload": w, "model": m, "speedup": float(v.iloc[0]),
                             "percentile_in_population": float((s < v.iloc[0]).mean() * 100),
                             "population_median": float(np.median(s))})
    return pd.DataFrame(rows)


def feature_corr(d):
    prof = pd.read_csv(os.path.join(ROOT, "data", "workload_profiles.csv"))
    prof = prof[prof.tokenizer.isin(["llama3", "native"])].drop_duplicates("workload")
    x = d.merge(prof[["workload", "in_mean", "out_mean", "prefix_hit_ratio", "nonprefix_extra_ratio"]], on="workload", how="left")
    rows = []
    for t, g in x[x.technique != "no_cache"].groupby("technique"):
        for f in ["in_mean", "out_mean", "prefix_hit_ratio", "kv_bytes_per_token", "active_b", "params_b"]:
            r = g[[f, "speedup_ttft_vs_nocache"]].dropna().rank().corr().iloc[0, 1]
            rows.append({"technique": t, "feature": f, "spearman": r})
    return pd.DataFrame(rows).pivot(index="technique", columns="feature", values="spearman").reset_index()


PROD = ["Mooncake conversation", "Mooncake tool&agent", "Mooncake synthetic", "Qwen-Bailian to-C (A)", "Qwen-Bailian to-B (B)",
        "Qwen-Bailian thinking", "Qwen-Bailian coder"]
MODERN = ["Llama-3.1-8B", "Llama-3.1-70B", "Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B", "Qwen3-8B", "Qwen3-32B",
          "Qwen3-30B-A3B", "Qwen3-4B", "Qwen3-14B", "DeepSeek-V2", "Llama-3.2-3B"]


def literature_vs_production(d):
    """Gain as the literature samples it (cells weighted by paper counts) vs production traces on current models."""
    sw = pd.read_csv(os.path.join(ROOT, "survey", "out", "top_workloads.csv")).set_index("name").papers
    sm = pd.read_csv(os.path.join(ROOT, "survey", "out", "top_models.csv")).set_index("name").papers
    d = d.assign(w_lit=d.workload.map(sw).fillna(0) * d.model.map(sm).fillna(0))

    def wmed(v, w):
        o = np.argsort(v)
        c = np.cumsum(w[o]) / w.sum()
        return v[o][np.searchsorted(c, 0.5)]
    rows = []
    for t, g in d[d.technique != "no_cache"].groupby("technique"):
        s = g.speedup_ttft_vs_nocache.values
        gp = g[g.workload.isin(PROD) & g.model.isin(MODERN)]
        rows.append({"technique": t, "uniform_median": float(np.median(s)),
                     "literature_weighted_median": float(wmed(s, g.w_lit.values + 1e-9)),
                     "production_traces_x_modern_models_median": float(gp.speedup_ttft_vs_nocache.median()) if len(gp) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- figures
def fig_spread(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    techs = [t for t in TECH_ORDER if t in set(d.technique)]
    data = [np.log10(d[d.technique == t]["speedup_ttft_vs_nocache"].dropna().clip(1e-3)) for t in techs]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(techs) + 1.0), facecolor=SURF)
    _style(ax)
    for i, (t, v) in enumerate(zip(techs, data)):
        y = len(techs) - 1 - i
        q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
        ax.plot([v.min(), v.max()], [y, y], color=GRID, lw=2, solid_capstyle="round")
        ax.plot([q10, q90], [y, y], color=SEQ[2], lw=4, solid_capstyle="round")
        ax.plot([q25, q75], [y, y], color=SEQ[5], lw=7, solid_capstyle="round")
        ax.plot([q50], [y], "o", ms=8, color=SURF, mec=INK, mew=2)
    ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(range(len(techs)))
    ax.set_yticklabels([TECH_LABEL[t] for t in techs][::-1], color=INK, fontsize=9)
    ticks = [-1, 0, 1, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{10 ** k:g}×" for k in ticks])
    ax.set_xlabel("Mean-TTFT speedup over no KV reuse (log scale) across all workload × model cells", color=INK2, fontsize=9)
    ax.set_title("Same technique, different setup: spread of reported improvement", loc="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.015, "grey: full range · light: p10–p90 · dark: IQR · dot: median", color=INK2, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"technique_spread.{ext}"), dpi=200)
    plt.close(fig)


def fig_heatmap(d, tech="tiered_ioaware", base="prefix_lru", top_w=25, top_m=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import yaml
    tops = yaml.safe_load(open(os.path.join(ROOT, "survey", "out", "top_lists_final.yaml")))
    p = d.pivot_table(index=["workload", "model"], columns="technique", values="ttft_mean")
    if tech not in p or base not in p:
        return
    r = np.log2(p[base] / p[tech]).rename("gain").reset_index()
    h = r.pivot(index="workload", columns="model", values="gain")
    ws = [w for w in tops["workloads"][:top_w] if w in h.index]
    ms = [m for m in tops["models"][:top_m] if m in h.columns]
    h = h.loc[ws, ms]
    lim = max(1.0, np.nanpercentile(np.abs(h.values), 98))
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(0.36 * len(ms) + 3, 0.3 * len(ws) + 2), facecolor=SURF)
    im = ax.imshow(h.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms, rotation=60, ha="right", fontsize=7, color=INK)
    ax.set_yticks(range(len(ws)))
    ax.set_yticklabels(ws, fontsize=7, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label(f"log2 TTFT gain of {TECH_LABEL[tech]}\nover {TECH_LABEL[base]} (blue = better)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"{TECH_LABEL[tech]} vs {TECH_LABEL[base]}: gain by workload × model", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"heatmap_{tech}_vs_{base}.{ext}"), dpi=200)
    plt.close(fig)


def fig_by_attention(d, tech="tiered_restore"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = d[d.technique == tech]
    order = [a for a in ["mha", "gqa", "mqa", "mla", "hybrid-swa", "hybrid-ssm"] if a in set(g.attn)]
    cats = [c for c in ["chat", "longctx", "rag", "synthetic", "reasoning", "code", "agent", "fewshot", "lm", "trace"] if c in set(g.category)]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURF)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    w = 0.8 / len(order)
    for j, a in enumerate(order):
        vals = [np.median(np.log2(g[(g.attn == a) & (g.category == c)]["speedup_ttft_vs_nocache"].clip(1e-3)))
                if len(g[(g.attn == a) & (g.category == c)]) else np.nan for c in cats]
        x = np.arange(len(cats)) + (j - (len(order) - 1) / 2) * w
        ax.bar(x, vals, width=w * 0.92, color=CAT[j], label=a.upper() if "-" not in a else a, edgecolor=SURF, linewidth=1)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=8, color=INK)
    ax.set_ylabel("median log2 TTFT speedup", fontsize=8, color=INK2)
    ax.set_title(f"{TECH_LABEL[tech]}: gain depends on workload category × attention type", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, ncol=len(order), loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"by_attention_{tech}.{ext}"), dpi=200)
    plt.close(fig)


def fig_sensitivity(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = s[s.tag.str.startswith("sens:load=")].copy()
    if s.empty:
        return
    s["load"] = s.tag.str.split("=").str[1].astype(float)
    techs = ["prefix_lru", "tiered_restore", "tiered_ioaware", "tiered_int4"]
    ws = list(dict.fromkeys(s.workload))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), facecolor=SURF, sharex=True)
    for ax, w in zip(axes.flat, ws):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        for j, t in enumerate(techs):
            g = s[(s.workload == w) & (s.technique == t)].groupby("load")["speedup_ttft_vs_nocache"].median()
            ax.plot(g.index, g.values, "-o", color=CAT[j], lw=2, ms=5, mec=SURF, mew=1.5, label=TECH_LABEL[t])
        ax.set_yscale("log")
        ax.set_title(w, fontsize=9, color=INK, loc="left")
        ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
    for ax in axes[-1]:
        ax.set_xlabel("offered load (× no-reuse capacity)", fontsize=8, color=INK2)
    axes[0, 0].set_ylabel("median TTFT speedup (log)", fontsize=8, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Speedup vs load, median over the 10 sensitivity models", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"sensitivity_load.{ext}"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    d = load()
    spread(d).to_csv(os.path.join(OUT, "technique_spread.csv"), index=False)
    spread(d, "speedup_service_vs_nocache").to_csv(os.path.join(OUT, "technique_spread_service.csv"), index=False)
    variance_shares(d).to_csv(os.path.join(OUT, "variance_shares.csv"), index=False)
    best, cnt = winners(d)
    best.to_csv(os.path.join(OUT, "winners.csv"), index=False)
    cnt.to_csv(os.path.join(OUT, "winner_counts.csv"), index=False)
    flips(d).to_csv(os.path.join(OUT, "pairwise_flips.csv"), index=False)
    typical_paper(d).to_csv(os.path.join(OUT, "typical_paper.csv"), index=False)
    literature_vs_production(d).to_csv(os.path.join(OUT, "literature_vs_production.csv"), index=False)
    feature_corr(d).to_csv(os.path.join(OUT, "feature_correlation.csv"), index=False)
    fig_spread(d)
    fig_heatmap(d, "tiered_ioaware", "prefix_lru")
    fig_heatmap(d, "tiered_restore", "prefix_lru")
    fig_by_attention(d, "tiered_restore")
    sp = os.path.join(RES, "sensitivity.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        s = s[s.technique.notna()]
        fig_sensitivity(s)
        sv = []
        for tag, g in s.groupby("tag"):
            sv.append(spread(g.assign(log_su=np.log(g.speedup_ttft_vs_nocache.clip(1e-6)))).assign(tag=tag))
        pd.concat(sv).to_csv(os.path.join(OUT, "sensitivity_spread.csv"), index=False)
    print(open(os.path.join(OUT, "technique_spread.csv")).read())
    print(open(os.path.join(OUT, "variance_shares.csv")).read())


if __name__ == "__main__":
    main()
