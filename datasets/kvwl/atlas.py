"""Full-coverage figures: every workload, every model, every technique.

    python -m kvwl.atlas [--results dataset/results]

figures/atlas_*.pdf|png
  01_survey         top-50 workloads and models by paper count (full-text share, colour = category / attention)
  02_workloads      all workloads: input & output token distributions + prefix / non-prefix reuse
  03_models         all models: KV bytes per token, cache-block size, weights
  04_kv_matrix      workload x model: mean KV per request (all cells)
  05_speedup_grid   workload x model heatmaps, one per technique (all cells)
  06_winner_map     workload x model: simplest technique within 2% of the best TTFT
  07_drivers        what explains the gain: reuse ratio, KV bytes/token, input length
  08_variance       share of log-speedup variance from workload / model / interaction
  09_sensitivity    load, GPU and SSD sweeps for every technique (all sensitivity cells)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm, ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .analysis import CAT, SEQ, INK, INK2, GRID, SURF, TECH_LABEL, EXCLUDE, variance_shares, load as load_main  # noqa: E402
from .models import load_models  # noqa: E402

from .paths import ROOT
FIG = os.path.join(ROOT, "figures")
CATEGORIES = ["chat", "agent", "trace", "longctx", "rag", "synthetic", "code", "reasoning", "fewshot", "lm"]
CAT_COLOR = dict(zip(CATEGORIES, CAT + ["#8a8984", "#b5b4ae"]))
ATTN = ["mha", "gqa", "mqa", "mla", "hybrid-swa", "hybrid-ssm"]
ATTN_COLOR = dict(zip(ATTN, CAT))
CORE = ["prefix_lru", "tiered_restore", "tiered_ioaware", "tiered_int4", "cacheblend"]
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
SEQMAP = LinearSegmentedColormap.from_list("seq", SEQ)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"atlas_{name}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def _clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _category(w):
    from .catalog import category
    return category(w)


# ------------------------------------------------------------------ 01 survey
def fig_survey():
    w = pd.read_csv(os.path.join(ROOT, "survey", "out", "top_workloads.csv"))
    m = pd.read_csv(os.path.join(ROOT, "survey", "out", "top_models.csv"))
    M = load_models()
    fig, axes = plt.subplots(1, 2, figsize=(11, 9.5))
    for ax, d, colf, title, legend in (
            (axes[0], w, lambda r: CAT_COLOR.get(r["category"], INK2), "Top-50 workloads", [(c, CAT_COLOR[c]) for c in CATEGORIES]),
            (axes[1], m, lambda r: ATTN_COLOR.get(M[r["name"]].attn, INK2), "Top-50 models", [(a.upper(), ATTN_COLOR[a]) for a in ATTN])):
        d = d.iloc[::-1].reset_index(drop=True)
        cols = [colf(r) for _, r in d.iterrows()]
        ax.barh(range(len(d)), d["papers"], color=cols, height=0.72, edgecolor=SURF, lw=0.5, alpha=0.45)
        ax.barh(range(len(d)), d["fulltext_eval"], color=cols, height=0.72, edgecolor=SURF, lw=0.5)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["name"], fontsize=6.5, color=INK)
        ax.set_xlabel("papers (solid = named in evaluation section, light = code artifact only)")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        _clean(ax)
        ax.legend(handles=[Patch(color=c, label=l) for l, c in legend], fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("What 2024–2026 KV-cache / serving papers evaluate on (273 papers)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "01_survey")


# --------------------------------------------------------------- 02 workloads
def workload_table():
    p = pd.read_csv(os.path.join(ROOT, "data", "workload_profiles.csv"))
    p = p[p.tokenizer.isin(["llama3", "native"]) & ~p.workload.isin(EXCLUDE)].drop_duplicates("workload").copy()
    p["category"] = p.workload.map(_category)
    p["cat_idx"] = p.category.map({c: i for i, c in enumerate(CATEGORIES)})
    return p.sort_values(["cat_idx", "in_mean"]).reset_index(drop=True)


def fig_workloads(p):
    n = len(p)
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.17 * n + 1.6), sharey=True, gridspec_kw={"width_ratios": [1.4, 1.1, 1.2]})
    y = np.arange(n)[::-1]
    for ax, pre, lab in ((axes[0], "in", "input tokens (log)"), (axes[1], "out", "output tokens (log)")):
        for i, r in p.iterrows():
            c = CAT_COLOR[r.category]
            ax.plot([max(1, r[f"{pre}_min"]), r[f"{pre}_p99"]], [y[i]] * 2, color=GRID, lw=2, solid_capstyle="round")
            ax.plot([max(1, r[f"{pre}_p50"] * 0.999), r[f"{pre}_p90"]], [y[i]] * 2, color=c, lw=3.5, solid_capstyle="round")
            ax.plot(r[f"{pre}_mean"], y[i], "o", ms=4.5, mfc=SURF if r.fidelity == "published" else c, mec=c, mew=1.3)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        _clean(ax)
    ax = axes[2]
    ax.barh(y, p.prefix_hit_ratio * 100, color=[CAT_COLOR[c] for c in p.category], height=0.72, edgecolor=SURF, lw=0.5)
    ax.barh(y, p.nonprefix_extra_ratio * 100, left=p.prefix_hit_ratio * 100, color=INK2, height=0.72, edgecolor=SURF, lw=0.5)
    for i, r in p.iterrows():
        tot = (r.prefix_hit_ratio + r.nonprefix_extra_ratio) * 100
        ax.text(tot + 1, y[i], f"{tot:.0f}%", va="center", fontsize=5.5, color=INK2)
    ax.set_xlim(0, 110)
    ax.set_xlabel("reusable input tokens, infinite cache (%)\ncolour = prefix reuse, grey = extra non-prefix (chunk) reuse")
    _clean(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{w}{' *' if f == 'published' else ''}" for w, f in zip(p.workload, p.fidelity)], fontsize=6.5, color=INK)
    axes[0].legend(handles=[Patch(color=CAT_COLOR[c], label=c) for c in CATEGORIES], fontsize=6, frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, -0.04 - 2.0 / n), ncol=5)
    fig.suptitle(f"Workload atlas: all {n} workloads (grey bar: min–p99, coloured: p50–p90, dot: mean; hollow dot / * = synthesized from published stats)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    _save(fig, "02_workloads")


# ----------------------------------------------------------------- 03 models
def model_table():
    M = load_models()
    top = yaml.safe_load(open(os.path.join(ROOT, "survey", "out", "top_lists_final.yaml")))["models"]
    rows = []
    for k in top:
        m = M[k]
        rows.append({"model": k, "attn": m.attn, "kv_kib": m.kv_bytes_per_token() / 1024, "block_mib": m.block_bytes() / 2 ** 20,
                     "weights_gb": m.weight_bytes() / 1e9, "active_b": m.active_b, "params_b": m.params_b, "ctx": m.ctx,
                     "kv32k_gb": m.kv_bytes(32768) / 1e9})
    return pd.DataFrame(rows).sort_values("kv_kib").reset_index(drop=True)


def fig_models(mt):
    n = len(mt)
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.17 * n + 1.4), sharey=True)
    y = np.arange(n)
    cols = [ATTN_COLOR[a] for a in mt.attn]
    for ax, col, lab, fmt in ((axes[0], "kv_kib", "KV per token (KiB, bf16, log)", "{:.0f}"),
                              (axes[1], "kv32k_gb", "KV for one 32K-token request (GB, log)", "{:.1f}"),
                              (axes[2], "weights_gb", "weights (GB, bf16, log)", "{:.0f}")):
        ax.barh(y, mt[col], color=cols, height=0.72, edgecolor=SURF, lw=0.5)
        for i, v in enumerate(mt[col]):
            ax.text(v * 1.08, i, fmt.format(v), va="center", fontsize=5.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        _clean(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{m} ({p:g}B)" for m, p in zip(mt.model, mt.params_b)], fontsize=6.5, color=INK)
    axes[1].legend(handles=[Patch(color=ATTN_COLOR[a], label=a.upper()) for a in ATTN if a in set(mt.attn)], fontsize=7,
                   frameon=False, loc="lower right")
    spread = mt.kv_kib.max() / mt.kv_kib.min()
    fig.suptitle(f"Model atlas: all {n} top-50 models — KV per token spans {spread:.0f}× (stock KVC-TierSim assumes 128 KiB for every model)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    _save(fig, "03_models")


# ------------------------------------------------------- shared heatmap util
def _heat(ax, mat, cmap, norm, rows, cols, row_colors=None, col_colors=None, cbar_label="", fig=None, small=False):
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=4.8 if small else 5.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=4.8 if small else 5.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if row_colors is not None:
        for i, c in enumerate(row_colors):
            ax.add_patch(plt.Rectangle((-1.6, i - 0.5), 0.9, 1, color=c, clip_on=False))
    if col_colors is not None:
        for j, c in enumerate(col_colors):
            ax.add_patch(plt.Rectangle((j - 0.5, -1.6), 1, 0.9, color=c, clip_on=False))
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
        cb.set_label(cbar_label, fontsize=7, color=INK2)
        cb.ax.tick_params(labelsize=6)
    return im


def _orders(wt, mt):
    return list(wt.workload), list(mt.model)


# --------------------------------------------------------------- 04 KV matrix
def fig_kv_matrix(wt, mt):
    mx = pd.read_csv(os.path.join(ROOT, "data", "workload_model_matrix.csv"))
    rows, cols = _orders(wt, mt)
    h = mx.pivot_table(index="workload", columns="model", values="kv_per_request_mean_MB").reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(11, 0.15 * len(rows) + 2.5))
    _heat(ax, h.values, SEQMAP, LogNorm(vmin=max(1, np.nanmin(h.values)), vmax=np.nanmax(h.values)), rows, cols,
          [CAT_COLOR[c] for c in wt.category], [ATTN_COLOR[a] for a in mt.attn], "mean KV per request (MB, log)", fig)
    ax.set_title("KV-cache footprint per request: every workload × every model (after context-window truncation)",
                 loc="left", fontsize=10, color=INK, pad=14)
    _save(fig, "04_kv_matrix")


# ------------------------------------------------------------ 05 speedup grid
def fig_speedup_grid(d, wt, mt):
    rows, cols = _orders(wt, mt)
    p = d.pivot_table(index=["workload", "model"], columns="technique", values="ttft_mean")
    panels = [("prefix_lru", "no_cache"), ("tiered_restore", "prefix_lru"), ("tiered_ioaware", "tiered_restore"),
              ("tiered_int4", "tiered_ioaware"), ("cacheblend", "tiered_ioaware")]
    fig, axes = plt.subplots(1, len(panels), figsize=(26, 0.15 * len(rows) + 3), sharey=True)
    lim = 4.0
    for ax, (t, b) in zip(axes, panels):
        if t not in p or b not in p:
            ax.axis("off")
            continue
        g = np.log2((p[b] / p[t]).clip(2 ** -lim, 2 ** lim)).rename("g").reset_index()
        h = g.pivot(index="workload", columns="model", values="g").reindex(index=rows, columns=cols)
        _heat(ax, h.values, DIV, TwoSlopeNorm(0, -lim, lim), rows, cols, [CAT_COLOR[c] for c in wt.category] if ax is axes[0] else None,
              [ATTN_COLOR[a] for a in mt.attn], small=True)
        ax.set_title(f"{TECH_LABEL[t]}\nvs {TECH_LABEL.get(b, 'no KV reuse')}", loc="left", fontsize=8, color=INK, pad=22)
    sm = plt.cm.ScalarMappable(cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    cb = fig.colorbar(sm, ax=axes, shrink=0.4, pad=0.005)
    cb.set_label("log2 mean-TTFT gain (blue = the newer technique is faster; clipped at ±16×)", fontsize=8, color=INK2)
    fig.suptitle("Incremental gain of each technique over its predecessor, for all workload × model cells "
                 "(rows grouped by workload category, columns ordered by KV bytes/token)", x=0.01, ha="left", fontsize=11, color=INK)
    _save(fig, "05_speedup_grid")


# ------------------------------------------------------------- 06 winner map
def fig_winner_map(d, wt, mt, tol=0.05):
    rows, cols = _orders(wt, mt)
    techs = ["no_cache"] + CORE
    p = d.pivot_table(index=["workload", "model"], columns="technique", values="ttft_mean")
    p = p[[t for t in techs if t in p]]
    best = p.min(axis=1)
    near = p.le(best * (1 + tol), axis=0) & p.notna()
    win = near.idxmax(axis=1).map({t: i for i, t in enumerate(techs)}).rename("w").reset_index()
    h = win.pivot(index="workload", columns="model", values="w").reindex(index=rows, columns=cols)
    colors = ["#d8d7d2"] + CAT[:len(CORE)]
    fig, ax = plt.subplots(figsize=(11, 0.15 * len(rows) + 2.5))
    _heat(ax, h.values, ListedColormap(colors), BoundaryNorm(np.arange(-0.5, len(techs)), len(techs)), rows, cols,
          [CAT_COLOR[c] for c in wt.category], [ATTN_COLOR[a] for a in mt.attn])
    share = win.w.value_counts(normalize=True)
    ax.legend(handles=[Patch(color=colors[i], label=f"{TECH_LABEL.get(t, 'no KV reuse (nothing helps)')} — {share.get(i, 0) * 100:.0f}%")
                       for i, t in enumerate(techs)], fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Which technique a paper would pick: simplest technique within 5% of the best mean TTFT, per workload × model",
                 loc="left", fontsize=10, color=INK, pad=14)
    _save(fig, "06_winner_map")


# --------------------------------------------------------------- 07 drivers
def fig_drivers(d, wt, mt):
    x = d.merge(wt[["workload", "prefix_hit_ratio", "nonprefix_extra_ratio", "in_mean"]], on="workload")
    p = x.pivot_table(index=["workload", "model"], columns="technique", values="ttft_mean")
    meta = x.drop_duplicates(["workload", "model"]).set_index(["workload", "model"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rng = np.random.default_rng(0)
    for ax, (t, b, f, lab, logx) in zip(axes, [
            ("prefix_lru", "no_cache", "prefix_hit_ratio", "workload prefix-reuse ratio (infinite cache)", False),
            ("tiered_restore", "prefix_lru", "kv_bytes_per_token", "model KV bytes per token (log)", True),
            ("cacheblend", "tiered_ioaware", "nonprefix_extra_ratio", "workload extra non-prefix (chunk) reuse ratio", False)]):
        if t not in p or b not in p:
            continue
        g = np.log2((p[b] / p[t])).dropna()
        m = meta.loc[g.index]
        xs = m[f].values * (1 + rng.normal(0, 0.02, len(m)))
        for a in ATTN:
            sel = (m.attn == a).values
            if sel.any():
                ax.scatter(xs[sel], g.values[sel], s=9, color=ATTN_COLOR[a], alpha=0.55, lw=0, label=a.upper())
        ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel(f"log2 TTFT gain: {TECH_LABEL[t]}\nvs {TECH_LABEL.get(b, 'no KV reuse')}", fontsize=7)
        rho = pd.Series(m[f].values).corr(pd.Series(g.values), method="spearman")
        ax.set_title(f"Spearman ρ = {rho:.2f}  ({len(g)} cells)", loc="left", fontsize=8, color=INK)
        _clean(ax, "y")
    axes[0].legend(fontsize=6.5, frameon=False, markerscale=1.6)
    fig.suptitle("What drives the gain: workload reuse sets the ceiling, model KV size decides whether offloading pays",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, "07_drivers")


# -------------------------------------------------------------- 08 variance
def fig_variance(d):
    v = variance_shares(d)
    v = v.set_index("technique").reindex([t for t in CORE if t in set(v.technique)])
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    left = np.zeros(len(v))
    for col, c, lab in (("share_workload", CAT[0], "workload"), ("share_model", CAT[1], "model"),
                        ("share_interaction", CAT[3], "workload × model")):
        ax.barh(range(len(v)), v[col] * 100, left=left, color=c, height=0.62, edgecolor=SURF, lw=1.5, label=lab)
        for i, (l, w) in enumerate(zip(left, v[col] * 100)):
            if w > 6:
                ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=7, color="white")
        left += v[col].values * 100
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels([TECH_LABEL[t] for t in v.index], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of variance in log TTFT speedup over no KV reuse (%)")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    _clean(ax, None)
    ax.set_title("Where the variation in reported gains comes from", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    _save(fig, "08_variance")


# ----------------------------------------------------------- 09 sensitivity
def fig_sensitivity(s):
    s = s[s.technique.notna()].copy()
    s["axis"] = s.tag.str.extract(r"sens:(\w+)=")[0]
    s["val"] = s.tag.str.split("=").str[1]
    axes_def = [("load", "offered load (× no-reuse capacity)", lambda v: float(v)),
                ("gpu", "GPU", {"A100": 0, "H100": 1, "H200": 2}.get),
                ("disk", "SSD", {"SATA": 0, "NVMe": 1, "NVMeT700": 2}.get)]
    fig, axes = plt.subplots(len(CORE), 3, figsize=(11, 2.0 * len(CORE)), sharey="row")
    for j, (ax_name, lab, conv) in enumerate(axes_def):
        g = s[s.axis == ax_name].copy()
        g["x"] = g.val.map(conv)
        for i, t in enumerate(CORE):
            ax = axes[i, j]
            gt = g[g.technique == t]
            if gt.empty:
                ax.axis("off")
                continue
            for (w, m), c in gt.groupby(["workload", "model"]):
                c = c.sort_values("x")
                ax.plot(c.x, np.log2(c.speedup_ttft_vs_nocache.clip(1e-3)), color=CAT_COLOR[_category(w)], lw=0.5, alpha=0.25)
            q = gt.groupby("x").speedup_ttft_vs_nocache.apply(lambda v: np.log2(v.clip(1e-3)).quantile([.25, .5, .75])).unstack()
            ax.fill_between(q.index, q[0.25], q[0.75], color=INK, alpha=0.12, lw=0)
            ax.plot(q.index, q[0.5], "-o", color=INK, lw=2, ms=4, mec=SURF)
            ax.axhline(0, color=INK2, lw=0.7, ls=(0, (3, 3)))
            _clean(ax, "y")
            if ax_name != "load":
                ax.set_xticks(range(3))
                ax.set_xticklabels(["A100", "H100", "H200"] if ax_name == "gpu" else ["SATA", "NVMe", "NVMe T700"])
            if j == 0:
                ax.set_ylabel(f"{TECH_LABEL[t]}\nlog2 TTFT speedup", fontsize=7)
            if i == len(CORE) - 1:
                ax.set_xlabel(lab)
    fig.suptitle("Sensitivity of every technique to load, GPU and SSD (thin lines: each of the 100 workload × model cells, coloured by workload "
                 "category; black: median, band: IQR)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    _save(fig, "09_sensitivity")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "data", "results"))
    a = ap.parse_args(argv)
    from . import analysis
    analysis.RES = a.results
    d = load_main()
    d = d[~d.workload.isin(EXCLUDE)]
    wt = workload_table()
    wt = wt[wt.workload.isin(set(d.workload))].reset_index(drop=True)
    mt = model_table()
    mt = mt[mt.model.isin(set(d.model))].reset_index(drop=True)
    fig_survey()
    fig_workloads(workload_table())
    fig_models(model_table())
    fig_kv_matrix(wt, mt)
    fig_speedup_grid(d, wt, mt)
    fig_winner_map(d, wt, mt)
    fig_drivers(d, wt, mt)
    fig_variance(d)
    sp = os.path.join(a.results, "sensitivity.csv")
    if os.path.exists(sp):
        fig_sensitivity(pd.read_csv(sp))
    print("atlas figures written to", FIG)


if __name__ == "__main__":
    main()
