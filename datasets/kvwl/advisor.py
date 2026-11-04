"""KV-cache evaluation advisor: learn from the atlas to help evaluate NEW workloads and models.

    python -m kvwl.advisor            # train + validate, writes dataset/advisor/*
    python -m kvwl.advisor --predict --workload-profile my_profile.json --model-key Llama-3.1-8B

Three tools, all trained on dataset/results/{main,sensitivity}.csv:

1. Gain predictor (regression): predicts log2 TTFT speedup of each technique over no reuse
   from a *fingerprint* of the workload (reuse ratios, length distribution, turns) and of the
   model (KV bytes/token, attention type, active params, context window) plus hardware/load.
   Validated leave-one-WORKLOAD-out and leave-one-MODEL-out, i.e. on data it has never seen.
2. Technique recommender (classification): predicts the simplest technique within 5% of the best.
3. Representative-suite selector: picks k workloads that cover the fingerprint space
   (k-medoids), and checks how well a k-workload evaluation reproduces the population
   result compared with the literature's top-k and with random picks.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .analysis import EXCLUDE
from .models import load_models
from .replay import Hardware, auto_gpu_count

from .paths import ROOT
OUT = os.path.join(ROOT, "data", "advisor")
TECHS = ["prefix_lru", "tiered_restore", "tiered_ioaware", "tiered_int4", "cacheblend"]
ORDER = ["no_cache"] + TECHS
ATTN = ["mha", "gqa", "mqa", "mla", "hybrid-swa", "hybrid-ssm"]
GPU = {"A100": (312e12, 2.039e12), "H100": (1979e12, 3.36e12), "H200": (4000e12, 4.8e12)}
DISK = {"SATA": 0.55e9, "NVMe": 2e9, "NVMeT700": 5e9}

WORKLOAD_FEATS = ["w_log_in_mean", "w_log_in_p99", "w_log_out_mean", "w_prefix", "w_chunk", "w_unique", "w_turns", "w_in_cv"]
MODEL_FEATS = ["m_log_kv_tok", "m_log_active", "m_log_ctx", "m_log_block", "m_gpus"] + [f"m_attn_{a}" for a in ATTN]
CELL_FEATS = ["c_trunc", "c_log_ws_over_hbm"]
HW_FEATS = ["h_log_flops", "h_log_hbm_bw", "h_log_disk_bw", "h_load"]
FEATS = WORKLOAD_FEATS + MODEL_FEATS + CELL_FEATS + HW_FEATS


# ------------------------------------------------------------- fingerprints
def workload_fingerprints():
    p = pd.read_csv(os.path.join(ROOT, "data", "workload_profiles.csv"))
    p = p[p.tokenizer.isin(["llama3", "native"])].drop_duplicates("workload")
    f = pd.DataFrame({
        "workload": p.workload,
        "w_log_in_mean": np.log10(p.in_mean.clip(1)), "w_log_in_p99": np.log10(p.in_p99.clip(1)),
        "w_log_out_mean": np.log10(p.out_mean.clip(1)), "w_prefix": p.prefix_hit_ratio, "w_chunk": p.nonprefix_extra_ratio,
        "w_unique": p.unique_block_ratio, "w_turns": np.log2(p.turns_per_session.clip(1)),
        "w_in_cv": (p.in_p90 / p.in_p50.clip(1)).clip(upper=50)})
    return f.set_index("workload")


def model_fingerprints():
    M = load_models()
    rows = []
    for k, m in M.items():
        r = {"model": k, "m_log_kv_tok": np.log10(m.kv_bytes_per_token()), "m_log_active": np.log10(m.active_b),
             "m_log_ctx": np.log10(m.ctx), "m_log_block": np.log10(m.block_bytes()), "m_gpus": np.log2(auto_gpu_count(m, Hardware()))}
        for a in ATTN:
            r[f"m_attn_{a}"] = float(m.attn == a)
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def cell_features():
    mx = pd.read_csv(os.path.join(ROOT, "data", "workload_model_matrix.csv"))
    M = load_models()
    hbm = {k: max(1e9, 80e9 * auto_gpu_count(m, Hardware()) * 0.9 - m.weight_bytes()) for k, m in M.items()}
    mx["c_trunc"] = mx.truncated_frac
    mx["c_log_ws_over_hbm"] = np.log10((mx.working_set_GB * 1e9 / mx.model.map(hbm)).clip(1e-4))
    return mx.set_index(["workload", "model"])[CELL_FEATS]


def build_table():
    frames = []
    for name in ("main", "sensitivity"):
        path = os.path.join(ROOT, "data", "results", f"{name}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path)
            frames.append(d[d.technique.notna()])
    d = pd.concat(frames, ignore_index=True)
    d = d[~d.workload.isin(EXCLUDE)]
    d["hw_gpu"] = d.get("hw_gpu", pd.Series(index=d.index, dtype=object)).fillna("H100")
    d["hw_disk"] = d.get("hw_disk", pd.Series(index=d.index, dtype=object)).fillna("NVMe")
    d["setting"] = d.tag.astype(str)
    piv = d.pivot_table(index=["setting", "workload", "model", "hw_gpu", "hw_disk", "rc_load"], columns="technique",
                        values="ttft_mean").reset_index()
    wf, mf, cf = workload_fingerprints(), model_fingerprints(), cell_features()
    X = piv.join(wf, on="workload").join(mf, on="model").join(cf, on=["workload", "model"])
    X["h_log_flops"] = np.log10(X.hw_gpu.map(lambda g: GPU[g][0]))
    X["h_log_hbm_bw"] = np.log10(X.hw_gpu.map(lambda g: GPU[g][1]))
    X["h_log_disk_bw"] = np.log10(X.hw_disk.map(DISK))
    X["h_load"] = X.rc_load
    for t in TECHS:
        X[f"y_{t}"] = np.log2(X["no_cache"] / X[t])
    near = X[ORDER].le(X[ORDER].min(axis=1) * 1.05, axis=0) & X[ORDER].notna()
    X["y_best"] = near.idxmax(axis=1)
    return X.dropna(subset=FEATS)


# ------------------------------------------------------------ validation
def grouped_cv(X, group_col, n_splits=5):
    """Leave-group-out CV: every test fold contains workloads (or models) never seen in training."""
    gkf = GroupKFold(n_splits=n_splits)
    reg_rows, cls_pred = [], pd.Series(index=X.index, dtype=object)
    for t in TECHS:
        y = X[f"y_{t}"]
        ok = y.notna()
        Xi, yi, gi = X.loc[ok, FEATS].values, y[ok].values, X.loc[ok, group_col].values
        pred = np.zeros_like(yi)
        for tr, te in gkf.split(Xi, yi, gi):
            m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)
            m.fit(Xi[tr], yi[tr])
            pred[te] = m.predict(Xi[te])
        err = np.abs(pred - yi)
        ss = 1 - ((pred - yi) ** 2).sum() / ((yi - yi.mean()) ** 2).sum()
        sign_ok = ((pred > 0.1) == (yi > 0.1)).mean()
        reg_rows.append({"technique": t, "holdout": group_col, "cells": int(ok.sum()), "R2_log2": ss,
                         "median_abs_err_x": float(2 ** np.median(err)), "p90_abs_err_x": float(2 ** np.quantile(err, .9)),
                         "helps_or_not_accuracy": float(sign_ok)})
    Xc, yc, gc = X[FEATS].values, X.y_best.values, X[group_col].values
    pred = np.empty(len(yc), dtype=object)
    for tr, te in gkf.split(Xc, yc, gc):
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        m.fit(Xc[tr], yc[tr])
        pred[te] = m.predict(Xc[te])
    # a recommendation is "good" if the predicted technique is within 5% of the best TTFT in that cell
    best = X[ORDER].min(axis=1)
    within = np.array([X.iloc[i][pred[i]] <= best.iloc[i] * 1.05 if pd.notna(X.iloc[i][pred[i]]) else False for i in range(len(X))])
    majority = pd.Series(yc).value_counts().index[0]
    maj_within = (X[majority] <= best * 1.05).mean()
    rule = np.where(X.w_prefix + X.w_chunk < 0.05, "no_cache",
                    np.where(X.w_chunk > 0.02, "cacheblend", np.where(X.c_log_ws_over_hbm > 0, "tiered_ioaware", "prefix_lru")))
    rule_within = np.array([X.iloc[i][rule[i]] <= best.iloc[i] * 1.05 if pd.notna(X.iloc[i][rule[i]]) else False for i in range(len(X))])
    cls = {"holdout": group_col, "exact_label_acc": float((pred == yc).mean()), "recommendation_within_5pct": float(within.mean()),
           "majority_class_within_5pct": float(maj_within), "majority_class": majority,
           "hand_rule_within_5pct": float(rule_within.mean())}
    return pd.DataFrame(reg_rows), cls


def importances(X):
    rows = []
    for t in TECHS:
        y = X[f"y_{t}"]
        ok = y.notna()
        m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)
        m.fit(X.loc[ok, FEATS].values, y[ok].values)
        imp = dict(zip(FEATS, m.feature_importances_))
        rows.append({"technique": t,
                     "workload_features": sum(v for k, v in imp.items() if k.startswith("w_")),
                     "model_features": sum(v for k, v in imp.items() if k.startswith("m_")),
                     "workload_x_model_features": sum(v for k, v in imp.items() if k.startswith("c_")),
                     "hardware_load": sum(v for k, v in imp.items() if k.startswith("h_")),
                     "top3": ", ".join(k for k, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:3])})
    return pd.DataFrame(rows)


# --------------------------------------------------- representative suite
def representative_suite(X, k=6, trials=300, seed=0):
    """k-medoids over workload fingerprints; compare suites on how well they reproduce
    the population's per-technique median log-speedup and its p10-p90 spread (main grid only)."""
    main = X[X.setting == "main"]
    wf = workload_fingerprints().loc[sorted(set(main.workload))]
    Z = StandardScaler().fit_transform(wf[WORKLOAD_FEATS].values)
    rng = np.random.default_rng(seed)

    def kmedoids(Z, k, iters=50):
        med = list(rng.choice(len(Z), k, replace=False))
        for _ in range(iters):
            dist = ((Z[:, None, :] - Z[None, med, :]) ** 2).sum(-1)
            lab = dist.argmin(1)
            new = []
            for c in range(k):
                idx = np.where(lab == c)[0]
                if not len(idx):
                    new.append(med[c]); continue
                dd = ((Z[idx][:, None, :] - Z[None, idx, :]) ** 2).sum(-1).sum(1)
                new.append(idx[dd.argmin()])
            if new == med:
                break
            med = new
        cost = ((Z[:, None, :] - Z[None, med, :]) ** 2).sum(-1).min(1).sum()
        return med, cost
    best = min((kmedoids(Z, k) for _ in range(30)), key=lambda r: r[1])[0]
    rep = list(wf.index[best])
    top = pd.read_csv(os.path.join(ROOT, "survey", "out", "top_workloads.csv")).name.tolist()
    lit = [w for w in top if w in wf.index][:k]

    def score(ws):
        e_med, e_spread = [], []
        for t in TECHS:
            y_all = main[f"y_{t}"].dropna()
            y_s = main[main.workload.isin(ws)][f"y_{t}"].dropna()
            if len(y_s) < 5:
                continue
            e_med.append(abs(y_s.median() - y_all.median()))
            e_spread.append(abs((y_s.quantile(.9) - y_s.quantile(.1)) - (y_all.quantile(.9) - y_all.quantile(.1))))
        return float(2 ** np.mean(e_med)), float(np.mean(e_spread))
    rnd = [score(list(rng.choice(wf.index, k, replace=False))) for _ in range(trials)]
    out = pd.DataFrame([
        {"suite": f"representative k-medoids (k={k})", "workloads": "; ".join(rep), "median_err_x": score(rep)[0], "spread_err_log2": score(rep)[1]},
        {"suite": f"literature top-{k}", "workloads": "; ".join(lit), "median_err_x": score(lit)[0], "spread_err_log2": score(lit)[1]},
        {"suite": f"random {k} (median of {trials})", "workloads": "", "median_err_x": float(np.median([r[0] for r in rnd])),
         "spread_err_log2": float(np.median([r[1] for r in rnd]))},
    ])
    return out, rep


def regime(wf):
    """Reuse regime of a workload fingerprint: the axis that decides KV-technique outcomes."""
    tot = wf.w_prefix + wf.w_chunk
    r = np.select([tot < 0.05, tot < 0.40, tot < 0.75], ["none", "low", "mid"], "high")
    long = np.where(wf.w_log_in_mean >= np.log10(8000), "long", "short")
    chunk = np.where(wf.w_chunk >= 0.02, "+chunk", "")
    return pd.Series([f"{a}/{b}{c}" for a, b, c in zip(r, long, chunk)], index=wf.index)


def stratified_suite(X, per_stratum=1, seed=0):
    """One medoid workload per reuse regime, reweighted by the regime's share of the population.
    Reports how well the weighted suite reproduces the population result."""
    main = X[X.setting == "main"]
    wf = workload_fingerprints().loc[sorted(set(main.workload))]
    reg = regime(wf)
    Z = pd.DataFrame(StandardScaler().fit_transform(wf[WORKLOAD_FEATS].values), index=wf.index)
    pick, weight = [], {}
    for r, ws in reg.groupby(reg):
        idx = list(ws.index)
        zz = Z.loc[idx].values
        order = ((zz[:, None, :] - zz[None, :, :]) ** 2).sum(-1).sum(1).argsort()
        chosen = [idx[i] for i in order[:per_stratum]]
        pick += chosen
        for w in chosen:
            weight[w] = len(idx) / len(chosen)
    e_med, e_spread = [], []
    for t in TECHS:
        y_all = main[f"y_{t}"].dropna()
        sub = main[main.workload.isin(pick)][["workload", f"y_{t}"]].dropna()
        if len(sub) < 5:
            continue
        w = sub.workload.map(weight).values
        o = np.argsort(sub[f"y_{t}"].values)
        v, c = sub[f"y_{t}"].values[o], np.cumsum(w[o]) / w.sum()
        q = lambda p: v[np.searchsorted(c, p)]
        e_med.append(abs(q(.5) - y_all.median()))
        e_spread.append(abs((q(.9) - q(.1)) - (y_all.quantile(.9) - y_all.quantile(.1))))
    return {"suite": f"stratified by reuse regime ({len(pick)} workloads, weighted)", "workloads": "; ".join(pick),
            "median_err_x": float(2 ** np.mean(e_med)), "spread_err_log2": float(np.mean(e_spread))}, reg


# ------------------------------------------------------------ prediction
def predict(profile: dict, model_key: str, gpu="H100", disk="NVMe", load=0.7):
    """Fingerprint a NEW workload (from kvwl.profile output row) on any catalog model."""
    X = build_table()
    M = load_models()
    m = M[model_key]
    mf = model_fingerprints().loc[model_key]
    row = {
        "w_log_in_mean": np.log10(max(1, profile["in_mean"])), "w_log_in_p99": np.log10(max(1, profile["in_p99"])),
        "w_log_out_mean": np.log10(max(1, profile["out_mean"])), "w_prefix": profile["prefix_hit_ratio"],
        "w_chunk": profile.get("nonprefix_extra_ratio", 0.0), "w_unique": profile.get("unique_block_ratio", 1.0),
        "w_turns": np.log2(max(1, profile.get("turns_per_session", 1))),
        "w_in_cv": min(50, profile.get("in_p90", profile["in_mean"]) / max(1, profile.get("in_p50", profile["in_mean"]))),
        **mf.to_dict(), "c_trunc": float(profile["in_mean"] > m.ctx),
        "c_log_ws_over_hbm": np.log10(max(1e-4, m.kv_bytes_per_token() * profile["in_mean"] * profile.get("requests", 1000)
                                          * profile.get("unique_block_ratio", 1.0) / max(1e9, 80e9 * auto_gpu_count(m, Hardware()) * 0.9 - m.weight_bytes()))),
        "h_log_flops": np.log10(GPU[gpu][0]), "h_log_hbm_bw": np.log10(GPU[gpu][1]), "h_log_disk_bw": np.log10(DISK[disk]), "h_load": load}
    x = np.array([[row[f] for f in FEATS]])
    out = {}
    for t in TECHS:
        if t == "cacheblend" and row["w_chunk"] < 0.005:
            out[t] = None   # no position-independent chunks to reuse: CacheBlend is out of its training domain
            continue
        y = X[f"y_{t}"]
        ok = y.notna()
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)
        reg.fit(X.loc[ok, FEATS].values, y[ok].values)
        out[t] = float(2 ** reg.predict(x)[0])
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
    clf.fit(X[FEATS].values, X.y_best.values)
    out["recommended"] = clf.predict(x)[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--workload-profile")
    ap.add_argument("--model-key")
    a = ap.parse_args(argv)
    if a.predict:
        print(json.dumps(predict(json.load(open(a.workload_profile)), a.model_key), indent=1))
        return
    os.makedirs(OUT, exist_ok=True)
    X = build_table()
    print(f"training table: {len(X)} cells, {X.workload.nunique()} workloads, {X.model.nunique()} models, {X.setting.nunique()} settings")
    regs, clss = [], []
    for g in ("workload", "model"):
        r, c = grouped_cv(X, g)
        regs.append(r)
        clss.append(c)
    reg = pd.concat(regs)
    reg.to_csv(os.path.join(OUT, "gain_predictor_cv.csv"), index=False)
    pd.DataFrame(clss).to_csv(os.path.join(OUT, "recommender_cv.csv"), index=False)
    imp = importances(X)
    imp.to_csv(os.path.join(OUT, "feature_group_importance.csv"), index=False)
    strat, reg = stratified_suite(X)
    reg.rename("regime").to_csv(os.path.join(OUT, "workload_regimes.csv"))
    k = len(strat["workloads"].split("; "))
    suite, rep = representative_suite(X, k=k)
    suite = pd.concat([pd.DataFrame([strat]), suite], ignore_index=True)
    suite.to_csv(os.path.join(OUT, "representative_suite.csv"), index=False)
    pd.set_option("display.width", 220)
    print(reg.round(3).to_string(index=False))
    print(pd.DataFrame(clss).round(3).to_string(index=False))
    print(imp.round(3).to_string(index=False))
    print(suite.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
