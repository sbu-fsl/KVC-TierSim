"""Combine full-text and artifact evidence into top-N workload/model rankings.

A paper counts as using an item when ANY of:
  (F) its evaluation section names it (paper_evidence.yaml, full text), or
  (R) its code artifact's README names it, or
  (C) >= 2 distinct non-vendored files in its artifact name it.
Each paper counts once per item. Ties break on full-text count, then name.

Usage: python -I rank.py <survey_dir> <out_dir> [N]
"""
import csv, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dictionaries import WORKLOADS, MODELS
import yaml

W_RE = {k: re.compile("|".join(f"(?:{p})" for p in v[1]), re.I) for k, v in WORKLOADS.items()}
M_RE = {k: re.compile("|".join(f"(?:{p})" for p in v), re.I) for k, v in MODELS.items()}
CODE_MIN_FILES = 2


def canon(text, table):
    return {k for k, rx in table.items() if rx.search(text)}


def collect(sdir):
    """Per-paper evidence: {id: {id, title, venue, year, technique, src, repo?, W: {item: {F,R,C}}, M: {...}}}."""
    pe = yaml.safe_load(open(os.path.join(sdir, "paper_evidence.yaml")))["papers"]
    art = json.load(open(os.path.join(sdir, "artifact_evidence.json")))
    papers = {}
    for p in pe:
        papers[p["id"]] = {"id": p["id"], "title": p["title"], "venue": p["venue"], "year": p["year"],
                           "technique": ",".join(p["technique"]), "src": {"F"},
                           "W": {k: {"F"} for k in canon(p["datasets"], W_RE)},
                           "M": {k: {"F"} for k in canon(p["models"], M_RE)}}
    for ax, a in art.items():
        rec = papers.setdefault(ax, {"id": ax, "title": a["title_line"][:120], "venue": "arXiv",
                                     "year": 2000 + int(ax[:2]), "technique": "", "src": set(), "W": {}, "M": {}})
        rec["repo"] = a["repo"]
        rec["src"].add("A")
        for kind, cnt_key, rd_key in (("W", "workloads", "workloads_readme"), ("M", "models", "models_readme")):
            for k in a.get(rd_key, []):
                rec[kind].setdefault(k, set()).add("R")
            for k, c in a[cnt_key].items():
                if c >= CODE_MIN_FILES:
                    rec[kind].setdefault(k, set()).add("C")
    return papers


def rank(papers, kind):
    """All items of `kind` ("W" or "M") ranked by #papers, then #full-text papers, then name."""
    agg = defaultdict(lambda: {"papers": 0, "F": 0, "R": 0, "C": 0, "years": defaultdict(int), "examples": [], "ids": []})
    for p in papers.values():
        for k, srcs in p[kind].items():
            a = agg[k]
            a["papers"] += 1
            a["ids"].append(p["id"])
            for s in srcs:
                a[s] += 1
            a["years"][p["year"]] += 1
            if "F" in srcs and len(a["examples"]) < 6:
                a["examples"].append(f'{p["title"]} ({p["venue"]})')
    return sorted(agg.items(), key=lambda kv: (-kv[1]["papers"], -kv[1]["F"], kv[0]))


def main(sdir, odir, n=50):
    n = int(n)
    papers = collect(sdir)
    os.makedirs(odir, exist_ok=True)
    summary = {}
    for kind, table, fname in (("W", WORKLOADS, "top_workloads.csv"), ("M", MODELS, "top_models.csv")):
        ranked = rank(papers, kind)
        with open(os.path.join(odir, fname), "w", newline="") as f:
            w = csv.writer(f)
            hdr = ["rank", "name", "category" if kind == "W" else "family", "papers", "fulltext_eval", "artifact_readme", "artifact_code",
                   "papers_2024", "papers_2025", "papers_2026", "example_fulltext_papers"]
            w.writerow(hdr)
            for i, (k, a) in enumerate(ranked[:n], 1):
                cat = table[k][0] if kind == "W" else ""
                w.writerow([i, k, cat, a["papers"], a["F"], a["R"], a["C"], a["years"].get(2024, 0), a["years"].get(2025, 0),
                            a["years"].get(2026, 0), " | ".join(a["examples"])])
        summary[kind] = [k for k, _ in ranked[:n]]
    with open(os.path.join(odir, "papers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "venue", "year", "technique", "evidence", "repo", "workloads", "models"])
        for p in sorted(papers.values(), key=lambda p: p["id"]):
            w.writerow([p["id"], p["title"], p["venue"], p["year"], p["technique"], "+".join(sorted(p["src"])), p.get("repo", ""),
                        "; ".join(sorted(p["W"])), "; ".join(sorted(p["M"]))])
    json.dump(summary, open(os.path.join(odir, "top_lists.json"), "w"), indent=1)
    print(f"papers={len(papers)} (fulltext={sum('F' in p['src'] for p in papers.values())}, artifacts={sum('A' in p['src'] for p in papers.values())})")
    for kind in ("W", "M"):
        print(kind, summary[kind])


if __name__ == "__main__":
    main(*sys.argv[1:])
