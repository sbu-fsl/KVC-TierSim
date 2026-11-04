"""The KVWL catalog: papers, the workloads (datasets/traces) they evaluate on, and models.

    catalog/papers.yaml     every surveyed paper -> the workloads and models it uses   (generated)
    catalog/workloads.yaml  every workload: how to build it + which papers use it
    catalog/models.yaml     every model:    KV geometry      + which papers use it
    catalog/sources.yaml    where each workload's raw data is fetched from
    catalog/links.csv       flat (workload|model, paper) table                       (generated)

`papers.yaml` and the `rank` / `papers` fields of workloads.yaml and models.yaml are
generated from the survey evidence (survey/paper_evidence.yaml + artifact_evidence.json)
by `python -m kvwl.catalog sync`; everything else in those files is curated by hand
and preserved. A workload whose survey name differs from its catalog key (e.g. the
four Mooncake traces all count as the survey's "Mooncake trace") says so with
`survey_name`.

    python -m kvwl.catalog sync            # regenerate paper links + papers.yaml
    python -m kvwl.catalog check           # validate cross-references
    python -m kvwl.catalog show LongBench  # one entry with its papers
    python -m kvwl.catalog stats
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from functools import lru_cache

import yaml

from .paths import CATALOG, SURVEY

PAPERS_YAML = os.path.join(CATALOG, "papers.yaml")
WORKLOADS_YAML = os.path.join(CATALOG, "workloads.yaml")
MODELS_YAML = os.path.join(CATALOG, "models.yaml")
SOURCES_YAML = os.path.join(CATALOG, "sources.yaml")
LINKS_CSV = os.path.join(CATALOG, "links.csv")
GENERATED = ("rank", "papers")  # fields owned by `sync`


# ------------------------------------------------------------------ loading
def _load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str) -> dict:
    import json
    with open(path) as f:
        return json.load(f)


def load_papers() -> dict:
    return _load(PAPERS_YAML)


def load_sources() -> dict:
    return _load(SOURCES_YAML)


def load_workloads(buildable: bool = False) -> dict:
    """name -> spec. `buildable=True` keeps only workloads with a stream builder."""
    out = {}
    for name, v in _load(WORKLOADS_YAML).items():
        spec = dict(v, args=v.get("args") or {}, papers=v.get("papers") or [])
        if buildable and not spec.get("builder"):
            continue
        out[name] = spec
    return out


@lru_cache(maxsize=1)
def _categories() -> dict:
    return {n: e.get("category", "trace") for n, e in load_workloads().items()}


def category(name: str) -> str:
    return _categories().get(name, "trace")


def survey_key(name: str, spec: dict) -> str:
    return spec.get("survey_name", name)


# ------------------------------------------------------------------ writing
def _header(path: str) -> str:
    """Leading comment block of an existing catalog file (kept verbatim on rewrite)."""
    lines = []
    if os.path.exists(path):
        for line in open(path):
            if line.startswith("#") or (lines and not line.strip()):
                lines.append(line)
            else:
                break
    return "".join(lines).rstrip("\n") + "\n\n" if lines else ""


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):  # never emit &anchors / *aliases
        return True


def dump(path: str, data: dict, header: str | None = None) -> None:
    body = yaml.dump(data, Dumper=_Dumper, sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)
    # blank line between top-level entries
    out = []
    for line in body.splitlines():
        if out and line and not line.startswith((" ", "-")):
            out.append("")
        out.append(line)
    with open(path, "w") as f:
        head = header.rstrip("\n") + "\n\n" if header else _header(path)
        f.write(head + "\n".join(out) + "\n")


# ------------------------------------------------------------------ sync
def _survey():
    sys.path.insert(0, SURVEY)
    import rank as survey_rank  # survey/rank.py
    from dictionaries import WORKLOADS as SURVEY_WORKLOADS
    return survey_rank, SURVEY_WORKLOADS


def _order(entries: dict) -> dict:
    return dict(sorted(entries.items(), key=lambda kv: (kv[1].get("rank") is None, kv[1].get("rank") or 0, kv[0].lower())))


def _link(entries: dict, ranked: list, stub) -> tuple[dict, list]:
    """Set rank/papers on every entry; add a stub for surveyed items with no entry yet."""
    idx = {k: (i, sorted(a["ids"])) for i, (k, a) in enumerate(ranked, 1)}
    covered = {survey_key(n, e) for n, e in entries.items()}
    added = []
    for k, _ in ranked:
        if k not in covered:
            entries[k] = stub(k)
            added.append(k)
    for n, e in entries.items():
        for g in GENERATED:
            e.pop(g, None)
        r, ids = idx.get(survey_key(n, e), (None, []))
        e["rank"], e["papers"] = r, ids
    return _order(entries), added


def clean_title(line: str) -> str:
    """Paper title from a curated-list markdown line (artifact-only papers)."""
    import re
    t = line.strip()
    m = re.search(r"\[\*\*(.+?)\*\*\]", t)
    if m:
        after = t[m.end():].strip()
        if after and not after.startswith("("):          # "[**Arrow**] Arrow: Adaptive ...(@Wu et al.)"
            t = re.split(r"\(@|<br>|\|", after)[0]
        else:                                             # "17. [**Title.**](https://arxiv.org/...)"
            t = m.group(1)
    elif t.startswith("|"):                               # "| 2025 | Title | method | ..."
        cells = [c.strip() for c in t.strip("|").split("|")]
        t = next((c for c in cells if c and not re.fullmatch(r"[\d.\s]+", c)), t)
    else:                                                 # "- Title <br> Arxiv 2024 [[Paper]](...)"
        t = re.split(r"<br>|\[\[", t.lstrip("-*0123456789. "))[0]
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)        # inline links -> text
    t = re.split(r"\(@|\s\|\s?|\|\s", t)[0]                 # "...(@PKU etc)", "... | NVIDIA"
    t = re.sub(r"^[^\w\[]+", "", t)                         # leading emoji / bullets
    m = re.match(r"\[([^\]]+)\]\s*(.*)", t)                 # "[MoA] MoA: ..." -> "MoA: ..."
    if m and m.group(2).lower().startswith(m.group(1).lower()):
        t = m.group(2)
    m = re.match(r"([^:]+):\s*\1:\s*(.*)", t)                 # "X: X: rest" -> "X: rest"
    if m:
        t = f"{m.group(1)}: {m.group(2)}"
    return re.sub(r"\s+", " ", t.replace("**", "")).strip(" .-:")


def sync(verbose: bool = True) -> None:
    survey_rank, survey_workloads = _survey()
    papers = survey_rank.collect(SURVEY)
    art = _load_json(os.path.join(SURVEY, "artifact_evidence.json"))
    for pid, p in papers.items():
        if "F" not in p["src"] and pid in art:
            p["title"] = clean_title(art[pid]["title_line"])

    out = {}
    for pid in sorted(papers):
        p = papers[pid]
        out[pid] = {"title": p["title"], "venue": p["venue"], "year": p["year"],
                    "technique": [t for t in p["technique"].split(",") if t],
                    "evidence": sorted(p["src"]), "repo": p.get("repo"),
                    "workloads": sorted(p["W"]), "models": sorted(p["M"])}
    dump(PAPERS_YAML, out)

    wl, added_w = _link(_load(WORKLOADS_YAML), survey_rank.rank(papers, "W"),
                        lambda k: {"category": survey_workloads[k][0], "hf": None, "source": [], "builder": None})
    dump(WORKLOADS_YAML, wl)
    md, added_m = _link(_load(MODELS_YAML), survey_rank.rank(papers, "M"), lambda k: {"hf": None, "status": "pending"})
    dump(MODELS_YAML, md)
    write_links(out, wl, md)
    if verbose:
        print(f"papers.yaml    {len(out)} papers")
        print(f"workloads.yaml {len(wl)} workloads ({sum(bool(e.get('builder')) for e in wl.values())} buildable)"
              + (f", new stubs: {added_w}" if added_w else ""))
        print(f"models.yaml    {len(md)} models ({sum(e.get('status') != 'pending' for e in md.values())} with KV geometry)"
              + (f", new stubs: {added_m}" if added_m else ""))


def write_links(papers: dict, wl: dict, md: dict) -> None:
    """catalog/links.csv: one row per (workload|model, paper) for spreadsheets / pandas."""
    import csv
    with open(LINKS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "rank", "paper_id", "title", "venue", "year", "technique"])
        for kind, table in (("workload", wl), ("model", md)):
            for name, e in table.items():
                for pid in e.get("papers") or []:
                    p = papers[pid]
                    w.writerow([kind, name, e.get("rank"), pid, p["title"], p["venue"], p["year"], "+".join(p["technique"])])


# ------------------------------------------------------------------ check
def check() -> list[str]:
    from .workloads import BUILDERS

    errs = []
    papers, wl, md, src = load_papers(), load_workloads(), _load(MODELS_YAML), load_sources()
    wkeys = defaultdict(list)
    for n, e in wl.items():
        wkeys[survey_key(n, e)].append(n)
        if e.get("builder") and e["builder"] not in BUILDERS:
            errs.append(f"workload {n!r}: unknown builder {e['builder']!r}")
        for s in e.get("source") or []:
            if s not in src:
                errs.append(f"workload {n!r}: unknown source {s!r}")
        for pid in e["papers"]:
            if pid not in papers:
                errs.append(f"workload {n!r}: unknown paper {pid!r}")
    for pid, p in papers.items():
        for w in p["workloads"]:
            if w not in wkeys:
                errs.append(f"paper {pid}: workload {w!r} not in workloads.yaml")
        for m in p["models"]:
            if m not in md:
                errs.append(f"paper {pid}: model {m!r} not in models.yaml")
    for n, e in md.items():
        for pid in e.get("papers") or []:
            if pid not in papers:
                errs.append(f"model {n!r}: unknown paper {pid!r}")
    top = _load(os.path.join(SURVEY, "out", "top_lists_final.yaml"))
    for m in top.get("models", []):
        if md.get(m, {}).get("status") == "pending" or m not in md:
            errs.append(f"top-list model {m!r} has no KV geometry in models.yaml")
    for w in top.get("workloads", []):
        if w not in wkeys:
            errs.append(f"top-list workload {w!r} not in workloads.yaml")
    return errs


def show(name: str) -> None:
    papers = load_papers()
    for kind, table in (("workload", load_workloads()), ("model", _load(MODELS_YAML))):
        if name in table:
            e = dict(table[name])
            ids = e.pop("papers", []) or []
            print(f"{kind}: {name}")
            for k, v in e.items():
                print(f"  {k}: {v}")
            print(f"  papers ({len(ids)}):")
            for pid in ids:
                p = papers.get(pid, {})
                print(f"    {pid:12s} {p.get('title', '?')[:60]:60s} {p.get('venue', '')}")
            return
    if name in papers:
        print(yaml.safe_dump({name: papers[name]}, sort_keys=False, allow_unicode=True))
        return
    raise SystemExit(f"{name!r} is not a workload, model or paper id")


def stats() -> None:
    papers, wl, md = load_papers(), load_workloads(), _load(MODELS_YAML)
    print(f"papers    {len(papers):4d}  (full-text {sum('F' in p['evidence'] for p in papers.values())}, "
          f"artifact {sum('A' in p['evidence'] for p in papers.values())})")
    print(f"workloads {len(wl):4d}  (buildable {sum(bool(e.get('builder')) for e in wl.values())}, "
          f"used by >=1 paper {sum(bool(e['papers']) for e in wl.values())})")
    print(f"models    {len(md):4d}  (with KV geometry {sum(e.get('status') != 'pending' for e in md.values())}, "
          f"used by >=1 paper {sum(bool(e.get('papers')) for e in md.values())})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    sub.add_parser("check")
    sub.add_parser("stats")
    s = sub.add_parser("show")
    s.add_argument("name")
    a = ap.parse_args(argv)
    if a.cmd == "sync":
        sync()
    elif a.cmd == "check":
        errs = check()
        for e in errs:
            print("ERROR", e)
        print("catalog OK" if not errs else f"{len(errs)} problem(s)")
        return 1 if errs else 0
    elif a.cmd == "show":
        show(a.name)
    else:
        stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
