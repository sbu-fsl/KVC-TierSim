"""Mine paper code artifacts for the workloads and models they evaluate on.

Usage: python -I mine_repos.py <corpus_raw.json> <repos_dir> <out.json>

Evidence rules (kept deliberately conservative):
  * only text files (md/py/sh/yaml/json/txt/toml/cfg/ipynb), <= 512 KB;
  * skip vendored / third-party trees and framework forks' model registries;
  * a file that names > 8 distinct models (or > 10 workloads) is treated as a
    registry/catalog and ignored;
  * for very large repos (framework forks) only README + experiment-ish
    directories are scanned.
A paper "uses" an item if any scanned file in its repo mentions it.
"""
import json, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dictionaries import WORKLOADS, MODELS

TEXT_EXT = {".md", ".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg", ".ipynb", ".rst", ".bash"}
SKIP_DIRS = re.compile(r"(^|/)(\.git|lm_eval|lm-evaluation-harness|lm_evaluation_harness|fastchat|opencompass|third_party|3rdparty|thirdparty|vendor|node_modules|site-packages|external|extern|submodules?|csrc|kernels?|cuda|include|build|dist|assets|docs?/api)(/|$)", re.I)
EXP_DIRS = re.compile(r"(^|/)(bench\w*|eval\w*|exp\w*|scripts?|configs?|examples?|run\w*|results?|artifact\w*|reproduce\w*|ae|osdi\w*|sosp\w*|nsdi\w*|eurosys\w*|fast\w*|atc\w*|mlsys\w*)(/|$)", re.I)

W_RE = {k: re.compile("|".join(f"(?:{p})" for p in v[1]), re.I) for k, v in WORKLOADS.items()}
M_RE = {k: re.compile("|".join(f"(?:{p})" for p in v), re.I) for k, v in MODELS.items()}


def scan_repo(path):
    files = []
    for root, dirs, fs in os.walk(path):
        rel = os.path.relpath(root, path)
        if SKIP_DIRS.search(rel.replace(os.sep, "/")):
            dirs[:] = []
            continue
        for f in fs:
            if os.path.splitext(f)[1].lower() in TEXT_EXT:
                files.append(os.path.join(rel, f))
    big = len(files) > 1500
    w_hits, m_hits = Counter(), Counter()
    w_readme, m_readme = set(), set()
    for rel in files:
        r = rel.replace(os.sep, "/")
        if big and not (r.count("/") <= 1 and r.lower().startswith(("./readme", "readme"))) and not EXP_DIRS.search(r):
            continue
        fp = os.path.join(path, rel)
        try:
            sz = os.path.getsize(fp)
            if sz > 256_000 or (fp.endswith((".json", ".txt")) and sz > 64_000):
                continue
            txt = open(fp, errors="ignore").read()
        except OSError:
            continue
        ws = {k for k, rx in W_RE.items() if rx.search(txt)}
        ms = {k for k, rx in M_RE.items() if rx.search(txt)}
        base = os.path.basename(r).lower()
        if base.startswith(("model2path", "model2maxlen", "dataset2")):
            ms = set()
        if len(ms) > 8:
            ms = set()
        if len(ws) > 10:
            ws = set()
        w_hits.update(ws)
        m_hits.update(ms)
        if base.startswith("readme"):
            w_readme |= ws; m_readme |= ms
    return {"n_files": len(files), "framework_like": big, "workloads": dict(w_hits), "models": dict(m_hits),
            "workloads_readme": sorted(w_readme), "models_readme": sorted(m_readme)}


def main(corpus_path, repos_dir, out_path):
    corpus = json.load(open(corpus_path))
    out = {}
    jobs = []
    for ax, e in corpus.items():
        # the paper's own repo is the first GitHub link on its list line(s)
        repo = None
        for line in e["lines"]:
            m = re.search(r"github\.com/([\w\-\.]+/[\w\-\.]+)", line)
            if m:
                cand = m.group(1).rstrip(".").removesuffix(".git")
                if os.path.isdir(os.path.join(repos_dir, cand.replace("/", "_"))):
                    repo = cand
                    break
        if not repo:
            continue
        title = re.sub(r"\s+", " ", e["lines"][0])[:200]
        jobs.append((ax, repo, title, os.path.join(repos_dir, repo.replace("/", "_"))))
    from multiprocessing import Pool
    with Pool(os.cpu_count()) as pool:
        results = pool.map(scan_repo, [j[3] for j in jobs], chunksize=1)
    for (ax, repo, title, _), res in zip(jobs, results):
        out[ax] = {"arxiv": ax, "repo": repo, "title_line": title, **res}
    json.dump(out, open(out_path, "w"), indent=1)
    wc, mc = Counter(), Counter()
    for p in out.values():
        wc.update(p["workloads"].keys()); mc.update(p["models"].keys())
    print(f"papers with artifacts: {len(out)}")
    print("TOP WORKLOADS", wc.most_common(60))
    print("TOP MODELS", mc.most_common(60))


if __name__ == "__main__":
    main(*sys.argv[1:4])
