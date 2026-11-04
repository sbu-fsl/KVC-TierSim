"""Download the raw sources listed in catalog/sources.yaml into datasets/data/raw/.

    python -m kvwl.fetch                    # every non-HF-dataset source (files, archives, GitHub dirs)
    python -m kvwl.fetch --hf               # also the `hf_dataset` sources (needs `pip install datasets`)
    python -m kvwl.fetch --only longbench leval
    python -m kvwl.fetch --check            # verify every source is reachable, download nothing
    python -m kvwl.fetch --list

Existing files are kept unless --force. HF_TOKEN (gated Hugging Face sets) and
GITHUB_TOKEN or GH_TOKEN (GitHub API rate limit for directory listings) are used when set;
`make` loads them from secrets.env.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
import zipfile

import yaml

from .paths import CATALOG, DOWNLOAD_CACHE as CACHE, RAW, ROOT

SOURCES = os.path.join(CATALOG, "sources.yaml")
HF = "https://huggingface.co"
_RAN_HF = False


def load_sources(path: str = SOURCES) -> dict:
    return yaml.safe_load(open(path))


# ------------------------------------------------------------------ http
def _request(url: str, method: str = "GET") -> urllib.request.Request:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "kvwl-fetch"})
    if url.startswith(HF) and os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    gh = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if url.startswith("https://api.github.com") and gh:
        req.add_header("Authorization", f"Bearer {gh}")
    return req


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=30) as r:
            return r.status < 400
    except urllib.error.HTTPError as e:
        if e.code == 405:  # HEAD not allowed: fall back to a 1-byte GET
            req = _request(url)
            req.add_header("Range", "bytes=0-0")
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status < 400
        return False
    except urllib.error.URLError:
        return False


def _download(url: str, dest: str, force: bool) -> str:
    if os.path.exists(dest) and not force:
        return "kept"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(_request(url), timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)
    return "downloaded"


# ------------------------------------------------------------------ templating
def _expand(mapping: dict, spec: dict) -> list[tuple[str, str]]:
    """Expand "{var}" placeholders in mapping keys/values over the list spec[var]."""
    out = []
    for src, dst in mapping.items():
        names = sorted(set(re.findall(r"{(\w+)}", src)))
        if not names:
            out.append((src, dst))
            continue
        # "{task}" expands over spec["task"] or, if absent, spec["tasks"]
        for combo in itertools.product(*(spec[n] if n in spec else spec[n + "s"] for n in names)):
            sub = dict(zip(names, combo))
            out.append((src.format(**sub), dst.format(**sub)))
    return out


def _hf_resolve(repo: str, path: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/main/{path}"


def _github_dir(repo: str, path: str, ref: str = "HEAD") -> list[tuple[str, str]]:
    api = f"https://api.github.com/repos/{repo}/contents/{path}" + (f"?ref={ref}" if ref != "HEAD" else "")
    with urllib.request.urlopen(_request(api), timeout=60) as r:
        items = json.load(r)
    return [(it["download_url"], it["name"]) for it in items if it["type"] == "file"]


# ------------------------------------------------------------------ per kind
def plan(key: str, spec: dict) -> list[tuple[str, str]]:
    """(url, raw-relative path) pairs a source resolves to. hf_zip yields the archive itself."""
    kind = spec["kind"]
    if kind == "url":
        return _expand(spec["files"], spec)
    if kind == "hf_file":
        return [(_hf_resolve(spec["repo"], s), d) for s, d in _expand(spec["files"], spec)]
    if kind == "hf_zip":
        return [(_hf_resolve(spec["repo"], spec["file"]), os.path.join(CACHE, spec["repo"].replace("/", "__"), spec["file"]))]  # absolute: kept out of raw/
    if kind == "github_dir":
        keep = re.compile(spec.get("match", r".*"))  # regex on file names, e.g. "\\.json$"
        return [(u, os.path.join(d, name)) for s, d in spec["dirs"].items()
                for u, name in _github_dir(spec["repo"], s, spec.get("ref", "HEAD")) if keep.search(name)]
    if kind == "hf_dataset":
        return []
    raise ValueError(f"{key}: unknown kind {kind!r}")


def fetch(key: str, spec: dict, force: bool = False, check: bool = False) -> bool:
    kind = spec["kind"]
    if kind == "hf_dataset":
        url = f"{HF}/api/datasets/{spec['repo']}"
        if check:
            ok = _reachable(url)
            print(f"  {'ok ' if ok else 'BAD'} {url}" + ("  (gated: needs HF_TOKEN)" if spec.get("gated") else ""))
            return ok or bool(spec.get("gated"))
        from . import hf_sources
        getattr(hf_sources, spec["normalizer"])()
        return True
    ok = True
    for url, rel in plan(key, spec):
        dest = os.path.normpath(os.path.join(RAW, rel))
        if check:
            good = _reachable(url)
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} {url}")
            continue
        print(f"  {_download(url, dest, force):10s} {dest}")
        if kind == "hf_zip":
            with zipfile.ZipFile(dest) as z:
                for member, out in _expand(spec["members"], spec):
                    target = os.path.join(RAW, out)
                    if os.path.exists(target) and not force:
                        continue
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with z.open(member) as src, open(target, "wb") as f:
                        shutil.copyfileobj(src, f)
                    print(f"  {'extracted':10s} {target}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="source keys (see --list)")
    ap.add_argument("--hf", action="store_true", help="include hf_dataset sources (normalized via kvwl.hf_sources)")
    ap.add_argument("--hf-only", action="store_true", help="only the hf_dataset sources")
    ap.add_argument("--check", action="store_true", help="check reachability only")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    sources = load_sources()
    if a.list:
        for k, s in sources.items():
            print(f"{k:18s} {s['kind']:11s} {s.get('repo', '')}")
        return 0
    failed, ran_hf = [], False
    for key, spec in sources.items():
        if a.only and key not in a.only:
            continue
        is_ds = spec["kind"] == "hf_dataset"
        if not a.only and not a.check and ((is_ds and not (a.hf or a.hf_only)) or (a.hf_only and not is_ds)):
            continue
        print(f"[{key}] {spec['kind']}")
        ran_hf |= is_ds and not a.check
        try:
            if not fetch(key, spec, force=a.force, check=a.check):
                failed.append(key)
        except Exception as e:  # keep going; report at the end
            print(f"  !! {e!r}")
            failed.append(key)
    global _RAN_HF
    _RAN_HF = ran_hf
    if failed:
        print(f"\nFAILED: {' '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nraw data in {RAW}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # `datasets` streaming can leave non-daemon HTTP threads behind, so the interpreter
    # never exits after an --hf run; everything is written by now, so exit hard.
    os._exit(rc) if _RAN_HF else sys.exit(rc)
