"""Hugging Face normalizers for the `hf_dataset` sources in catalog/sources.yaml.

Called by datasets/scripts/fetch.py (`make fetch-hf`). Each normalizer streams a
dataset with `datasets.load_dataset` and writes data/raw/hf/<workload>.jsonl in one
of four record shapes consumed by kvwl.workloads.build_hf:
  chat     {"session", "turns": [user, assistant, ...]}
  doc_qa   {"doc_id", "doc", "question", "answer"}            (doc shared by questions)
  mc       {"ctx", "choices": [...]}                          (choices share ctx)
  agent    {"session", "steps": [{"input_append", "output"}]} (history re-sent each step)
`python -m kvwl.profile --use-hf` then rebuilds those workloads as `measured`.
Gated sets (lmsys/lmsys-chat-1m, Idavidrein/gpqa) need HF_TOKEN and an accepted license.
"""
import json
import os

from .paths import RAW

OUT = os.path.join(RAW, "hf")
N = 2000


def load_dataset(*args, **kwargs):
    from datasets import load_dataset as _ld  # optional dependency: pip install datasets
    return _ld(*args, **kwargs)


def dump(name, shape, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        f.write(json.dumps({"shape": shape}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} rows ({shape})")


def take(ds, n=N):
    for i, r in enumerate(ds):
        if i >= n:
            break
        yield r


def sharegpt():
    ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                      data_files="ShareGPT_V3_unfiltered_cleaned_split.json", split="train", streaming=True)
    rows = []
    for r in take(ds, 3 * N):
        t = [c["value"] for c in r["conversations"]]
        if len(t) >= 2:
            rows.append({"session": r["id"], "turns": t})
    dump("ShareGPT", "chat", rows[:N])


def lmsys():
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    dump("LMSYS-Chat-1M", "chat", [{"session": r["conversation_id"], "turns": [c["content"] for c in r["conversation"]]}
                                    for r in take(ds) if r["language"] == "English"])


def _hf_jsonl_head(repo, path, n):
    """First n records of a JSONL file in a HF dataset repo, streamed over HTTP (no full download)."""
    import urllib.request
    req = urllib.request.Request(f"https://huggingface.co/datasets/{repo}/resolve/main/{path}", headers={"User-Agent": "kvwl-fetch"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    out = []
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def infinitebench():
    """Per-task JSONL files read directly: `load_dataset` cannot unify the tasks' schemas."""
    rows = []
    for task in ("longbook_qa_eng", "longbook_choice_eng", "longbook_sum_eng", "passkey", "kv_retrieval", "code_debug", "math_find"):
        for r in _hf_jsonl_head("xinrongzhang2022/InfiniteBench", f"{task}.jsonl", 60):
            rows.append({"doc_id": f"{task}:{hash(r['context'][:2000])}", "doc": r["context"], "question": r["input"],
                         "answer": str(r["answer"])[:2000]})
    dump("InfiniteBench", "doc_qa", rows)


def longbench_v2():
    ds = load_dataset("zai-org/LongBench-v2", split="train")
    dump("LongBench v2", "doc_qa", [{"doc_id": r["_id"], "doc": r["context"],
                                     "question": r["question"] + "\n" + "\n".join(f"({k}) {r['choice_' + k]}" for k in "ABCD"),
                                     "answer": r["answer"]} for r in ds])


def booksum():
    ds = load_dataset("kmfoda/booksum", split="test", streaming=True)
    dump("BookSum", "doc_qa", [{"doc_id": r["bid"], "doc": r["chapter"], "question": "Summarize the chapter.",
                                "answer": r["summary_text"]} for r in take(ds, 300)])


def scbench():
    rows = []
    for cfg in ("scbench_kv", "scbench_qa_eng", "scbench_repoqa", "scbench_summary", "scbench_many_shot"):
        ds = load_dataset("microsoft/SCBench", cfg, split="test", streaming=True)
        for r in take(ds, 30):
            for q in r["multi_turns"]:
                rows.append({"doc_id": f"{cfg}:{r['id']}", "doc": r["context"], "question": q["input"], "answer": str(q["answer"])})
    dump("SCBench", "doc_qa", rows)


def loogle():
    """One row per question (current Hub schema): doc_id, context, question, answer."""
    rows = []
    for cfg in ("longdep_qa", "shortdep_qa"):
        ds = load_dataset("bigai-nlco/LooGLE", cfg, split="test", streaming=True)
        for r in take(ds, 600):
            rows.append({"doc_id": r["doc_id"], "doc": r["context"], "question": r["question"], "answer": r["answer"]})
    dump("LooGLE", "doc_qa", rows)


def swebench_agent():
    """Real agent trajectories (SWE-agent on SWE-bench) -> per-step appended context."""
    ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
    rows = []
    for r in take(ds, 300):
        steps, pending = [], ""
        for m in r["trajectory"]:
            if m["role"] in ("system", "user"):
                pending += m.get("content") or m.get("text") or ""
            elif m["role"] == "ai":
                steps.append({"input_append": pending, "output": m.get("content") or m.get("text") or ""})
                pending = ""
        rows.append({"session": r["instance_id"], "steps": steps})
    dump("SWE-bench", "agent", rows)


def gpqa():
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    dump("GPQA", "doc_qa", [{"doc_id": str(i), "doc": "", "question": r["Question"], "answer": r["Explanation"]} for i, r in enumerate(ds)])


def mc(name, path, cfg, split, ctx, choices):
    ds = load_dataset(path, cfg, split=split, streaming=True) if cfg else load_dataset(path, split=split, streaming=True)
    dump(name, "mc", [{"ctx": ctx(r), "choices": choices(r)} for r in take(ds, 500)])


def hellaswag():
    mc("HellaSwag", "Rowan/hellaswag", None, "validation", lambda r: r["ctx"], lambda r: r["endings"])


def winogrande():
    mc("WinoGrande", "allenai/winogrande", "winogrande_xl", "validation",
       lambda r: "", lambda r: [r["sentence"].replace("_", r["option1"]), r["sentence"].replace("_", r["option2"])])


def arc():
    mc("ARC", "allenai/ai2_arc", "ARC-Challenge", "test", lambda r: "Question: " + r["question"] + "\nAnswer:",
       lambda r: r["choices"]["text"])


def boolq():
    mc("BoolQ", "google/boolq", None, "validation",
       lambda r: f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", lambda r: [" yes", " no"])


def copa():
    mc("COPA", "pkavumba/balanced-copa", None, "test", lambda r: r["premise"] + (" because" if r["question"] == "cause" else " therefore"),
       lambda r: [r["choice1"], r["choice2"]])
