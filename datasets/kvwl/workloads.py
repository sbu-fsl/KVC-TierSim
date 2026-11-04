"""Workload builders: turn each surveyed dataset/trace into a request stream.

Every builder reproduces how serving/KV papers *replay* that dataset (prompt
template, few-shot prefix, multi-turn history, document shared by several
questions, retrieved chunks), because that construction -- not the dataset
name -- decides the input lengths and the KV reuse a technique can exploit.

Three fidelity levels, recorded in each stream's meta["fidelity"]:
  measured   real text tokenized here, or a production trace with block hashes
  lengths    real per-request token counts, no content (reuse = none / modeled)
  published  no data reachable from this sandbox; lengths & reuse synthesized
             from statistics published in the dataset paper (see catalog notes)
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import os
import random
from functools import lru_cache

import numpy as np

from .streams import Request, chain_blocks, chunk_hash, continue_blocks, BLOCK_TOKENS

from .paths import RAW


# --------------------------------------------------------------------- tokenizers
@lru_cache(maxsize=None)
def tokenizer(kind: str):
    """'llama3' ~ 128-150K BPE vocab (Llama-3/Qwen/DeepSeek/GLM families) via
    Mistral's Tekken (131K, tiktoken-style); 'llama2' ~ 32K SentencePiece
    (Llama-2/Mistral-v0.x/Vicuna/OPT-era) via Mistral's v1 SentencePiece model.
    Both ship inside the `mistral_common` wheel, so no network is needed."""
    import mistral_common
    base = os.path.join(os.path.dirname(mistral_common.__file__), "data")
    if kind == "llama3":
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        tk = Tekkenizer.from_file(os.path.join(base, "tekken_240911.json"))
        return lambda s: tk.encode(s, bos=False, eos=False)
    if kind == "llama2":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=os.path.join(base, "tokenizer.model.v1"))
        return lambda s: sp.encode(s)
    raise KeyError(kind)


FAMILY_TOKENIZER = {"llama3": "llama3", "qwen": "llama3", "other": "llama3", "llama2": "llama2", "mistral": "llama2"}


class Builder:
    """Accumulates prompt segments into Requests for one tokenizer."""

    def __init__(self, tok_kind: str, block_tokens: int = BLOCK_TOKENS):
        self.tok = tokenizer(tok_kind)
        self.bt = block_tokens
        self.reqs: list[Request] = []
        self._cache: dict[str, list[int]] = {}
        self.keep_out = False   # set by multi-turn builders: generated KV is reused by the next turn

    def ids(self, text: str) -> list[int]:
        t = self._cache.get(text)
        if t is None:
            t = self.tok(text)
            if len(text) < 200_000:
                self._cache[text] = t
        return t

    def add(self, segments, output, session="s", turn=1, arrival=None, max_len=None):
        """segments: list of (text, reusable_chunk: bool)."""
        toks, chunk_of_tok = [], []
        for text, reusable in segments:
            ids = self.ids(text)
            cid = chunk_hash(ids) if reusable and len(ids) >= self.bt else -1
            toks.extend(ids)
            chunk_of_tok.extend([cid] * len(ids))
        if max_len and len(toks) > max_len:      # LongBench-style middle truncation
            half = max_len // 2
            toks = toks[:half] + toks[-half:]
            chunk_of_tok = chunk_of_tok[:half] + chunk_of_tok[-half:]
        out_ids = [] if isinstance(output, int) else self.ids(output)
        out_len = output if isinstance(output, int) else max(1, len(out_ids))
        blocks = chain_blocks(toks, self.bt)
        chunks = [chunk_of_tok[i] for i in range(0, len(toks), self.bt)]
        self.reqs.append(Request(rid=len(self.reqs), session=str(session), turn=turn, input_len=len(toks),
                                 output_len=int(out_len), blocks=blocks, chunk_ids=chunks, arrival=arrival,
                                 block_tokens=self.bt, out_blocks=continue_blocks(toks, out_ids, self.bt) if self.keep_out else []))


def _jsonl(path, n=None):
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if n and len(out) >= n:
                    break
    return out


def _lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


# ======================================================================= builders
LB_DIR = os.path.join(RAW, "longbench")
LB_TASKS_EN = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
               "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
LB_RAG = {"hotpotqa", "2wikimqa", "musique"}


def build_longbench(b: Builder, tasks=None, n=200, max_len=None, seed=0, **_):
    prompts = json.load(open(os.path.join(LB_DIR, "dataset2prompt.json")))
    maxlen = json.load(open(os.path.join(LB_DIR, "dataset2maxlen.json")))
    tasks = tasks or LB_TASKS_EN
    per = max(1, n // len(tasks)) if len(tasks) > 1 else n
    rng = random.Random(seed)
    items = []
    for t in tasks:
        rows = _jsonl(os.path.join(LB_DIR, f"{t}.jsonl"))
        rng.shuffle(rows) if len(tasks) > 1 else None
        items += [(t, r) for r in rows[:per]]
    for i, (t, r) in enumerate(items):
        tpl = prompts[t]
        pre, post = tpl.split("{context}") if "{context}" in tpl else ("", tpl)
        segs = [(pre, False)]
        ctx = r["context"]
        if t in LB_RAG:  # passages are independently reusable chunks (CacheBlend/EPIC/KVLink setting)
            parts = ctx.split("Passage ")
            segs += [(("Passage " + p) if j else p, True) for j, p in enumerate(parts) if p]
        else:
            segs.append((ctx, True))
        segs.append((post.replace("{input}", r.get("input", "")), False))
        b.add(segs, maxlen[t], session=f"{t}:{i}", max_len=max_len)


def build_ruler(b: Builder, length=8192, tasks=None, n=200, **_):
    d = os.path.join(RAW, "ruler", str(length))
    tasks = tasks or sorted(x[:-6] for x in os.listdir(d) if x.endswith(".jsonl"))
    per = max(1, n // len(tasks))
    for t in tasks:
        for i, r in enumerate(_jsonl(os.path.join(d, f"{t}.jsonl"), per)):
            outs = _lit(r.get("outputs", ""))
            b.add([(r["input"], False)], " ".join(map(str, outs)) if isinstance(outs, list) else str(outs) or 32,
                  session=f"{t}:{i}")


def _haystack_text(target_tokens_llama3=150_000):
    essays = json.load(open(os.path.join(RAW, "misc", "PaulGrahamEssays.json")))["text"]
    return essays


def build_niah(b: Builder, lengths=(4096, 8192, 16384, 32768), depths=(0, 0.25, 0.5, 0.75, 1.0), **_):
    """gkamradt-style sweep: same haystack prefix, needle inserted at depth d of length L.
    Requests for one L share the haystack up to the needle -> prefix reuse grows with depth."""
    hay = b.ids(_haystack_text())
    needle = b.ids("\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n")
    q = b.ids("\n\nWhat is the best thing to do in San Francisco? Answer:")
    i = 0
    for L in lengths:
        body = L - len(needle) - len(q)
        for d in depths:
            cut = int(body * d)
            toks = hay[:cut] + needle + hay[cut:body] + q
            blocks = chain_blocks(toks, b.bt)
            b.reqs.append(Request(rid=len(b.reqs), session=f"niah:{L}", turn=1, input_len=len(toks), output_len=32,
                                  blocks=blocks, chunk_ids=[-1] * len(blocks), block_tokens=b.bt))
            i += 1


def build_passkey(b: Builder, lengths=(2048, 4096, 8192, 16384, 32768), per_len=10, seed=0, **_):
    """Landmark/KIVI-style passkey: repeated filler sentences, key at a random position."""
    rng = random.Random(seed)
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    fid = b.ids(filler)
    for L in lengths:
        reps = max(1, L // len(fid))
        for k in range(per_len):
            key = rng.randint(10000, 99999)
            pos = rng.randint(0, reps - 1)
            pre = b.ids("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. ")
            info = b.ids(f"The pass key is {key}. Remember it. {key} is the pass key. ")
            toks = pre + fid * pos + info + fid * (reps - pos) + b.ids("What is the pass key? The pass key is")
            blocks = chain_blocks(toks, b.bt)
            b.reqs.append(Request(rid=len(b.reqs), session=f"pk:{L}:{k}", turn=1, input_len=len(toks), output_len=8,
                                  blocks=blocks, chunk_ids=[-1] * len(blocks), block_tokens=b.bt))


def build_sharegpt_bench(b: Builder, n=1000, **_):
    """vLLM benchmark_serving.py usage: FIRST turn of each conversation only."""
    for i, r in enumerate(_jsonl(os.path.join(RAW, "misc", "sharegpt_vllm_bench_1k.jsonl"), n)):
        b.add([(r["prompt"], False)], int(r.get("gpt len") or 128), session=f"sg:{i}")


def _multiturn(b, convs, name, n_conv, sys_prompt=None, max_turns=20):
    b.keep_out = True
    for ci, conv in enumerate(convs[:n_conv]):
        hist = [(sys_prompt, True)] if sys_prompt else []
        turn = 0
        msgs = conv
        for j in range(0, len(msgs) - 1, 2):
            u, a = msgs[j], msgs[j + 1]
            turn += 1
            if turn > max_turns:
                break
            hist.append((f"USER: {u}\n", False))
            hist.append(("ASSISTANT:", False))
            b.add(list(hist), f" {a}\n", session=f"{name}:{ci}", turn=turn)
            hist.append((f" {a}\n", False))


def build_ultrachat(b: Builder, n_conv=300, **_):
    data = json.load(open(os.path.join(RAW, "misc", "ultrachat_2k.json")))
    convs = [[t["content"] for t in d["turns"]] for d in data]
    _multiturn(b, convs, "uc", n_conv)


def build_sharegpt_multiturn(b: Builder, n_conv=300, **_):
    """Multi-turn ShareGPT replay (CachedAttention/Pensieve/FastSwitch style).
    The full ShareGPT dump is HF-only here, so turn structure + text come from
    UltraChat conversations and lengths are rescaled to ShareGPT-V3 first-turn
    statistics measured from the vLLM sample (published fidelity)."""
    build_ultrachat(b, n_conv=n_conv)


def build_alpaca(b: Builder, n=500, **_):
    d = json.load(open(os.path.join(RAW, "misc", "alpaca.json")))["instances"][:n]
    for i, r in enumerate(d):
        txt = r["text"]
        inp, _, out = txt.partition("Output:")
        b.add([(inp + "Output:", False)], out or 64, session=f"alp:{i}")


def build_mtbench(b: Builder, **_):
    b.keep_out = True
    for i, r in enumerate(_jsonl(os.path.join(RAW, "misc", "mtbench_question.jsonl"))):
        t1, t2 = r["turns"][0], r["turns"][1]
        ans = " " + "answer " * 300   # placeholder ~300-token first answer
        b.add([(f"USER: {t1}\nASSISTANT:", False)], ans, session=f"mt:{i}", turn=1)
        b.add([(f"USER: {t1}\nASSISTANT:", False), (ans, False), (f"\nUSER: {t2}\nASSISTANT:", False)],
              300, session=f"mt:{i}", turn=2)


GSM_FEWSHOT_N = 8


def build_gsm8k(b: Builder, n=500, shots=GSM_FEWSHOT_N, **_):
    """lm-eval gsm8k_cot-style: the same 8 exemplars prefix every question."""
    train = _jsonl(os.path.join(RAW, "misc", "gsm8k_train.jsonl"), shots) if os.path.exists(
        os.path.join(RAW, "misc", "gsm8k_train.jsonl")) else []
    test = _jsonl(os.path.join(RAW, "misc", "gsm8k_test.jsonl"))
    if not train:
        train, test = test[:shots], test[shots:]
    fs = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in train[:shots]) if shots else ""
    for i, r in enumerate(test[:n]):
        b.add([(fs, True), (f"Question: {r['question']}\nAnswer:", False)], r["answer"], session=f"gsm:{i}")


def build_math500(b: Builder, **_):
    for i, r in enumerate(_jsonl(os.path.join(RAW, "misc", "math_500.jsonl"))):
        b.add([(f"Solve the problem and put the final answer in \\boxed{{}}.\n{r['question']}", False)], 2048,
              session=f"m5:{i}")


def build_aime(b: Builder, out_len=16384, **_):
    """Reasoning-model setting (R-KV/R1 evals): short prompt, very long output."""
    for i, r in enumerate(_jsonl(os.path.join(RAW, "misc", "aime24.jsonl"))):
        b.add([(r["question"], False)], out_len, session=f"aime:{i}")


def build_mmlu(b: Builder, shots=5, n=600, **_):
    """5-shot per subject: the subject's exemplars form a shared prefix."""
    rows = _jsonl(os.path.join(RAW, "misc", "mmlu_stem_test.jsonl"))
    by = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r)

    def fmt(r, ans=True):
        ch = _lit(r["choices"])
        s = r["question"] + "\n" + "".join(f"{'ABCD'[k]}. {c}\n" for k, c in enumerate(ch)) + "Answer:"
        return s + (f" {'ABCD'[int(r['answer'])]}\n\n" if ans else "")
    i = 0
    for subj, rs in sorted(by.items()):
        head = f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
        fs = head + "".join(fmt(r) for r in rs[:shots])
        for r in rs[shots:]:
            b.add([(fs, True), (fmt(r, False), False)], 1, session=f"mmlu:{subj}")
            i += 1
            if i >= n:
                return


def build_squad(b: Builder, n=600, **_):
    """SQuAD: ~5 questions per paragraph -> context-first template gives prefix reuse."""
    d = json.load(open(os.path.join(RAW, "misc", "squad_v2_dev.json")))["data"]
    i = 0
    for art in d:
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                ans = (qa.get("answers") or [{"text": "unanswerable"}])[0]["text"]
                b.add([(f"Context: {p['context']}\n", True), (f"Question: {qa['question']}\nAnswer:", False)], ans,
                      session=f"sq:{art['title']}")
                i += 1
                if i >= n:
                    return


def build_nq_rag(b: Builder, n=500, k=10, **_):
    """NQ-open with 10 retrieved Wikipedia passages (Lost-in-the-middle / KVLink setup)."""
    for i, r in enumerate(_jsonl(os.path.join(RAW, "misc", "nq-open-10_0.jsonl"), n)):
        segs = [("Write a high-quality answer for the given question using only the provided search results.\n\n", False)]
        for j, c in enumerate(_lit(r["ctxs"])[:k]):
            segs.append((f"Document [{j + 1}](Title: {c['title']}) {c['text']}\n", True))
        segs.append((f"\nQuestion: {r['question']}\nAnswer:", False))
        b.add(segs, (_lit(r["answers"]) or ["x"])[0], session=f"nq:{i}")


def build_humaneval(b: Builder, **_):
    with gzip.open(os.path.join(RAW, "misc", "HumanEval.jsonl.gz"), "rt") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            b.add([(r["prompt"], False)], r["canonical_solution"], session=f"he:{i}")


def build_mbpp(b: Builder, shots=3, n=500, **_):
    rows = _jsonl(os.path.join(RAW, "misc", "mbpp.jsonl"))
    fs_rows = [r for r in rows if 2 <= r["task_id"] <= 4][:shots]

    def fmt(r, sol=True):
        s = f"You are an expert Python programmer, and here is your task: {r['text']} Your code should pass these tests:\n\n" + "\n".join(r["test_list"]) + "\n"
        return s + (f"[BEGIN]\n{r['code']}\n[DONE]\n" if sol else "[BEGIN]\n")
    fs = "".join(fmt(r) for r in fs_rows)
    for i, r in enumerate([r for r in rows if 11 <= r["task_id"] <= 510][:n]):
        b.add([(fs, True), (fmt(r, False), False)], r["code"], session=f"mbpp:{i}")


def build_fewshot_jsonl(b: Builder, file, field="article", out_len=64, n=300, **_):
    for i, r in enumerate(_jsonl(os.path.join(RAW, "misc", file), n)):
        txt = r.get(field) or json.dumps(r)
        b.add([(txt, False)], out_len, session=f"{file}:{i}")


def build_cnndm(b: Builder, n=300, **_):
    for i, r in enumerate(_jsonl(os.path.join(RAW, "misc", "cnndm.jsonl"), n)):
        req = _lit(r["request"]) if isinstance(r["request"], str) else r["request"]
        b.add([(req["prompt"], False)], int(req.get("max_tokens", 128)), session=f"cnn:{i}")


def build_text_windows(b: Builder, file, window=4096, n=64, **_):
    """Perplexity / long-sequence speed tests: fixed windows over a corpus, no reuse."""
    toks = []
    with open(os.path.join(RAW, "misc", file)) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            toks += d["input_tokens"] if "input_tokens" in d else b.ids(d["text"])
            if len(toks) > window * n:
                break
    for i in range(0, min(len(toks), window * n) - window + 1, window):
        seg = toks[i:i + window]
        blocks = chain_blocks(seg, b.bt)
        b.reqs.append(Request(rid=len(b.reqs), session=f"{file}:{i}", turn=1, input_len=len(seg), output_len=1,
                              blocks=blocks, chunk_ids=[-1] * len(blocks), block_tokens=b.bt))


def build_leval(b: Builder, tasks=None, n=300, **_):
    """L-Eval: each long document comes with several instructions -> doc-shared prefix."""
    d = os.path.join(RAW, "leval")
    tasks = tasks or sorted(x[:-6] for x in os.listdir(d))
    for t in tasks:
        for di, r in enumerate(_jsonl(os.path.join(d, f"{t}.jsonl"))):
            for qi, (ins, out) in enumerate(zip(r["instructions"], r["outputs"])):
                b.add([("Now you are given a very long document. Please follow the instruction based on this document.\n", False),
                       (r["input"], True), (f"\nInstruction: {ins}\nAnswer:", False)], out, session=f"{t}:{di}", turn=qi + 1)
                if len(b.reqs) >= n:
                    return


def build_bbh(b: Builder, per_task=12, **_):
    """BIG-Bench-Hard 3-shot CoT (Suzgun et al.): each task's CoT prompt is a shared prefix."""
    d = os.path.join(RAW, "bbh")
    for t in sorted(os.listdir(os.path.join(d, "data"))):
        name = t[:-5]
        cot = open(os.path.join(d, "cot", name + ".txt")).read().split("-----", 1)[-1].strip() + "\n\n"
        for i, ex in enumerate(json.load(open(os.path.join(d, "data", t)))["examples"][:per_task]):
            b.add([(cot, True), (f"Q: {ex['input']}\nA: Let's think step by step.", False)], 256, session=f"bbh:{name}")


def build_piqa(b: Builder, n=400, shots=5, **_):
    """lm-eval PIQA loglikelihood: goal + each of 2 solutions (goal shared); 5-shot exemplar prefix."""
    rows = _jsonl(os.path.join(RAW, "misc", "piqa_valid.jsonl"))
    fs = "".join(f"Question: {r['goal']}\nAnswer: {r['sol1']}\n\n" for r in rows[:shots])
    for i, r in enumerate(rows[shots:shots + n // 2]):
        for sol in (r["sol1"], r["sol2"]):
            b.add([(fs, True), (f"Question: {r['goal']}\nAnswer:", False), (" " + sol, False)], 1, session=f"piqa:{i}")


def build_hf(b: Builder, name, n=600, shots_prefix="", **_):
    """Consume data/raw/hf/<name>.jsonl written by kvwl.hf_sources (make fetch-hf)."""
    rows = _jsonl(os.path.join(RAW, "hf", f"{name}.jsonl"))
    shape, rows = rows[0]["shape"], rows[1:]
    if shape == "chat":
        _multiturn(b, [r["turns"] for r in rows], name, n_conv=n)
    elif shape == "doc_qa":
        for r in rows[:n]:
            b.add([("Read the document and answer.\n", False), (r["doc"], True), (f"\nQuestion: {r['question']}\nAnswer:", False)],
                  r["answer"] or 64, session=f"{name}:{r['doc_id']}")
    elif shape == "mc":
        for i, r in enumerate(rows[: max(1, n // 4)]):
            for c in r["choices"]:
                b.add([(shots_prefix, True), (r["ctx"], False), (" " + c, False)], 1, session=f"{name}:{i}")
    elif shape == "agent":
        b.keep_out = True
        for r in rows[:n]:
            hist = []
            for t, st in enumerate(r["steps"], 1):
                hist.append((st["input_append"], False))
                b.add(list(hist), st["output"], session=f"{name}:{r['session']}", turn=t)
                hist.append((st["output"], False))


# -------------------------------------------------------- production traces
def build_trace_hashed(b: Builder, file, n=None, block_tokens=16, **_):
    """Mooncake (512-token blocks) / Qwen-Bailian (16-token blocks): real hash ids."""
    rows = _jsonl(os.path.join(RAW, "traces", file), n)
    t0 = None
    for r in rows:
        ts = r["timestamp"] / (1000.0 if "chat_id" not in r else 1.0)  # Mooncake ms -> s
        t0 = ts if t0 is None else t0
        hs = [int(h) for h in r["hash_ids"]]
        b.reqs.append(Request(rid=len(b.reqs), session=str(r.get("chat_id", len(b.reqs))) if r.get("parent_chat_id", -1) == -1
                              else str(r.get("parent_chat_id")), turn=int(r.get("turn", 1)), input_len=int(r["input_length"]),
                              output_len=int(r["output_length"]), blocks=hs, chunk_ids=[-1] * len(hs),
                              arrival=ts - t0, block_tokens=block_tokens))


def build_trace_lengths(b: Builder, file, n=5000, in_col=None, out_col=None, ts_col=None, **_):
    """Azure/BurstGPT: per-request token counts only (no content -> no reuse info)."""
    path = os.path.join(RAW, "traces", file)
    with open(path) as f:
        rd = csv.DictReader(f)
        t0 = None
        for r in rd:
            il, ol = int(float(r[in_col])), int(float(r[out_col]))
            if il <= 0:
                continue
            ts = r.get(ts_col)
            try:
                from datetime import datetime
                tsv = float(ts) if ts and ts.replace(".", "").isdigit() else datetime.fromisoformat(ts[:26]).timestamp()
            except Exception:
                tsv = None
            if tsv is not None:
                t0 = tsv if t0 is None else t0
            nb = -(-il // b.bt)
            uniq = [hash((file, len(b.reqs), k)) & ((1 << 63) - 1) for k in range(nb)]
            b.reqs.append(Request(rid=len(b.reqs), session=str(len(b.reqs)), turn=1, input_len=il, output_len=max(1, ol),
                                  blocks=uniq, chunk_ids=[-1] * nb, arrival=(tsv - t0) if tsv is not None else None,
                                  block_tokens=b.bt))
            if len(b.reqs) >= n:
                break


# ---------------------------------------------------- published-stats synthesis
def build_synthetic(b: Builder, n=300, in_mean=1000, in_cv=0.5, out_mean=128, out_cv=0.5, docs=None, q_per_doc=1,
                    shared_prefix=0, turns=1, turn_append=0, chunks_per_req=0, chunk_pool=0, chunk_len=0, zipf=1.0,
                    in_max=None, seed=0, name="syn", **_):
    """Lognormal lengths + explicit reuse structure:
       shared_prefix  tokens common to every request (system prompt / few-shot)
       q_per_doc      requests sharing one document (doc length ~ in_mean)
       turns          multi-turn sessions; each turn re-sends history + turn_append new tokens
       chunks_per_req RAG: k chunks drawn Zipf(zipf) from a pool of chunk_pool, chunk_len each
    """
    rng = np.random.default_rng(seed)

    def ln(mean, cv, size=None):
        s2 = np.log(1 + cv ** 2)
        return np.maximum(1, rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), size)).astype(int)
    tok = 1_000_003

    def fresh(k):
        nonlocal tok
        out = list(range(tok, tok + int(k)))
        tok += int(k)
        return out
    sp = fresh(shared_prefix) if shared_prefix else []
    pool = [fresh(chunk_len) for _ in range(chunk_pool)] if chunk_pool else []
    if chunk_pool:
        w = 1.0 / np.arange(1, chunk_pool + 1) ** zipf
        w /= w.sum()
    produced, s = 0, 0
    while produced < n:
        s += 1
        doc_len = int(ln(in_mean, in_cv)) - shared_prefix - (chunks_per_req * chunk_len)
        if in_max:
            doc_len = min(doc_len, in_max)
        doc = fresh(max(16, doc_len))
        for q in range(max(1, q_per_doc)):
            hist = []
            for t in range(max(1, turns)):
                parts, cids = [], []
                parts.append(sp); cids.append(1 if sp else -1)
                if chunk_pool:
                    for c in rng.choice(chunk_pool, size=chunks_per_req, replace=False, p=w):
                        parts.append(pool[c]); cids.append(int(c) + 7)
                parts.append(doc); cids.append(-1 if q_per_doc == 1 else 3)
                parts.append(hist)
                cids.append(-1)
                qtok = fresh(max(8, turn_append if t else 24))
                parts.append(qtok); cids.append(-1)
                toks, ctok = [], []
                for p, c in zip(parts, cids):
                    toks += p; ctok += [c if c == -1 else chunk_hash(p)] * len(p)
                blocks = chain_blocks(toks, b.bt)
                out_len = int(ln(out_mean, out_cv))
                out_toks = fresh(out_len)
                b.reqs.append(Request(rid=len(b.reqs), session=f"{name}:{s}", turn=t + 1, input_len=len(toks),
                                      output_len=out_len, blocks=blocks,
                                      chunk_ids=[ctok[i] for i in range(0, len(toks), b.bt)], block_tokens=b.bt,
                                      out_blocks=continue_blocks(toks, out_toks, b.bt) if turns > 1 else []))
                hist = hist + qtok + out_toks
                produced += 1
                if produced >= n:
                    return


BUILDERS = {k[6:]: v for k, v in globals().items() if k.startswith("build_")}
