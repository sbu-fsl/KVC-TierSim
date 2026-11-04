"""Diff catalog/models.yaml against each model's Hugging Face config.json.

    HF_TOKEN=... python -m kvwl.verify_models     # needs huggingface.co (gated repos need a token)
"""
import json
import os
import urllib.request

from .models import load_models

FIELDS = {"layers": ["num_hidden_layers", "n_layer", "num_layers"],
          "hidden": ["hidden_size", "n_embd", "d_model"],
          "q_heads": ["num_attention_heads", "n_head"],
          "kv_heads": ["num_key_value_heads", "multi_query_group_num", "num_kv_heads"],
          "head_dim": ["head_dim"]}


def fetch(repo):
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/config.json")
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    cfg = json.load(urllib.request.urlopen(req, timeout=30))
    return cfg.get("text_config", cfg)


def main():
    bad = 0
    for k, m in load_models().items():
        if "/" not in m.hf or " " in m.hf:
            continue
        try:
            cfg = fetch(m.hf)
        except Exception as e:
            print(f"?? {k:32s} {m.hf}: {e}")
            continue
        for f, keys in FIELDS.items():
            v = next((cfg[x] for x in keys if x in cfg), None)
            if f == "head_dim" and v is None:
                v = cfg.get("hidden_size", 0) // max(1, cfg.get("num_attention_heads", 1))
            if f == "kv_heads" and v is None:
                v = cfg.get("num_attention_heads")
            if m.attn == "mla" and f in ("kv_heads", "head_dim"):
                continue
            if v is not None and int(v) != int(getattr(m, f)):
                bad += 1
                print(f"!! {k:32s} {f}: catalog={getattr(m, f)} config={v}")
    print("mismatches:", bad)


if __name__ == "__main__":
    main()
