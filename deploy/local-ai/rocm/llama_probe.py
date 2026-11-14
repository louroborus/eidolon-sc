"""Feasibility probe: load the local GGUFs with the ROCm torch/transformers stack.

Run with D:\\AI\\ComfyUI\\venv\\Scripts\\python.exe. Each model is loaded in a
separate subprocess (this script takes a model name as argv[1]) so a failure in
one does not poison the others and VRAM is released between probes.
"""
import sys
import time
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

GGUF = r"F:\eidolon-sc\deploy\local-ai\gguf"
MODELS = {
    "embed": "qwen3-embed-0.6b-q8_0.gguf",
    "rerank": "qwen3-reranker-0.6b-q8_0.gguf",
    "judge": "qwen3-4b-instruct-2507-q4_k_m.gguf",
}

which = sys.argv[1]
path = f"{GGUF}\\{MODELS[which]}"
t0 = time.time()
print(f"[{which}] torch {torch.__version__} hip={torch.version.hip} cuda={torch.cuda.is_available()}", flush=True)
print(f"[{which}] loading {path}", flush=True)

# transformers resolves a bare GGUF path when gguf_file points at the file and
# the directory holds it; config/tokenizer come from the model id below.
import os
directory, filename = os.path.split(path)
kwargs = dict(gguf_file=filename, dtype=torch.float16, low_cpu_mem_usage=True)

if which == "judge":
    model = AutoModelForCausalLM.from_pretrained(
        directory, **kwargs
    )
else:
    model = AutoModel.from_pretrained(directory, **kwargs)
model = model.to("cuda")
print(f"[{which}] loaded in {time.time() - t0:.1f}s, "
      f"vram={torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
print(f"[{which}] class={type(model).__name__} "
      f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)

tok = AutoTokenizer.from_pretrained(directory, gguf_file=filename)
texts = ["制作人说想去海边", "明天要早起开会"]
enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**enc)
print(f"[{which}] last_hidden_state shape={tuple(out.last_hidden_state.shape)}", flush=True)
if which == "embed":
    vec = out.last_hidden_state[:, -1, :]
    print(f"[{which}] last-token vector dim={vec.shape[-1]} "
          f"norm={vec.norm(dim=-1).tolist()}", flush=True)
elif which == "rerank":
    print(f"[{which}] logits shape={tuple(out.logits.shape) if hasattr(out, 'logits') else 'n/a'}", flush=True)
else:
    gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
    print(f"[{which}] generated: {tok.decode(gen[0], skip_special_tokens=True)[:120]!r}", flush=True)
print(f"[{which}] OK total {time.time() - t0:.1f}s", flush=True)
