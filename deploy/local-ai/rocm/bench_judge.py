"""Where does the judge's 77s go on ROCm: the GPU, or per-token Python overhead?

Measures tokens/second with plain greedy decoding, plus batch throughput for
embedding and rerank, so the report can say what the ROCm path actually costs.
"""
import sys
import time

sys.path.insert(0, r"F:\eidolon-sc\deploy\local-ai\rocm")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from server import FILES, GGUF_DIR, JUDGE_MODEL, EMBED_MODEL

tok = AutoTokenizer.from_pretrained(str(GGUF_DIR), gguf_file=FILES[JUDGE_MODEL])
model = AutoModelForCausalLM.from_pretrained(
    str(GGUF_DIR), gguf_file=FILES[JUDGE_MODEL], dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()

text = "用一句话说明你是谁。"
enc = tok(text, return_tensors="pt").to("cuda")
print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
print(f"prompt tokens: {enc['input_ids'].shape[1]}", flush=True)

# warm up so the first-call graph/shader compile does not count
with torch.no_grad():
    model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)

for n in (32, 64):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    produced = out.shape[1] - enc["input_ids"].shape[1]
    print(f"max_new={n}: {elapsed:.2f}s, generated {produced} tokens, "
          f"{produced / elapsed:.1f} tok/s", flush=True)

print(f"vram allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
