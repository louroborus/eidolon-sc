"""Find the pooling that reproduces llama-server's Qwen3-Embedding vectors.

If one variant reaches cosine ~1.0, the ROCm service can reproduce the stored
vectors and no reembed is needed. Otherwise switching backends costs a reembed.
"""
import numpy as np
import torch
import httpx
from transformers import AutoModel, AutoTokenizer

GGUF_DIR = r"F:\eidolon-sc\deploy\local-ai\gguf"
GGUF_FILE = "qwen3-embed-0.6b-q8_0.gguf"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
texts = ["制作人说想去海边", "[嬉しそうに]いいですね、プロデューサーさん！"]

r = httpx.post(
    "http://127.0.0.1:8094/v1/embeddings",
    json={"model": MODEL_ID, "input": texts, "encoding_format": "float"},
    timeout=120,
)
r.raise_for_status()
server = np.array([d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])])

tok = AutoTokenizer.from_pretrained(GGUF_DIR, gguf_file=GGUF_FILE)
model = AutoModel.from_pretrained(
    GGUF_DIR, gguf_file=GGUF_FILE, dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()
print("eos_token:", tok.eos_token, tok.eos_token_id, "pad_token:", tok.pad_token, tok.pad_token_id)

variants = {}
for name, add_eos in (("no_eos", False), ("with_eos", True)):
    ids = []
    for t in texts:
        seq = tok(t, add_special_tokens=False)["input_ids"]
        if add_eos and tok.eos_token_id is not None:
            seq = seq + [tok.eos_token_id]
        ids.append(seq)
    maxlen = max(len(s) for s in ids)
    padded = torch.full((len(ids), maxlen), tok.pad_token_id or tok.eos_token_id or 0, dtype=torch.long)
    mask = torch.zeros_like(padded)
    for i, s in enumerate(ids):
        padded[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    padded, mask = padded.to("cuda"), mask.to("cuda")
    with torch.no_grad():
        h = model(input_ids=padded, attention_mask=mask).last_hidden_state.float()
    for pool in ("last_nonpad", "last_position", "mean"):
        if pool == "last_nonpad":
            idx = mask.sum(1) - 1
            v = h[torch.arange(len(ids)), idx, :]
        elif pool == "last_position":
            v = h[:, -1, :]
        else:
            v = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        cos = (v * server).sum(1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(server, axis=1))
        variants[f"{name}/{pool}"] = cos
        print(f"{name:10s} {pool:12s} cosine={cos.round(5).tolist()}  lens={[len(s) for s in ids]}")

best = max(variants.items(), key=lambda kv: kv[1].mean())
print(f"\nBEST: {best[0]} mean cosine {best[1].mean():.5f}")
