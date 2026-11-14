"""Go/no-go probe: are ROCm-torch Qwen3-Embedding vectors the same space as the
running llama-server's? If cosine ~= 1 the existing 14k stored vectors stay valid;
if not, switching costs a full `eidolon reembed`.

Run with the ROCm venv Python. Does not touch the 8094 service.
"""
import numpy as np
import torch
import httpx
from transformers import AutoModel, AutoTokenizer

GGUF_DIR = r"F:\eidolon-sc\deploy\local-ai\gguf"
GGUF_FILE = "qwen3-embed-0.6b-q8_0.gguf"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

texts = [
    "制作人说想去海边",
    "[嬉しそうに]いいですね、プロデューサーさん！",
    "Instruct: 根据问题，检索相关的剧情台词\nQuery: 那次演唱会怎么样",
]

# --- llama-server (Vulkan) ---
r = httpx.post(
    "http://127.0.0.1:8094/v1/embeddings",
    json={"model": MODEL_ID, "input": texts, "encoding_format": "float"},
    timeout=120,
)
r.raise_for_status()
server = np.array([d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])])
print(f"llama-server vectors: {server.shape}, norms={np.linalg.norm(server, axis=1).round(3).tolist()}")

# --- ROCm torch ---
tok = AutoTokenizer.from_pretrained(GGUF_DIR, gguf_file=GGUF_FILE)
model = AutoModel.from_pretrained(
    GGUF_DIR, gguf_file=GGUF_FILE, dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()

enc = tok(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to("cuda")
with torch.no_grad():
    hidden = model(**enc).last_hidden_state
# last non-padding token (Qwen3-Embedding uses last-token pooling)
last_idx = enc["attention_mask"].sum(dim=1) - 1
raw = hidden[torch.arange(len(texts)), last_idx, :].float()
normalized = torch.nn.functional.normalize(raw, dim=-1)

print(f"torch raw norms={raw.norm(dim=-1).tolist()}")
for label, mat in (("raw", raw), ("normalized", normalized)):
    local = mat.cpu().numpy()
    cos = (local * server).sum(axis=1) / (
        np.linalg.norm(local, axis=1) * np.linalg.norm(server, axis=1)
    )
    print(f"cosine(llama-server, torch {label}) = {cos.round(5).tolist()}")

# Also: does llama-server normalize its vectors?
print(f"llama-server is L2-normalized: {np.allclose(np.linalg.norm(server, axis=1), 1.0, atol=1e-3)}")
