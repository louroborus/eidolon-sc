"""Real retrieval check: query the stored vectors with ROCm embeddings.

The story corpus in the database was embedded by llama-server. If ROCm's vectors
land in the same space (cosine 0.9996), a query embedded here must find the same
passages. This reads the database read-only and does not write to it.
"""
import os
import sqlite3
import shutil
import sys
import tempfile

import httpx
import numpy as np

DB = r"F:\eidolon-sc\backend\data\eidolon.db"

# Read-only access: the sandbox blocks sqlite from opening the file in place, so work
# on a copy and never write back.
copy = os.path.join(tempfile.gettempdir(), "dsh_eval_retrieval.db")
shutil.copyfile(DB, copy)
conn = sqlite3.connect(copy)

rows = conn.execute(
    "SELECT p.episode_id, p.seq, p.text, p.embedding, p.model, e.title, e.source "
    "FROM episode_passages p JOIN episodes e ON e.id = p.episode_id "
    "WHERE p.embedding IS NOT NULL LIMIT 60000"
).fetchall()
print(f"passages with vectors: {len(rows)}", flush=True)
models = {r[4] for r in rows}
print(f"vector models: {models}", flush=True)


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


vectors = np.stack([unpack(r[3]) for r in rows])
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
print(f"matrix: {vectors.shape}", flush=True)

for query in ["演唱会上她唱了什么歌", "制作人和她第一次见面", "关于猫的话题"]:
    r = httpx.post(
        "http://127.0.0.1:8095/v1/embeddings",
        json={"model": "Qwen/Qwen3-Embedding-0.6B",
              "input": ["Instruct: 根据问题，检索相关的剧情台词\nQuery: " + query],
              "encoding_format": "float"},
        timeout=120,
    )
    q = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = vectors @ q
    top = np.argsort(-scores)[:3]
    print(f"\nquery: {query}", flush=True)
    for i in top:
        ep, seq, text, _, _, title, source = rows[i]
        print(f"  {scores[i]:.4f}  {source}「{title}」", flush=True)
        print(f"          {text[:70]}", flush=True)

conn.close()
os.unlink(copy)
