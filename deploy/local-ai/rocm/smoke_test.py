"""Smoke-test the ROCm server's three endpoints against the contracts eidolon uses."""
import json
import time
import httpx

base = "http://127.0.0.1:8095"
for _ in range(120):
    try:
        r = httpx.get(base + "/health", timeout=3)
        if r.status_code == 200:
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise SystemExit("server never became healthy")
print("health", r.status_code, r.json(), flush=True)

print("-- embeddings --", flush=True)
r = httpx.post(base + "/v1/embeddings", json={
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["制作人说想去海边", "明天要早起开会"],
    "encoding_format": "float",
}, timeout=300)
print(r.status_code, "count", len(r.json().get("data", [])),
      "dim", len(r.json()["data"][0]["embedding"]), flush=True)

print("-- rerank --", flush=True)
r = httpx.post(base + "/v1/rerank", json={
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "制作人说想去海边",
    "documents": ["一起去海边吧", "明天要早起开会，别迟到"],
    "top_n": 2,
    "return_documents": False,
}, timeout=300)
print(r.status_code, r.text[:400], flush=True)

print("-- chat/completions (judge JSON) --", flush=True)
schema = {
    "type": "object",
    "properties": {"needs_retrieval": {
        "type": "object",
        "properties": {"noul": {"type": "number"}},
        "required": ["noul"],
        "additionalProperties": False,
    }},
    "required": ["needs_retrieval"],
    "additionalProperties": False,
}
prompt = (
    "对话状态：{}\n需要你回答的问题：\n\n"
    "[needs_retrieval] 这是一条测试消息吗？\n  给一个 0 到 1 之间的数：越接近 1 越偏向「是」。\n"
    "只输出一个 JSON 对象，键是：needs_retrieval。"
)
r = httpx.post(base + "/v1/chat/completions", json={
    "model": "local-judge",
    "messages": [
        {"role": "system", "content": "你是一个角色扮演对话里的判断模型。只输出一个 JSON 对象，不要解释。"},
        {"role": "user", "content": prompt},
    ],
    "response_format": {"type": "json_schema",
                        "json_schema": {"name": "judge", "schema": schema, "strict": True}},
    "temperature": 0,
    "max_tokens": 700,
}, timeout=900)
print(r.status_code, r.text[:600], flush=True)
if r.status_code == 200:
    content = r.json()["choices"][0]["message"]["content"]
    try:
        print("parsed JSON:", json.loads(content), flush=True)
    except ValueError as e:
        print("JSON PARSE FAILED:", e, flush=True)
