"""Compare ROCm reranker scores with llama-server's on the same query/documents.

eidolon's calibrated threshold for Qwen3-Reranker-0.6B is references=0.3. If the
ROCm scores land in a different range, the switch would silently drop every
retrieved passage, so the score mapping must be aligned.
"""
import httpx

pairs = [
    ("制作人说想去海边", ["一起去海边吧", "明天要早起开会，别迟到"]),
    ("那次演唱会怎么样", ["ライブで歌った曲の話", "朝ごはんはパンでした"]),
    ("猫を飼っている", ["猫が好きです", "犬の散歩に行った"]),
]
for query, docs in pairs:
    body = {
        "model": "Qwen/Qwen3-Reranker-0.6B", "query": query,
        "documents": docs, "top_n": len(docs), "return_documents": False,
    }
    out = {}
    for name, url in (("vulkan", "http://127.0.0.1:8094"), ("rocm", "http://127.0.0.1:8095")):
        try:
            r = httpx.post(url + "/v1/rerank", json=body, timeout=120)
            out[name] = [(x["index"], round(x["relevance_score"], 6)) for x in r.json()["results"]]
        except Exception as e:
            out[name] = f"ERROR {type(e).__name__}: {e}"
    print(f"query={query!r}")
    print(f"  vulkan: {out['vulkan']}")
    print(f"  rocm:   {out['rocm']}")
