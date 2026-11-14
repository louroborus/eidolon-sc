"""Full acceptance of the desktop entry point.

Checks everything a user touches after double-clicking the shortcut: the app page,
the character list, the stand assets the call layer needs, the model endpoints the
backend calls, and one real streamed turn. Prints PASS/FAIL per item and exits
non-zero if any fails.
"""
import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"
SITE = "http://127.0.0.1:3000"
STANDS = "http://127.0.0.1:8765"
MODELS = "http://127.0.0.1:8094"
CHARACTER = "mano"
# Writes must carry the backend's own origin: in development `origin_allowed` accepts
# http:// + a local host, and once the session cookie is set the Origin may not be
# missing. The web app reaches the backend same-origin, so name that.
HEADERS = {"Origin": BASE}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


with httpx.Client(timeout=httpx.Timeout(180, connect=10), follow_redirects=True,
                  headers=HEADERS) as c:
    # 1. the page
    try:
        r = c.get(SITE)
        check("frontend page", r.status_code == 200 and "Eidolon" in r.text,
              f"{r.status_code}, {len(r.text)} bytes")
    except Exception as e:
        check("frontend page", False, str(e)[:80])

    # 2. models
    try:
        r = c.get(f"{MODELS}/health")
        check("model server", r.status_code == 200, r.text[:40])
    except Exception as e:
        check("model server", False, str(e)[:80])

    try:
        r = c.post(f"{MODELS}/v1/embeddings", json={
            "model": "Qwen/Qwen3-Embedding-0.6B", "input": ["测试"], "encoding_format": "float"})
        dim = len(r.json()["data"][0]["embedding"])
        check("embedding", r.status_code == 200 and dim == 1024, f"dim {dim}")
    except Exception as e:
        check("embedding", False, str(e)[:80])

    try:
        r = c.post(f"{MODELS}/v1/rerank", json={
            "model": "Qwen/Qwen3-Reranker-0.6B", "query": "看海",
            "documents": ["去海边", "开会"], "top_n": 2})
        scores = [x["relevance_score"] for x in r.json()["results"]]
        check("rerank", r.status_code == 200 and scores[0] > 0.3, f"top {scores[0]:.3f}")
    except Exception as e:
        check("rerank", False, str(e)[:80])

    # 3. backend + characters
    try:
        chars = c.get(f"{BASE}/api/characters").json()
        check("characters", len(chars) == 29, f"{len(chars)} characters")
    except Exception as e:
        chars = []
        check("characters", False, str(e)[:80])

    # 4. stand assets (the call layer needs these)
    try:
        r = c.get(f"{STANDS}/spine/idols/stand/1030010010/data.png")
        check("stand assets", r.status_code == 200 and len(r.content) > 10000,
              f"{len(r.content)} bytes")
    except Exception as e:
        check("stand assets", False, str(e)[:80])

    # 5. one real turn. Always the same dev account, so no throwaway users pile up;
    # the conversation created below is deleted afterward to keep it that way.
    try:
        c.post(f"{BASE}/api/auth/dev-login", json={"name": "test-runner"})
        c.post(f"{BASE}/api/account/consent",
               json={"agree_to_terms": True, "at_least_13": True})
        r = c.post(f"{BASE}/api/conversations", json={"character_id": CHARACTER})
        if r.status_code >= 400:
            check("one real turn", False, f"create {r.status_code}: {r.text[:120]}")
            raise SystemExit(0)
        body = r.json()
        conversation_id = body.get("id") or body.get("conversation_id")
        if not conversation_id:
            check("one real turn", False, f"unexpected create body: {str(body)[:160]}")
            raise SystemExit(0)
        lines = []
        with c.stream("POST", f"{BASE}/api/conversations/{conversation_id}/turns",
                      json={"text": "你好"}) as res:
            status = res.status_code
            if status >= 400:
                check("one real turn", False, f"turn {status}: {res.read()[:160]}")
                raise SystemExit(0)
            for line in res.iter_lines():
                if line.strip():
                    lines.append(line)
                if len(lines) >= 200:
                    break
        blob = "\n".join(lines)
        kana = any("\u3040" <= ch <= "\u30ff" for ch in blob)
        check("one real turn", status == 200 and kana,
              f"{len(lines)} stream lines, japanese={kana}")
        # Leave the fixed account clean: this conversation was only a probe.
        c.delete(f"{BASE}/api/conversations/{conversation_id}")
    except SystemExit:
        pass
    except Exception as e:
        check("one real turn", False, str(e)[:120])

failed = [n for n, ok, _ in results if not ok]
print()
print(f"{len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
