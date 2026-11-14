"""End-to-end check through the running stack: does a real turn come back?

Walks the backend's own HTTP surface the way the web app does — dev login, create a
conversation, send one turn, read the streamed reply. Proves the desktop entry point
brings up something that actually answers, not just three listening ports.
"""
import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"
CHARACTER = "mano"
# Writes must carry the backend's own origin: in development `origin_allowed`
# accepts http:// + a local host, and once the session cookie is set the Origin
# may not be missing.
HEADERS = {"Origin": BASE}

with httpx.Client(timeout=httpx.Timeout(180, connect=10), follow_redirects=True,
                  headers=HEADERS) as c:
    # One fixed dev account for all checks, so no throwaway users pile up.
    r = c.post(f"{BASE}/api/auth/dev-login", json={"name": "test-runner"})
    print(f"dev-login: {r.status_code}", flush=True)
    if r.status_code >= 400:
        print(r.text[:400], flush=True)
        sys.exit(1)

    # A first sign-in waits for consent to the privacy policy and terms.
    r = c.post(f"{BASE}/api/account/consent",
               json={"agree_to_terms": True, "at_least_13": True})
    print(f"consent: {r.status_code}", flush=True)
    if r.status_code >= 400:
        print(r.text[:400], flush=True)
        sys.exit(1)

    r = c.post(f"{BASE}/api/conversations", json={"character_id": CHARACTER})
    print(f"create conversation: {r.status_code}", flush=True)
    if r.status_code >= 400:
        print(r.text[:400], flush=True)
        sys.exit(1)
    conversation = r.json()
    print(f"  conversation: {conversation.get('id')}", flush=True)

    payload = {"text": "你好，今天过得怎么样？"}
    got = []
    with c.stream("POST", f"{BASE}/api/conversations/{conversation['id']}/turns",
                  json=payload) as res:
        print(f"turn: {res.status_code}", flush=True)
        if res.status_code >= 400:
            print(res.read()[:600], flush=True)
            sys.exit(1)
        for line in res.iter_lines():
            if line.strip():
                got.append(line)
            if len(got) >= 300:
                break
    # Leave the fixed account clean: this conversation was only a probe.
    c.delete(f"{BASE}/api/conversations/{conversation['id']}")

blob = "\n".join(got)
print("---- first 700 chars ----", flush=True)
print(blob[:700], flush=True)
kana = any("\u3040" <= ch <= "\u30ff" for ch in blob)
print(f"\ncarried a Japanese reply: {kana}", flush=True)
sys.exit(0 if kana else 2)
