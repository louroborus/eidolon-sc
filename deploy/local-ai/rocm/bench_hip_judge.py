"""Measure a HIP llama-server on eidolon's real judge prompt.

Pass the base URL as the first argument (default http://127.0.0.1:8097, the router
service). If the full 8-question prompt lands well under JUDGE_TIMEOUT=30 s, the
judge can run on ROCm rather than Vulkan; that is what RESULTS.md records.
"""
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(r"F:\eidolon-sc")
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.append(str(ROOT / "backend" / ".venv" / "Lib" / "site-packages"))
from eidolon.judge import _local_prompt, _local_schema, _LOCAL_SYSTEM  # noqa: E402

STATE = {
    "制作人": "制作人说：周末去看海吧",
    "角色": "櫻木真乃",
    "上一句": "[嬉しそうに]いいですね、プロデューサーさん！",
    "好感度": "trust",
}
QUESTIONS = {
    "needs_retrieval": {"type": "noul", "instructions": "这句话需要检索剧情、角色资料或记忆吗？",
                        "criteria": {"是": "提到过去的事、作品设定或用户说过的事", "否": "只是闲聊"}},
    "recall": {"type": "noul", "instructions": "这句话在问某段原作剧情是怎么发展的吗？",
               "criteria": {"是": "问过去某件事的经过", "否": "没有问过去"}},
    "about_producer": {"type": "noul", "instructions": "这句话在说制作人自己的事吗？",
                       "criteria": {"是": "说了自己的情况", "否": "没有"}},
    "affinity": {"type": "score", "instructions": "这句话让她的好感度怎样变化？",
                 "criteria": ["明显变亲近", "略微变亲近", "不变", "略微疏远", "明显疏远"]},
    "emotion": {"type": "choice", "instructions": "她这段回复是什么情绪？",
                "criteria": {"平静": "平常", "开心": "高兴", "害羞": "不好意思", "惊讶": "意外",
                             "难过": "低落", "生气": "不高兴", "思考": "在想"}},
    "nod": {"type": "choice", "instructions": "要不要点头？",
            "criteria": {"点头": "肯定", "摇头": "否定", "不点": "都不"}},
    "photo": {"type": "noul", "instructions": "制作人在要一张照片吗？",
              "criteria": {"是": "明确要照片", "否": "没有"}},
    "considerate": {"type": "noul", "instructions": "这句话需要体贴她吗？",
                    "criteria": {"是": "她状态不好要照顾", "否": "不需要"}},
}

schema = _local_schema(QUESTIONS)
prompt = _local_prompt(STATE, QUESTIONS)
body = {
    "model": "local-judge",
    "messages": [
        {"role": "system", "content": _LOCAL_SYSTEM},
        {"role": "user", "content": prompt},
    ],
    "response_format": {"type": "json_schema",
                        "json_schema": {"name": "judge", "schema": schema, "strict": True}},
    "temperature": 0,
    "max_tokens": 700,
}

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8097"

for attempt in (1, 2):
    t0 = time.time()
    r = httpx.post(f"{BASE}/v1/chat/completions", json=body, timeout=300)
    elapsed = time.time() - t0
    print(f"attempt {attempt}: {elapsed:.2f}s status={r.status_code}", flush=True)
    if r.status_code == 200:
        payload = r.json()
        content = payload["choices"][0]["message"]["content"]
        timings = payload.get("timings") or {}
        if timings:
            print(f"  prompt {timings.get('prompt_n')} tok @ {timings.get('prompt_per_second', 0):.0f}/s, "
                  f"gen {timings.get('predicted_n')} tok @ {timings.get('predicted_per_second', 0):.1f}/s",
                  flush=True)
        try:
            parsed = json.loads(content)
            nested = all(isinstance(v, dict) for v in parsed.values())
            print(f"  keys: {list(parsed)}", flush=True)
            print(f"  nested structure (what eidolon requires): {nested}", flush=True)
        except ValueError as e:
            print(f"  JSON parse failed: {e}: {content[:200]!r}", flush=True)
    else:
        print("  body:", r.text[:400], flush=True)
