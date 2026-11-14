"""Can the judge finish the full 8-question prompt within JUDGE_TIMEOUT=30s?

Constrained decoding is correct but far too slow (~640 s) because lm-format-enforcer
scans the 151k vocabulary per generated token. This measures the alternative:
unconstrained greedy decoding with a tight max_new_tokens, on the exact prompt,
schema and question set eidolon's LocalJudge sends.
"""
import json
import sys
import time

sys.path.insert(0, r"F:\eidolon-sc\deploy\local-ai\rocm")
# The ROCm venv lacks tzdata (pip install is blocked), but eidolon's judge module
# imports zoneinfo("Asia/Tokyo"). Borrow it from the project venv rather than
# modifying the ROCm environment.
sys.path.append(r"F:\eidolon-sc\backend\.venv\Lib\site-packages")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from server import FILES, GGUF_DIR, JUDGE_MODEL
from test_local_judge import QUESTIONS, STATE

# Same prompt construction as eidolon's judge.py (_local_prompt / _local_schema).
sys.path.insert(0, r"F:\eidolon-sc\backend\src")
from eidolon.judge import _local_prompt, _local_schema, _LOCAL_SYSTEM  # noqa: E402

schema = _local_schema(QUESTIONS)
prompt = _local_prompt(STATE, QUESTIONS)
print("schema keys:", list(schema.get("properties", {})), flush=True)
print("prompt chars:", len(prompt), flush=True)

tok = AutoTokenizer.from_pretrained(str(GGUF_DIR), gguf_file=FILES[JUDGE_MODEL])
model = AutoModelForCausalLM.from_pretrained(
    str(GGUF_DIR), gguf_file=FILES[JUDGE_MODEL], dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()

messages = [{"role": "system", "content": _LOCAL_SYSTEM}, {"role": "user", "content": prompt}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
print("prompt tokens:", enc["input_ids"].shape[1], flush=True)

for budget in (256, 400):
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    content = tok.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"\nmax_new_tokens={budget}: {elapsed:.1f}s, generated {len(new_tokens)} tokens", flush=True)
    print(f"  content: {content[:300]!r}", flush=True)
    try:
        parsed = json.loads(content)
        ok = isinstance(parsed, dict) and all(isinstance(v, dict) for v in parsed.values())
        print(f"  valid nested JSON: {ok}", flush=True)
    except ValueError as e:
        print(f"  JSON parse failed: {e}", flush=True)
