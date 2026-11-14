"""Unit-check the JSON-Schema constrained decoding before wiring it into the server."""
import sys
import time

sys.path.insert(0, r"F:\eidolon-sc\deploy\local-ai\rocm")
import torch
from server import _constraint, FILES, GGUF_DIR, JUDGE_MODEL
from transformers import AutoModelForCausalLM, AutoTokenizer

schema = {
    "type": "object",
    "properties": {"check": {
        "type": "object",
        "properties": {"noul": {"type": "number"}},
        "required": ["noul"],
        "additionalProperties": False,
    }},
    "required": ["check"],
    "additionalProperties": False,
}

tok = AutoTokenizer.from_pretrained(str(GGUF_DIR), gguf_file=FILES[JUDGE_MODEL])
t0 = time.time()
fn = _constraint(schema, tok, len(tok))
print(f"enforcer built in {time.time() - t0:.1f}s (vocab {len(tok)})", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    str(GGUF_DIR), gguf_file=FILES[JUDGE_MODEL], dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()

prompt = (
    '对话状态：\n{\n "测试": "启动检查"\n}\n\n'
    "需要你回答的问题：\n\n"
    "[check] 这是一条测试消息吗？\n  给一个 0 到 1 之间的数：越接近 1 越偏向「是」，越接近 0 越偏向「否」。\n\n"
    "只输出一个 JSON 对象，键是：check。"
)
text = tok.apply_chat_template(
    [{"role": "system", "content": "你是一个角色扮演对话里的判断模型。只输出一个 JSON 对象。"},
     {"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True,
)
enc = tok(text, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
for label, fn_arg in (("unconstrained", None), ("constrained", fn)):
    t = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=700, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.eos_token_id,
                             prefix_allowed_tokens_fn=fn_arg)
    content = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"{label}: {time.time() - t:.1f}s -> {content!r}", flush=True)
    if label == "constrained":
        import json
        parsed = json.loads(content)
        ok = isinstance(parsed.get("check"), dict) and "noul" in parsed["check"]
        print(f"  nested-structure OK: {ok}", flush=True)
