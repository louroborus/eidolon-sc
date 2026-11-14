"""Measure the torch/HIP judge with the card to itself.

Earlier measurement (77 s for the full 8-question prompt, 10.1 tok/s) was taken while
other model servers shared the 12 GB card. Now that everything else is stopped, this
re-measures to decide whether the torch route can carry the judge.
"""
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(r"F:\eidolon-sc")
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.append(str(ROOT / "backend" / ".venv" / "Lib" / "site-packages"))
from eidolon.judge import _local_prompt, _local_schema, _LOCAL_SYSTEM  # noqa: E402

GGUF = ROOT / "deploy" / "local-ai" / "gguf"
FILE = "qwen3-4b-instruct-2507-q4_k_m.gguf"

STATE = {"制作人": "制作人说：周末去看海吧", "角色": "櫻木真乃",
         "上一句": "[嬉しそうに]いいですね、プロデューサーさん！", "好感度": "trust"}
QUESTIONS = {
    "needs_retrieval": {"type": "noul", "instructions": "这句话需要检索吗？", "criteria": {"是": "要", "否": "不要"}},
    "affinity": {"type": "score", "instructions": "好感度怎么变？", "criteria": ["明显亲近", "略亲近", "不变", "略疏远", "明显疏远"]},
    "emotion": {"type": "choice", "instructions": "什么情绪？", "criteria": {"平静": "平常", "开心": "高兴", "害羞": "害羞", "惊讶": "惊讶", "难过": "低落", "生气": "生气", "思考": "思考"}},
    "nod": {"type": "choice", "instructions": "点头吗？", "criteria": {"点头": "肯定", "摇头": "否定", "不点": "都不"}},
}

print(f"free before: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(str(GGUF), gguf_file=FILE)
model = AutoModelForCausalLM.from_pretrained(
    str(GGUF), gguf_file=FILE, dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()
print(f"loaded in {time.time() - t0:.1f}s, vram {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
print(f"free after load: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB", flush=True)

prompt = _local_prompt(STATE, QUESTIONS)
messages = [{"role": "system", "content": _LOCAL_SYSTEM}, {"role": "user", "content": prompt}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
print(f"prompt tokens: {enc['input_ids'].shape[1]}", flush=True)

# Warm up, then time generation.
with torch.no_grad():
    model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)

for budget in (200, 300):
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    produced = out.shape[1] - enc["input_ids"].shape[1]
    content = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"budget={budget}: {elapsed:.2f}s, {produced} tok, {produced / elapsed:.1f} tok/s", flush=True)
    print(f"  JSON valid: {content.startswith('{')} ... {content[-40:]!r}", flush=True)
