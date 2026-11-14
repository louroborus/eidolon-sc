"""Torch/HIP judge throughput on the exact 8-question set eidolon sends.

Establishes the ceiling for the torch route so the decision rests on a number rather
than on whether this card's HIP build happens to start.
"""
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(r"F:\eidolon-sc")
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT / "deploy" / "local-ai" / "rocm"))
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

GGUF = ROOT / "deploy" / "local-ai" / "gguf"
FILE = "qwen3-4b-instruct-2507-q4_k_m.gguf"

tok = AutoTokenizer.from_pretrained(str(GGUF), gguf_file=FILE)
model = AutoModelForCausalLM.from_pretrained(
    str(GGUF), gguf_file=FILE, dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()
print(f"vram {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
      f"free {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB", flush=True)

prompt = _local_prompt(STATE, QUESTIONS)
text = tok.apply_chat_template(
    [{"role": "system", "content": _LOCAL_SYSTEM}, {"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True,
)
enc = tok(text, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
print(f"questions: {len(QUESTIONS)}, prompt tokens: {enc['input_ids'].shape[1]}", flush=True)

with torch.no_grad():
    model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)

for budget in (250, 350):
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    produced = out.shape[1] - enc["input_ids"].shape[1]
    content = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        import json
        parsed = json.loads(content)
        nested = all(isinstance(v, dict) for v in parsed.values())
        ok = f"JSON ok, {len(parsed)} keys, nested={nested}"
    except ValueError as e:
        ok = f"JSON INVALID ({e})"
    print(f"budget={budget}: {elapsed:.2f}s  {produced} tok  {produced / elapsed:.1f} tok/s  {ok}", flush=True)
