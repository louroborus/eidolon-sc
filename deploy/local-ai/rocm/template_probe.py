"""Find the reranker prompt that reproduces llama-server's scores on ROCm torch.

llama.cpp's `reranking=1` uses its own prompt for Qwen3-Reranker; the standard
Qwen chat template plus a `<think>` generation prefix is the likely match.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF_DIR = r"F:\eidolon-sc\deploy\local-ai\gguf"
FILE = "qwen3-reranker-0.6b-q8_0.gguf"
tok = AutoTokenizer.from_pretrained(GGUF_DIR, gguf_file=FILE)
model = AutoModelForCausalLM.from_pretrained(
    GGUF_DIR, gguf_file=FILE, dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()

SYSTEM = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
QUERY = "制作人说想去海边"
DOCS = ["一起去海边吧", "明天要早起开会，别迟到"]
TARGET = [0.973956, 9e-05]  # llama-server, for reference

yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
no_id = tok("no", add_special_tokens=False)["input_ids"][0]
print("yes_id", yes_id, "no_id", no_id, "target", TARGET, flush=True)


def score(texts):
    enc = tok(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits.float()
    idx = enc["attention_mask"].sum(1) - 1
    last = logits[torch.arange(len(texts)), idx, :]
    p = torch.softmax(last[:, [no_id, yes_id]], dim=-1)[:, 1]
    return p.cpu().tolist()


def chat_template(prefix):
    out = []
    for doc in DOCS:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<Instruct>: {INSTRUCT}\n<Query>: {QUERY}\n<Document>: {doc}"},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out.append(text + prefix)
    return out


variants = {
    "chat_only": chat_template(""),
    "plus_think": chat_template("<think:6124c78e>\n\n"),
    "plus_think_closed": chat_template("<think:6124c78e>\n\n</think:6124c78e>\n\n"),
    "raw_qwen": [
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<Instruct>: {INSTRUCT}\n<Query>: {QUERY}\n<Document>: {d}<|im_end|>\n<|im_start|>assistant\n<think:6124c78e>\n\n"
        for d in DOCS
    ],
    "raw_no_think": [
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<Instruct>: {INSTRUCT}\n<Query>: {QUERY}\n<Document>: {d}<|im_end|>\n<|im_start|>assistant\n"
        for d in DOCS
    ],
}
for name, texts in variants.items():
    print(f"\n--- {name} ---", flush=True)
    print("tail:", repr(texts[0][-70:]), flush=True)
    try:
        s = score(texts)
        print("scores:", [round(x, 6) for x in s], flush=True)
    except Exception as e:
        print("ERROR", type(e).__name__, str(e)[:200], flush=True)
