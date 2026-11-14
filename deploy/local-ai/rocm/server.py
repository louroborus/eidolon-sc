"""An OpenAI-compatible inference server backed by ROCm (HIP) torch.

NOT IN USE. The project's `.env` points at the Vulkan llama-server on 8094
(`deploy/local-ai/start-llama-server.ps1`). This server was written to test whether
torch could replace it; it works for embeddings and rerank, but the judge is far too
slow (see below), so it never became the default. Kept as a working reference.

What it does: serves `/v1/embeddings`, `/v1/rerank` and `/v1/chat/completions` from
the same GGUF files `models.ini` lists, using D:\\AI\\ComfyUI\\venv (torch 2.11 +
ROCm 7.13, transformers 5.17) rather than llama.cpp.

Endpoints (the exact contracts in backend/src/eidolon/{embedding,rerank,judge}.py):
  POST /v1/embeddings        {"model", "input": [str], "encoding_format": "float"}
                             -> {"data": [{"index", "embedding", "object"}], "usage": {...}}
  POST /v1/rerank            {"model", "query", "documents", "top_n", "return_documents"}
                             -> {"results": [{"index", "relevance_score"}], "meta": {...}}
  POST /v1/chat/completions  {"model", "messages", "response_format": {json_schema}, ...}
                             -> {"choices": [{"message": {"content": "<JSON>"}}], "usage": {...}}
  GET  /health, /v1/models

Model files: deploy/local-ai/gguf/*.gguf. Loaded lazily on first request per model and
kept resident; ~10.5 GB VRAM for all three on a 12 GB card. All inference runs under a
single asyncio lock: the GPU is one resource and eidolon's calls are sequential in a
turn anyway.

Measured (RX 6750 XT, card to itself):
  embeddings  0.19 s per query    works, and matches llama-server's vectors
  rerank      0.07 s             works
  judge       8.8 tok/s, ~21 s for the full 8-question prompt, and the JSON comes
              back flat where eidolon needs it nested. Vulkan does the same job in
              2.6 s, so the judge stays on Vulkan.

Verified: this server's Qwen3-Embedding output matches llama-server's to cosine
0.9995+ when the EOS token is appended and last-non-padding pooling is used (see
pooling_probe.py). Existing stored vectors therefore stay valid; no `reembed`.

Run:  D:\\AI\\ComfyUI\\venv\\Scripts\\python.exe deploy\\local-ai\\rocm\\server.py
Stop: Ctrl+C, or the -Stop switch in start-rocm-server.ps1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
GGUF_DIR = ROOT / "deploy" / "local-ai" / "gguf"
# `deploy/local-ai/logs/` is not writable under the sandboxed agent, so logs go to
# the temp directory; pass ROCM_LOG_DIR to put them anywhere else.
LOG_DIR = Path(os.environ.get("ROCM_LOG_DIR") or (Path(tempfile.gettempdir()) / "eidolon-rocm"))

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
JUDGE_MODEL = "local-judge"

FILES = {
    EMBED_MODEL: "qwen3-embed-0.6b-q8_0.gguf",
    RERANK_MODEL: "qwen3-reranker-0.6b-q8_0.gguf",
    JUDGE_MODEL: "qwen3-4b-instruct-2507-q4_k_m.gguf",
}
served: set[str] = set(FILES)
"""Which models this process answers for; `--only` narrows it (the judge is too slow
on ROCm, so the split setup serves just embed and rerank here)."""
MAX_LENGTH = 8192
JUDGE_MAX_TOKENS = 700

_enforcer_data: dict[int, Any] = {}
"""The lm-format-enforcer tokenizer data, by vocabulary size. Kept apart from
`_models` so that `/health` lists loaded models and nothing else."""

log = logging.getLogger("rocm-server")
app = FastAPI(title="eidolon local-ai (ROCm)")
lock = asyncio.Lock()
_models: dict[str, dict[str, Any]] = {}


def _load(model_id: str) -> dict[str, Any]:
    """Load (and cache) one GGUF model onto the ROCm device."""
    if model_id in _models:
        return _models[model_id]
    filename = FILES[model_id]
    path = GGUF_DIR / filename
    if not path.is_file():
        raise RuntimeError(f"找不到模型文件 {path}")
    t0 = time.time()
    log.info("loading %s -> %s on cuda", model_id, path.name)
    tok = AutoTokenizer.from_pretrained(str(GGUF_DIR), gguf_file=filename)
    # Reranker and judge are both causal LMs: the former scores yes/no tokens, so it
    # needs lm_head too (AutoModel would give a bare Qwen3Model without logits).
    cls = AutoModel if model_id == EMBED_MODEL else AutoModelForCausalLM
    model = cls.from_pretrained(
        str(GGUF_DIR), gguf_file=filename, dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    entry = {"tokenizer": tok, "model": model, "vocab_size": len(tok)}
    _models[model_id] = entry
    vram = torch.cuda.memory_allocated() / 1e9
    log.info("loaded %s in %.1fs (vram now %.2f GB)", model_id, time.time() - t0, vram)
    return entry


def _encode(tok, texts: list[str], *, add_eos: bool) -> dict[str, torch.Tensor]:
    """Tokenize with right padding, appending EOS (Qwen3-Embedding/Reranker need it)."""
    eos = tok.eos_token_id
    sequences = []
    for text in texts:
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        if add_eos and eos is not None:
            ids = ids + [eos]
        sequences.append(ids)
    longest = max(len(s) for s in sequences)
    padded = torch.full((len(sequences), longest), eos or 0, dtype=torch.long)
    mask = torch.zeros_like(padded)
    for i, ids in enumerate(sequences):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = 1
    return {"input_ids": padded.to("cuda"), "attention_mask": mask.to("cuda")}


def _last_hidden(entry: dict[str, Any], texts: list[str], *, add_eos: bool) -> torch.Tensor:
    """Last-non-padding hidden state of each text, L2-normalized (float32 CPU)."""
    enc = _encode(entry["tokenizer"], texts, add_eos=add_eos)
    with torch.no_grad():
        hidden = entry["model"](**enc).last_hidden_state.float()
    idx = enc["attention_mask"].sum(1) - 1
    vectors = hidden[torch.arange(len(texts), device=hidden.device), idx, :]
    return torch.nn.functional.normalize(vectors, dim=-1).cpu()


def _embeddings_sync(model_id: str, texts: list[str]) -> torch.Tensor:
    return _last_hidden(_load(model_id), texts, add_eos=True)


# ── /v1/embeddings ────────────────────────────────────────────────────────


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    body = await request.json()
    model_id = body.get("model", EMBED_MODEL)
    if model_id not in served:
        raise HTTPException(400, f"unknown model {model_id}")
    inputs = body.get("input") or []
    texts = [str(i) for i in inputs] if isinstance(inputs, list) else [str(inputs)]
    if not texts:
        raise HTTPException(400, "input is empty")
    async with lock:
        vectors = await asyncio.to_thread(_embeddings_sync, model_id, texts)
    data = [
        {"object": "embedding", "index": i, "embedding": v.tolist()}
        for i, v in enumerate(vectors)
    ]
    usage = {
        "prompt_tokens": sum(len(t) for t in texts) // 2,
        "total_tokens": sum(len(t) for t in texts) // 2,
    }
    return JSONResponse({"object": "list", "model": model_id, "data": data, "usage": usage})


# ── /v1/rerank ────────────────────────────────────────────────────────────


def _rerank_scores(entry: dict[str, Any], query: str, documents: list[str]) -> list[float]:
    """Qwen3-Reranker: score yes/no for the query/document pair at the last token.

    llama-server's rerank endpoint does the same with its own chat template; we use
    the model's chat template so the token layout matches what it was trained on.
    """
    tok = entry["tokenizer"]
    model = entry["model"]
    pairs = []
    for doc in documents:
        messages = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc}"},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # llama.cpp's `reranking=1` appends the Qwen3 `<think>` generation prefix before
        # scoring yes/no. Without it every score collapses ~7x (0.98 -> 0.14) and
        # eidolon's calibrated 0.3 threshold would drop all relevant passages.
        # Verified against the Vulkan llama-server (see template_probe.py).
        pairs.append(text + "<think:6124c78e>\n\n")
    yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok("no", add_special_tokens=False)["input_ids"][0]
    enc = tok(pairs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits.float()
    idx = enc["attention_mask"].sum(1) - 1
    last = logits[torch.arange(len(pairs), device=logits.device), idx, :]
    scores = torch.softmax(last[:, [no_id, yes_id]], dim=-1)[:, 1]
    return scores.cpu().tolist()


@app.post("/v1/rerank")
async def rerank(request: Request) -> JSONResponse:
    body = await request.json()
    model_id = body.get("model", RERANK_MODEL)
    if model_id not in served:
        raise HTTPException(400, f"unknown model {model_id}")
    query = str(body.get("query", ""))
    documents = [str(d) for d in (body.get("documents") or [])]
    if not documents:
        return JSONResponse({"results": [], "meta": {"tokens": {"input_tokens": 0}}})
    async with lock:
        scores = await asyncio.to_thread(
            lambda: _rerank_scores(_load(model_id), query, documents)
        )
    results = [
        {"index": i, "relevance_score": float(score)} for i, score in enumerate(scores)
    ]
    top_n = body.get("top_n")
    if isinstance(top_n, int) and top_n > 0:
        results = sorted(results, key=lambda r: -r["relevance_score"])[:top_n]
    else:
        results.sort(key=lambda r: -r["relevance_score"])
    if body.get("return_documents"):
        for r in results:
            r["document"] = {"text": documents[r["index"]]}
    return JSONResponse({
        "results": results,
        "meta": {"tokens": {"input_tokens": (len(query) + sum(map(len, documents))) // 2}},
    })


# ── /v1/chat/completions (judge) ──────────────────────────────────────────

JUDGE_SYSTEM = (
    "你是一个角色扮演对话里的判断模型。用户会给你对话状态和若干个打分问题，"
    "你只输出一个 JSON 对象，不要解释，不要输出任何 JSON 之外的东西。"
)


def _schema_fields(schema: dict[str, Any] | None) -> str:
    """A short reminder of the required keys, appended to the prompt."""
    if not isinstance(schema, dict):
        return ""
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        return "\n只输出一个 JSON 对象，键是：" + "、".join(props) + "。"
    return ""


def _token_enforcer_data(tok, vocab_size: int) -> Any:
    """What lm-format-enforcer needs to know about this tokenizer, computed once.

    Its `integrations.transformers` module imports `PreTrainedTokenizerBase` from
    `transformers.tokenization_utils`, which transformers 5 moved, so that module
    raises ImportError. This is the same construction, inlined and cached: scanning
    a 151k-token vocabulary per request would cost seconds.
    """
    from lmformatenforcer import TokenEnforcerTokenizerData

    if vocab_size in _enforcer_data:
        return _enforcer_data[vocab_size]
    token_0 = tok.encode("0")[-1]
    special = set(tok.all_special_ids)
    regular = []
    for token_id in range(vocab_size):
        if token_id in special:
            continue
        after_0 = tok.decode([token_0, token_id])[1:]
        plain = tok.decode([token_id])
        regular.append((token_id, after_0, len(after_0) > len(plain)))

    def decode(tokens: list[int]) -> str:
        return tok.decode(tokens).rstrip("\ufffd")

    data = TokenEnforcerTokenizerData(regular, decode, tok.eos_token_id, False, vocab_size)
    _enforcer_data[vocab_size] = data
    return data


def _constraint(schema: dict[str, Any], tok, vocab_size: int) -> Any:
    """A `prefix_allowed_tokens_fn` that forces the reply to be JSON of `schema`.

    llama-server enforces `response_format.json_schema` with a grammar; transformers
    does not, so without this the 4B judge answers `{"check": 1}` where eidolon needs
    the nested `{"check": {"noul": 1}}` its `_reply` validates, and it keeps generating
    up to `max_tokens` instead of stopping at the closing brace (which overran the 30 s
    `JUDGE_TIMEOUT` on the full 8-question prompt). lm-format-enforcer filters logits
    per prefix and ends the object as soon as it is complete.
    """
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer

    data = _token_enforcer_data(tok, vocab_size)
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))

    def allowed(_batch_id: int, sent: Any) -> list[int]:
        return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

    return allowed


def _judge_generate(entry: dict[str, Any], messages: list[dict], schema: dict | None,
                    max_tokens: int) -> tuple[str, int]:
    tok = entry["tokenizer"]
    model = entry["model"]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
    constraints = _constraint(schema, tok, entry["vocab_size"]) if schema else None
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=min(max_tokens, JUDGE_MAX_TOKENS),
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
            prefix_allowed_tokens_fn=constraints,
        )
    generated = out[0][enc["input_ids"].shape[1]:]
    content = tok.decode(generated, skip_special_tokens=True).strip()
    return content, int(enc["input_ids"].shape[1])


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    model_id = body.get("model", JUDGE_MODEL)
    if model_id not in served:
        raise HTTPException(400, f"unknown model {model_id}")
    messages = body.get("messages") or []
    schema = None
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema")
        if schema:
            messages = list(messages)
            if messages and messages[-1].get("role") == "user":
                messages[-1] = dict(messages[-1])
                messages[-1]["content"] = str(messages[-1]["content"]) + _schema_fields(schema)
    max_tokens = body.get("max_tokens") or JUDGE_MAX_TOKENS
    async with lock:
        content, prompt_tokens = await asyncio.to_thread(
            lambda: _judge_generate(_load(model_id), messages, schema, max_tokens)
        )
    # eidolon does json.loads(content); make sure it is a bare JSON object.
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]
    return JSONResponse({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(content),
            "total_tokens": prompt_tokens + len(content),
        },
    })


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "loaded": list(_models), "device": "cuda", "backend": "rocm-hip"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [
        {"id": mid, "object": "model", "owned_by": "local", "status": "loaded" if mid in _models else "unloaded"}
        for mid in sorted(served)
    ]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--preload", action="store_true", help="load all three models at startup")
    parser.add_argument(
        "--only", nargs="+", choices=["embed", "rerank", "judge"],
        help="serve only these models (judge is too slow on ROCm; see RESULTS.md)",
    )
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "rocm-server.log", encoding="utf-8")],
    )
    log.info("ROCm server starting on %s:%s (torch %s, hip %s, cuda=%s)",
             args.host, args.port, torch.__version__, torch.version.hip, torch.cuda.is_available())
    if not torch.cuda.is_available():
        log.error("CUDA/HIP is not available; aborting")
        return 1
    if args.only:
        wanted = {
            "embed": EMBED_MODEL,
            "rerank": RERANK_MODEL,
            "judge": JUDGE_MODEL,
        }
        served.clear()
        served.update(wanted[o] for o in args.only)
        log.info("serving only: %s", ", ".join(sorted(served)))
    if args.preload:
        for mid in served:
            _load(mid)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_keep_alive=120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
