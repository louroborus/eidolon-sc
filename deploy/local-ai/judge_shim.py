#!/usr/bin/env python3
"""judge_shim.py —— 把 TypeSafe System One 的 /systemone 翻译成 OpenAI 兼容的 /chat/completions。

> ⚠️ 这个脚本是**方案 A 的回退**，不是默认路径。`local-ai` 分支已经把同样的
> schema/prompt 逻辑做进后端（`backend/src/eidolon/judge.py` 的 `LocalJudge`，
> `JUDGE_PROVIDER=local`），不需要额外进程。`LocalJudge` 那份没有再导入这里的代码
> —— 它是本脚本 `schema_for` / `prompt_for` 的移植，改一处记得看另一处。
> 需要「一行代码都不改后端」时才用这个脚本，用法见文末。

为什么需要它
------------
eidolon 的判断模型客户端（`backend/src/eidolon/judge.py` 里的 `JevJudge`）只会说
System One 协议：

    POST {base_url}/systemone
    body {"model": ..., "state": {...}, "questions": {key: {...}}}
    ← {"answers": {key: dict}, "usage": {"input_tokens": N}}

而 llama-server 只提供 `/v1/chat/completions`。这个脚本在中间做翻译，于是 eidolon
**一行代码都不用改**，只靠 `.env` 就能用本机 4B 模型做检索判断、好感度、照片判断
和通话中的情绪判断。

翻译的做法
----------
1. 按每个问题的 `type` 生成 **strict JSON Schema**，交给 llama-server 的
   `response_format: {"type": "json_schema", ...}` 强制输出合法 JSON：
   - `noul`   → `{"noul": number}`                       （0..1，越接近 1 越"是"）
   - `score`  → `{"probabilities": {"0": n, "1": n, ...}}`（档位序号为键）
   - `choice` → `{"probabilities": {"选项名": n, ...}}`    （中文选项名原样为键）
2. 把 `state`（角色 / 上文 / 制作人最新的一句话）和每题的 instructions + criteria
   拼成一段中文 prompt。
3. `temperature: 0`，`max_tokens: 700`，把上游的 `usage.prompt_tokens` 填进
   `usage.input_tokens`（eidolon 用它记账）。

用法
----
以下路径都相对仓库根（`F:\eidolon-sc`）：

    # 1) 先起 llama-server（三模型同进程）
    powershell -File deploy\local-ai\start-llama-server.ps1

    # 2) 再起本 shim
    python deploy\local-ai\judge_shim.py

    # 3) eidolon 的 .env 指过来（见 deploy\local-ai\env.local-ai.example）
    #    方案 A 用 JEV_BASE_URL + JUDGE_PROVIDER=jev 指向 8097，不是 LOCAL_JUDGE_*

    # 自检：不监听端口，直接问上游一次
    python deploy\local-ai\judge_shim.py --selftest

环境变量
--------
    JUDGE_SHIM_PORT      监听端口，默认 8097
    JUDGE_SHIM_UPSTREAM  上游 OpenAI 兼容地址，默认 http://127.0.0.1:8094/v1
    JUDGE_SHIM_MODEL     上游模型名，默认 local-judge（必须与 models.ini 的段名一致）
    JUDGE_SHIM_API_KEY   上游 key，默认 local（llama-server 不校验，但不能为空）
    JUDGE_SHIM_TIMEOUT   上游超时秒数，默认 120

实测（AMD RX 6750 XT / Vulkan）：3 个 noul 问题热 0.51–0.53 s，全量 8 问题热 2.6 s。
eidolon 的 `JUDGE_TIMEOUT` 默认 5 s（本分支已可配，本地建议 30），冷启动会超，
所以 `.env` 里要放宽。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PORT = int(os.environ.get("JUDGE_SHIM_PORT") or 8097)
UPSTREAM = (os.environ.get("JUDGE_SHIM_UPSTREAM") or "http://127.0.0.1:8094/v1").rstrip("/")
MODEL = os.environ.get("JUDGE_SHIM_MODEL") or "local-judge"
API_KEY = os.environ.get("JUDGE_SHIM_API_KEY") or "local"
TIMEOUT = float(os.environ.get("JUDGE_SHIM_TIMEOUT") or 120)

MAX_TOKENS = 700

SYSTEM = (
    "你是一个角色扮演对话里的判断模型。用户会给你对话状态和若干个打分问题，"
    "你只输出一个 JSON 对象，不要解释，不要输出任何 JSON 之外的东西。"
)

#: eidolon 启动/后台保存时用来试调的那道题（`judge.py:40 _CHECK_QUESTION`）。
CHECK_QUESTION: dict[str, Any] = {"type": "noul", "instructions": "这是一条测试消息吗？"}


# ── schema ──────────────────────────────────────────────────────────────


def _probabilities(names: list[str]) -> dict[str, Any]:
    """给一组名字生成 `{"probabilities": {名字: number}}` 的 strict schema。"""
    properties = {name: {"type": "number"} for name in names}
    return {
        "type": "object",
        "properties": {"probabilities": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }},
        "required": ["probabilities"],
        "additionalProperties": False,
    }


def schema_for(questions: dict[str, Any]) -> dict[str, Any]:
    """把 eidolon 的 questions 翻成 llama-server 能吃的 strict JSON Schema。"""
    properties: dict[str, Any] = {}
    for key, question in questions.items():
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "noul":
            properties[key] = {
                "type": "object",
                "properties": {"noul": {"type": "number"}},
                "required": ["noul"],
                "additionalProperties": False,
            }
        elif kind == "score":
            levels = criteria if isinstance(criteria, list) else []
            properties[key] = _probabilities([str(index) for index in range(len(levels))])
        elif kind == "choice":
            options = criteria if isinstance(criteria, dict) else {}
            properties[key] = _probabilities(list(options))
        else:
            raise ValueError(f"不认识的问题类型 {kind!r}（键 {key}）")
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


# ── prompt ──────────────────────────────────────────────────────────────


def prompt_for(state: Any, questions: dict[str, Any]) -> str:
    """把 state 和 questions 拼成一段中文 prompt。"""
    lines = ["对话状态：", json.dumps(state, ensure_ascii=False, indent=1), "", "需要你回答的问题："]
    for key, question in questions.items():
        kind = question.get("type")
        criteria = question.get("criteria")
        lines.append("")
        lines.append(f"[{key}] {question.get('instructions', '')}")
        if kind == "noul":
            lines.append("  给一个 0 到 1 之间的数：越接近 1 越偏向「是」，越接近 0 越偏向「否」。")
            if isinstance(criteria, dict):
                for name, description in criteria.items():
                    lines.append(f"  - {name}：{description}")
        elif kind == "score":
            levels = criteria if isinstance(criteria, list) else []
            lines.append(f"  按 {len(levels)} 档评估，每档给一个 0 到 1 之间的概率，加起来接近 1。")
            for index, description in enumerate(levels):
                lines.append(f"  - 第 {index} 档：{description}")
        elif kind == "choice":
            options = criteria if isinstance(criteria, dict) else {}
            lines.append("  每个选项给一个 0 到 1 之间的概率，加起来接近 1。")
            for name, description in options.items():
                lines.append(f"  - {name}：{description}")
        else:
            raise ValueError(f"不认识的问题类型 {kind!r}（键 {key}）")
    lines.append("")
    lines.append(f"只输出一个 JSON 对象，键是：{', '.join(questions)}。")
    return "\n".join(lines)


# ── 上游请求 ────────────────────────────────────────────────────────────


def ask(state: Any, questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """把一组问题丢给 llama-server，返回 System One 形状的 `{"answers", "usage"}`。"""
    body = {
        "model": model or MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_for(state, questions)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judge", "schema": schema_for(questions), "strict": True},
        },
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    request = urllib.request.Request(
        f"{UPSTREAM}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"上游 {UPSTREAM} 返回 {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"连不上上游 {UPSTREAM}：{error.reason}") from error

    content = payload["choices"][0]["message"]["content"]
    try:
        answers = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"上游返回的不是 JSON：{content[:300]}") from error
    if not isinstance(answers, dict):
        raise RuntimeError(f"上游返回的不是 JSON 对象：{content[:300]}")

    usage = payload.get("usage") or {}
    tokens = usage.get("prompt_tokens")
    return {"answers": answers, "usage": {"input_tokens": tokens if isinstance(tokens, int) else None}}


# ── HTTP 服务 ───────────────────────────────────────────────────────────


def log(message: str) -> None:
    print(f"[judge-shim] {message}", file=sys.stderr, flush=True)


def _route(path: str) -> str:
    """去掉查询串和尾部斜杠。

    eidolon 的 `JevJudge` 把路径拼成 `base_url.rstrip("/") + "/systemone"`，所以
    `JEV_BASE_URL=http://127.0.0.1:8097/v1` 实际打到的是 `POST /v1/systemone`。
    这里两种前缀都收，避免用户漏写或多写 `/v1` 就 404。
    """
    return path.split("?", 1)[0].rstrip("/")


class Handler(BaseHTTPRequestHandler):
    """只实现 eidolon 真正会打的那两条：`POST /systemone`、`GET /health`。"""

    protocol_version = "HTTP/1.0"  # 一次请求一个连接，避免 body 读错导致 keep-alive 错位

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 规定的名字
        if _route(self.path).endswith("/health"):
            self._send(200, {"status": "ok", "upstream": UPSTREAM, "model": MODEL})
        else:
            self._send(404, {"error": {"message": f"没有 {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 规定的名字
        started = time.monotonic()
        if not _route(self.path).endswith("/systemone"):
            self._send(404, {"error": {"message": f"没有 {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            body = json.loads(raw)
            questions = body.get("questions") or {}
            if not isinstance(questions, dict):
                raise ValueError("questions 必须是对象")
            answer = ask(body.get("state") or {}, questions, body.get("model"))
        except Exception as error:  # noqa: BLE001 - 任何异常都要回给 eidolon，让它自己报错
            elapsed = time.monotonic() - started
            log(f"{self.path} 失败（{elapsed:.2f}s）：{error!r}")
            self._send(502, {"error": {"message": str(error) or repr(error)}})
            return
        elapsed = time.monotonic() - started
        log(f"{self.path} {len(questions)} 问 → 200（{elapsed:.2f}s）")
        self._send(200, answer)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        """默认实现会往 stderr 打每条访问日志，这里静音，自己上面打。"""


# ── 入口 ────────────────────────────────────────────────────────────────


def selftest() -> int:
    """不监听端口，直接问上游一次，确认整条链路通。"""
    print(f"上游 {UPSTREAM}，模型 {MODEL}")
    started = time.monotonic()
    try:
        answer = ask({"测试": "保存前试调"}, {"check": CHECK_QUESTION})
    except Exception as error:  # noqa: BLE001 - 自检就是要看到真实错因
        print(f"失败：{error}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started
    print(f"成功（{elapsed:.2f}s）：{json.dumps(answer, ensure_ascii=False)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="System One → chat completions 翻译服务")
    parser.add_argument("--selftest", action="store_true", help="直接问上游一次，不监听端口")
    parser.add_argument("--port", type=int, default=PORT, help=f"监听端口，默认 {PORT}")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if args.selftest:
        return selftest()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    log(f"监听 http://127.0.0.1:{args.port} → 上游 {UPSTREAM}（模型 {MODEL}）")
    log("eidolon 的 .env 要写：JEV_BASE_URL=http://127.0.0.1:%d/v1" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("收到 Ctrl+C，退出")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
