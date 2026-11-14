"""Test the split configuration: embed+rerank on ROCm (8095), judge on Vulkan (8094).

eidolon reads SILICONFLOW_BASE_URL for both embedding and rerank, and
LOCAL_JUDGE_BASE_URL separately for the judge. So the two can point at different
servers, letting the fast 0.6B models use ROCm while the 4B judge stays on
llama.cpp. This measures whether that arrangement actually works end to end.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\eidolon-sc")
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.append(str(ROOT / "backend" / ".venv" / "Lib" / "site-packages"))
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v)

# Embed + rerank -> ROCm (8095); judge -> Vulkan (8094)
os.environ["SILICONFLOW_BASE_URL"] = "http://127.0.0.1:8095/v1"
os.environ["LOCAL_JUDGE_BASE_URL"] = "http://127.0.0.1:8094/v1"

from eidolon.config import EmbeddingSettings, RerankSettings, LocalJudgeSettings  # noqa: E402
from eidolon.embedding import Document, open_embedder  # noqa: E402
from eidolon.judge import LocalJudge, _CHECK_QUESTION  # noqa: E402
from eidolon.rerank import SiliconFlowReranker  # noqa: E402


async def main():
    embed_settings = EmbeddingSettings.required_from_env()
    embedder = open_embedder(embed_settings)
    rerank_settings = RerankSettings.from_env()
    reranker = SiliconFlowReranker(
        rerank_settings.api_key, rerank_settings.model, rerank_settings.base_url
    )
    judge_settings = LocalJudgeSettings.from_env()
    judge = LocalJudge(judge_settings.model, judge_settings.base_url, judge_settings.api_key)
    try:
        t0 = time.time()
        docs = [Document(text="制作人想去看海，约她一起去。"), Document(text="今天练习很辛苦。")]
        vectors = await embedder.embed_documents(docs)
        print(f"embed via ROCm: {len(vectors)} vectors, dim {len(vectors[0])}, "
              f"{time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        query = await embedder.embed_query("要不要去看海")
        print(f"embed_query via ROCm: dim {len(query)}, {time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        ranked = await reranker.rerank("要不要去看海", ["制作人想去看海，约她一起去。", "今天练习很辛苦。"], 2)
        print(f"rerank via ROCm: {[(i, round(s, 4)) for i, s in ranked]}, "
              f"{time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        answers = await judge.ask({"测试": "启动检查"}, {"check": _CHECK_QUESTION})
        print(f"judge via Vulkan: {answers}, {time.time() - t0:.2f}s", flush=True)

        print("\nALL FOUR OK: embed+rerank on ROCm, judge on Vulkan", flush=True)
    finally:
        await embedder.aclose()
        await reranker.aclose()
        await judge.aclose()


asyncio.run(main())
