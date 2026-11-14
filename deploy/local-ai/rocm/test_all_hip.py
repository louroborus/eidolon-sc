"""End-to-end check of the all-HIP llama-server (8097) with eidolon's own clients.

Embeddings, rerank and judge all against the same ROCm/HIP llama-server, using the
project's client classes so prompts, schemas and parsing are exactly production's.
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

BASE = "http://127.0.0.1:8097/v1"
os.environ["SILICONFLOW_BASE_URL"] = BASE
os.environ["LOCAL_JUDGE_BASE_URL"] = BASE

from eidolon.config import EmbeddingSettings, LocalJudgeSettings, RerankSettings  # noqa: E402
from eidolon.embedding import Document, open_embedder  # noqa: E402
from eidolon.judge import LocalJudge, _CHECK_QUESTION  # noqa: E402
from eidolon.rerank import SiliconFlowReranker  # noqa: E402

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


async def main():
    embedder = open_embedder(EmbeddingSettings.required_from_env())
    rerank_settings = RerankSettings.from_env()
    reranker = SiliconFlowReranker(
        rerank_settings.api_key, rerank_settings.model, rerank_settings.base_url
    )
    judge_settings = LocalJudgeSettings.from_env()
    judge = LocalJudge(judge_settings.model, judge_settings.base_url, judge_settings.api_key)
    ok = True
    try:
        t0 = time.time()
        vectors = await embedder.embed_documents(
            [Document(text="制作人想去看海，约她一起去。"), Document(text="今天练习很辛苦。")]
        )
        print(f"embed_documents (cold): {len(vectors)} vec, dim {len(vectors[0])}, "
              f"{time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        query = await embedder.embed_query("要不要去看海")
        print(f"embed_query: dim {len(query)}, {time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        ranked = await reranker.rerank(
            "要不要去看海", ["制作人想去看海，约她一起去。", "今天练习很辛苦。"], 2
        )
        print(f"rerank: {[(i, round(s, 4)) for i, s in ranked]}, {time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        answers = await judge.ask({"测试": "启动检查"}, {"check": _CHECK_QUESTION})
        print(f"judge check: {answers}, {time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        full = await judge.ask(STATE, QUESTIONS)
        elapsed = time.time() - t0
        print(f"judge full 8 questions: {elapsed:.2f}s", flush=True)
        print(f"  keys: {list(full)}", flush=True)
        nested = all(isinstance(v, dict) for v in full.values())
        print(f"  nested (eidolon requirement): {nested}", flush=True)
        if elapsed > 30:
            print("  WARNING: over JUDGE_TIMEOUT=30", flush=True)
            ok = False
        if not nested:
            ok = False
    finally:
        await embedder.aclose()
        await reranker.aclose()
        await judge.aclose()
    print("\nALL THREE ON HIP:", "OK" if ok else "PROBLEM", flush=True)
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
