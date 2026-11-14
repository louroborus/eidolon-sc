"""Exercise eidolon's real LocalJudge client against the ROCm server (8095).

Uses the project's own judge module (backend/src), so the prompt, strict schema,
response parsing and timeout are exactly what production uses.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\eidolon-sc")
sys.path.insert(0, str(ROOT / "backend" / "src"))
# The ROCm venv lacks tzdata (installing is blocked by the sandbox), but eidolon's
# judge module imports zoneinfo("Asia/Tokyo"). Borrow it from the project venv.
sys.path.append(str(ROOT / "backend" / ".venv" / "Lib" / "site-packages"))
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line.startswith("LOCAL_JUDGE_BASE_URL="):
        continue
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v)
os.environ["LOCAL_JUDGE_BASE_URL"] = "http://127.0.0.1:8095/v1"

from eidolon.config import LocalJudgeSettings  # noqa: E402
from eidolon.judge import LocalJudge, _CHECK_QUESTION  # noqa: E402

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
    settings = LocalJudgeSettings("local-judge", "http://127.0.0.1:8095/v1", None)
    judge = LocalJudge(settings.model, settings.base_url, settings.api_key)
    try:
        for label, questions in (("startup-check", {"check": _CHECK_QUESTION}),
                                 ("full-8-questions", QUESTIONS)):
            t0 = time.time()
            try:
                answers = await judge.ask(STATE, questions)
                print(f"{label}: OK in {time.time() - t0:.1f}s", flush=True)
                print("   ", str(answers)[:500], flush=True)
            except Exception as e:
                print(f"{label}: FAILED in {time.time() - t0:.1f}s: "
                      f"{type(e).__name__}: {str(e)[:400]}", flush=True)
    finally:
        await judge.aclose()


asyncio.run(main())
