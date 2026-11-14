"""The retrieval services' own protocols: SiliconFlow's rerank endpoint, and Jev through
TypeSafe's System One endpoint (OpenRouter's by default), and how Jev is configured."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from eidolon.characters import Character
from eidolon.config import JevSettings, Settings, rerank_timeout_from_env
from eidolon.producer import Birthday
from eidolon.rerank import RerankError, SiliconFlowReranker, rerank_timeout
from eidolon.judge import JevJudge, JudgeError
from eidolon.retrieval import ModelRetrievalJudge
from eidolon.store import Message
from tests.test_embedding import HANG, Answer, scripted

Handler = Callable[[httpx.Request], httpx.Response]


def client(handle: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def recording(requests: list[dict], response: httpx.Response) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append({"url": str(request.url), "auth": request.headers["Authorization"],
                         **json.loads(request.content)})
        return response

    return handle


async def test_reranker_returns_the_best_documents_first():
    requests: list[dict] = []
    # Not sorted on purpose: the order comes from the scores.
    results = [{"index": 0, "relevance_score": 0.1}, {"index": 2, "relevance_score": 0.9},
               {"index": 1, "relevance_score": 0.5}]
    reranker = SiliconFlowReranker(
        "sk-test", "Qwen/Qwen3-Reranker-0.6B", "http://sf.test/v1",
        client(recording(requests, httpx.Response(200, json={"results": results}))),
    )

    best = await reranker.rerank("鳩の話", ["a", "b", "c"], top_n=2)

    assert best == [(2, 0.9), (1, 0.5)]
    [request] = requests
    assert request["url"] == "http://sf.test/v1/rerank"
    assert request["auth"] == "Bearer sk-test"
    assert (request["model"], request["query"], request["documents"], request["top_n"]) == (
        "Qwen/Qwen3-Reranker-0.6B", "鳩の話", ["a", "b", "c"], 2
    )


@pytest.mark.parametrize(
    "response, error",
    [
        (httpx.Response(401, text="unauthorized"), "401"),
        (httpx.Response(200, json={"oops": []}), "无法解析"),
        (httpx.Response(200, json={"results": [{"index": 7, "relevance_score": 1}]}), "不存在"),
    ],
)
async def test_reranker_errors_are_reported(response: httpx.Response, error: str):
    reranker = SiliconFlowReranker("k", "m", "http://sf.test/v1", client(lambda _: response))

    with pytest.raises(RerankError, match=error):
        await reranker.rerank("q", ["a"], top_n=1)



def ranked(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})


def scripted_reranker(answers: list[Answer]) -> SiliconFlowReranker:
    scripted_client = httpx.AsyncClient(transport=httpx.MockTransport(scripted(answers, ranked)))
    return SiliconFlowReranker("k", "m", "http://sf.test/v1", scripted_client, timeout=0.05)


@pytest.mark.parametrize(
    "failure",
    [
        HANG,
        httpx.ReadError("connection reset"),
        httpx.Response(429, text="rate limited"),
        httpx.Response(500, text="internal error"),
    ],
)
async def test_a_rerank_failing_in_passing_is_asked_again_at_once(failure: Answer):
    answers = [failure]

    assert await scripted_reranker(answers).rerank("q", ["a"], top_n=1) == [(0, 0.9)]
    assert answers == []


@pytest.mark.parametrize(
    "answers, error",
    [
        ([HANG, HANG], "没有回应"),
        ([HANG, httpx.Response(503, text="unavailable")], "503"),
        ([httpx.Response(403, text="forbidden")], "403"),
        ([httpx.Response(200, json={"oops": []})], "无法解析"),
    ],
)
async def test_a_rerank_failing_again_or_refused_is_reported(answers: list[Answer], error: str):
    answers = [*answers, httpx.Response(200)]

    with pytest.raises(RerankError, match=error):
        await scripted_reranker(answers).rerank("q", ["a"], top_n=1)
    assert len(answers) == 1  # never asked a third time, nor a second when refused


@pytest.fixture
def rerank_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("RERANK_TIMEOUT", raising=False)
    return monkeypatch


def test_rerank_timeout_defaults_to_three_seconds(rerank_env: pytest.MonkeyPatch):
    assert rerank_timeout_from_env() == 3.0
    assert rerank_timeout() == 3.0


def test_rerank_timeout_is_configurable(rerank_env: pytest.MonkeyPatch):
    rerank_env.setenv("RERANK_TIMEOUT", "10")

    assert rerank_timeout_from_env() == 10.0
    assert rerank_timeout() == 10.0


def test_a_bad_rerank_timeout_stops_the_start(rerank_env: pytest.MonkeyPatch):
    for bad in ("0", "-1", "abc"):
        rerank_env.setenv("RERANK_TIMEOUT", bad)
        with pytest.raises(RuntimeError, match="环境变量 RERANK_TIMEOUT 需要是大于 0 的秒数"):
            rerank_timeout_from_env()


def test_a_blank_rerank_timeout_uses_the_default(rerank_env: pytest.MonkeyPatch):
    rerank_env.setenv("RERANK_TIMEOUT", "   ")

    assert rerank_timeout_from_env() == 3.0


async def test_the_reranker_uses_the_configured_timeout(rerank_env: pytest.MonkeyPatch):
    rerank_env.setenv("RERANK_TIMEOUT", "0.05")

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    slow_client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    reranker = SiliconFlowReranker("k", "m", "http://sf.test/v1", slow_client)

    with pytest.raises(RerankError, match="没有回应"):
        await reranker.rerank("q", ["a"], top_n=1)


MANO = Character("mano", "櫻木真乃", "", "voice", "sheet", None, "#ffbad6", Birthday(4, 25), 16)


def message(role: str, text: str) -> Message:
    if role == "user":
        return Message(1, "user", text, None, None, "2026-09-25T00:00:00+00:00")
    return Message(2, "character", None, text, "", "2026-09-25T00:00:00+00:00")


def judge_answering(
    requests: list[dict], retrieve: float, recall: float = 0.01, about_producer: float = 0.02
) -> ModelRetrievalJudge:
    body = {
        "id": "gen-dec-1",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            "needs_retrieval": {"type": "noul", "noul": retrieve},
            "recall": {"type": "noul", "noul": recall},
            "about_producer": {"type": "noul", "noul": about_producer},
        },
        "usage": {"input_tokens": 500, "output_tokens": 0, "cost": 0.00002},
    }
    return ModelRetrievalJudge(JevJudge(
        "jev-key", "typesafe/jev-1.13", "http://jev.test/api/v1",
        client(recording(requests, httpx.Response(200, json=body))),
    ))


@pytest.mark.parametrize(
    "retrieve, recall, about_producer, expected",
    [
        (0.5, 0.2, 0.1, "references"),
        (0.89, 0.03, 0.95, "references"),  # which carries memories too
        (0.21, 0.01, 0.1, "nothing"),
        (0.21, 0.01, 0.95, "memories"),
        (0.91, 0.96, 0.95, "recall"),
        (0.4, 0.5, 0.1, "recall"),
    ],
)
async def test_jev_answers_what_the_turn_needs(
    retrieve: float, recall: float, about_producer: float, expected: str
):
    requests: list[dict] = []

    judge = judge_answering(requests, retrieve, recall, about_producer)
    assert await judge.judge(MANO, [], "那次比赛") == (expected, None, False)
    [request] = requests  # all questions in one call
    assert request["url"] == "http://jev.test/api/v1/systemone"
    assert request["auth"] == "Bearer jev-key"
    assert request["model"] == "typesafe/jev-1.13"
    for question in request["questions"].values():
        assert question["type"] == "noul"
        assert question["criteria"].keys() == {"true", "false"}  # System One wants both


async def test_jev_sees_the_latest_messages_and_what_was_just_said():
    requests: list[dict] = []
    history = [message("user", f"old {i}") for i in range(5)] + [message("character", "ほわっ")]

    await judge_answering(requests, 0.1).judge(MANO, history, "那次比赛")

    state = requests[0]["state"]
    assert state["角色"] == "櫻木真乃"
    assert state["上文"][-1] == "櫻木真乃：ほわっ" and "制作人：old 4" in state["上文"]
    assert "制作人：old 1" not in state["上文"]
    assert state["制作人最新的一句话"] == "那次比赛"


async def test_jev_errors_are_reported():
    judge = ModelRetrievalJudge(JevJudge(
        "k", "typesafe/jev-1.13", "http://jev.test/api/v1",
        client(lambda _: httpx.Response(503, text="busy")),
    ))

    with pytest.raises(JudgeError, match="503"):
        await judge.judge(MANO, [], "在吗")


async def test_unparsable_jev_answers_are_reported():
    judge = ModelRetrievalJudge(JevJudge(
        "k", "typesafe/jev-1.13", "http://jev.test/api/v1",
        client(lambda _: httpx.Response(200, json={"answers": {}})),
    ))

    with pytest.raises(JudgeError, match="无法解析"):
        await judge.judge(MANO, [], "在吗")


def test_jev_goes_through_openrouter_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JEV_MODEL", raising=False)
    monkeypatch.delenv("JEV_BASE_URL", raising=False)
    monkeypatch.setenv("JEV_API_KEY", "or-key")

    assert JevSettings.from_env() == JevSettings(
        "or-key", "typesafe/jev-1.13", "https://openrouter.ai/api/v1"
    )


def test_jev_can_be_reached_elsewhere(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JEV_API_KEY", "ts-key")
    monkeypatch.setenv("JEV_MODEL", "jev-1.13.0")
    monkeypatch.setenv("JEV_BASE_URL", "https://api.typesafe.ai/v1")

    assert JevSettings.from_env() == JevSettings(
        "ts-key", "jev-1.13.0", "https://api.typesafe.ai/v1"
    )


def test_the_old_gateway_key_alone_stops_the_start(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
    monkeypatch.setenv("CHAT_LLM_BASE_URL", "http://llm.invalid/v1")
    monkeypatch.setenv("CHAT_LLM_API_KEY", "chat-key")
    monkeypatch.setenv("CHAT_LLM_MODEL", "chat-model")

    with pytest.raises(RuntimeError, match="AI_GATEWAY_API_KEY.*JEV_API_KEY"):
        Settings.from_env()
