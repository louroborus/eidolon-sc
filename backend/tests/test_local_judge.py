"""The self-hosted local judge (`JUDGE_PROVIDER=local`): an OpenAI-compatible Chat
Completions server asked with a strict JSON Schema instead of System One, and how it is
chosen and configured. This is what `deploy/local-ai/` serves with llama-server."""

import json
import logging

import httpx
import pytest

from eidolon.billing import Billing, billing
from eidolon.config import LocalJudgeSettings, Settings
from eidolon.judge import (
    JudgeError,
    JudgeRefused,
    LocalJudge,
    check_judge,
    judge_timeout,
    judge_timeout_from_env,
    open_judge,
)
from tests.test_retrieval_services import client, recording

URL = "http://127.0.0.1:8094/v1/chat/completions"
SETTINGS = LocalJudgeSettings("local-judge", "http://127.0.0.1:8094/v1", None)


def answered(answers: dict) -> httpx.Response:
    """A Chat Completions response whose content is the judge's JSON answers."""
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(answers)}}],
        "usage": {"prompt_tokens": 312, "completion_tokens": 40, "total_tokens": 352},
    })


def local(handle) -> LocalJudge:
    judge = open_judge(SETTINGS, client(handle))
    assert isinstance(judge, LocalJudge)
    return judge


async def test_the_local_judge_is_asked_at_the_chat_completions_endpoint():
    requests: list[dict] = []
    answers = {
        "needs_retrieval": {"noul": 0.2},
        "recall": {"noul": 0.1},
        "about_producer": {"noul": 0.9},
    }
    judge = local(recording(requests, answered(answers)))

    assert await judge.ask({"对话": ["制作人：你好"]}, {
        "needs_retrieval": {"type": "noul", "instructions": "需要检索吗？"},
        "recall": {"type": "noul", "instructions": "要回忆吗？"},
        "about_producer": {"type": "noul", "instructions": "和制作人有关吗？"},
    }) == answers

    [request] = requests
    assert request["url"] == URL
    assert request["auth"] == "Bearer local"  # no key needed, but the header is well formed
    assert request["model"] == "local-judge"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 700
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "judge",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    key: {"type": "object", "properties": {"noul": {"type": "number"}},
                          "required": ["noul"], "additionalProperties": False}
                    for key in answers
                },
                "required": list(answers),
                "additionalProperties": False,
            },
        },
    }
    [system, user] = request["messages"]
    assert system["role"] == "system" and "只输出一个 JSON 对象" in system["content"]
    assert user["role"] == "user"
    assert "需要检索吗？" in user["content"] and "只输出一个 JSON 对象" in user["content"]


@pytest.mark.parametrize(
    "question, schema",
    [
        ({"type": "noul", "instructions": "x"},
         {"type": "object", "properties": {"noul": {"type": "number"}},
          "required": ["noul"], "additionalProperties": False}),
        ({"type": "score", "instructions": "x", "criteria": ["差", "中", "好"]},
         {"type": "object", "properties": {"probabilities": {
             "type": "object",
             "properties": {"0": {"type": "number"}, "1": {"type": "number"}, "2": {"type": "number"}},
             "required": ["0", "1", "2"], "additionalProperties": False}},
          "required": ["probabilities"], "additionalProperties": False}),
        ({"type": "choice", "instructions": "x", "criteria": {"喜": "高兴", "怒": "生气"}},
         {"type": "object", "properties": {"probabilities": {
             "type": "object",
             "properties": {"喜": {"type": "number"}, "怒": {"type": "number"}},
             "required": ["喜", "怒"], "additionalProperties": False}},
          "required": ["probabilities"], "additionalProperties": False}),
    ],
    ids=["noul", "score", "choice"],
)
async def test_each_question_type_asks_for_its_own_schema(question: dict, schema: dict):
    requests: list[dict] = []
    judge = local(recording(requests, answered({"q": {"noul": 0.4}})))

    await judge.ask({}, {"q": question})

    assert requests[0]["response_format"]["json_schema"]["schema"] == {
        "type": "object", "properties": {"q": schema}, "required": ["q"],
        "additionalProperties": False,
    }


async def test_a_question_of_no_known_type_is_reported_not_asked():
    asked: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        return answered({})

    with pytest.raises(JudgeError, match="本地判断模型的提问无法转换.*other"):
        await local(handle).ask({}, {"q": {"type": "other"}})

    assert asked == []  # the schema is built before anything is sent


async def test_the_prompt_tokens_are_billed():
    judge = local(lambda _: answered({"q": {"noul": 0.4}}))
    billed = Billing()
    with billing(billed):
        await judge.ask({}, {"q": {"type": "noul", "instructions": "x"}})

    assert billed.unit == "tokens" and billed.input == 312


async def test_a_refusal_is_named_the_local_judge():
    response = httpx.Response(503, text="模型正在加载")
    with pytest.raises(JudgeRefused, match="本地判断模型 返回 503"):
        await local(lambda _: response).ask({}, {"q": {"type": "noul", "instructions": "x"}})


async def test_an_unparseable_answer_is_a_judge_error():
    response = httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
    with pytest.raises(JudgeError, match="本地判断模型 的返回无法解析"):
        await local(lambda _: response).ask({}, {"q": {"type": "noul", "instructions": "x"}})


async def test_a_missing_choices_key_is_a_judge_error():
    response = httpx.Response(200, json={"usage": {"prompt_tokens": 1}})
    with pytest.raises(JudgeError, match="本地判断模型 的返回无法解析"):
        await local(lambda _: response).ask({}, {"q": {"type": "noul", "instructions": "x"}})


async def test_the_local_judge_is_not_tried_as_the_server_starts(caplog: pytest.LogCaptureFixture):
    """`check_judge` tries Cloudflare's and Model Studio's credentials only: a local judge
    is a server the deployer runs, and its absence must not stop the start."""
    with caplog.at_level(logging.WARNING):
        await check_judge(SETTINGS, client(lambda _: httpx.Response(500)))

    assert caplog.text == ""  # as Jev: silent, and asked nothing


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ("JUDGE_PROVIDER", "LOCAL_JUDGE_MODEL", "LOCAL_JUDGE_BASE_URL",
                 "LOCAL_JUDGE_API_KEY", "JUDGE_TIMEOUT", "JEV_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHAT_LLM_BASE_URL", "http://llm.invalid/v1")
    monkeypatch.setenv("CHAT_LLM_API_KEY", "chat-key")
    monkeypatch.setenv("CHAT_LLM_MODEL", "chat-model")
    return monkeypatch


def test_local_is_chosen_without_any_key(env: pytest.MonkeyPatch):
    env.setenv("JUDGE_PROVIDER", "local")

    assert Settings.from_env().judge == SETTINGS


def test_local_settings_are_read_from_the_env(env: pytest.MonkeyPatch):
    env.setenv("JUDGE_PROVIDER", "local")
    env.setenv("LOCAL_JUDGE_MODEL", "my-judge")
    env.setenv("LOCAL_JUDGE_BASE_URL", "http://localhost:9000/v1/")
    env.setenv("LOCAL_JUDGE_API_KEY", "secret")

    assert Settings.from_env().judge == LocalJudgeSettings("my-judge", "http://localhost:9000/v1", "secret")


async def test_the_key_is_sent_as_the_bearer_when_one_is_configured():
    requests: list[dict] = []
    judge = LocalJudge(
        "local-judge", "http://localhost:9000/v1", "secret",
        client(recording(requests, answered({"q": {"noul": 0.4}}))),
    )
    await judge.ask({}, {"q": {"type": "noul", "instructions": "x"}})

    assert requests[0]["auth"] == "Bearer secret"


def test_judge_timeout_defaults_to_five_seconds(env: pytest.MonkeyPatch):
    assert judge_timeout_from_env() == 5.0
    assert judge_timeout().read == 5.0


def test_judge_timeout_is_configurable(env: pytest.MonkeyPatch):
    env.setenv("JUDGE_TIMEOUT", "30")

    assert judge_timeout_from_env() == 30.0
    assert judge_timeout().read == 30.0
    assert judge_timeout().connect == 3.0  # never longer than the old connect timeout


def test_a_short_judge_timeout_keeps_connect_inside_it(env: pytest.MonkeyPatch):
    env.setenv("JUDGE_TIMEOUT", "1")

    assert judge_timeout().connect == 1.0


def test_a_bad_judge_timeout_stops_the_start(env: pytest.MonkeyPatch):
    # Every bad value reaches the one guard in `_seconds`: not a number and not a
    # positive number alike.
    for bad in ("0", "-1", "abc"):
        env.setenv("JUDGE_TIMEOUT", bad)
        with pytest.raises(RuntimeError, match="环境变量 JUDGE_TIMEOUT 需要是大于 0 的秒数"):
            judge_timeout_from_env()


def test_a_blank_judge_timeout_uses_the_default(env: pytest.MonkeyPatch):
    env.setenv("JUDGE_TIMEOUT", "   ")

    assert judge_timeout_from_env() == 5.0
