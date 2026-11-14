"""The judge model (GLOSSARY.md「判断模型」): asked typed questions about a state, it answers
each with probabilities, and writes nothing. Retrieval asks it what a turn needs looked up,
and calls ask it how she feels in each segment of a reply; what is asked is theirs, how it
is asked is here.

The deployer chooses one service for the whole deployment (`JUDGE_PROVIDER`): Jev through
any endpoint that serves TypeSafe's System One API, OpenRouter's by default; Cloudflare's
Clef or Clef-flash on Workers AI, which take the same questions; or Alibaba Cloud Model
Studio's decision model, served over System One itself. None stands in for another when it
fails.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import httpx

from eidolon.billing import bill
from eidolon.characters import Character
from eidolon.config import (
    BailianJudgeSettings,
    CloudflareJudgeSettings,
    EmotionSettings,
    JudgeSettings,
    LocalJudgeSettings,
    judge_timeout_from_env,
)
from eidolon.store import Message

log = logging.getLogger(__name__)

CLOUDFLARE_API_URL = "https://api.cloudflare.com/client/v4"

_ACCOUNT_ERROR_CODES = {7000, 7003}
"""Cloudflare's codes for a path it cannot route, as with an account id that is wrong."""
_QUOTA_ERROR_CODES = {4006}
"""Cloudflare's code for the day's free allocation of neurons being used up."""

_CHECK_QUESTION = {"type": "noul", "instructions": "这是一条测试消息吗？"}

JUDGE_HISTORY = 4
"""How many of the latest messages the judge model sees along with what the user just said."""

SAID = "制作人最新的一句话"
"""What the user just said, as the judge model is told it about a turn (`turn_state`)."""


def turn_state(character: Character, history: Sequence[Message], text: str) -> dict[str, Any]:
    """What the judge model is told of a turn, whatever it is asked about it: who she is,
    the latest messages before it, and what the user just said (`SAID`)."""
    recent = []
    for message in history[-JUDGE_HISTORY:]:
        if message.role == "user":
            recent.append(f"制作人：{message.text}")
        else:
            recent.append(f"{character.name}：{message.without_cues().ja}")
    return {"角色": character.name, "上文": recent or "（对话刚开始）", SAID: text}


@dataclass(frozen=True)
class JudgeThresholds:
    """The judge model's probabilities at or above which its answer counts as yes. Each
    model is sure of itself in its own way, so each has thresholds of its own."""

    needs_retrieval: float
    """Whether the turn needs references looked up."""
    recall: float
    """Whether the turn asks how a past episode went."""
    about_producer: float
    """Whether the turn is about the user's own life, so their memories are found."""
    emotion: float
    """The likeliest emotion's probability below which she is calm, and the likeliest
    nod's below which there is none (overridden by `JUDGE_EMOTION_MIN_PROBABILITY`)."""
    out_of_line: float
    """Whether what the user said is out of line (GLOSSARY.md「不该说的话」)."""
    manipulation: float
    """Whether what the user said tries to dictate her affinity, or poses as the system."""
    photo: float
    """Whether the user asks her for a photo (`photo_judge`)."""


JudgeFamily = Literal["jev", "clef", "clef-flash", "decision-model"]
"""The judge models with thresholds of their own, by whatever name and service they are
reached: Clef-flash through OpenRouter judges as it does on Workers AI."""

JUDGE_THRESHOLDS: dict[JudgeFamily, JudgeThresholds] = {
    # Set by the judge eval (backend/eval/judge.toml); how each was chosen from the spread
    # of probabilities is in .scratch/cloudflare-judge/issues/05-apply-eval-and-docs.md,
    # and affinity's two (out_of_line, manipulation) in
    # .scratch/affinity/issues/04-judge-eval-and-thresholds.md. Out of line is set high
    # where its yes and no come close, as a turn wrongly out of line costs her affinity
    # most: Jev finds a bare demand for affinity somewhat out of line, and Clef-flash a
    # user telling her she is only a program. Asking for a photo, chosen in
    # .scratch/photos/issues/23-photo-judge-calibration.md, is set where no turn not asking
    # is taken for asking, better a photo asked for and not sent than one sent unasked: Jev
    # is sure of no, and not so sure of a request made by asking to see something.
    "jev": JudgeThresholds(
        needs_retrieval=0.5, recall=0.5, about_producer=0.5, emotion=0.5,
        out_of_line=0.7, manipulation=0.4, photo=0.5,
    ),
    # Clef and Clef-flash are surer of recall than they should be, less sure of the user's
    # own life, and less sure of the emotion they hear. Clef-flash finds the user saying a
    # photo came close to asking for one, so it misses some asking rather than take that.
    "clef": JudgeThresholds(
        needs_retrieval=0.5, recall=0.8, about_producer=0.1, emotion=0.4,
        out_of_line=0.5, manipulation=0.5, photo=0.8,
    ),
    "clef-flash": JudgeThresholds(
        needs_retrieval=0.5, recall=0.7, about_producer=0.3, emotion=0.4,
        out_of_line=0.6, manipulation=0.6, photo=0.8,
    ),
    # Model Studio's decision model, chosen in .scratch/bailian-judge/thresholds.md, answers
    # the same every time, so each sits mid-gap. It is sure a turn needs nothing looked up,
    # and seldom sure one does, so a little is enough; it finds most turns somewhat about a
    # past episode. Asking for a photo it cannot tell from talk of photos: only 0.92 takes
    # no turn not asking for asking, and misses half the asking.
    "decision-model": JudgeThresholds(
        needs_retrieval=0.06, recall=0.8, about_producer=0.1, emotion=0.4,
        out_of_line=0.3, manipulation=0.4, photo=0.92,
    ),
}
FALLBACK_FAMILY: JudgeFamily = "jev"
"""Whose thresholds a model not in `JUDGE_THRESHOLDS` gets."""


def judge_family(model: str) -> JudgeFamily | None:
    """The group of `model`, named as any service names it (`typesafe/jev-1.13`,
    `cloudflare/clef-flash`, `clef`); None for a model of none of them."""
    name = model.rsplit("/", 1)[-1].lower()
    if name in ("clef", "clef-flash"):
        return name  # pyright: ignore[reportReturnType]  # checked above
    if name == "jev" or name.startswith("jev-"):  # Jev is named by version, or as latest
        return "jev"
    if name == "decision-model" or name.startswith("decision-model-"):  # as preview, or not
        return "decision-model"
    return None


def judge_thresholds_for(model: str) -> JudgeThresholds:
    """The thresholds `model` judges by; Jev's for a model not known (see
    `warn_if_uncalibrated`)."""
    return JUDGE_THRESHOLDS[judge_family(model) or FALLBACK_FAMILY]


def warn_if_uncalibrated(model: str) -> None:
    """Warn that `model`, taking effect, has no thresholds of its own, if it has none."""
    if judge_family(model) is None:
        log.warning(
            "judge model %s has no thresholds of its own; it judges by %s's: %s",
            model, FALLBACK_FAMILY, JUDGE_THRESHOLDS[FALLBACK_FAMILY],
        )


def same_judge_family(set_for: str, model: str) -> bool:
    """Whether thresholds set for judge model `set_for` hold for `model`: the two are of
    one family, or, either being of none, are the one model."""
    families = judge_family(set_for), judge_family(model)
    if None in families:
        return set_for == model
    return families[0] == families[1]


def judge_thresholds(
    settings: JudgeSettings | None, emotion: EmotionSettings
) -> JudgeThresholds:
    """The thresholds the judge model `settings` choose judges by, with the emotion
    settings' minimum probability, if set, over every model's."""
    thresholds = JUDGE_THRESHOLDS[FALLBACK_FAMILY]
    if settings is not None:
        thresholds = judge_thresholds_for(settings.model)
    if emotion.min_probability is not None:
        thresholds = replace(thresholds, emotion=emotion.min_probability)
    return thresholds


class JudgeError(RuntimeError):
    pass


class JudgeRefused(JudgeError):
    """A System One endpoint answered with an error: its HTTP status, and its body."""

    def __init__(self, message: str, status: int, body: str) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class JudgeReply:
    """The judge model's answers, and how many tokens of input it read, if it said."""

    answers: dict[str, Any]
    input_tokens: int | None


_LOCAL_SYSTEM = (
    "你是一个角色扮演对话里的判断模型。用户会给你对话状态和若干个打分问题，"
    "你只输出一个 JSON 对象，不要解释，不要输出任何 JSON 之外的东西。"
)
_LOCAL_MAX_TOKENS = 700
"""The local judge is asked by strict JSON Schema, as the validated `judge_shim.py` in
`deploy/local-ai/` did: the schema is built from each question's `type` (`noul`, `score`,
`choice`), and the model's answer is parsed straight back into System One's `answers`."""


def _probabilities_schema(names: list[str]) -> dict[str, Any]:
    """A strict `{"probabilities": {name: number}}` schema for `score`/`choice`."""
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


def _local_schema(questions: dict[str, dict]) -> dict[str, Any]:
    """The strict JSON Schema the local judge must answer with, from `questions`. Mirrors
    `judge_shim.schema_for` (validated end-to-end against Qwen3-4B on llama-server)."""
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
            properties[key] = _probabilities_schema([str(i) for i in range(len(levels))])
        elif kind == "choice":
            options = criteria if isinstance(criteria, dict) else {}
            properties[key] = _probabilities_schema(list(options))
        else:
            raise ValueError(f"不认识的问题类型 {kind!r}（键 {key}）")
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _local_prompt(state: dict[str, Any], questions: dict[str, dict]) -> str:
    """The Chinese prompt the local judge is given, from `state` and `questions`. Mirrors
    `judge_shim.prompt_for`."""
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


class JudgeModel(Protocol):
    async def ask(self, state: dict[str, Any], questions: dict[str, dict]) -> dict[str, Any]:
        """The answer to each of `questions` about `state`, by the question's key, all in
        one request. Raises JudgeError if the model cannot be reached or its answer read."""
        ...

    async def aclose(self) -> None: ...


def judge_timeout() -> httpx.Timeout:
    """A client's timeout for the judge model: `JUDGE_TIMEOUT` seconds, with a connect
    timeout no larger than it (and no larger than the old 3 s). Local judges need this
    raised from the 5 s cloud default; see `deploy/local-ai/notes/05-代码插桩点.md`."""
    seconds = judge_timeout_from_env()
    return httpx.Timeout(seconds, connect=min(3.0, seconds))


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=judge_timeout())


def _reply(answers: Any, usage: Any, service: str, res: httpx.Response) -> JudgeReply:
    if not isinstance(answers, dict) or not all(isinstance(a, dict) for a in answers.values()):
        raise JudgeError(f"{service} 的返回无法解析（{len(res.text)} 字）")
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    return JudgeReply(answers, tokens if isinstance(tokens, int) else None)


class JevJudge:
    """Jev through TypeSafe's System One endpoint (`/systemone`, which OpenRouter, Vercel AI
    Gateway and TypeSafe itself all serve alike): Jev is an evaluation model, answering
    typed questions about a state with probabilities, and is not served by the Chat
    Completions endpoint."""

    service = "Jev"
    """The service, as its failures name it."""

    def __init__(
        self, api_key: str, model: str, base_url: str, client: httpx.AsyncClient | None = None
    ) -> None:
        self._url = base_url.rstrip("/") + "/systemone"
        self._api_key = api_key
        self._model = model
        self._client = client or _client()

    async def ask(self, state: dict[str, Any], questions: dict[str, dict]) -> dict[str, Any]:
        reply = await self.reply(state, questions)
        bill("tokens", reply.input_tokens)
        return reply.answers

    async def reply(self, state: dict[str, Any], questions: dict[str, dict]) -> JudgeReply:
        """As `ask`, with the input tokens the model read."""
        body = {"model": self._model, "state": state, "questions": questions}
        try:
            res = await self._client.post(
                self._url, json=body, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.HTTPError as e:
            raise JudgeError(f"无法连接 {self.service}：{e!r}") from e
        if res.status_code >= 400:
            raise JudgeRefused(
                f"{self.service} 返回 {res.status_code}: {res.text[:500]}", res.status_code, res.text
            )
        try:
            body = res.json()
            answers = body["answers"]
        except (ValueError, KeyError, TypeError) as e:
            raise JudgeError(f"{self.service} 的返回无法解析（{len(res.text)} 字）") from e
        return _reply(answers, body.get("usage"), self.service, res)

    async def aclose(self) -> None:
        await self._client.aclose()


class BailianJudge(JevJudge):
    """Model Studio's decision model at the workspace's endpoint (`BailianJudgeSettings`):
    asked and answering as Jev does, refusing in OpenAI's way (`{error: {code, message}}`)."""

    service = "阿里云百炼"

    async def check(self) -> None:
        """Tries the key, the workspace id and the model with the least of requests. Raises
        RuntimeError naming the setting that is wrong if Model Studio refuses one; only
        warns if it cannot be told, say on a timeout, an outage or being throttled."""
        try:
            await self.ask({"测试": "启动检查"}, {"check": _CHECK_QUESTION})
        except JudgeError as e:
            if isinstance(e, JudgeRefused):
                code, message = _bailian_error(e.body)
                # A workspace id malformed is invalid (400); one well formed but not the
                # key's, or not in the region, is denied (403).
                if "workspace" in message.lower():
                    raise RuntimeError(
                        "BAILIAN_JUDGE_WORKSPACE_ID 不对，或者 BAILIAN_JUDGE_API_KEY 不属于这个业务空间"
                        f"（两者都要在 BAILIAN_JUDGE_REGION 这个地域）：{e}"
                    ) from None
                if e.status in (401, 403):
                    raise RuntimeError(f"BAILIAN_JUDGE_API_KEY 无效：{e}") from None
                if code == "model_not_found":
                    raise RuntimeError(f"BAILIAN_JUDGE_MODEL 不对，百炼没有这个模型：{e}") from None
            log.warning("could not check the Bailian judge model; starting anyway: %s", e)


def _bailian_error(body: str) -> tuple[str, str]:
    """The code and message of Model Studio's refusal; empty if it said neither."""
    try:
        error = json.loads(body)["error"]
        return str(error.get("code") or ""), str(error.get("message") or "")
    except (ValueError, KeyError, TypeError, AttributeError):
        return "", ""


class CloudflareRefused(JudgeError):
    """Workers AI answered with an error: its HTTP status and Cloudflare's error codes."""

    def __init__(self, message: str, status: int, codes: set[int]) -> None:
        super().__init__(message)
        self.status = status
        self.codes = codes


class CloudflareJudge:
    """Cloudflare's Clef or Clef-flash on Workers AI, over its REST API: the questions are
    System One's, but the model is named by its bare name and the answers come wrapped in
    Cloudflare's envelope (`{result, success, errors, messages}`)."""

    def __init__(
        self, api_token: str, account_id: str, model: str, client: httpx.AsyncClient | None = None
    ) -> None:
        self._url = f"{CLOUDFLARE_API_URL}/accounts/{account_id}/ai/run/@cf/cloudflare/{model}"
        self._api_token = api_token
        self._model = model
        self._client = client or _client()

    async def ask(self, state: dict[str, Any], questions: dict[str, dict]) -> dict[str, Any]:
        reply = await self.reply(state, questions)
        bill("tokens", reply.input_tokens)
        return reply.answers

    async def reply(self, state: dict[str, Any], questions: dict[str, dict]) -> JudgeReply:
        """As `ask`, with the input tokens Workers AI counted (after any truncating)."""
        body = {"model": self._model, "state": state, "questions": questions}
        try:
            res = await self._client.post(
                self._url, json=body, headers={"Authorization": f"Bearer {self._api_token}"}
            )
        except httpx.HTTPError as e:
            raise JudgeError(f"无法连接 Cloudflare Workers AI：{e!r}") from e
        try:
            envelope = res.json()
        except ValueError:
            envelope = None
        if not isinstance(envelope, dict):
            if res.status_code >= 400:
                raise CloudflareRefused(
                    f"Cloudflare Workers AI 返回 {res.status_code}: {res.text[:500]}",
                    res.status_code, set(),
                )
            raise JudgeError(f"Cloudflare Workers AI 的返回无法解析（{len(res.text)} 字）")
        if res.status_code >= 400 or envelope.get("success") is False:
            raise _refused(res.status_code, envelope.get("errors"))
        result = envelope.get("result")
        if not isinstance(result, dict):
            result = {}
        return _reply(result.get("answers"), result.get("usage"), "Cloudflare Workers AI", res)

    async def check(self) -> None:
        """Tries the token and the account id with the least of requests. Raises
        RuntimeError naming the setting that is wrong if Cloudflare refuses either; only
        warns if it cannot be told, say on a timeout, an outage or the day's free
        allocation used up."""
        try:
            await self.ask({"测试": "启动检查"}, {"check": _CHECK_QUESTION})
        except JudgeError as e:
            if isinstance(e, CloudflareRefused):
                if e.status == 404 or e.codes & _ACCOUNT_ERROR_CODES:
                    raise RuntimeError(
                        f"CLOUDFLARE_ACCOUNT_ID 不对，Cloudflare 找不到这个账号：{e}"
                    ) from None
                if e.status in (401, 403) and await self._token_active():
                    raise RuntimeError(
                        "CLOUDFLARE_ACCOUNT_ID 不对，或者 CLOUDFLARE_API_TOKEN 没有这个账号的 Workers AI"
                        f" 权限（token 本身有效）：{e}"
                    ) from None
                if e.status in (401, 403):
                    raise RuntimeError(
                        "CLOUDFLARE_API_TOKEN 无效或缺少 Workers AI 权限（需要 Workers AI 的 Read 和"
                        f" Edit），或者它不属于 CLOUDFLARE_ACCOUNT_ID 这个账号：{e}"
                    ) from None
            log.warning("could not check the Cloudflare judge model; starting anyway: %s", e)

    async def _token_active(self) -> bool:
        """Whether Cloudflare says the token itself is good: Workers AI refuses a wrong
        account id just as it does a wrong token ("Authentication error"). False if it
        cannot be told, as for a token the account rather than a user owns."""
        try:
            res = await self._client.get(
                f"{CLOUDFLARE_API_URL}/user/tokens/verify",
                headers={"Authorization": f"Bearer {self._api_token}"},
            )
            return res.json()["result"]["status"] == "active"
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


def _refused(status: int, errors: Any) -> CloudflareRefused:
    """What Workers AI's `errors` say went wrong, in so many words when it is the day's
    free allocation used up."""
    errors = errors if isinstance(errors, list) else []
    codes = {e["code"] for e in errors if isinstance(e, dict) and isinstance(e.get("code"), int)}
    said = str(errors)[:500]
    if codes & _QUOTA_ERROR_CODES or "neurons" in said.lower():
        return CloudflareRefused(
            "Workers AI 每日的免费额度（10,000 neurons）已用完，今天的判断都会失败；"
            f"公开部署请开 Workers Paid：{status} {said}",
            status, codes,
        )
    return CloudflareRefused(f"Cloudflare Workers AI 返回 {status}: {said}", status, codes)


class LocalJudge:
    """A self-hosted OpenAI-compatible Chat Completions server as the judge model
    (`deploy/local-ai/`). Jev speaks TypeSafe's System One protocol; a local llama-server
    has only `/v1/chat/completions`, so the typed System One questions are translated
    here into one strict-JSON-Schema chat request. This is the logic the validated
    `judge_shim.py` used, now a first-class provider: no extra process to run."""

    service = "本地判断模型"

    def __init__(
        self, model: str, base_url: str, api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key or "local"  # llama-server does not check it, but a header is sent
        self._model = model
        self._client = client or _client()

    async def ask(self, state: dict[str, Any], questions: dict[str, dict]) -> dict[str, Any]:
        reply = await self.reply(state, questions)
        bill("tokens", reply.input_tokens)
        return reply.answers

    async def reply(self, state: dict[str, Any], questions: dict[str, dict]) -> JudgeReply:
        """As `ask`, with the prompt tokens the local server counted."""
        try:
            schema = _local_schema(questions)
            prompt = _local_prompt(state, questions)
        except ValueError as e:
            raise JudgeError(f"{self.service}的提问无法转换：{e}") from e
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _LOCAL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "judge", "schema": schema, "strict": True},
            },
            "temperature": 0,
            "max_tokens": _LOCAL_MAX_TOKENS,
        }
        try:
            res = await self._client.post(
                self._url, json=body, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.HTTPError as e:
            raise JudgeError(f"无法连接 {self.service}：{e!r}") from e
        if res.status_code >= 400:
            raise JudgeRefused(
                f"{self.service} 返回 {res.status_code}: {res.text[:500]}", res.status_code, res.text
            )
        try:
            payload = res.json()
            content = payload["choices"][0]["message"]["content"]
            answers = json.loads(content)
            usage = payload.get("usage") or {}
            tokens = usage.get("prompt_tokens")
        except (ValueError, KeyError, TypeError, IndexError) as e:
            raise JudgeError(f"{self.service} 的返回无法解析（{len(res.text)} 字）") from e
        return _reply(answers, {"input_tokens": tokens}, self.service, res)

    async def aclose(self) -> None:
        await self._client.aclose()


def open_judge(
    settings: JudgeSettings, client: httpx.AsyncClient | None = None
) -> JevJudge | CloudflareJudge | LocalJudge:
    """The judge model `settings` choose."""
    if isinstance(settings, CloudflareJudgeSettings):
        return CloudflareJudge(settings.api_token, settings.account_id, settings.model, client)
    if isinstance(settings, BailianJudgeSettings):
        return BailianJudge(settings.api_key, settings.model, settings.base_url, client)
    if isinstance(settings, LocalJudgeSettings):
        return LocalJudge(settings.model, settings.base_url, settings.api_key, client)
    return JevJudge(settings.api_key, settings.model, settings.base_url, client)


async def try_judge_model(model: JudgeModel) -> None:
    """Try the judge model before the admin console keeps it: Cloudflare's and Model
    Studio's credentials as the server starts tries them (their `check`), any other by
    asking it the least of questions. Raises JudgeError saying why it does not work."""
    if isinstance(model, CloudflareJudge | BailianJudge):
        try:
            await model.check()
        except RuntimeError as e:
            raise JudgeError(str(e)) from None
        return
    await model.ask({"测试": "保存前试调"}, {"check": _CHECK_QUESTION})


async def check_judge(
    settings: JudgeSettings | None, client: httpx.AsyncClient | None = None
) -> None:
    """Tries Cloudflare's or Model Studio's credentials as the server starts (see their
    `check`); Jev and the local judge are not tried."""
    if not isinstance(settings, CloudflareJudgeSettings | BailianJudgeSettings):
        return
    judge = open_judge(settings, client)
    assert isinstance(judge, CloudflareJudge | BailianJudge)
    try:
        await judge.check()
    finally:
        await judge.aclose()
