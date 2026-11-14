"""`eidolon doctor`: tries each of the deployer's services `.env` configures, the way the
server reaches it, and says what is wrong in words a newcomer, or the agent deploying for
them, can act on (docs/agents/deploying.md「排障」).

Each service is tried as `.env` routes it, through the proxy or direct. Failing to reach
it, or its provider refusing the server's region, it is tried the other way, if there is
a proxy: what works then is what `EIDOLON_PROXY_SERVICES` should say, and the doctor says
so. Every failure is put down to one `Problem`, from the errors under it and what the
provider said; the provider's own words are shown too, the classing being a guess.

Only `.env` is tried: what the admin console overrides is not (see docs/deploy.md「后台和
`.env` 的关系」). Image generation is not tried, every picture being paid for; the admin
console tries it as it is saved. Nothing is recorded, and the database is only read, for
how much of the story corpus there is; the server need not be running, nor stopped.
"""

import asyncio
import json
import re
import sqlite3
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import httpx
from websockets.exceptions import WebSocketException

from eidolon.config import (
    PROXIED_SERVICES,
    ProxySettings,
    Registration,
    ServiceName,
    Settings,
)
from eidolon.database import read_only
from eidolon.proxy import ProxyRoutes, masked, open_client

Problem = Literal["network", "region", "key", "balance", "model", "other"]
"""Why a service failed: it could not be reached; its provider does not serve the server's
region; the key is wrong or may not do this; the account has no balance or quota left;
the model is wrong, or does not do what the server needs; anything else."""

Route = Literal["direct", "proxy"]

Check = Callable[[str | None], Awaitable[None]]
"""Tries a service through the proxy given, or direct for None; raises if it fails."""

CHECK_TIMEOUT = 90.0
"""How long, in seconds, trying a service one way may take: the chat model's check asks
it as many as five times."""
PROXY_TIMEOUT = 3.0
"""How long, in seconds, connecting to the proxy itself may take."""

LABELS: dict[ServiceName, str] = {
    "chat_llm": "对话模型",
    "memory_llm": "后台模型",
    "judge": "判断模型",
    "embedding": "向量化",
    "rerank": "重排",
    "tts": "语音合成（Fish Audio）",
    "stt": "语音识别",
    "image": "生图",
}

PROBLEM_NAMES: dict[Problem, str] = {
    "network": "连不上",
    "region": "地区不支持",
    "key": "key 不对",
    "balance": "余额不足",
    "model": "模型不对",
    "other": "其他错误",
}

GUIDE = "docs/agents/deploying.md"

HINTS: dict[Problem, str] = {
    "network": "连不上这家服务。服务器所在的地区可能要经代理才能访问它：配好 EIDOLON_PROXY_URL，"
    f"再把这一项加进 EIDOLON_PROXY_SERVICES（{GUIDE}「网络和代理」）",
    "region": "这家服务不对服务器所在的地区开放：让这一项经代理访问，或者换一家服务"
    f"（{GUIDE}「网络和代理」）",
    "key": "key 不对，或者没有这项权限：到服务商的控制台重新复制一遍（别带空格和引号），"
    "确认填在了对应的变量上",
    "balance": "余额或额度用完了：到服务商的控制台充值，或者换一个免费的模型",
    "model": "模型名不对，或者这个模型做不到后端要的事（比如工具调用）："
    "对照服务商的模型列表核对，或者换一个模型",
    "other": "看上面服务返回的原话；拿不准时把这一段整个交给帮你部署的 Agent",
}


@dataclass(frozen=True)
class Finding:
    """How trying one service went."""

    service: ServiceName
    label: str
    ok: bool
    route: Route
    """The way it worked, or the way `.env` routes it, if it failed both."""
    problem: Problem | None = None
    detail: str = ""
    """What failed, in the provider's or the client's words."""
    hint: str = ""
    move: Route | None = None
    """The way it should go instead of `.env`'s, having worked that way only."""


async def diagnose(
    checks: Mapping[ServiceName, Check], proxy: ProxySettings, *, proxy_reachable: bool = True
) -> list[Finding]:
    """Try each of `checks` as `proxy` routes it, and, failing to reach it or refused for
    the region, the other way, if the proxy is set and could be reached; all at once."""
    other_way = proxy.url is not None and proxy.enabled and proxy_reachable

    async def one(service: ServiceName, check: Check) -> Finding:
        label = LABELS[service]
        through = proxy.proxy_for(service)
        route: Route = "direct" if through is None else "proxy"
        failure = await _try(check, through)
        if failure is None:
            return Finding(service, label, True, route)
        problem = classify(failure)
        detail = masked(_said(failure), proxy.url)
        if other_way and problem in ("network", "region"):
            other: Route = "proxy" if route == "direct" else "direct"
            if await _try(check, proxy.url if other == "proxy" else None) is None:
                return Finding(service, label, True, other, problem, detail, _moved(other), other)
        return Finding(service, label, False, route, problem, detail, HINTS[problem])

    return list(await asyncio.gather(*(one(s, c) for s, c in checks.items())))


def suggested_proxy_services(proxy: ProxySettings, findings: list[Finding]) -> str | None:
    """What `EIDOLON_PROXY_SERVICES` should say for the services that worked only the
    other way to go that way, the rest of it as it is; None if it is right as it is."""
    services = set(proxy.services)
    for f in findings:
        if f.move == "proxy":
            services.add(f.service)
        elif f.move == "direct":
            services.discard(f.service)
    if services == set(proxy.services):
        return None
    return ",".join(s for s in PROXIED_SERVICES if s in services)


async def _try(check: Check, proxy: str | None) -> BaseException | None:
    try:
        async with asyncio.timeout(CHECK_TIMEOUT):
            await check(proxy)
    except Exception as e:
        return e
    return None


def _moved(route: Route) -> str:
    if route == "proxy":
        return "直连不行，经代理能用：把这一项加进 EIDOLON_PROXY_SERVICES（见最后的建议）"
    return "经代理不行，直连能用：把这一项从 EIDOLON_PROXY_SERVICES 里去掉（见最后的建议）"


_REGION = re.compile(
    r"location is not supported|unsupported_country|country, region, or territory"
    r"|not available in your (country|region)|request not allowed|地区不支持|不支持.*地区",
    re.IGNORECASE,
)
_BALANCE = re.compile(
    r"(?<!\d)402(?!\d)|insufficient|quota|balance|credit|arrearage|余额|额度|欠费",
    re.IGNORECASE,
)
_KEY = re.compile(
    r"(?<!\d)40[13](?!\d)|unauthori[sz]ed|forbidden|invalid.{0,10}(api.?key|token)"
    r"|incorrect api key|authentication|permission|拒绝了这个 key|key 不对",
    re.IGNORECASE,
)
_MODEL = re.compile(
    r"(?<!\d)404(?!\d)|model.{0,20}(not.?found|not exist|does not exist|not supported)"
    r"|no such model|unknown model|工具调用|需要支持流式输出|缓存标记|思考强度|模型.{0,10}(不存在|没有开通|未开通)",
    re.IGNORECASE,
)


def classify(failure: BaseException) -> Problem:
    """What `failure` most likely comes down to: by the errors under it, which the
    services' own errors are raised over, and failing those, by what was said."""
    chain = _chain(failure)
    said = " ".join(str(e) for e in chain)
    if _REGION.search(said):
        return "region"
    if any(isinstance(e, _UNREACHABLE) for e in chain) or "连不上" in said:
        return "network"
    if _BALANCE.search(said):
        return "balance"
    if _KEY.search(said):
        return "key"
    if _MODEL.search(said):
        return "model"
    return "other"


_UNREACHABLE = (httpx.TransportError, OSError, TimeoutError, WebSocketException)
"""What not reaching a provider is raised as, under whatever the service raises over it:
httpx's connecting, timing out, the proxy failing; a websocket's; DNS and sockets'."""


def _chain(failure: BaseException) -> list[BaseException]:
    """`failure` and every error it was raised from or over, though raised `from None`."""
    found: list[BaseException] = []
    todo: list[BaseException | None] = [failure]
    while todo:
        e = todo.pop()
        if e is None or any(e is f for f in found):
            continue
        found.append(e)
        todo += [e.__cause__, e.__context__]
        if isinstance(e, BaseExceptionGroup):
            todo += list(e.exceptions)
    return found


def _said(failure: BaseException) -> str:
    text = str(failure) or repr(failure)
    return text if len(text) <= 600 else text[:600] + "…"


async def proxy_failure(url: str) -> str | None:
    """Why the proxy at `url` cannot be connected to, or None if it can."""
    address = httpx.URL(url)
    port = address.port or (443 if address.scheme == "https" else 80)
    try:
        async with asyncio.timeout(PROXY_TIMEOUT):
            _, writer = await asyncio.open_connection(address.host, port)
    except (OSError, TimeoutError) as e:
        why = f"{PROXY_TIMEOUT:g} 秒内没有回应" if isinstance(e, TimeoutError) else repr(e)
        hint = f"连不上代理 {address.host}:{port}（{why}）。"
        if address.host in ("127.0.0.1", "localhost") and Path("/.dockerenv").exists():
            hint += (
                "后端在容器里，用 docker-compose.yml（本机体验）时容器里的 127.0.0.1 是容器自己："
                f"改成 http://host.docker.internal:{port}。用 docker-compose.prod.yml 时容器和服务器"
                "共用网络，127.0.0.1 没问题，要看代理软件是不是在运行、端口对不对。"
            )
        else:
            hint += "看代理软件是不是在运行、端口对不对。"
        return hint + (
            "代理装在另一台机器（或 Linux 的宿主机）上时，要在代理软件里打开「允许局域网连接」"
            f"（{GUIDE}「网络和代理」）"
        )
    writer.close()
    return None


def story_episodes(database: Path) -> int | None:
    """How many episodes the story corpus has, reading the database; None without one."""
    if not database.exists():
        return None
    try:
        db = read_only(database)
        try:
            return db.execute("SELECT count(*) FROM episodes").fetchone()[0]
        finally:
            db.close()
    except sqlite3.OperationalError:  # no such table: the server never started on it
        return 0


def checks_for(settings: Settings) -> dict[ServiceName, Check]:
    """The check of each service `settings` configure, but image generation's."""
    from eidolon.call import SpeechServices
    from eidolon.embedding import EMBEDDING_TIMEOUT, open_embedder
    from eidolon.judge import judge_timeout, open_judge, try_judge_model
    from eidolon.llm.providers import LLM_TIMEOUT, check_chat_models, open_chat_models, open_llm
    from eidolon.rerank import RERANK_TIMEOUT, SiliconFlowReranker
    from eidolon.service_status import embed_a_word, first_word, rerank_a_word

    checks: dict[ServiceName, Check] = {}

    async def chat_llm(proxy: str | None) -> None:
        models = open_chat_models(settings.chat_model, open_client(proxy, LLM_TIMEOUT))
        try:
            await check_chat_models(models, settings.chat_model)
        finally:
            await models.aclose()

    checks["chat_llm"] = chat_llm

    if settings.background_llm != settings.chat_llm:

        async def memory_llm(proxy: str | None) -> None:
            async with open_client(proxy, LLM_TIMEOUT) as client:
                await first_word(open_llm(settings.background_llm, client))

        checks["memory_llm"] = memory_llm

    if (judge_settings := settings.judge) is not None:

        async def judge(proxy: str | None) -> None:
            async with open_client(proxy, judge_timeout()) as client:
                await try_judge_model(open_judge(judge_settings, client))

        checks["judge"] = judge

    if (embedding := settings.embedding) is not None:

        async def embed(proxy: str | None) -> None:
            async with open_client(proxy, EMBEDDING_TIMEOUT) as client:
                await embed_a_word(open_embedder(embedding, client))

        checks["embedding"] = embed

    if (rerank_settings := settings.rerank) is not None:

        async def rerank(proxy: str | None) -> None:
            async with open_client(proxy, RERANK_TIMEOUT) as client:
                reranker = SiliconFlowReranker(
                    rerank_settings.api_key, rerank_settings.model, rerank_settings.base_url,
                    client,
                )
                await rerank_a_word(reranker)

        checks["rerank"] = rerank

    @asynccontextmanager
    async def speech(proxy: str | None) -> AsyncIterator[SpeechServices]:
        """The speech services, both through `proxy`, or direct for None."""
        routes = ProxyRoutes(ProxySettings(proxy, proxy is not None, frozenset({"tts", "stt"})))
        try:
            yield SpeechServices(settings.bailian_stt_model, routes=routes)
        finally:
            await routes.aclose()

    if (fish_key := settings.fish_audio_api_key) is not None:

        async def tts(proxy: str | None) -> None:
            async with speech(proxy) as services:
                await services.check_tts(fish_key, settings.fish_audio_model, own=False)

        checks["tts"] = tts

    if (stt_key := settings.stt_key) is not None:

        async def stt(proxy: str | None) -> None:
            async with speech(proxy) as services:
                await services.check_stt(stt_key.provider, stt_key.api_key, own=False)

        checks["stt"] = stt

    return checks


@dataclass(frozen=True)
class Report:
    """What the doctor found of the deployment `.env` configures."""

    findings: list[Finding]
    proxy_error: str | None
    """Why the proxy could not be connected to; None if it could, or there is none."""
    suggested_proxy_services: str | None
    story_episodes: int | None
    stand_assets_url: str | None
    image_configured: bool
    public: bool
    admins: bool
    """Whether there is an admin console (`Settings.admin_console`)."""
    everyone_admin: bool
    registration: Registration

    @property
    def invite_without_admins(self) -> bool:
        """Only those invited may sign up, but there is no admin console to make an invite
        link in: no one but the first user can. Only warned of, for the first user signs in
        before their identity can be put on the admin list."""
        return self.registration == "invite" and not self.admins

    @property
    def ok(self) -> bool:
        """Everything tried works, the way `.env` routes it."""
        return self.proxy_error is None and all(f.ok and f.move is None for f in self.findings)

    def as_json(self) -> dict:
        found = asdict(self)
        found["services"] = found.pop("findings")
        del found["admins"], found["everyone_admin"]
        found["invite_without_admins"] = self.invite_without_admins
        return {"ok": self.ok, "config_error": None, **found}

    def lines(self) -> list[str]:
        """The report as the newcomer reads it."""
        said = ["Eidolon-SC 诊断：按 .env 逐项试一次各项服务"]
        if self.everyone_admin:
            said.append("  （开发登录：所有人都能进后台。后台配置页改过的项以后台为准，这里只查 .env）")
        elif self.admins:
            said.append("  （设了管理者：后台配置页改过的项以后台为准，这里只查 .env）")
        said.append("")
        if self.proxy_error is not None:
            said += [f"✗ 代理：{self.proxy_error}", ""]
        for f in self.findings:
            if f.ok and f.move is None:
                said.append(f"✓ {f.label}：{'直连' if f.route == 'direct' else '经代理'}正常")
                continue
            assert f.problem is not None
            said += [
                f"{'!' if f.ok else '✗'} {f.label}（{f.service}）："
                f"{PROBLEM_NAMES[f.problem]}（{f.problem}）",
                f"  原话：{f.detail}",
                f"  → {f.hint}",
            ]
        if self.image_configured:
            said.append("- 生图：不试（每张都要花钱）。后台配置页「生图」保存时会真生成一张来检查")
        said.append("")
        if self.suggested_proxy_services is not None:
            said += [
                f"建议把 .env 里的这一行改成：EIDOLON_PROXY_SERVICES={self.suggested_proxy_services}",
                "  改完重启后端才生效（用 Docker 的要 docker compose up -d，只 restart 读不到新的 .env）",
                "",
            ]
        if self.story_episodes is None:
            said.append("剧情库：还没有数据库（后端还没启动过）")
        elif self.story_episodes == 0:
            said.append(
                "剧情库：空的。剧情不随仓库分发，要自己同步一次，"
                f"在这之前角色只凭人设回复（{GUIDE}「剧情库」）"
            )
        else:
            said.append(f"剧情库：{self.story_episodes} 段剧情")
        if self.stand_assets_url is None:
            said.append(f"立绘：没配 EIDOLON_STAND_ASSETS_URL，通话时只显示背景图（{GUIDE}「立绘」）")
        else:
            said.append(f"立绘：从 {self.stand_assets_url} 加载")
        if self.invite_without_admins:
            said += [
                "! 注册：设了只让受邀的人注册（EIDOLON_REGISTRATION=invite），却没设管理者名单"
                "（EIDOLON_ADMINS）",
                "  → 没有后台就生成不了邀请链接，除了第一位用户谁都注册不了。"
                f"先登录一次，再配好管理者名单（{GUIDE}「公网部署」）",
            ]
        elif self.registration == "invite":
            said.append("注册：只让受邀的人注册，邀请链接在后台配置页「注册」里生成")
        said += [
            "",
            "全部正常" if self.ok else f"有问题，按上面的提示改；改完再运行一次。说明在 {GUIDE}「排障」",
        ]
        return said


async def examine(settings: Settings) -> Report:
    """Try every service `settings` configure, and the proxy, if any."""
    proxy = settings.proxy
    proxy_error = None
    if proxy.url is not None and proxy.enabled:
        proxy_error = await proxy_failure(proxy.url)
    findings = await diagnose(checks_for(settings), proxy, proxy_reachable=proxy_error is None)
    return Report(
        findings=findings,
        proxy_error=proxy_error,
        suggested_proxy_services=suggested_proxy_services(proxy, findings),
        story_episodes=story_episodes(settings.database_path),
        stand_assets_url=settings.stand_assets_url,
        image_configured=settings.image is not None,
        public=settings.domain is not None,
        admins=settings.admin_console,
        everyone_admin=settings.everyone_admin,
        registration=settings.registration,
    )


async def run(as_json: bool) -> int:
    """Diagnose the deployment `.env` configures, printing what was found, as JSON if
    `as_json`; 0 if everything tried works, the way `.env` routes it, else 1."""
    from loguru import logger

    logger.remove()  # Pipecat's, which says hello as the speech services are imported
    logger.add(sys.stderr, level="WARNING")
    try:
        settings = Settings.from_env()
    except RuntimeError as e:
        if as_json:
            print(json.dumps({"ok": False, "config_error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"✗ .env 有错，后端不会启动：{e}")
            print("  按上面说的改 .env，改完再运行一次 eidolon doctor")
        return 1
    report = await examine(settings)
    if as_json:
        print(json.dumps(report.as_json(), ensure_ascii=False, indent=2))
    else:
        print("\n".join(report.lines()))
    return 0 if report.ok else 1
