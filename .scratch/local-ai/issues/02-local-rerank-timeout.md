# 02: 本地重排的回合超时可配，并实测本地化效果

**What to build:** 把重排的回合超时从写死的 3 秒改成可配的 `RERANK_TIMEOUT`，让本机
llama-server 能处理真实长度的剧情候选；顺便用 `eidolon eval` 实测本地化方案的效果，把结论
写进 `deploy/local-ai/README.md`。

**Blocked by:** None (can start immediately)

**Status:** done

- [x] 查明 `eidolon eval rerank` 的两个失败原因：
      - ① `input (631 tokens) is too large to process. increase the physical batch size
        (current batch size: 512)` —— `models.ini` 三段都缺 `ubatch-size`，默认 512；
        三段各补 `ubatch-size = 2048`。
      - ② `重排服务 3 秒内没有回应` —— `retry.TURN_TIMEOUT = 3.0` 是给云端重排的预算；
        本机 RX 6750 XT / Vulkan 跑 36 条 × 173 字（真实剧情段落长度）要 4.2–4.8 s，
        36 条 × 50 字只要 2.5 s，60 条约 3.9 s，耗时随总 token 数增长，必然超时。
- [x] `config.py` 加 `DEFAULT_RERANK_TIMEOUT = 3.0` 与 `rerank_timeout_from_env()`
      （复用 `_seconds`，坏值启动期报错）；`rerank.py` 加 `rerank_timeout()`，
      `SiliconFlowReranker.__init__` 的 `timeout` 改成 `float | None = None`，不给就按环境读。
      4 个生产构造点（assembly / doctor / 两处 eval）不传 timeout，自动获得新默认值。
- [x] `.env.example`、三语 `docs/configuration*.md`、`deploy/local-ai/env.local-ai.example`、
      `deploy/local-ai/README.md` 补 `RERANK_TIMEOUT`；本地建议 10 秒，写清本机实测数字。
- [x] `backend/tests/test_retrieval_services.py` 加 5 个测试：默认 3.0、可配 10.0、坏值
      （0 / -1 / abc）启动期报错、空白回落默认、以及真的按配置超时（0.05 s 卡 0.3 s 的
      handler）。
- [x] `F:\eidolon-sc\.env` 补 `RERANK_TIMEOUT=10`（该文件被 `.gitignore` 排除，不入库）。
- [x] `sh scripts/affected-tests.sh 8e4e3829` → EXIT=0：ruff、pyright、后端 336 passed、
      前端 866 passed 全绿；`backend/uv.lock` 的镜像重写副产物已回滚。
- [x] 实跑验证：`uv run --env-file=../.env eidolon eval rerank`（60 cases）→ **EXIT=0**，
      完整打印分数分布，不再出现 3 秒超时。
- [x] 实跑 `eidolon eval judge --provider local --only retrieval`（47 例）并留档效果结论。

## Comments

### 实现说明（票 02）

- **接口**：`rerank_timeout_from_env() -> float` 与 `judge_timeout_from_env()` 同构，读
  `RERANK_TIMEOUT`，空串/空白落 `DEFAULT_RERANK_TIMEOUT`，其余经
  `config.py` 的 `_seconds(name, value)` 校验（0 / 负数 / 非数字都报
  `环境变量 RERANK_TIMEOUT 需要是大于 0 的秒数`）。`rerank_timeout()` 是 `rerank.py` 里的
  一层薄封装，不直接 import `retry.TURN_TIMEOUT`，两个常量各自独立、注释里写明对齐关系。
- **判断**：不改 `retry.TURN_TIMEOUT` 本身（它是全局回合预算，改了会放宽所有服务），只让
  重排这一类能单独放宽；`RERANK_TIMEOUT = httpx.Timeout(30, connect=10)` 是 httpx 层、和
  `answered("重排服务", self._timeout, RerankError)` 用的 `self._timeout` 是两回事，后者才是
  这次修的对象。
- **测试写法**：行为测试里 `slow` 是 async handler，不能传给只接受同步 `Handler` 的
  `client()` 辅助（pyright 会报 `CoroutineType... cannot be assigned to Handler`），改成直接
  `httpx.AsyncClient(transport=httpx.MockTransport(slow))`，仿 `test_embedding.py` 的
  `scripted_embedder`。

### 本地化效果实测（2026-11-10，AMD RX 6750 XT / Vulkan，三个 Qwen3 0.6B/4B GGUF）

- **重排**：`eidolon eval rerank` 60 cases 在 `RERANK_TIMEOUT=10` 下完整跑完；相关/无关分数
  区分明显（如 `0.9998 人物关系 / 制作人` 对 `0.0000 演出…`），校准阈值仍用现成的 0.3/0.01。
- **判断模型（检索判断，47 例）**：整体准确率 **65.96%**（短例 62.8%，4 个长例 100%）。
  逐类看：`recall`（剧情回顾）几乎全对、`references`（查资料）约一半、`memories`（要不要
  检索记忆）几乎全错 —— 本地 `local-judge`（qwen3-4b-instruct-2507-q4_k_m）沿用 jev 族阈值
  `0.5/0.5/0.5/0.5/0.7/0.4/0.5` 时，`about_producer` 概率普遍偏低，本该判「查记忆」的话
  被判成「不检索」。单请求 662–698 token 约 1.5–3.7 s，长例 2831–3154 token 10.4–12.8 s，
  都在 `JUDGE_TIMEOUT=30` 内。
- **结论与边界（如实）**：文字链路能跑通、检索/重排质量可用，但**判断模型沿用云端阈值时效果
  明显不足**，尤其影响「是否检索记忆」。这不是超时或配置问题，是 4B 本地模型的能力上限；
  要真正可用，需给本地模型单独标定阈值（`eidolon eval judge --json` 已支持逐条概率），或换
  更大的本地模型。`eval memory` 与 judge 的其余四类产品（emotion / affinity / photo /
  considerate）尚未跑，留待后续。
- **语音**：TTS/STT 仍上云，见 `deploy/local-ai/README.md`「边界」，本轮不动。

### 票 02 交接

- 若继续提升效果：先跑全量 `eidolon eval judge --provider local`（399 例）拿逐类准确率，
  再用 `--json` 的概率分布为 `local-judge` 单独定一族阈值，而不是继续套 jev 阈值。
- `deploy/local-ai/start-llama-server.ps1` 的预热段（依次温起 embedding / rerank / judge）
  已加并校验语法，但还没实跑计时；下次起服务时留意打印的三个加载耗时。
- 剧情库 `corpus/` 仍为空（doctor 提示），上述重排/判断评测用的是仓库自带 eval 集，不是
  真实剧情库上的端到端检索。
