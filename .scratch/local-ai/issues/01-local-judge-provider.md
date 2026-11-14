# 01: 原生本地判断模型 + 资产进仓库

**What to build:** 把 `F:\local-ai` 的本地化调研资产搬进 `deploy/local-ai/`，并让判断模型
本地化成为一等公民：`JUDGE_PROVIDER=local` 走原生 `LocalJudge`（直接打 OpenAI 兼容
`/chat/completions`），`JUDGE_TIMEOUT` 可配，`judge_shim.py` 只作零改码方案 A 回退。
文字聊天链路（embedding / rerank / judge）可做到只有 DeepSeek 上云；语音链路仍上云。

**Blocked by:** None (can start immediately)

**Status:** done

- [x] 开 `local-ai` 分支（HEAD 从 `8e4e3829`）；修好 `.git/refs/heads` 的 ACL 后开出
- [x] 资产搬到 `deploy/local-ai/`：README、env.local-ai.example、judge_shim.py、models.ini、
      start-llama-server.ps1（BOM 保留）、notes/；bin/ 49 MB 与 gguf/ 3.6 GB
      复制到同目录但由 `.gitignore` 排除。早期 logs/ 下的 .log 是本机产物，
      后来随 `start-app.ps1` 的运行时日志一起改为整个 `logs/` 不入库
- [x] `config.py`：`JudgeProvider` 加 `local`；新增 `LocalJudgeSettings`、`DEFAULT_LOCAL_JUDGE_*`、
      `judge_timeout_from_env()`（复用 `_seconds`，坏值启动期报错）；`Settings.from_env()`、
      `judge_from_env()` 接线；`JudgeSettings` 联合类型加 `LocalJudgeSettings`
- [x] `judge.py`：删掉硬编码 `JUDGE_TIMEOUT = 5`，改 `judge_timeout()`（connect 不超过 3s）；
      新增 `_LOCAL_SYSTEM` / `_local_schema` / `_local_prompt`（照搬 shim 的 schema_for/prompt_for）、
      `LocalJudge`（`service = "本地判断模型"`，`base_url.rstrip("/") + "/chat/completions"`，
      `response_format` strict JSON Schema、`temperature: 0`、`max_tokens: 700`，读
      `choices[0].message.content` 作 json.loads、`usage.prompt_tokens` 计费）；
      `open_judge()` 接线、`check_judge()` docstring 注明本地不试
- [x] 3 个外部调用点改 `judge_timeout()`：`service_clients.py:71`、`doctor.py:299`、
      `__main__.py:257`；第 4 处在 `judge.py:305` 的 `_client()` 里（原来的 `JUDGE_TIMEOUT`
      常量与 client 合并成 `judge_timeout()` 工厂，那两处引用一起消失了）
      （`grep -rn JUDGE_TIMEOUT backend/src backend/tests` 无残留符号引用）
- [x] `live_config.py`：`_Judge.fields` 加 `base_url`、`_per_service` 加 `base_url`；`env()` 对
      `LocalJudgeSettings` 返回 `provider=local` + model/base_url/api_key；`candidate()` 对 local
      免 key、填默认 `base_url`、非 local 清空 `base_url`；`_judge_settings()` 构造 `LocalJudgeSettings`
- [x] `admin_config_api.py`：`JudgeInput` 加 `base_url`（`checked_base_url` 校验），`_change()` 的
      None 删除列表加 `base_url`
- [x] 前端：`adminConfig.ts` judge spec 的 rows/options/form 加 `base_url` 与 `local`；
      `i18n/zh.ts` 加 `local: "本地模型"` 与三条 formHint；`components/AdminConfig.tsx` 的
      `fieldHint()` 加 `judge_base_url` 分支
- [x] 文档：`.env.example` 加 4 个本地变量与 `JUDGE_TIMEOUT`；`docs/configuration.md` /
      `.en.md` / `.ja.md` 判断模型表格加第 5 列「本地模型」+ 顶部服务一览行 + 超时说明；
      `GLOSSARY.md` 判断模型/自带密钥条目补「本机也算一家」
- [x] `deploy/local-ai/README.md` / `env.local-ai.example` / `start-llama-server.ps1` 重写对齐
      方案 B（默认 `JUDGE_PROVIDER=local`、一个端口 8094、不需要 shim、路径改成仓库内）；
      notes/04、05、07 加「历史记录、命令行已过时」声明；写明 TTS/STT 边界
- [x] 实测 `models.ini` 相对路径按 **CWD** 解析（从仓库根起报 `model name=... failed to load`，
      从 `deploy/local-ai/` 起 `dims=1024`）；`start-llama-server.ps1` 加 `-WorkingDirectory $root`
      固定子进程 CWD，参数去掉 `-Shim`/`ShimPort`/`Python`，`-Stop -Port $Port`
- [x] 端到端实测：脚本起在 18096 后 `/v1/embeddings` 1024 维、`/v1/rerank` 0.9932 vs 0.00016、
      `/v1/chat/completions`（`model=local-judge`）返回合法 JSON；测试进程已停
- [x] 测试：`backend/tests/test_local_judge.py`（请求形状与三种问题类型的 schema、
      计费、错误路径、配置解析、`judge_timeout_from_env` 的默认/可配/坏值）、
      `frontend/src/lib/adminConfig.test.ts` 的 `local` 用例
- [x] 自检：ruff、pyright、`sh scripts/affected-tests.sh 8e4e3829`（285 后端 + 866 前端全绿）、
      `frontend/scripts/check-discipline.sh`

## Comments

### 实现说明（票 01）

- **接口**：`LocalJudgeSettings(model, base_url, api_key)` 是不可变 dataclass，`from_env()` 读
  `LOCAL_JUDGE_MODEL` / `LOCAL_JUDGE_BASE_URL`（rstrip `/`）/ `LOCAL_JUDGE_API_KEY`（空→None）；
  默认 `http://127.0.0.1:8094/v1`、`local-judge`。`judge_timeout_from_env()` 返回 float，
  `judge_timeout()` 包成 `httpx.Timeout(seconds, connect=min(3.0, seconds))`。
- **模块**：`judge.py` 的 `LocalJudge` 实现 `JudgeModel` Protocol（`ask` / `aclose`），
  `open_judge()` 按 settings 类型分派；`try_judge_model()` 对非 Cloudflare/Bailian 走 `model.ask()`，
  所以后台保存时会试调本地模型；`check_judge()` 启动期不试本地模型（和 Jev 一致）。
- **判定形状**：`_local_schema` 按问题 `type` 生成 strict JSON Schema —— `noul` 要
  `{"noul": number}`，`score`/`choice` 要 `{"probabilities": {...}}`，与 `judge_shim.py`
  的 `schema_for` 及 consumers（retrieval.py / affinity_judge.py / emotion.py）读的形状一致。
- **判断调用**：`LocalJudge` 不用 `JevJudge` 的 `/systemone`，直接 `/chat/completions` +
  `response_format={"type":"json_schema",...}`、`temperature: 0`、`max_tokens: 700`；
  答案从 `choices[0].message.content` 里 `json.loads`，`usage.prompt_tokens` 折成 `input_tokens`
  走 `bill("tokens", ...)`。
- **judgement calls**：① 选方案 B 而不是只靠 shim；② 本地免 key 但仍发占位 Bearer；
  ③ 相对路径不回退成绝对路径，改用脚本固定 CWD；④ TTS/STT 明确不做；⑤ `uv.lock`
  的镜像重写与任务无关，已 `git checkout --` 回滚。
- **闭票时的实测**：`start-llama-server.ps1 -Port 18096` 退出码 0、三端点全通，等于
  用真实 llama-server 验证了 `LocalJudge` 的请求形状能落地。

### 票 01 交接

- 换机器时 `bin/` 与 `gguf/` 不入库，需自行补齐（模型只能从 ModelScope 下，HuggingFace
  全封）；`deploy/local-ai/README.md`「资产清单」有精确字节数。
- 唯一仍上云的是 `CHAT_LLM_*`（DeepSeek）与语音 TTS/STT；若将来要做语音本地化，起点是
  `deploy/local-ai/notes/04-本地化调研.md` 第 5 节列的硬编码三处与缺 `vad_analyzer`。
- `notes/` 是方案 A 时代的历史记录，命令行已过时，只保留结论与实测数据。
