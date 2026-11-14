# local-ai —— 把 eidolon 的文字链路搬到本机

这套东西让 eidolon 的**向量化、重排、判断模型**都跑在本机（一个 `llama-server` 进程、
一个端口），文字聊天只有对话 LLM（DeepSeek）需要上云。语音通话的 TTS/STT 仍上云，
见下面「边界」。

- 分支：`local-ai`（从 `8e4e3829` 开出）。
- 来源：2026-11-02～04 那次「本地化调研」会话的全部落盘记录，原本在 `F:\local-ai\`
  （仓库外、非 git 仓库、会话已删）。本分支把它搬进版本库，`notes/` 保留原始调研记录。
- 代码落地：采用 **方案 B `JUDGE_PROVIDER=local`** —— 在后端加了原生 `LocalJudge`
  （直接打 OpenAI 兼容 `/chat/completions` + strict JSON Schema），不再需要 `judge_shim.py`
  翻译。`judge_shim.py` 保留作零改码的**方案 A 回退**（见文末）。

## 先看哪个

| 你要做的事 | 打开 |
|---|---|
| 把这套配置在本机跑起来 | 本文件「启动」一节 |
| 逐行 `.env` 覆盖项 | `env.local-ai.example` |
| 改代码前了解仓库规矩和本机工具链坑 | `notes/06-仓库与工具链.md` |
| 本地化可行性结论与实测数据（原始调研） | `notes/04-本地化调研.md` |
| 后端/前端确切改了哪些文件:行 | `notes/05-代码插桩点.md`（方案 A 时代的插桩点，方案 B 已落地，仅供对照） |
| 本机环境、平台、启动 eidolon 的坑 | `notes/01`～`03`、`notes/07` |

> `notes/` 是**历史调研记录**，写于方案 A 时代：里面提到 `JEV_BASE_URL` / 8097 shim /
> `JUDGE_TIMEOUT` 硬编码 5s / `F:\local-ai` 路径，都是当时的事实，现在已被本分支的代码和
> 脚本取代。结论和实测数据仍有效，命令行要按本文件来。

## 目录内容

```
deploy/local-ai/                 （在 F:\eidolon-sc 仓库内）
├── README.md                    ← 本文件（当前状态 + 启动步骤）
├── env.local-ai.example         要覆盖进仓库根 .env 的那几行
├── start-app.ps1                ← 一键启动全部（无窗口，只出现应用界面）
├── start-llama-server.ps1       只起 llama-server（8094），调试用
├── rocm\                        ROCm/HIP 那条路，见下面「ROCm」一节
├── models.ini                   llama-server 三模型 preset（embed + rerank + judge）
├── judge_shim.py                方案 A 回退：/systemone → /chat/completions 翻译（已实测）
├── .gitignore                   bin/ gguf/ logs/ 不入库
├── notes\                       8 份调研记录：01–07 是方案 A 时代的历史，08 是换模型/ROCm 的交接
├── bin\llama-server\            49 MB，llama.cpp 2.36.0（Vulkan，AMD 可用）—— 不入库
├── gguf\                       3.6 GB，三个 GGUF —— 不入库
└── logs\                        运行时日志（含 start-app.ps1 的 logs/desktop/），不入库
```

## ROCm（试过，本机不通）

2026-11-13～14 试过把推理搬到 ROCm。**结论是这张卡上不行**，已回到 Vulkan：

- RX 6750 XT 是 `gfx1031`，AMD 的
  [Windows ROCm 支持表](https://raw.githubusercontent.com/ROCm/rocm-install-on-windows/develop/docs/reference/system-requirements.rst)
  标为不支持；
- llama.cpp 的 HIP 预编译版能认到显卡，但显存报 `0 MiB`，一加载模型就崩
  （`cudaMemGetInfo failed`，见 llama.cpp #26963 / #24836）；
- 用 torch 直接加载 GGUF 能跑，但判断模型只有 8.8 tok/s、全量 8 问题约 21 秒，
  而 Vulkan 是 2.6 秒。

完整实测、可复现步骤、以及「认到设备不等于能用」这个坑，见
[`rocm/RESULTS.md`](rocm/RESULTS.md)。`rocm/` 下那批脚本是调研产物，不进 CI。

下面「启动」仍是本目录的正式用法。

## 资产清单

| 文件 | 大小 | 说明 |
|---|---|---|
| `bin/llama-server/llama-server.exe` | 21 KB（薄壳）+ 同级 DLL | llama.cpp 2.36.0，Vulkan 后端 |
| `gguf/qwen3-embed-0.6b-q8_0.gguf` | 639 150 592 B | 向量化，模型名必须是 `Qwen/Qwen3-Embedding-0.6B` |
| `gguf/qwen3-reranker-0.6b-q8_0.gguf` | 639 153 184 B | 重排，模型名必须是 `Qwen/Qwen3-Reranker-0.6B` |
| `gguf/qwen3-4b-instruct-2507-q4_k_m.gguf` | 2 497 281 120 B | 判断模型（本地 4B），preset 里叫 `local-judge` |

`bin/llama-server/` 从已卸载的 LM Studio 残留整目录搬出，**不依赖 LM Studio**。
`bin/` 和 `gguf/` 体积大、参照 ADR-0003「剧情库与音色不入库」精神不进版本库 —— 换机器
需自行补齐（历史记录里记了模型只能从 ModelScope 下，HuggingFace 全封）。

## 启动

**日常用 `start-app.ps1`。** 它把四个服务都放到后台跑（不弹任何 PowerShell 窗口），
然后以 Chrome/Edge 的 `--app` 模式打开一个应用窗口——没有地址栏，像原生程序：

```powershell
powershell -ExecutionPolicy Bypass -File deploy\local-ai\start-app.ps1
```

桌面上的 `Eidolon-SC` 快捷方式就是这条命令，**一个入口兼作开关**：服务在跑时双击它
就停掉，没在跑时双击就启动。脚本里也有 `-Stop`（总是停）和 `-NoToggle`（总是启动），
供脚本调用。
起的顺序是 8094（模型）→ 8765（立绘素材）→ 8000（后端）→ 3000（前端），
每个都等到就绪才继续。日志在 `logs/desktop/`（`llama`、`stands`、`backend`、`frontend`），
只在某项起不来时才会弹一次窗口报错。

`eidolon serve` 只读进程环境变量、不读 `.env`，所以 `start-app.ps1` 自己把 `.env`
加载进环境（和 `uv run --env-file` 一样的规则：已有的值优先）。

下面这节讲的是**手动分步起**，调试单个服务时用。

## 手动启动（方案 B，默认，不需要 shim）

```powershell
# 1. 起本地模型服务（三模型同进程，一个端口 8094）
powershell -ExecutionPolicy Bypass -File deploy\local-ai\start-llama-server.ps1

# 2. 把 env.local-ai.example 里那几行覆盖进仓库根的 .env（同名旧行删掉，别写两遍）
#    唯一省不掉的线上 key 是 CHAT_LLM_API_KEY（DeepSeek）；本机用 EIDOLON_DEV_LOGIN=1 免 OAuth

# 3. 起后端（必须在 backend/ 下）、前端
cd backend;  uv run --env-file=../.env eidolon serve
cd ../frontend;  npm run start

# 4. 验三项
cd ../backend;  uv run eidolon doctor      # embedding / rerank / judge

# 停模型服务
powershell -ExecutionPolicy Bypass -File deploy\local-ai\start-llama-server.ps1 -Stop -Port 8094
```

`start-llama-server.ps1` 启动后会打印一整段 `.env` 覆盖项，照抄即可。脚本用
`-WorkingDirectory` 把子进程 CWD 固定到脚本目录 —— **因为 `models.ini` 里的 `gguf/...`
相对路径是按 CWD 解析的**（实测：从仓库根启动会加载失败，从 `deploy/local-ai/` 启动成功）。

`env.local-ai.example` 的要点：

```
SILICONFLOW_BASE_URL=http://127.0.0.1:8094/v1   # embedding + rerank 共用
SILICONFLOW_API_KEY=fake                          # 必须非空，空了重排会静默关掉
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B        # 须与 models.ini [段名] 一致
RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
RERANK_TIMEOUT=10                                 # 默认 3 秒不够：本机 36 条剧情长度候选要 4~5 秒
JUDGE_PROVIDER=local                              # 本分支新增，走原生 LocalJudge
LOCAL_JUDGE_BASE_URL=http://127.0.0.1:8094/v1    # 同一个端口，拼 /chat/completions
LOCAL_JUDGE_MODEL=local-judge
JUDGE_TIMEOUT=30                                 # 本分支改成可配，解决本地 4B 冷启动超时
```

## 本分支改了什么（能力覆盖）

| 环节 | 现状 | 靠什么 |
|---|---|---|
| 向量化 embedding | ✅ 本地 | `SILICONFLOW_BASE_URL` 指向本机，零改码 |
| 重排 rerank | ✅ 本地 | 同上；`SILICONFLOW_API_KEY=fake` 必须非空 |
| 判断模型 judge | ✅ 本地、一等公民 | `JUDGE_PROVIDER=local` + `LocalJudge`，`JUDGE_TIMEOUT` 可配 |
| 对话 LLM | ☁️ 上云 | DeepSeek，唯一保留的线上 key |
| 语音 TTS / STT | ☁️ 上云 | 见「边界」，本分支明确不改；替代方案见 `LOCAL-AI-ALTERNATIVES.md` |
| 生图 | 🔧 本来可配 | `NOVELAI_API_FORMAT=openai` + `NOVELAI_BASE_URL` 能接兼容本地服务；注意本机 SD-WebUI 是 A1111 格式，需一层适配，见 `LOCAL-AI-ALTERNATIVES.md` |

→ **打字聊天整条链路（剧情检索、记忆、好感度、通话情绪判断）只有 DeepSeek 上云**；
语音通话仍上云。

代码落地点：`backend/src/eidolon/` 下 `config.py`（`LocalJudgeSettings` /
`judge_timeout_from_env()` / `judge_from_env()`）、`judge.py`（`LocalJudge`、
`judge_timeout()`、shim 的 schema/prompt 逻辑）、`service_clients.py` / `doctor.py` /
`__main__.py`（4 处超时调用点）、`live_config.py` / `admin_config_api.py`（`base_url`
字段 + 本地免 key）、前端 `adminConfig.ts` / `i18n/zh.ts` / `components/AdminConfig.tsx`
（`local` 选项与提示）。文档：`docs/configuration.md` 三语、`GLOSSARY.md`、`.env.example`。

## 边界（为什么语音不本地化）

- **TTS**：`FISH_TTS_URL` 硬编码三处（`call.py:797`、`greetings.py:45`、
  `pipecat/services/fish/tts.py:203`），Fish Audio 用 ormsgpack event 私有协议，
  `call.py:1277-1282` 还缺 `vad_analyzer`。要换得写新 `TTSService` + 加 VAD +
  改 12 个后端文件的 `FishAudioModel Literal` + 5 个前端文件，`readings.py` 读音表也会失效。
- **STT**：`config.py:40` `SttProvider = Literal["soniox", "bailian"]`，没有第三个值。
- 改动面与风险远超收益，本分支**只把边界写清楚，不动语音链路**。

本机其实已有 GPT-SoVITS、SD-WebUI 等 ROCm 服务，接进来的可行性与工作量另写在
[`LOCAL-AI-ALTERNATIVES.md`](LOCAL-AI-ALTERNATIVES.md)：生图只需一层薄适配，
语音则相当于一张独立的大票。那是**后续提案**，不属于本分支已落地的范围。

## 实测数据（原始调研，2026-11-04，AMD RX 6750 XT / Vulkan）

用 eidolon 自己的客户端类跑通，不是手搓 curl：

```
POST /v1/embeddings   → 1024 维，非全零
POST /v1/rerank       → 0.9765（"制作人说想去海边"） vs 0.000088（"明天要早起开会"）
POST /v1/chat/completions (judge, temperature:0 + json_schema)
  3 个 noul 问        冷 0.98 s / 热 0.51–0.53 s
  全量 8 问           冷 3.56 s / 热 2.64–2.66 s
检索三问  1.53 s → {"needs_retrieval":{"noul":0.1},"recall":{"noul":0.0},"about_producer":{"noul":0.0}}
情绪+点头 2.00 s → 7 情绪 + 3 nod 选项，概率各自相加为 1
```

2026-11-09 补测（真实候选规模）：重排耗时主要随「候选条数 × 每条长度」增长 —— 36 条 × 50 字
稳定 2.5 s，36 条 × 173 字（接近剧情段落长度）要 4.2–4.8 s，60 条约 3.9 s。后端原本给重排
只有 3 秒（`retry.TURN_TIMEOUT`），所以本地跑真实长度的剧情检索**必然超时**。本分支把
`RERANK_TIMEOUT` 做成可配，`env.local-ai.example` 建议 `10`。

阈值：`judge_thresholds_for("local-judge")` 落到 `jev` 族（0.5/0.5/0.5/0.5/0.7/0.4/0.5），
启动时会打印「没有自己的阈值」警告 —— 本地模型沿用 Jev 阈值，属预期。
`JUDGE_TIMEOUT` 原来硬编码 5s，本地 4B 冷启动 3.56s、长上文会超时；本分支改成可配，
`env.local-ai.example` 建议 `30`。

## 三个踩过的坑（改这些文件前必看）

1. **带中文的 `.ps1` 必须存成 UTF-8 with BOM。** PowerShell 5.1 对没有 BOM 的
   `.ps1` 按 ANSI(GBK) 解码，中文注释和字符串会把引号咬坏，报
   `表达式或语句中包含意外的标记"}"`。用 `write`/`edit` 工具改完会掉 BOM，掉了就用
   `python -c "d=open(p,'rb').read(); open(p,'wb').write(b'\xef\xbb\xbf'+d)"` 补回去。
   `start-app.ps1` 和 `rocm/` 下两个脚本走的是另一条路：**全文纯 ASCII**，所以不需要 BOM，PS 5.1 按哪种
   编码读都不会坏。改它们时保持 ASCII，或者补 BOM，二者选一。
   （`judge_shim.py` 不需要 BOM，Python 3 源码默认按 UTF-8 读。）
2. **`models.ini` 的相对路径按 CWD 解析，不是按 preset 所在目录。** 脚本已用
   `-WorkingDirectory` 固定；若改成从别处 `Start-Process`，三个 GGUF 会加载失败
   （实测报 `model name=... failed to load`）。
3. **模型名是路由键。** `EMBEDDING_MODEL` / `RERANK_MODEL` / `LOCAL_JUDGE_MODEL` 原样进请求体，
   llama-server 拿它匹配 `models.ini` 的 `[段名]`，差一个字就 500。

## 方案 A 回退（零改码，不用本分支代码）

如果不想用 `JUDGE_PROVIDER=local`，可以退回原始调研里的做法：保持 `JUDGE_PROVIDER=jev`，
起 `judge_shim.py` 把 eidolon 的 System One 协议（`POST {base_url}/systemone`）翻译成本机的
`/chat/completions`，占 8097：

```
python deploy\local-ai\judge_shim.py
JEV_BASE_URL=http://127.0.0.1:8097/v1   # 注意写到 /v1，JevJudge 会拼 /systemone
JEV_API_KEY=local
JEV_MODEL=local-judge
```

方案 A 下 `JUDGE_TIMEOUT` 是硬编码 5s（`notes/05` 记了怎么做成可配）；本分支已经把它做成
可配并新增原生 `LocalJudge`，所以默认走方案 B。shim 只作为不改后端代码时的兼容后路。
