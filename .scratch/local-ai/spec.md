# 把判断模型本地化，并把本地化调研资产搬进仓库

Status: done

术语以 `GLOSSARY.md` 为准；相关背景见 `deploy/local-ai/notes/04-本地化调研.md`（原始调研）
与 ADR-0003（剧情库与音色不入库，本地模型权重同此精神不入库）。

## Problem Statement

2026-11-02～04 有一次「除 DeepSeek 外都用本机」的本地化调研，全部产出落盘在
`F:\local-ai\`（仓库外、非 git 仓库），会话已删，那里是唯一副本，随时可能丢。调研结论是：
文字聊天链路（向量化、重排、判断模型）可以全部跑在本机，语音通话做不到。但当时只做到
「零改码方案 A」——用 `judge_shim.py` 把 System One 协议翻译成 `/chat/completions`，还要
另起一个进程占 8097；`JUDGE_TIMEOUT` 仍是硬编码 5 秒，本地 4B 冷启动 3.56 秒、长上文会超时。

要把这次调研变成仓库里可持续维护的东西：资产进版本库、判断模型本地化成为一等公民、
让超时可配，并把「还做不到」和「怎么跑」写清楚。

## Solution

1. 开出 `local-ai` 分支；把 `F:\local-ai` 全部搬到 `deploy/local-ai/`（文本/脚本入库，
   `bin/` 49 MB + `gguf/` 3.6 GB 复制到同目录但用 `.gitignore` 排除，参照 ADR-0003 精神）。
2. 采用**方案 B**：新增 `JUDGE_PROVIDER=local` 与原生 `LocalJudge`，把 `judge_shim.py` 的
   schema/prompt 逻辑搬进后端，直接打 OpenAI 兼容 `/chat/completions`，不再需要翻译进程；
   `judge_shim.py` 保留作零改码的方案 A 回退。
3. `JUDGE_TIMEOUT` 做成可配环境变量，坏值在启动期就拦下，解决本地模型冷启动超时。
4. 配置、后台管理页、三语文档、GLOSSARY、`.env.example` 全部补 `local`；写出启动步骤与
   「语音链路仍上云」的边界。
5. 实测 `models.ini` 的相对路径解析基准，修正启动脚本。

## Decisions

- 选方案 B（原生 `LocalJudge`）而非只靠 shim：本地化要成为一等公民，不该让部署者多跑一个
  进程；shim 留作不改后端代码时的兼容后路。
- embedding / rerank **零改码**：本来就能用 `SILICONFLOW_BASE_URL` 指向本机 llama-server
  （`embedding.py` 拼 `/embeddings`、`rerank.py` 拼 `/rerank`），共用 8094 一个端口。
- 本地判断模型**免 API key**：`LocalJudgeSettings.api_key` 可空，`LocalJudge` 仍发一个
  占位的 Bearer；后台管理页对 `local` 不要求 key，但要能填 `base_url`。
- `models.ini` 里的 `gguf/...` 是**相对 CWD 的路径**（实测：从仓库根起加载失败，从
  `deploy/local-ai/` 起成功）。`start-llama-server.ps1` 用 `-WorkingDirectory` 把子进程
  CWD 固定到脚本目录；不改回绝对路径，这样换机器只要整个目录一起搬。
- `start-llama-server.ps1` 存成 UTF-8 with BOM（PowerShell 5.1 无 BOM 会按 GBK 解码咬坏中文）。
- 阈值：`judge_family("local-judge")` 落到 jev 族（0.5/0.5/0.5/0.5/0.7/0.4/0.5），启动时
  会打「没有自己的阈值」警告，属预期，不新增一族。
- **TTS/STT 不做地址可配改造**：Fish Audio 的 `FISH_TTS_URL` 硬编码三处、用 ormsgpack
  私有协议、`call.py:1277-1282` 还缺 `vad_analyzer`，`config.py:40` 的 `SttProvider` 只有
  `soniox`/`bailian`；改动面 12 个后端 + 5 个前端文件且 `readings.py` 读音表会失效，风险
  远超收益。只在交接文档写明边界。生图本就有 `NOVELAI_BASE_URL` + `NOVELAI_API_FORMAT=openai`
  可接本地兼容服务，不动。
- `backend/uv.lock` 的镜像重写（`uv run` 顺带产生）与任务无关，已回滚。

## Out of Scope

- 语音合成 / 语音识别的本地化（见 Decisions，明确不做；本机替代方案只在
  `LOCAL-AI-ALTERNATIVES.md` 里研究，**不改代码**）。
- 对话 LLM 的本地化（继续用 DeepSeek，唯一保留的线上 key）。
- 生图的本地化（`NOVELAI_API_FORMAT=openai` 本来就支持兼容服务；本机 SD-WebUI
  是 A1111 格式、需要一层适配，作为后续工作记在 `LOCAL-AI-ALTERNATIVES.md`）。
- 把三个 GGUF 提交进版本库（体积太大，参照 ADR-0003）。

## Issues

- `issues/01-local-judge-provider.md`
- `issues/02-local-rerank-timeout.md`（本地重排超时可配 + `eidolon eval` 效果实测）
- `issues/03-rocm.md`（试把本地推理搬到 ROCm：本机不通，结论见票）

## 后续追加的工作（不属原始五条 Solution）

桌面入口合并成一个开关、清掉验收留下的测试账号并固定 `test-runner`、以及本地
语音/生图替代方案的调研，都是这一波里按用户要求追加的，没有单独开票；
改动分别见提交信息与 `deploy/local-ai/LOCAL-AI-ALTERNATIVES.md`。
