# ROCm 本地推理：实测结论（这条路在本机不通）

2026-11-13～14 实测。回答「跑模型用 ROCm」。

## 结论

**这张卡上用不了 ROCm 的 llama.cpp 后端。** 换回 Vulkan 的 `llama-server`
（`deploy/local-ai/start-llama-server.ps1`，端口 8094）。

本目录保留下来是因为调研过程中确认了几件有价值的事（见下面「确认下来的事实」），
以及一份尽量完整的失败记录，免得以后有人再走一遍（脚本里硬编码了本机路径，换机器要改）。

## 为什么不通

三条原因叠在一起，且都不在我们能改的层面：

1. **这张卡不在 AMD 的 Windows ROCm 支持表里。** RX 6750 XT 是 `gfx1031`
   （`torch.cuda.get_device_properties(0).gcnArchName` 确认），
   [官方系统需求表](https://raw.githubusercontent.com/ROCm/rocm-install-on-windows/develop/docs/reference/system-requirements.rst)
   把它标为 ❌ 不支持。RDNA2 消费卡在 Windows 上没有官方 ROCm。
2. **llama.cpp 的 HIP 后端在 Windows 上查不到可用显存。**
   `ggml_backend_cuda_device_get_memory: cudaMemGetInfo failed (invalid argument)`，
   设备报 `0 MiB, 0 MiB free`，然后加载模型时访问冲突退出（`0xC0000005` / `-1073741819`）。
   这是两个已知未修的问题：
   [#26963](https://github.com/ggml-org/llama.cpp/issues/26963)（预编译 ROCm Windows 版崩、
   Vulkan 版正常，与本机症状一致）、
   [#24836](https://github.com/ggml-org/llama.cpp/issues/24836)（`hipMemGetInfo()` 在 Windows
   上返回不可靠）。
3. **试过的绕法都不行**（都被实测排除）：
   - `HSA_OVERRIDE_GFX_VERSION=10.3.0`（把 gfx1031 当 gfx1030）→ 仍是 0 MiB
   - 换成 ROCm SDK 里的新版 `amdhip64_7.dll`（19.3 MB）替代包内自带的旧版（15.8 MB）→ 仍是 0 MiB
   - 直接起单个模型进程、绕过 router → 同样崩在 `load_model`

有一个容易误判的地方：`llama-server.exe --list-devices` 能**认到**显卡并打印型号，
看起来像成功；但显存报 0，一加载模型就崩。**「认到设备」不等于「能用」。**

## 确认下来的事实（仍然有效）

调研过程中这几件事被验证过，换了方案也用得上：

### 向量化：torch 能跑，且与 llama-server 的向量空间一致

- 用 `D:\AI\ComfyUI\venv` 的 torch 2.11+rocm7.13 加载同一个 GGUF（0.6B Q8，1.19 GB）可用
- 关键：Qwen3-Embedding 要**追加 EOS** 并取**最后一个非 padding** 位置，再 L2 归一化，
  这样得到的向量与 llama-server 的 cosine 是 **0.9995–0.9997**（`pooling_probe.py`）
- 推论：换后端**不需要** `eidolon reembed`，库里已有的 14525 段向量继续有效
- 错误做法：不加 EOS 只有 0.48–0.76；加 EOS 但取 last_position 是 0.94

### 重排：必须补 `<think>` 前缀

- Qwen3-Reranker 是因果模型，取 `yes`/`no` 两个 token 的 softmax 概率；要用
  `AutoModelForCausalLM`（`AutoModel` 没有 lm_head，取不到 logits）
- **必须**在 chat template 之后追加 `<think:6124c78e>\n\n`，这是 llama.cpp `reranking=1`
  的做法。不追加时所有分数塌缩约 7 倍（0.98 → 0.14），eidolon 的 0.3 阈值会把
  所有相关段落全部滤掉（`template_probe.py`）
- 对齐后相关文档 0.98 / 0.75，无关文档远低于 0.3

### 判断模型：torch 路线的真实上限

独占整块卡（无其它进程）时，eidolon 真实的 8 问集（765 prompt token、184 输出 token）：

| | 数值 |
|---|---|
| 吞吐 | **8.8–9.3 tok/s** |
| 全量 8 问题 | **约 21 秒** |
| 显存 | 8.10 GB（加载后剩 4.61 GB） |

21 秒在 `JUDGE_TIMEOUT=30` 以内，但只余 9 秒；而且输出是 `nested=False`
（结构不符合 eidolon 的 `_reply` 要求），需约束解码，那会更慢——lm-format-enforcer 每题
**642 秒**，因为它每生成一个 token 都要在 151k 词表上重算允许集合。

作为对照：Vulkan 的 llama.cpp 跑同样任务是 ~2.6 秒。

## 一个未解开的疑点

2026-11-14 01:57 那次 `start-hip-server.ps1` 曾成功起过服务并报出
`ROCm0: AMD Radeon RX 6750 XT (12272 MiB, 12131 MiB free)`，当时的基准脚本报
`99.4 tok/s`、全量 8 问题 `2.76 秒`。之后同样的命令再也起不来（稳定复现 0 MiB + 崩溃），
那个数字**无法复现，不应作为结论引用**。

## 如果以后要再试

- 换一张 AMD 官方 Windows ROCm 支持的卡（RX 7000/9000 系，gfx11xx/gfx12xx）
- 或者走 Linux / WSL2，那里的 `hipMemGetInfo()` 行为不同
- 或者等 llama.cpp 修掉 #24836 / #26963

## 本目录的脚本（都是调研产物，不进 CI）

`backend/pyproject.toml` 的 `testpaths` 只有 `tests`，这些脚本不会跑。

| 文件 | 作用 |
|---|---|
| `server.py` | torch+ROCm 的 OpenAI 兼容服务（embed/rerank 可用，judge 太慢） |
| `start-rocm-server.ps1` | 起上面那个（只做 embed+rerank） |
| `start-hip-server.ps1` | 起 llama.cpp HIP router 服务（**在本机起不来**） |
| `fetch_llama_hip.py` | 下载 llama.cpp 的 Windows HIP 预编译包到 `corpus/llama-hip/` |
| `serve-stands.cjs` | 立绘静态服务（零依赖，端口 8765），与 ROCm 无关 |
| `compare_embedding.py` | 对比 ROCm 与 llama-server 的向量 |
| `pooling_probe.py` | 找出与 llama-server 一致的 pooling（EOS + last-non-pad） |
| `compare_rerank.py` | 对比重排分数 |
| `template_probe.py` | 找出重排的 `<think>` 前缀 |
| `llama_probe.py` | 用 transformers 挨个加载三个 GGUF |
| `test_constrained.py` | JSON Schema 约束解码（结构对，642 秒） |
| `test_judge_budget.py` | transformers 生成耗时 |
| `bench_judge.py` | transformers 吞吐（10.1 tok/s，含争用） |
| `bench_torch_full8.py` | **torch 路线真实上限**（8.8 tok/s / 21 秒） |
| `bench_torch_judge_exclusive.py` | 独占显卡时的判模型耗时 |
| `bench_hip_judge.py` | HIP llama-server 吞吐（端口作参数） |
| `test_split.py` | 混合配置：ROCm embed/rerank + Vulkan judge |
| `test_local_judge.py` | 用 eidolon 真实的 `LocalJudge` 客户端打服务，测 schema/prompt/超时是否对得上 |
| `test_all_hip.py` | 全 HIP 端到端（需要能起的 HIP 服务） |
| `test_retrieval.py` | 用库里 14525 段向量做真实语义检索 |
| `inspect_judge.py` | 把 judge 原始返回原样打出来 |
| `judge_timing.py` | 单问判模型冷/热耗时 |
| `smoke_test.py` | torch 服务三端点连通性 |

跑它们要用 ROCm 的 venv：`D:\AI\ComfyUI\venv\Scripts\python.exe`
（需要额外装 `fastapi`、`uvicorn`、`accelerate`、`lm-format-enforcer`）。
