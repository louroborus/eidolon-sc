# 03: 试把本地推理搬到 ROCm（结论：本机不通，不采用）

Status: wontfix

## 背景

`spec.md` 落地方案 B 后，文字链路的三个模型跑在 `bin/llama-server`
（llama.cpp 2.36.0 **Vulkan**）上，共用 8094 一个端口。用户要求「跑模型用 ROCm」。

`notes/08` 在 2026-11-09 判过「ROCm 路线已排除」，理由是缺 `cl.exe` / `cmake` / `ninja`。
这次发现了绕过编译的办法：llama.cpp 每个 release 都发预编译的 Windows HIP 包
（`llama-<build>-bin-win-rocm-<ver>-x64.zip`），下载即用。

## 做了什么

1. 下载 `llama-b12361-bin-win-rocm-10.0-x64.zip`（280 MB）到 `corpus/llama-hip/`
   （不入库，`rocm/fetch_llama_hip.py` 可重下）。它的 `ggml-hip.dll` 依赖 ROCm 运行时，
   把 `HIP_PATH` / `PATH` 指到 `D:\AI\ComfyUI\venv` 里的 ROCm SDK 即可。
2. `rocm/start-hip-server.ps1`：一个 router 服务（8097）按 `models.ini` 承载三个模型。
3. 另写 `rocm/server.py`：用 torch 直接加载同样的 GGUF，作为对照。

## 结果：没能采用

**llama.cpp HIP 版在本机起不来。** 能认到显卡（`--list-devices` 打印出
`ROCm0: AMD Radeon RX 6750 XT`），但显存报 `0 MiB, 0 MiB free`，
一加载模型就访问冲突退出（`0xC0000005` / `-1073741819`）。

根因三条，都不在我们能改的层面：

1. RX 6750 XT 是 `gfx1031`，AMD 的
   [Windows ROCm 支持表](https://raw.githubusercontent.com/ROCm/rocm-install-on-windows/develop/docs/reference/system-requirements.rst)
   标为 ❌ 不支持；
2. llama.cpp [#26963](https://github.com/ggml-org/llama.cpp/issues/26963)：预编译 ROCm
   Windows 版报 `cudaMemGetInfo failed` 后崩溃，Vulkan 版正常——与本机症状一致；
3. llama.cpp [#24836](https://github.com/ggml-org/llama.cpp/issues/24836)：`hipMemGetInfo()`
   在 Windows 上返回不可靠。

试过且被排除的绕法：`HSA_OVERRIDE_GFX_VERSION=10.3.0`、换 SDK 里的新版
`amdhip64_7.dll` 替代包内旧版、绕过 router 直接起单模型进程——全部仍是 0 MiB。

**torch 这条替代路能用但不够快**：独占整卡时判断模型 8.8 tok/s，全量 8 问题约 21 秒
（Vulkan 2.6 秒），且在 `JUDGE_TIMEOUT=30` 边缘；结构化输出还要约束解码，更慢。

## 决定

**回到 Vulkan**，`.env` 改回 8094。`CORPUS/llama-hip` 和 `rocm/` 下的脚本作为调研记录保留。

## 保留下来仍然有效的发现

调研中验证过、以后换方案也用得上的两件事：

- **向量化**：Qwen3-Embedding 要追加 EOS 并取最后一个非 padding 位置再归一化，这样
  torch 出来的向量与 llama-server 的 cosine 是 0.9995+，换后端**不用 reembed**
- **重排**：Qwen3-Reranker 必须补 `<think:6124c78e>\n\n` 前缀（llama.cpp `reranking=1`
  的做法），否则分数塌缩 7 倍（0.98 → 0.14），0.3 阈值会滤掉所有相关段落

## 一个未解开的疑点

2026-11-14 01:57 那次曾成功起过服务并报出 `12272 MiB, 12131 MiB free`，基准脚本报
`99.4 tok/s` / `2.76 秒`。之后同样命令再也起不来，那个数字**不可复现，不作结论引用**。
`RESULTS.md` 里已如实标注。

## Comments

`code-review` 修的问题保留（README 加 ROCm 一节、`-Stop` 不再误杀 8094 的进程、
`server.py` 删死常量并分离 enforcer 缓存、`bench_hip_judge.py` 端口参数化）。

`spec.md` 的 `## Issues` 已加上本票。
