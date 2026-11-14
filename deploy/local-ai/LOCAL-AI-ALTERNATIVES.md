# 本地语音与生图接入替代：调研结论

2026-11-14。回答「不买 Fish Audio / NovelAI 的 key，能不能用本机已有的东西把语音通话和照片跑起来」。

术语和架构以仓库的 `GLOSSARY.md`、`docs/configuration.md` 为准。

## 一句话结论

**生图能接，工作量小；语音通话能接，但工作量相当于一张独立的大票。**
本机 `D:\AI` 下已经有 ROCm 版的 SD-WebUI、ComfyUI 和 GPT-SoVITS，模型都齐了，
不用重新下载。

---

## 生图：可行性高

### 项目侧的接口契约

`backend/src/eidolon/image_generation.py` 已经支持两种格式，由 `NOVELAI_API_FORMAT` 选择：

| 格式 | 地址 | 请求 | 返回 |
|---|---|---|---|
| `novelai`（默认） | `POST {base_url}/ai/generate-image` | NovelAI V4 形状，带 negative prompt、832×1216、23 steps、scale 7、Euler Ancestral | ZIP，里面一张图 |
| `openai` | `POST {base_url}/images/generations` | OpenAI 形状，`model` / `prompt` / `size` / `n` | `b64_json`（或 URL，由客户端去下载） |

`NOVELAI_BASE_URL` 和 `NOVELAI_API_FORMAT` 在 `.env.example` 里都有，不需要改代码。
返回被严格校验：**必须恰好一张图、且尺寸是要求的 832×1216**，否则算 `bad_image`。
`TIME_LIMIT = 90` 秒（含排队），`GPT_TIME_LIMIT = 240` 秒。

### 本机已有资产

| 服务 | 位置 | 状态 |
|---|---|---|
| SD-WebUI（A1111，ROCm 版） | `D:\AI\stable-diffusion-webui-amdgpu` | `webui-user-rocm.bat` 已带 `--api` |
| ComfyUI（ROCm 版） | `D:\AI\ComfyUI` | `run-comfyui-rocm.bat`，端口 8188 |
| 模型 | SD-WebUI `models/Stable-diffusion/` 下 10 个（含 animagine-xl-4.0、Lustify-SDXL、CyberRealistic、Counterfeit-V2.5、WAI-illustrious-SDXL 等，SD1.5 与 SDXL 都有） | 已就位 |
| GPU | RX 6750 XT 12 GB，ROCm 原生 | 实测 512×512 / 12 steps ≈ 2 秒，首图约 90 秒预热 |

### 缺口

**A1111 的 `--api` 暴露的是 `/sdapi/v1/txt2img`，不是 `/images/generations`。** 请求和返回形状都不一样：

- 请求要 `steps`、`cfg_scale`、`sampler_name`、`width`、`height`、`negative_prompt`
- 返回是 `{"images": ["<base64 png>", ...], "parameters": {...}}`，不是 `{"data":[{"b64_json": ...}]}`

所以**不能直接把 `NOVELAI_BASE_URL` 指向 SD-WebUI**。需要一层薄适配：

```
POST /v1/images/generations   ← eidolon 发的（OpenAI 形状）
        ↓ 转换
POST /sdapi/v1/txt2img        ← SD-WebUI 收的（A1111 形状）
        ↓ 包装
{"data": [{"b64_json": "..."}]}
```

### 落地方案

写一个独立的小服务（约 100–150 行 Python/FastAPI），放在 `deploy/local-ai/` 下：

1. 收 `/images/generations`，把 `prompt`、`size` 拆成 A1111 的 `width`×`height`；
   `832×1216` 直接透传（SD1.5 也能出，只是慢些；SDXL 建议对齐到 1024 倍数再裁回）。
2. 用 `NOVELAI_STYLE` 里那套画风 tag 拼进 `prompt`，`negative_prompt` 用仓库 `photo_prompt.py` 的那份。
3. 转发到 `http://127.0.0.1:7860/sdapi/v1/txt2img`，把返回的 `images[0]` 包成 `b64_json`。
4. `.env` 设 `NOVELAI_API_FORMAT=openai`、`NOVELAI_BASE_URL=http://127.0.0.1:<适配端口>/v1`、
   `NOVELAI_API_KEY=fake`（本地不校验）。

**注意点**：
- 尺寸必须严格 832×1216，否则 `image_generation.py` 判 `bad_image`。适配层负责缩放/裁切。
- 首图约 90 秒预热，逼近 `TIME_LIMIT=90`。建议服务常驻并预热一次，或把 `TIME_LIMIT` 放宽。
- 生成的是「画风接近」的图，不是原作角色本身；项目要求全年龄，需要在 negative prompt 里守住。
- 朋友圈配图（`GPT_IMAGE_*`）也走 OpenAI 形状，同一个适配层可以复用。

**评估：改动只在 `.env` + 新增一个适配脚本，不碰后端 12 个文件。风险低。**

---

## 语音合成（TTS）：可行但改动大

### 项目侧现状（`notes/04-本地化调研.md` 第 5 节）

- `FISH_TTS_URL` 硬编码三处：`call.py:797`（`wss://api.fish.audio/v1/tts/live`）、
  `greetings.py:45`、`pipecat/services/fish/tts.py:203`。**全仓没有 `FISH_AUDIO_BASE_URL`。**
- Pipecat 的 `FishAudioTTSService` 用 ormsgpack event 私有协议，不是普通 HTTP。
- `call.py:1277-1282` 的 `TransportParams` **缺 `vad_analyzer`**，而 `SegmentedSTTService`
  明确要求 VAD。
- `FishAudioModel` 这个 `Literal` 波及后端 12 个文件 + 前端 5 个文件。
- `readings.py` 的读音表（OpenJTalk 音素 + 音高数字，如 `ha0shi1ga0`）是随每个 Fish 请求
  发 `pronunciation_dictionary` 的，换服务就失效。
- `CallMeter` / `Origin` / activity 记账写死了 Fish URL。

### 本机已有资产：GPT-SoVITS

| 项 | 内容 |
|---|---|
| 位置 | `D:\AI\gpt-sovits`，启动 `start-gsv.bat` → `127.0.0.1:9883` |
| 接口 | `GET/POST /tts`、`/set_refer_audio`、`/set_gpt_weights`、`/characters`、`/set_character`、`/health`、`/speakers` |
| 协议 | 普通 HTTP，返回 WAV 音频；支持 `text_lang=ja`、参考音频 + `prompt_text` |
| 音色 | `voices/` 下 13 位角色（china / hiro / kotone / lilja / mao / misuzu / rinami / saki / sena / sumika / temari / tsubame / ume），**按情绪分类**的日语参考音频，每条带 `.json` 的 prompt_text |
| 情绪 | 每个角色 20–60 条，涵盖害羞、沮丧、欣喜、惊讶、认真、迟疑等，正好对得上项目的「随情绪换表情」 |

**但这 13 位是《学园偶像大师》的角色，不是《闪耀色彩》的 29 位。** 见
`voices/_学马仕13小偶像索引.md`。所以音色库只能验证技术链路，
不能直接替换 `characters.toml` 里的 `voice_id`；要真正用起来，得为 29 位角色各准备参考音频，
那属于版权与素材工作，不在代码范围内。

### 落地方案（如果要做）

新的 `TTSService` 要同时满足：流式（Pipecat 按句推送）、低延迟、能带情绪选参考音频。
GPT-SoVITS 的 `/tts` 是整段返回，不是流式 WebSocket，所以要在适配层里按 `segments` 逐句请求、
按句回传音频帧。

工作量（沿用 `notes/04` 的估算）：

1. 写 pipecat `TTSService` 子类，接 GPT-SoVITS `/tts`（或换 pipecat 自带的 `kokoro` / `xtts`）
2. 改 `call.py:721 tts()` 工厂
3. 加 `vad_analyzer`
4. 把 `FishAudioModel` 从 12 个后端文件 + 5 个前端文件里剥掉
5. 读音表改用别的方式（或接受专有名词念不准）
6. `greetings.py` 的落地页招呼、`read_aloud.py` 的朗读、`letters.py` 的信都要一起换
7. 三语文档 + GLOSSARY

**评估：约等于一张独立的大票，涉及前后端 + 文档。建议先做生图，语音另开会话。**

---

## 语音识别（STT）：可行，改动中等

- `config.py:40` `SttProvider = Literal["soniox", "bailian"]`，没有第三个值。
- `call.py:700 stt()`：bailian → `BailianSTTService`，否则 `ProxiedSoniox`。
- Pipecat 自带 `whisper`（`faster-whisper`）和 `funasr`，但项目 venv 里都没装。
- AMD 无 CUDA，`faster-whisper` 只能 CPU 或 DirectML；本机 ROCm 虽能用（见 `rocm/RESULTS.md`
  的 torch 验证），但 whisper 的 ROCm 路径未验证。

**最小路径**：`faster-whisper` small/base 跑 CPU，加一个 `local` 分支到 `stt()`。
改动集中在 `config.py` + `call.py` + 一个服务类，不涉及 `FishAudioModel` 那种 12 文件连锁。
日语识别质量需要实测。

---

## 建议的优先级

| 优先级 | 事项 | 理由 |
|---|---|---|
| 1 | **生图适配层** | 后端已支持 OpenAI 格式，只加一个转换服务；本机模型齐全、ROCm 实测 2 秒/图 |
| 2 | **STT 本地分支** | 改动面小，可独立验证 |
| 3 | **TTS 本地分支** | 改动面最大，且音色素材不对应本项目角色，建议单独开一张票 |

## 另：本机 ROCm 的已知限制

见 `rocm/RESULTS.md`：RX 6750 XT（gfx1031）不在 AMD 的 Windows ROCm 支持表里，
llama.cpp 的 HIP 预编译版起不来；但 **torch 2.11+rocm7.13 能正常用**
（`cuda.is_available()=True`、matmul 正确）。SD-WebUI / ComfyUI / GPT-SoVITS 走的都是
torch 这条路，所以本节的三套服务不受那个限制影响。
