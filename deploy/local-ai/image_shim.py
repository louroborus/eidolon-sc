"""OpenAI-images-compatible shim in front of a local A1111 SD-WebUI (ROCm).

Why this exists
---------------
`eidolon` can already talk OpenAI's image format: with
`NOVELAI_API_FORMAT=openai` it POSTs to `{NOVELAI_BASE_URL}/images/generations`
and reads `data[0].b64_json` (backend/src/eidolon/image_generation.py). But the
local SD-WebUI exposes A1111's `/sdapi/v1/txt2img`, whose request and response
shapes differ. This shim translates between them, so enabling photos needs only
three .env lines and no change to the backend.

Contract (verified against the backend)
--------------------------------------
Request it receives:
    {"model", "prompt", "negative_prompt", "size": "832x1216",
     "n": 1, "response_format": "b64_json"}
Response it must give:
    {"data": [{"b64_json": "<base64 png>"}]}

The backend then checks the decoded picture is EXACTLY 832x1216 (fit=False for
photos), so this shim resizes the A1111 output to that before encoding. It also
serves GPT Image's moments requests (`/images/generations` with `quality` and no
negative_prompt / response_format); those allow other sizes (fit=True), but this
shim keeps the same exact-size guarantee, which the backend also accepts.

What it sends to A1111 (from D:\\AI\\...\\check-output.py):
    prompt, negative_prompt, steps, width, height, cfg_scale,
    sampler_name, batch_size, seed
A1111 answers {"images": ["<base64 png>", ...], "parameters": {...}}.

Usage
-----
    python deploy/local-ai/image_shim.py --port 8090
    # .env:
    NOVELAI_API_FORMAT=openai
    NOVELAI_BASE_URL=http://127.0.0.1:8090/v1
    NOVELAI_API_KEY=fake
    NOVELAI_MODEL=<checkpoint name shown by /sdapi/v1/options>

Env vars: SD_WEBUI_URL (default http://127.0.0.1:7860), SD_STEPS (default 23),
SD_SAMPLER (default Euler a), SD_CFG (default 7), IMAGE_SHIM_PORT (default 8090).

Only the standard library is used, so it runs on any Python 3.9+ already on the
machine; it does not need the project's venv or the ROCm venv.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow", file=sys.stderr)
    raise

DEFAULT_SD_URL = os.environ.get("SD_WEBUI_URL", "http://127.0.0.1:7860").rstrip("/")
STEPS = int(os.environ.get("SD_STEPS", "23"))
SAMPLER = os.environ.get("SD_SAMPLER", "Euler a")
CFG = float(os.environ.get("SD_CFG", "7"))
TIMEOUT = int(os.environ.get("SD_TIMEOUT", "240"))

# The same all-ages guard the project uses for photos. Kept here so pictures made
# locally do not silently lose the constraint the cloud service had.
NEGATIVE_FALLBACK = (
    "nsfw, nude, naked, explicit, sexual, gore, blood, violence, "
    "lowres, blurry, worst quality, low quality, watermark, text, signature"
)

log_prefix = "[image-shim]"


def say(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {log_prefix} {message}", flush=True)


def parse_size(value: Any) -> tuple[int, int]:
    """The requested size, as (width, height); fall back to the photo size."""
    if isinstance(value, str) and "x" in value.lower():
        w, h = value.lower().split("x", 1)
        try:
            return max(64, int(w)), max(64, int(h))
        except ValueError:
            pass
    return (832, 1216)


def call_sd(body: dict[str, Any], width: int, height: int) -> bytes:
    """One picture from A1111's txt2img, as raw PNG bytes."""
    seed = random.SystemRandom().randint(0, 2**31 - 1)
    payload = {
        "prompt": body.get("prompt") or "",
        "negative_prompt": body.get("negative_prompt") or NEGATIVE_FALLBACK,
        "steps": STEPS,
        "width": width,
        "height": height,
        "cfg_scale": CFG,
        "sampler_name": SAMPLER,
        "batch_size": 1,
        "n_iter": 1,
        "seed": seed,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{DEFAULT_SD_URL}/sdapi/v1/txt2img",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
        answer = json.loads(res.read().decode("utf-8"))
    images = answer.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError("A1111 returned no images")
    return base64.b64decode(images[0])


def fit_png(png: bytes, size: tuple[int, int]) -> bytes:
    """Re-encode as PNG at exactly `size`, so the backend's size check passes."""
    with Image.open(io.BytesIO(png)) as opened:
        opened.load()
        rgb = opened.convert("RGB")
        if rgb.size != size:
            # Center-crop to the target aspect, then scale — same idea as the
            # backend's ImageOps.fit, so faces stay framed.
            from PIL import ImageOps

            rgb = ImageOps.fit(rgb, size, Image.Resampling.LANCZOS)
        out = io.BytesIO()
        rgb.save(out, "PNG")
        return out.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "EidolonImageShim/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        say(fmt % args)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send(200, {"status": "ok", "upstream": DEFAULT_SD_URL})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        # Accept both with and without the /v1 prefix, so either .env spelling works.
        if self.path.rstrip("/") not in ("/images/generations", "/v1/images/generations"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, UnicodeDecodeError) as e:
            self._send(400, {"error": f"bad request: {e}"})
            return

        size = parse_size(body.get("size"))
        t0 = time.time()
        try:
            png = call_sd(body, *size)
            png = fit_png(png, size)
        except urllib.error.HTTPError as e:
            say(f"upstream HTTP {e.code}: {e.read()[:200]!r}")
            self._send(502, {"error": f"A1111 returned {e.code}"})
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            say(f"upstream unreachable: {e}")
            self._send(502, {"error": f"cannot reach A1111 at {DEFAULT_SD_URL}: {e}"})
            return
        except Exception as e:  # noqa: BLE001 - report, never crash the server
            say(f"generation failed: {type(e).__name__}: {e}")
            self._send(500, {"error": str(e)})
            return

        encoded = base64.b64encode(png).decode("ascii")
        say(f"{size[0]}x{size[1]} in {time.time() - t0:.1f}s, {len(png)} bytes")
        self._send(200, {"created": int(time.time()), "data": [{"b64_json": encoded}]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("IMAGE_SHIM_PORT", "8090")))
    parser.add_argument("--sd-url", default=None, help="A1111 base URL")
    args = parser.parse_args()
    if args.sd_url:
        globals()["DEFAULT_SD_URL"] = args.sd_url.rstrip("/")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    say(f"listening on http://127.0.0.1:{args.port}, upstream {DEFAULT_SD_URL}")
    say(f"steps={STEPS} sampler={SAMPLER} cfg={CFG} timeout={TIMEOUT}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        say("stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
