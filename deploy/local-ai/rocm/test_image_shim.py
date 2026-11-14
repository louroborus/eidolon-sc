"""Contract test for image_shim: does it return exactly what eidolon expects?

Mocks A1111's /sdapi/v1/txt2img (so no GPU/ROCm needed), starts the shim on a
test port, sends the OpenAI-shaped request the backend really sends, and checks
the answer with the backend's own parsing rules (data[0].b64_json, 832x1216).
"""
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

REPO = Path(r"F:\eidolon-sc")
sys.path.insert(0, str(REPO / "deploy" / "local-ai"))

# --- mock A1111 -----------------------------------------------------------
class MockSD(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length))
        # A1111 returns a picture at whatever size it was asked for.
        w, h = payload["width"], payload["height"]
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (120, 140, 200)).save(buf, "PNG")
        answer = json.dumps({"images": [base64.b64encode(buf.getvalue()).decode()],
                             "parameters": payload}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)


mock = ThreadingHTTPServer(("127.0.0.1", 0), MockSD)
mock_port = mock.server_address[1]
threading.Thread(target=mock.serve_forever, daemon=True).start()

port = 18090
env = dict(os.environ)
env["SD_WEBUI_URL"] = f"http://127.0.0.1:{mock_port}"
env["PYTHONIOENCODING"] = "utf-8"
proc = subprocess.Popen(
    [sys.executable, str(REPO / "deploy" / "local-ai" / "image_shim.py"), "--port", str(port)],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
try:
    for _ in range(40):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            break
        except Exception:
            time.sleep(0.25)
    else:
        raise SystemExit("shim did not start")

    # Exactly the body eidolon's openai_request() builds.
    body = {
        "model": "local-sd",
        "prompt": "1girl, smile, outdoors",
        "negative_prompt": "nsfw, lowres",
        "size": "832x1216",
        "n": 1,
        "response_format": "b64_json",
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/images/generations",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer fake"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        answer = json.loads(res.read())
        print("status:", res.status)

    # The backend's checks.
    data = answer.get("data")
    assert isinstance(data, list) and len(data) == 1, f"data shape wrong: {answer}"
    encoded = data[0].get("b64_json")
    assert isinstance(encoded, str), "no b64_json"
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as img:
        print("decoded image:", img.size, img.format)
        assert img.size == (832, 1216), f"size mismatch: {img.size}"

    # GPT Image's moments body has no negative_prompt / response_format.
    gpt = {"model": "local-sd", "prompt": "a street at dusk", "size": "832x1216",
           "quality": "medium", "n": 1}
    req2 = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/images/generations",
        data=json.dumps(gpt).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer fake"},
        method="POST",
    )
    with urllib.request.urlopen(req2, timeout=60) as res:
        a2 = json.loads(res.read())
    assert Image.open(io.BytesIO(base64.b64decode(a2["data"][0]["b64_json"]))).size == (832, 1216)
    print("GPT-image-shaped request: OK")
    print("\nCONTRACT TEST PASSED")
finally:
    proc.terminate()
    mock.shutdown()
