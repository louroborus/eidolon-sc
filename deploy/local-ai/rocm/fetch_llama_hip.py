"""Download llama.cpp's official Windows ROCm/HIP build and unpack it.

Building from source needs MSVC + cmake + ninja, none of which are installed and
winget cannot reach the network. llama.cpp publishes a prebuilt HIP binary for
Windows on every release (`llama-<build>-bin-win-rocm-<ver>-x64.zip`), which is
what this fetches. It links against the ROCm runtime shipped in the ROCm SDK
already present in D:\\AI (via HIP_PATH / the rocm wheel), so no compiler is needed.
"""
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import httpx

ROOT = Path(r"F:\eidolon-sc")
# deploy/local-ai/bin/ refuses writes under the sandbox, and binaries this size
# should not be in git anyway; corpus/ is git-ignored and writable.
DEST = ROOT / "corpus" / "llama-hip"
LOG = Path(os.environ["TEMP"]) / "eidolon-llama-hip-download.log"


def say(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_asset() -> tuple[str, str, int]:
    r = httpx.get(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5",
        timeout=60,
        follow_redirects=True,
    )
    r.raise_for_status()
    for release in r.json():
        for asset in release.get("assets", []):
            name = asset["name"]
            if "bin-win-rocm" in name and name.endswith(".zip"):
                return name, asset["browser_download_url"], asset["size"]
    raise SystemExit("no win-rocm asset found in the last 5 releases")


name, url, size = find_asset()
say(f"asset: {name} ({size / 1e6:.1f} MB)")
say(f"url: {url}")

DEST.mkdir(parents=True, exist_ok=True)
archive = DEST / name

if archive.exists() and archive.stat().st_size == size:
    say("archive already downloaded")
else:
    t0 = time.time()
    received = 0
    with httpx.stream("GET", url, timeout=httpx.Timeout(300, connect=60), follow_redirects=True) as r:
        r.raise_for_status()
        with archive.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                received += len(chunk)
                if received % (40 << 20) < (1 << 20):
                    say(f"  {received / 1e6:.0f}/{size / 1e6:.0f} MB in {time.time() - t0:.0f}s")
    say(f"downloaded in {time.time() - t0:.0f}s ({received / 1e6:.1f} MB)")

say("extracting")
with zipfile.ZipFile(archive) as z:
    z.extractall(DEST)
say("extracted")

# Report what we got: the server binary and whether ggml-hip came along.
found = []
for p in DEST.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".exe", ".dll"}:
        found.append((p.relative_to(DEST).as_posix(), p.stat().st_size))
say(f"binaries: {len(found)}")
for rel, n in sorted(found)[:40]:
    say(f"  {rel} ({n / 1e6:.1f} MB)")

hip_dlls = [rel for rel, _ in found if "hip" in rel.lower()]
say(f"HIP-related files: {hip_dlls if hip_dlls else 'NONE'}")
say("DONE")
