"""Download the Smart Turn v3 ONNX weights into ./models.

Run from the repo root:

    python models/download_smart_turn.py

The media worker resolves SMART_TURN_MODEL_PATH (default
`models/smart-turn-v3.1.onnx`); this script saves there. If the download fails or
the file is missing at runtime, the worker still runs — it just skips end-of-turn
detection and treats every VAD pause as a complete turn.

The exported filename has drifted across Smart Turn releases, so we try a few
known URLs and save the first that works. Override with SMART_TURN_URL if the
weights move again.
"""
import os
import sys
from pathlib import Path

import requests

TARGET = Path(__file__).resolve().parent / "smart-turn-v3.1.onnx"

# Tried in order. First hit wins. The v3.1 export is the int8 CPU model; we run
# on CPU. (Files are named *-cpu.onnx / *-gpu.onnx in the repo.)
CANDIDATE_URLS = [
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.1-cpu.onnx",
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.0.onnx",
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
]


def _download(url: str, dest: Path) -> bool:
    try:
        print(f"trying {url}")
        with requests.get(url, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                print(f"  -> HTTP {resp.status_code}")
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            tmp.replace(dest)
        print(f"  -> saved {dest} ({dest.stat().st_size} bytes)")
        return True
    except Exception as e:  # network / TLS / disk
        print(f"  -> failed: {e}")
        return False


def main() -> int:
    urls = [os.environ["SMART_TURN_URL"]] if os.environ.get("SMART_TURN_URL") else CANDIDATE_URLS
    for url in urls:
        if _download(url, TARGET):
            return 0
    print(
        "\nCould not download Smart Turn weights. Set SMART_TURN_URL to a valid "
        "ONNX URL, or place the file at models/smart-turn-v3.1.onnx yourself.\n"
        "The interview still runs without it (no end-of-turn refinement).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
