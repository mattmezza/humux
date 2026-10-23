"""Real-engine HTTP smoke test; also runs inside the image with networking disabled."""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sample = Path(sys.argv[1]).read_bytes()
base_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
for attempt in range(60):
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            assert json.load(response)["model"] == "whistle"
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(0.5)
else:
    raise RuntimeError("Whistle did not become healthy")

boundary = "whistle-smoke"
body = (
    (
        f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
        f"whisper-1\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="audio.ogg"\r\n'
        "Content-Type: audio/ogg\r\n\r\n"
    ).encode()
    + sample
    + f"\r\n--{boundary}--\r\n".encode()
)
request = urllib.request.Request(
    f"{base_url}/v1/audio/transcriptions",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
start = time.perf_counter()
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
text = result["text"].lower()
for digit, word in [("0", "zero"), ("1", "one"), ("2", "two")]:
    text = text.replace(digit, word)
words = "".join(char if char.isalpha() else " " for char in text).split()
assert words == ["zero", "one", "two"], result
print(f"Whistle: {result['text']!r} in {time.perf_counter() - start:.3f}s")
