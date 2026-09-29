"""Download ai-imagine-models release assets into a cache folder (same URLs + SHA-256 as the app)."""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aii.models import MIRRORS, USER_AGENT, assets, sha256_file  # noqa: E402


def main(dest: Path):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for a in assets():
        out = dest / a.name
        if out.is_file() and out.stat().st_size == a.bytes and sha256_file(out) == a.sha256:
            print(f"OK cached {a.name}")
            continue
        part = Path(str(out) + ".part")
        url = MIRRORS[0] + a.name
        print(f"GET {url}")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        have = part.stat().st_size if part.is_file() else 0
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=120) as r:
            mode = "ab" if r.status == 206 and have else "wb"
            if r.status == 200:
                have = 0
            with open(part, mode) as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        if part.stat().st_size != a.bytes:
            raise SystemExit(f"size mismatch {a.name}: {part.stat().st_size} != {a.bytes}")
        got = sha256_file(part)
        if got != a.sha256:
            part.unlink()
            raise SystemExit(f"sha mismatch {a.name}: {got} != {a.sha256}")
        part.replace(out)
        print(f"OK {a.name}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ".release-cache"))
