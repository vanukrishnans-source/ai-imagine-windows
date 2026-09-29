"""Model store for AI Imagine — resumable downloads from the ai-imagine-models GitHub release,
SHA-256 verification, safety-checker unpack (same .wts rebuild as AI Image Create)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIRRORS = [
    "https://github.com/vanukrishnans-source/ai-imagine-models/releases/download/models-v1/",
]
REPO_URL = "https://github.com/vanukrishnans-source/ai-imagine-models"
USER_AGENT = "AIImagine/1.0 (Windows)"

# Release assets (individual files / zips). Sizes + SHA-256 filled by scripts/make_manifest.py
# Placeholder zeros are replaced when the models release is published; the app embeds the final values.


@dataclass(frozen=True)
class Asset:
    name: str          # release asset filename
    label: str
    bytes: int
    sha256: str
    kind: str          # "file" | "zip"
    dest: str          # relative path under models dir (file path or extract folder)


# Populated below after we hash the local copies during packaging. The constants here are the
# expected production values; scripts/make_manifest.py regenerates this block.


def _manifest() -> dict:
    from .resources_path import res
    return json.loads(res("manifest.json").read_text(encoding="utf-8"))


def assets() -> list[Asset]:
    return [Asset(**a) for a in _manifest()["assets"]]


def download_bytes() -> int:
    return int(_manifest()["download_bytes"])


def installed_bytes_estimate() -> int:
    """Rough on-disk size after install (ONNX + safety fp32)."""
    return int(_manifest()["installed_bytes"])


class ChecksumError(IOError):
    pass


class NotEnoughStorage(IOError):
    pass


class Cancelled(Exception):
    pass


def default_dir() -> Path:
    env = os.environ.get("AII_MODELS")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "AIImagine" / "models"


def sha256_file(p: Path, cancel=None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            b = f.read(4 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ssl_ctx():
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:  # noqa: BLE001
        pass
    return ctx


def rebuild_fp32(manifest: dict, blob_path: Path, out_path: Path, cancel=None, progress=None):
    """Same maths as AI Image Create / reference rebuild_fp32."""
    tmp = Path(str(out_path) + ".part")
    blob = np.memmap(blob_path, dtype=np.uint8, mode="r")
    with open(tmp, "wb") as f:
        pos = 0
        for t in manifest["tensors"]:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            pad = t["fp32_offset"] - pos
            if pad < 0:
                raise IOError("tensor table out of order")
            if pad:
                f.write(b"\0" * pad)
            pos = t["fp32_offset"]
            n = t["count"]
            o = t["w_offset"]
            if t["kind"] == "f16":
                a = np.frombuffer(blob[o:o + 2 * n].tobytes(), np.float16).astype(np.float32)
            elif t["kind"] == "q8":
                ns = t["nscale"]
                sc = np.frombuffer(blob[o:o + 4 * ns].tobytes(), np.float32)
                q = np.frombuffer(blob[o + 4 * ns:o + 4 * ns + n].tobytes(), np.int8).astype(np.float32)
                shp = t["shape"]
                if t["axis"] == 1:
                    a = (q.reshape(shp[0], -1) * sc[None, :]).reshape(-1)
                else:
                    a = (q.reshape(shp[0], -1) * sc[:, None]).reshape(-1)
            else:
                raise IOError(f"unknown tensor kind {t['kind']}")
            f.write(a.astype(np.float32).tobytes())
            pos += 4 * n
            if progress:
                progress(pos / manifest["fp32_bytes"])
        if manifest["fp32_bytes"] > pos:
            f.write(b"\0" * (manifest["fp32_bytes"] - pos))
    del blob
    if tmp.stat().st_size != manifest["fp32_bytes"]:
        raise IOError(f"{manifest['name']}: rebuilt {tmp.stat().st_size} != {manifest['fp32_bytes']}")
    if out_path.exists():
        out_path.unlink()
    os.replace(tmp, out_path)


@dataclass
class Progress:
    stage: str  # download | verify | unpack | done
    done: int
    total: int
    file: str
    bps: float = 0.0
    fraction: float = 0.0


# Required relative paths that must exist when fully installed
REQUIRED_FILES = [
    "text_encoder/model.onnx",
    "text_encoder/model.onnx_data",
    "text_encoder_2/model.onnx",
    "text_encoder_2/model.onnx_data",
    "unet/model.onnx",
    "unet/model.onnx_data",
    "unet/model.onnx_data_1",
    "unet/model.onnx_data_2",
    "vae_decoder/model.onnx",
    "vae_decoder/model.onnx_data",
    "vae_encoder/model.onnx",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "safety.onnx",
    "safety.fp32.bin",
    "safety.ok",
]


class ModelStore:
    def __init__(self, d: Path | None = None, mirrors=None):
        self.dir = Path(d) if d else default_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mirrors = list(mirrors or MIRRORS)

    def path(self, rel: str) -> Path:
        return self.dir / rel

    def is_installed(self) -> bool:
        try:
            ok = self.path("safety.ok")
            if not ok.is_file():
                return False
            for rel in REQUIRED_FILES:
                p = self.path(rel)
                if not p.is_file() or p.stat().st_size < 16:
                    return False
            return True
        except OSError:
            return False

    def missing_labels(self) -> list[str]:
        out = []
        for a in assets():
            if a.kind == "file":
                if not self.path(a.dest).is_file():
                    out.append(a.label)
            else:
                # zip: consider missing if any expected member marker missing — use dest folder existence of a marker
                marker = self.path(a.dest) / ".ok" if a.dest else None
                # For our zips, dest is a folder; check via REQUIRED_FILES instead
                pass
        if not self.is_installed():
            # coarse
            for rel in REQUIRED_FILES:
                if not self.path(rel).is_file():
                    out.append(rel)
        return out

    def bytes_present(self) -> int:
        tot = 0
        for a in assets():
            part = self.dir / (a.name + ".part")
            done = self.dir / a.name
            if done.is_file():
                tot += min(done.stat().st_size, a.bytes)
            elif part.is_file():
                tot += min(part.stat().st_size, a.bytes)
            elif a.kind == "file" and self.path(a.dest).is_file():
                tot += a.bytes
        return tot

    def free_space(self) -> int:
        try:
            return shutil.disk_usage(self.dir).free
        except OSError:
            return -1

    def free_space_needed(self) -> int:
        # download remainder + safety unpack headroom
        have = self.bytes_present()
        return max(0, download_bytes() - have) + 500_000_000

    def clear(self):
        if self.dir.exists():
            for p in self.dir.iterdir():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                except OSError:
                    pass

    def ensure(self, progress=None, cancel: threading.Event | None = None):
        cancel = cancel or threading.Event()
        progress = progress or (lambda p: None)
        if self.is_installed():
            return
        need = self.free_space_needed()
        free = self.free_space()
        if 0 < free < need:
            raise NotEnoughStorage(
                f"Not enough free disk space: setup needs about {need / 1e9:.1f} GB free, "
                f"only {free / 1e9:.1f} GB is available."
            )
        total = download_bytes()
        base = 0
        for a in assets():
            dest_dl = self.dir / a.name
            if a.kind == "file" and self.path(a.dest).is_file() and self.path(a.dest).stat().st_size == a.bytes:
                base += a.bytes
                continue
            if a.kind == "zip" and (self.dir / (a.name + ".extracted")).is_file():
                base += a.bytes
                continue
            self._fetch_file(a.name, a.bytes, a.sha256, dest_dl, base, total, cancel, progress)
            progress(Progress("unpack", base + a.bytes, total, a.name))
            self._install_asset(a, dest_dl, cancel, progress)
            try:
                dest_dl.unlink()
            except OSError:
                pass
            base += a.bytes
        # safety unpack if needed
        if not self.path("safety.fp32.bin").is_file():
            self._unpack_safety(cancel, progress)
        progress(Progress("done", total, total, ""))
        if not self.is_installed():
            raise IOError(f"models still incomplete: {self.missing_labels()[:8]}")

    def _install_asset(self, a: Asset, downloaded: Path, cancel, progress):
        if a.kind == "file":
            dest = self.path(a.dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            shutil.copyfile(downloaded, dest)
            if a.dest.endswith("safety.wts"):
                self._unpack_safety(cancel, progress)
        elif a.kind == "zip":
            dest_dir = self.path(a.dest) if a.dest else self.dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(downloaded) as z:
                z.extractall(dest_dir)
            (self.dir / (a.name + ".extracted")).write_text(a.sha256)
            if "safety" in a.name:
                self._unpack_safety(cancel, progress)

    def _unpack_safety(self, cancel, progress):
        wts = self.path("safety.wts")
        man_path = self.path("safety.json")
        onnx = self.path("safety.onnx")
        if not (wts.is_file() and man_path.is_file() and onnx.is_file()):
            return
        man = json.loads(man_path.read_text(encoding="utf-8"))
        out = self.path("safety.fp32.bin")
        progress(Progress("unpack", download_bytes(), download_bytes(), "safety"))
        rebuild_fp32(man, wts, out, cancel, lambda fr: progress(Progress("unpack", download_bytes(), download_bytes(), "safety", fraction=fr)))
        self.path("safety.ok").write_text(man["wts_sha256"])
        try:
            wts.unlink()
        except OSError:
            pass

    def import_from(self, folder: Path, progress=None, cancel=None):
        folder = Path(folder)
        cancel = cancel or threading.Event()
        progress = progress or (lambda p: None)
        base = 0
        total = download_bytes()
        for a in assets():
            src = folder / a.name
            if not src.is_file():
                # also accept already-extracted layout
                if a.kind == "file" and (folder / a.dest).is_file():
                    dest = self.path(a.dest)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(folder / a.dest, dest)
                    base += a.bytes
                    continue
                continue
            progress(Progress("verify", base, total, a.name))
            if sha256_file(src, cancel) != a.sha256:
                raise ChecksumError(f"{a.name} failed the checksum check")
            self._install_asset(a, src, cancel, progress)
            base += a.bytes
        if not self.path("safety.fp32.bin").is_file():
            # copy safety pieces if present as loose files
            for n in ("safety.wts", "safety.json", "safety.onnx"):
                s = folder / n
                if s.is_file():
                    shutil.copyfile(s, self.path(n))
            self._unpack_safety(cancel, progress)
        return [a.name for a in assets()]

    def _fetch_file(self, asset, size, sha, dest: Path, base, total, cancel, progress):
        part = Path(str(dest) + ".part")
        last_err = None
        for attempt in range(3):
            for m in self.mirrors:
                try:
                    self._download(m + asset, size, part, base, total, cancel, progress, asset)
                    last_err = None
                    break
                except Cancelled:
                    raise
                except (IOError, urllib.error.URLError, TimeoutError) as e:
                    last_err = e
            if last_err is None:
                break
            if cancel.is_set():
                raise Cancelled()
            time.sleep(2 * (attempt + 1))
        if last_err is not None:
            raise IOError(f"download of {asset} failed: {last_err}")
        if part.stat().st_size != size:
            raise IOError(f"{asset}: size {part.stat().st_size} != {size}")
        progress(Progress("verify", base + size, total, asset))
        got = sha256_file(part, cancel)
        if got != sha:
            part.unlink()
            raise ChecksumError(
                f"{asset} failed the checksum check (got {got[:12]}…). "
                "It was deleted — press Retry to download it again."
            )
        if dest.exists():
            dest.unlink()
        os.replace(part, dest)

    def _download(self, url, size, part: Path, base, total, cancel, progress, label):
        have = part.stat().st_size if part.is_file() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            return
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            r = urllib.request.urlopen(req, timeout=60, context=_ssl_ctx())
        except urllib.error.HTTPError as e:
            if e.code == 416:
                part.unlink(missing_ok=True)
            raise IOError(f"HTTP {e.code} for {label}") from e
        with r:
            code = r.status
            append = code == 206 and have > 0
            if code == 200:
                have = 0
            elif not append:
                raise IOError(f"HTTP {code} for {label}")
            with open(part, "ab" if append else "wb") as out:
                got = have
                t0 = time.time()
                start = have
                last = 0.0
                while True:
                    if cancel.is_set():
                        raise Cancelled()
                    b = r.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
                    got += len(b)
                    if got > size:
                        raise IOError(f"{label}: server sent more data than expected")
                    now = time.time()
                    if now - last > 0.25:
                        last = now
                        dt = now - t0
                        progress(Progress("download", base + got, total, label, (got - start) / dt if dt > 0 else 0.0))
