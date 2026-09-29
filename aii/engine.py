"""ONNX Runtime sessions: DirectML for UNet + VAE when available, CPU fallback. Safety always on CPU."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as M

log = logging.getLogger("aii")

# Models that prefer DirectML (heavy). Text encoders + safety stay on CPU for parity / low overhead.
GPU_MODELS = ("unet", "vae_decoder")
CPU_MODELS = ("text_encoder", "text_encoder_2", "safety")


@dataclass
class DeviceInfo:
    requested: str = "auto"
    active: str = "CPU"
    adapter: str = ""
    fallback_reason: str = ""
    per_model: dict = field(default_factory=dict)

    def label(self):
        if self.active == "DirectML":
            return "GPU · DirectML" + (f" · {self.adapter}" if self.adapter else "")
        return "CPU" + (" (GPU unavailable)" if self.fallback_reason else "")


def gpu_adapter_name() -> str:
    if os.name != "nt":
        return ""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '"],
            capture_output=True, text=True, timeout=15, creationflags=0x08000000,
        ).stdout.strip()
        return out
    except Exception:  # noqa: BLE001
        return ""


class Engine:
    def __init__(self, store: M.ModelStore, device: str = "auto", threads: int = 0):
        import onnxruntime as ort
        self.ort = ort
        self.store = store
        self.device = device
        self.threads = threads or int(os.environ.get("ORT_THREADS", "0") or 0)
        self.lock = threading.RLock()
        self.sessions = {}
        self.timing = {}
        self.info = DeviceInfo(requested=device)
        self.dml_available = "DmlExecutionProvider" in ort.get_available_providers()
        if device in ("auto", "dml") and not self.dml_available:
            self.info.fallback_reason = "DirectML is not available in this onnxruntime build"
        self.info.active = "DirectML" if device in ("auto", "dml") and self.dml_available else "CPU"
        if self.info.active == "DirectML":
            self.info.adapter = gpu_adapter_name().split(";")[0].strip()

    def _options(self, dml: bool):
        so = self.ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if dml:
            so.enable_mem_pattern = False
            so.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            so.enable_cpu_mem_arena = False
            so.inter_op_num_threads = 1
            if self.threads:
                so.intra_op_num_threads = self.threads
        return so

    def _onnx_path(self, name: str) -> Path:
        if name == "safety":
            return self.store.path("safety.onnx")
        return self.store.path(name) / "model.onnx"

    def _fallback(self, name, e):
        if self.device == "dml":
            raise e
        log.warning("DirectML failed for %s: %s -> CPU", name, e)
        self.info.fallback_reason = f"{type(e).__name__}: {str(e)[:200]}"
        if all(self.info.per_model.get(m) != "DirectML" for m in GPU_MODELS if m != name):
            self.info.active = "CPU"

    def _open(self, name: str, force_cpu=False):
        path = str(self._onnx_path(name))
        if not Path(path).is_file():
            raise FileNotFoundError(f"model not installed: {name} ({path})")
        t0 = time.time()
        # Safety always CPU. Text encoders: CPU (fast enough, avoids DML quirks with many outputs).
        want_dml = (not force_cpu) and name in GPU_MODELS and self.info.active == "DirectML"
        if want_dml:
            try:
                sess = self.ort.InferenceSession(
                    path, self._options(True),
                    providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
                )
                used = sess.get_providers()
                if not used or used[0] != "DmlExecutionProvider":
                    raise RuntimeError(f"DirectML provider not used (got {used})")
                self.info.per_model[name] = "DirectML"
                self.timing["load_" + name] = time.time() - t0
                return sess
            except Exception as e:  # noqa: BLE001
                self._fallback(name, e)
        sess = self.ort.InferenceSession(path, self._options(False), providers=["CPUExecutionProvider"])
        self.info.per_model[name] = "CPU"
        self.timing["load_" + name] = time.time() - t0
        return sess

    def session(self, name):
        with self.lock:
            if name not in self.sessions:
                self.sessions[name] = self._open(name)
            return self.sessions[name]

    def run(self, name, feeds, output_names=None):
        sess = self.session(name)
        with self.lock:
            t0 = time.time()
            try:
                if output_names:
                    out = sess.run(output_names, feeds)
                else:
                    out = sess.run(None, feeds)
            except Exception as e:  # noqa: BLE001
                if self.info.per_model.get(name) != "DirectML":
                    raise
                self._fallback(name, e)
                self.sessions[name] = sess = self._open(name, force_cpu=True)
                out = sess.run(output_names, feeds) if output_names else sess.run(None, feeds)
            self.timing[name] = self.timing.get(name, 0.0) + time.time() - t0
            return out

    def warm_up(self, names=("text_encoder", "text_encoder_2", "unet", "vae_decoder", "safety"), cb=None):
        for i, n in enumerate(names):
            if cb:
                cb(n, i, len(names))
            self.session(n)

    def close(self):
        with self.lock:
            self.sessions.clear()
