"""AI Imagine — Qt GUI for Windows, touch-first for the ROG Ally X (7" 1080p, 150% scaling).

Two modes: (1) text-to-image from a prompt alone, (2) optional photo + prompt remake (img2img)
with a strength slider. Aspect presets, quality/speed, Create. Options: negative prompt,
safety strictness, size, processor. Result: Save / Copy / Regenerate / Edit prompt.
NSFW safety filter is always on — Relaxed/Standard only; no way to disable.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QStandardPaths, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from .. import models as M
from ..pipeline import Cancelled as GenCancelled, GenParams, GenResult, load_photo, new_seed, prepare_photo
from .. import sdxl as S
from .theme import QSS

log = logging.getLogger("aii")
PAGE_SETUP, PAGE_MAIN, PAGE_PROGRESS, PAGE_RESULT, PAGE_BLOCKED = range(5)


def qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()


def pix(rgb, w, h):
    dpr = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    p = QPixmap.fromImage(qimage(rgb)).scaled(
        int(w * dpr), int(h * dpr), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    p.setDevicePixelRatio(dpr)
    return p


def pictures_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation) or str(Path.home() / "Pictures")
    return Path(base) / "AIImagine"


class Worker(QThread):
    progressed = Signal(object)
    done = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancel = threading.Event()

    def run(self):
        try:
            self.done.emit(self.fn(self.cancel, self.progressed.emit))
        except (M.Cancelled, GenCancelled):
            self.failed.emit("cancelled", "")
        except Exception as e:  # noqa: BLE001
            if self.cancel.is_set():
                self.failed.emit("cancelled", "")
            else:
                self.failed.emit(f"{e}", traceback.format_exc())


def button(text, kind=None, min_w=0, tip=None):
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if min_w:
        b.setMinimumWidth(min_w)
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def label(text="", kind=None, wrap=False):
    l = QLabel(text)
    if kind:
        l.setObjectName(kind)
    l.setWordWrap(wrap)
    return l


def card():
    f = QFrame()
    f.setObjectName("card")
    return f


class Segmented(QWidget):
    changed = Signal(object)

    def __init__(self, items):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for text, value in items:
            b = button(text, "seg")
            b.setCheckable(True)
            b.setMinimumHeight(50)
            self.group.addButton(b)
            lay.addWidget(b, 1)
            self.buttons[value] = b
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))

    def set(self, value):
        if value in self.buttons:
            self.buttons[value].setChecked(True)

    def value(self):
        for v, b in self.buttons.items():
            if b.isChecked():
                return v



class PhotoSlot(QFrame):
    """Optional photo for remake mode. Tap / drop / paste. Empty = text-to-image only."""
    clicked = Signal()
    dropped = Signal(object)
    cleared = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label("Photo (optional)", "section"))
        top.addStretch(1)
        self.clear_btn = button("Clear", tip="Remove photo — back to text-to-image")
        self.open_btn = button("Open…")
        top.addWidget(self.clear_btn)
        top.addWidget(self.open_btn)
        lay.addLayout(top)
        self.image = QLabel("No photo — prompt only\n\nTap to attach a photo\nto remake it with your prompt")
        self.image.setObjectName("slot")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(180, 180)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.image, 1)
        self.info = label("Text-to-image mode", "hint", True)
        lay.addWidget(self.info)
        self.rgb = None
        self.open_btn.clicked.connect(self.clicked.emit)
        self.clear_btn.clicked.connect(self._clear)

    def _clear(self):
        self.rgb = None
        self.image.setPixmap(QPixmap())
        self.image.setText("No photo — prompt only\n\nTap to attach a photo\nto remake it with your prompt")
        self.info.setText("Text-to-image mode")
        self.cleared.emit()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() or md.hasImage():
            e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and md.urls():
            self.dropped.emit(md.urls()[0].toLocalFile())
        elif md.hasImage():
            self.dropped.emit(QImage(md.imageData()))

    def set_image(self, rgb):
        self.rgb = rgb
        self._render()
        self.info.setText(f"Photo remake · {rgb.shape[1]}×{rgb.shape[0]}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def _render(self):
        if self.rgb is not None:
            self.image.setPixmap(pix(self.rgb, max(80, self.image.width() - 8), max(80, self.image.height() - 8)))


class MainWindow(QMainWindow):
    def __init__(self, store: M.ModelStore, device="auto", settings: QSettings | None = None):
        super().__init__()
        self.store = store
        self.qs = settings or QSettings("vanu", "AIImagine")
        self.device = device if device in ("cpu", "dml") else self._get("device", "auto", ("auto", "cpu"))
        self.pipe = None
        self.engine = None
        self.ready = threading.Event()
        self.warm = None
        self.worker = None
        self.photo = None
        self.photo_name = ""
        self.result: GenResult | None = None
        self.last_params: GenParams | None = None
        self.saved_path = None
        self.sec_per_eval = None
        self.setWindowTitle(f"AI Imagine {__version__}")
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._topbar())
        self.stack = QStackedWidget()
        v.addWidget(self.stack, 1)
        for page in (self._page_setup(), self._page_main(), self._page_progress(), self._page_result(), self._page_blocked()):
            self.stack.addWidget(page)
        self.setCentralWidget(central)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save)
        QShortcut(QKeySequence.StandardKey.Paste, self, activated=self._paste_photo)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._pick_photo)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._create)
        self._load_options()
        if store.is_installed():
            self.go(PAGE_MAIN)
            self._start_warmup()
        else:
            self.go(PAGE_SETUP)
        self._set_chip()

    def _get(self, key, default, allowed=None):
        v = self.qs.value(key, default)
        if isinstance(default, bool):
            v = str(v).lower() in ("1", "true", "yes")
        elif isinstance(default, int):
            v = int(v) if str(v).lstrip("-").isdigit() else default
        elif isinstance(default, float):
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = default
        if allowed is not None and v not in allowed:
            v = default
        return v

    def _topbar(self):
        bar = QFrame()
        bar.setObjectName("topbar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 6, 12, 6)
        lay.addWidget(label("AI Imagine", "apptitle"))
        lay.addSpacing(12)
        self.chip = label("…", "chip")
        lay.addWidget(self.chip)
        lay.addStretch(1)
        about = button("About")
        about.clicked.connect(self._about)
        lay.addWidget(about)
        return bar

    def _set_chip(self, text=None):
        if text:
            self.chip.setText(text)
            self.chip.setProperty("state", "")
        elif self.engine is None:
            dml = False
            try:
                import onnxruntime as ort
                dml = "DmlExecutionProvider" in ort.get_available_providers()
            except Exception:  # noqa: BLE001
                pass
            self.chip.setText("Processor: GPU (DirectML) when ready" if dml and self.device != "cpu" else "Processor: CPU")
            self.chip.setProperty("state", "" if dml and self.device != "cpu" else "cpu")
        else:
            i = self.engine.info
            if i.active == "DirectML":
                self.chip.setText("GPU · DirectML" + (f" · {i.adapter}" if i.adapter else ""))
                self.chip.setProperty("state", "")
            else:
                self.chip.setText("CPU" + (" (GPU unavailable)" if i.fallback_reason and i.requested != "cpu" else ""))
                self.chip.setProperty("state", "cpu")
            self.chip.setToolTip(i.fallback_reason or "")
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)

    def go(self, page):
        self.stack.setCurrentIndex(page)
        if page == PAGE_MAIN:
            self._refresh_main()
            self.prompt.setFocus()
        focus = {
            PAGE_SETUP: self.btn_dl, PAGE_PROGRESS: self.btn_cancel,
            PAGE_RESULT: self.btn_save, PAGE_BLOCKED: self.btn_blk_edit,
        }.get(page)
        if focus is not None and focus.isEnabled():
            focus.setFocus()

    def _about(self):
        QMessageBox.about(
            self, "About AI Imagine",
            f"<b>AI Imagine {__version__}</b> for Windows (x64) by vanu krishnan.<br><br>"
            "Type a prompt (and optionally attach a photo to remake it) — entirely on this PC (nothing is uploaded).<br><br>"
            "<b>Model</b>: ByteDance SDXL-Lightning 4-step (CreativeML Open RAIL++-M) + SDXL base components "
            "(Open RAIL++-M) + madebyollin SDXL VAE fp16-fix (MIT) + Stable Diffusion safety checker "
            "(CreativeML OpenRAIL-M). ONNX export redistributed under the same licences.<br><br>"
            "The safety filter is always on. Uses ONNX Runtime + DirectML (MIT), Qt 6 / PySide6 (LGPLv3), "
            "Pillow, NumPy. See THIRD_PARTY.md next to the app."
        )

    # ---- setup ----
    def _page_setup(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(40, 22, 40, 22)
        outer.setSpacing(12)
        dl = M.download_bytes()
        outer.addWidget(label(f"One-time setup: download the AI models (~{dl / 1e9:.1f} GB)", "title"))
        outer.addWidget(label(
            "Models come from the public GitHub release “ai-imagine-models / models-v1”, are checked with "
            "SHA-256 and unpacked into your user folder. You can pause and resume, or import from a folder.",
            "subtitle", True))
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 14, 18, 14)
        rows = [
            ("SDXL-Lightning UNet (4-step)", "~5.1 GB"),
            ("Text encoders (CLIP-L + OpenCLIP-G)", "~1.6 GB"),
            ("VAE decoder", "~99 MB"),
            ("Tokenizers", "~6 MB"),
            ("Safety filter (NSFW classifier)", "~86 MB packed → ~345 MB"),
        ]
        for i, (name, size) in enumerate(rows):
            g.addWidget(label(name), i, 0)
            g.addWidget(label(size, "hint"), i, 1, Qt.AlignmentFlag.AlignRight)
        g.setColumnStretch(0, 1)
        outer.addWidget(c)
        self.setup_bar = QProgressBar()
        self.setup_bar.setRange(0, 1000)
        outer.addWidget(self.setup_bar)
        self.setup_status = label("", "subtitle", True)
        outer.addWidget(self.setup_status)
        outer.addWidget(label(
            "By downloading you accept the model licences (CreativeML Open RAIL++-M / OpenRAIL-M use "
            "restrictions: no illegal, harmful or non-consensual content). The safety filter is always on.",
            "hint", True))
        outer.addStretch(1)
        row = QHBoxLayout()
        self.btn_import = button("Import from folder…", tip="Use the release files copied from another PC")
        self.btn_import.clicked.connect(self._import_models)
        row.addWidget(self.btn_import)
        row.addStretch(1)
        self.btn_pause = button("Pause")
        self.btn_pause.clicked.connect(self._cancel_worker)
        self.btn_pause.setEnabled(False)
        row.addWidget(self.btn_pause)
        self.btn_dl = button("Download", "primary", 260)
        self.btn_dl.clicked.connect(self._download)
        row.addWidget(self.btn_dl)
        outer.addLayout(row)
        self._refresh_setup()
        return w

    def _refresh_setup(self):
        have = self.store.bytes_present()
        tot = max(1, M.download_bytes())
        self.setup_bar.setValue(int(1000 * have / tot))
        if have and not self.store.is_installed():
            self.btn_dl.setText("Resume")
            self.setup_status.setText(f"{have / 1e6:,.0f} of {tot / 1e6:,.0f} MB already here.")

    def _download(self, import_dir=None):
        if self.worker and self.worker.isRunning():
            return
        self.btn_dl.setEnabled(False)
        self.btn_import.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.setup_status.setText("Connecting…")
        store = self.store

        def work(cancel, emit):
            cb = lambda p: emit(("dl", p))
            if import_dir:
                store.import_from(Path(import_dir), cb, cancel)
            store.ensure(cb, cancel)
            return True

        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._setup_progress)
        self.worker.done.connect(self._setup_done)
        self.worker.failed.connect(self._setup_failed)
        self.worker.start()

    def _setup_progress(self, d):
        kind, p = d
        self.setup_bar.setValue(int(1000 * p.done / max(1, p.total)))
        if p.stage == "download":
            eta = (p.total - p.done) / p.bps if p.bps > 0 else None
            self.setup_status.setText(
                f"Downloading {p.file} · {p.done / 1e6:,.0f} of {p.total / 1e6:,.0f} MB"
                + (f" · {p.bps / 1e6:.1f} MB/s · about {int(eta // 60)} min {int(eta % 60)} s left" if eta else "")
            )
        elif p.stage == "verify":
            self.setup_status.setText(f"Checking {p.file} (SHA-256)…")
        elif p.stage == "unpack":
            self.setup_status.setText(f"Unpacking {p.file}… {p.fraction * 100:.0f}%")

    def _setup_done(self, _):
        self.btn_pause.setEnabled(False)
        self.btn_dl.setEnabled(True)
        self.btn_import.setEnabled(True)
        self._refresh_setup()
        self.setup_status.setText("All set.")
        self.go(PAGE_MAIN)
        self._start_warmup()

    def _setup_failed(self, msg, tb):
        if tb:
            log.error("setup failed: %s", tb)
        self.btn_dl.setEnabled(True)
        self.btn_import.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self._refresh_setup()
        if msg == "cancelled":
            self.setup_status.setText("Paused — tap Resume to continue where it stopped.")
            self.btn_dl.setText("Resume")
        else:
            self.setup_status.setText(f"Setup stopped: {msg}\nCheck the internet connection / free disk space and tap Resume.")

    def _import_models(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with the model release files")
        if d:
            self._download(import_dir=d)

    # ---- warmup ----
    def _start_warmup(self):
        if self.warm is not None:
            return
        self.ready.clear()
        store = self.store

        def work(cancel, emit):
            from ..engine import Engine
            from ..pipeline import Pipeline
            eng = Engine(store, self.device)
            pipe = Pipeline(eng)
            self.engine, self.pipe = eng, pipe
            emit(("chip", None))
            try:
                eng.warm_up(cb=lambda n, i, t: emit(("warm", (n, i, t))))
            finally:
                self.ready.set()
            return True

        self.warm = Worker(work, self)
        self.warm.progressed.connect(self._warm_progress)
        self.warm.done.connect(lambda _: (self._set_chip(), self._refresh_main()))
        self.warm.failed.connect(lambda m, tb: (log.error("warm-up failed: %s %s", m, tb), self.ready.set(), self._set_chip()))
        self.warm.start()

    def _warm_progress(self, d):
        kind, v = d
        if kind == "chip":
            self._set_chip()
        elif kind == "warm":
            self.status_hint.setText(f"Loading AI models… ({v[1] + 1}/{v[2]})")

    # ---- main ----
    def _page_main(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(28, 16, 28, 16)
        lay.setSpacing(12)
        lay.addWidget(label("What do you want to see?", "title"))
        # Photo (optional) + prompt side by side on wide screens
        mid = QHBoxLayout(); mid.setSpacing(12)
        self.slot = PhotoSlot()
        self.slot.setMaximumWidth(320)
        self.slot.clicked.connect(self._pick_photo)
        self.slot.dropped.connect(self._open_any)
        self.slot.cleared.connect(self._photo_cleared)
        mid.addWidget(self.slot, 2)
        right = QVBoxLayout(); right.setSpacing(10)
        pc = card()
        pl = QVBoxLayout(pc)
        pl.setContentsMargins(14, 12, 14, 12)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(
            "e.g. a cozy cabin in the snowy woods at dusk, cinematic lighting, highly detailed\n"
            "a watercolor painting of a red bicycle under cherry blossoms\n"
            "anime-style cat astronaut floating above Earth"
        )
        self.prompt.setMinimumHeight(160)
        self.prompt.textChanged.connect(self._refresh_main)
        pl.addWidget(self.prompt)
        right.addWidget(pc, 3)
        # Strength (only meaningful with a photo)
        self.strength_card = card()
        sl = QVBoxLayout(self.strength_card); sl.setContentsMargins(12, 8, 12, 8); sl.setSpacing(0)
        stop = QHBoxLayout(); stop.addWidget(label("How much to change", "section")); stop.addStretch(1)
        self.s_val = label("0.65", "section"); stop.addWidget(self.s_val); sl.addLayout(stop)
        self.s_strength = QSlider(Qt.Orientation.Horizontal)
        self.s_strength.setRange(30, 95); self.s_strength.setValue(65)
        self.s_strength.valueChanged.connect(self._strength_changed)
        sl.addWidget(self.s_strength)
        ends = QHBoxLayout(); ends.addWidget(label("keep photo", "hint")); ends.addStretch(1); ends.addWidget(label("follow prompt", "hint"))
        sl.addLayout(ends)
        self.o_likeness = QCheckBox("Keep likeness (lower change)"); self.o_likeness.setChecked(True)
        sl.addWidget(self.o_likeness)
        right.addWidget(self.strength_card)
        self.strength_card.setVisible(False)
        mid.addLayout(right, 5)
        lay.addLayout(mid, 4)

        row = QHBoxLayout()
        row.addWidget(label("Aspect", "section"))
        self.o_aspect = Segmented([("1:1", "1:1"), ("16:9", "16:9"), ("9:16", "9:16"), ("4:3", "4:3"), ("3:4", "3:4")])
        row.addWidget(self.o_aspect, 1)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(label("Quality", "section"))
        self.o_quality = Segmented([("Best", "best"), ("Fast", "fast")])
        row2.addWidget(self.o_quality, 1)
        lay.addLayout(row2)

        self.btn_opts = button("▸  Options", "toggle")
        self.btn_opts.setCheckable(True)
        self.btn_opts.toggled.connect(self._toggle_options)
        lay.addWidget(self.btn_opts)
        self.opts_scroll = QScrollArea()
        self.opts_scroll.setWidgetResizable(True)
        self.opts_scroll.setWidget(self._options_panel())
        self.opts_scroll.setVisible(False)
        self.opts_scroll.setMinimumHeight(220)
        lay.addWidget(self.opts_scroll, 2)

        self.status_hint = label("", "hint", True)
        lay.addWidget(self.status_hint)
        self.btn_create = button("Create", "primary")
        self.btn_create.setMinimumHeight(64)
        self.btn_create.clicked.connect(self._create)
        lay.addWidget(self.btn_create)
        return w

    def _options_panel(self):
        c = card()
        grid = QHBoxLayout(c)
        grid.setContentsMargins(14, 10, 14, 12)
        grid.setSpacing(18)
        cols = [QVBoxLayout(), QVBoxLayout()]
        for col in cols:
            col.setSpacing(4)
            grid.addLayout(col, 1)

        def row(ci, title, widget, hint=None):
            cols[ci].addWidget(label(title, "section"))
            cols[ci].addWidget(widget)
            if hint:
                cols[ci].addWidget(label(hint, "hint", True))

        cols[0].addWidget(label("Negative prompt (optional)", "section"))
        self.o_negative = QPlainTextEdit()
        self.o_negative.setPlaceholderText("Things to avoid (nudity terms are always added automatically)")
        self.o_negative.setMaximumHeight(90)
        cols[0].addWidget(self.o_negative)
        self.o_size = Segmented([("Standard", "standard"), ("Large", "large")])
        row(0, "Output size", self.o_size, "Standard ≈ 512–768 px. Large ≈ 768–1024 px — more detail, slower.")
        self.o_strict = Segmented([("Relaxed", "relaxed"), ("Standard", "standard")])
        row(1, "Safety filter strictness", self.o_strict,
            "Always on. Relaxed (default) blocks clear nudity; Standard also blocks borderline pictures.")
        cols[1].addWidget(label("Seed", "section"))
        self.o_random = QCheckBox("New random seed each time")
        cols[1].addWidget(self.o_random)
        self.o_seed = QSpinBox()
        self.o_seed.setRange(1, 2 ** 31 - 1)
        self.o_seed.setMinimumWidth(170)
        cols[1].addWidget(self.o_seed)
        self.o_device = Segmented([("Auto (GPU)", "auto"), ("CPU only", "cpu")])
        row(1, "Processor", self.o_device, "Applies after restarting the app.")
        for col in cols:
            col.addStretch(1)
        for sgn in (self.o_aspect.changed, self.o_quality.changed, self.o_size.changed, self.o_strict.changed, self.o_device.changed):
            sgn.connect(lambda _v: self._save_options())
        self.o_random.toggled.connect(lambda v: (self.o_seed.setEnabled(not v), self._save_options()))
        self.o_seed.valueChanged.connect(lambda _v: self._save_options())
        self.o_negative.textChanged.connect(lambda: self._save_options())
        return c

    def _toggle_options(self, on):
        self.opts_scroll.setVisible(on)
        self.btn_opts.setText("▾  Options" if on else "▸  Options")

    def _load_options(self):
        self.o_aspect.set(self._get("aspect", "1:1", tuple(S.ASPECTS)))
        self.o_quality.set(self._get("quality", "best", ("best", "fast")))
        self.o_size.set(self._get("size", "standard", ("standard", "large")))
        self.o_strict.set(self._get("strictness", "relaxed", ("relaxed", "standard")))
        self.o_device.set(self._get("device", "auto", ("auto", "cpu")))
        self.o_random.setChecked(self._get("random_seed", True))
        self.o_seed.setValue(int(self._get("seed", 1234)))
        self.o_seed.setEnabled(not self.o_random.isChecked())
        self.o_negative.setPlainText(str(self.qs.value("negative", "") or ""))
        self.prompt.setPlainText(str(self.qs.value("prompt", "") or ""))

    def _save_options(self):
        self.qs.setValue("aspect", self.o_aspect.value() or "1:1")
        self.qs.setValue("quality", self.o_quality.value() or "best")
        self.qs.setValue("size", self.o_size.value() or "standard")
        self.qs.setValue("strictness", self.o_strict.value() or "relaxed")
        self.qs.setValue("device", self.o_device.value() or "auto")
        self.qs.setValue("random_seed", self.o_random.isChecked())
        self.qs.setValue("seed", self.o_seed.value())
        self.qs.setValue("negative", self.o_negative.toPlainText())
        self.qs.setValue("prompt", self.prompt.toPlainText())

    def _refresh_main(self):
        ready = self.store.is_installed() and (self.ready.is_set() or self.pipe is not None)
        has = bool(self.prompt.toPlainText().strip())
        self.btn_create.setEnabled(ready and has)
        if not self.store.is_installed():
            self.status_hint.setText("Download the AI models first.")
        elif not self.ready.is_set():
            self.status_hint.setText("Loading AI models…")
        elif not has:
            self.status_hint.setText("Type a prompt, then tap Create.")
        else:
            W, H = S.target_size(self.o_aspect.value() or "1:1", self.o_size.value() or "standard")
            mode = "photo remake" if self.photo is not None else "text-to-image"
            self.status_hint.setText(f"Ready · {mode} · {W}×{H} · 4 steps · safety filter on")

    def _params(self) -> GenParams:
        seed = new_seed() if self.o_random.isChecked() else int(self.o_seed.value())
        return GenParams(
            prompt=self.prompt.toPlainText().strip(),
            negative=self.o_negative.toPlainText().strip(),
            aspect=self.o_aspect.value() or "1:1",
            quality=self.o_quality.value() or "best",
            size=self.o_size.value() or "standard",
            seed=seed,
            strictness=self.o_strict.value() or "relaxed",
            strength=self.s_strength.value() / 100.0,
            keep_likeness=self.o_likeness.isChecked(),
        )


    def _paste_photo(self):
        cb = QGuiApplication.clipboard()
        img = cb.image()
        if not img.isNull():
            self._open_any(img)

    def _strength_changed(self, v):

        self.s_val.setText(f"{v / 100:.2f}")
        self._save_options()

    def _photo_cleared(self):
        self.photo = None
        self.photo_name = ""
        self.strength_card.setVisible(False)
        self._refresh_main()

    def _pick_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a photo", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)")
        if path:
            self._open_any(path)

    def _open_any(self, src):
        try:
            if isinstance(src, QImage):
                img = src.convertToFormat(QImage.Format.Format_RGB888)
                w, h, bpl = img.width(), img.height(), img.bytesPerLine()
                a = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)[:, :w * 3].reshape(h, w, 3).copy()
                self.photo = a
                self.photo_name = "clipboard"
            else:
                self.photo = load_photo(src)
                self.photo_name = Path(src).name
            W, H = S.target_size(self.o_aspect.value() or "1:1", self.o_size.value() or "standard")
            preview = prepare_photo(self.photo, W, H)
            self.slot.set_image(preview)
            self.strength_card.setVisible(True)
            self._refresh_main()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Photo", f"Could not open photo: {e}")

    def _create(self):
        if not self.btn_create.isEnabled():
            return
        if not self.ready.wait(timeout=0.05) and self.pipe is None:
            self.status_hint.setText("Still loading models — try again in a moment.")
            return
        self._save_options()
        self.last_params = self._params()
        self.go(PAGE_PROGRESS)
        self.prog_bar.setValue(0)
        self.prog_label.setText("Starting…")
        self.prog_eta.setText("")
        params = self.last_params
        pipe = self.pipe
        hint = self.sec_per_eval

        photo = None if self.photo is None else self.photo.copy()

        def work(cancel, emit):
            def cb(label, frac, eta):
                emit((label, frac, eta))
            return pipe.generate(params, photo=photo, progress=cb, cancel=cancel, sec_per_eval_hint=hint)

        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._gen_progress)
        self.worker.done.connect(self._gen_done)
        self.worker.failed.connect(self._gen_failed)
        self.worker.start()

    def _gen_progress(self, d):
        label_, frac, eta = d
        self.prog_bar.setValue(int(1000 * (frac or 0)))
        self.prog_label.setText(label_)
        if eta is not None:
            self.prog_eta.setText(f"About {int(eta)} s left" if eta >= 1 else "Almost done…")

    def _gen_done(self, r: GenResult):
        self.result = r
        if r.timings.get("sec_per_unet_eval"):
            self.sec_per_eval = r.timings["sec_per_unet_eval"]
        self._set_chip()
        if r.blocked:
            self.blk_msg.setText(
                f"This picture was blocked by the safety filter (score {r.nsfw:.2f} > {r.threshold:.2f}). "
                "Nothing was saved. Try a different prompt."
            )
            self.go(PAGE_BLOCKED)
        else:
            self.saved_path = None
            self.res_image.setPixmap(pix(r.image, max(200, self.res_image.width()), max(200, self.res_image.height())))
            self.res_meta.setText(
                f"{r.size[0]}×{r.size[1]} · seed {r.seed} · {r.seconds:.1f}s · {r.provider} · safety {r.nsfw:.3f}"
            )
            self.go(PAGE_RESULT)

    def _gen_failed(self, msg, tb):
        if tb:
            log.error("generate failed: %s", tb)
        self.go(PAGE_MAIN)
        if msg != "cancelled":
            QMessageBox.warning(self, "Create failed", msg)

    def _cancel_worker(self):
        if self.worker:
            self.worker.cancel.set()

    # ---- progress / result / blocked ----
    def _page_progress(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.addStretch(1)
        lay.addWidget(label("Creating your picture…", "big"))
        self.prog_label = label("…", "subtitle", True)
        lay.addWidget(self.prog_label)
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 1000)
        lay.addWidget(self.prog_bar)
        self.prog_eta = label("", "hint")
        lay.addWidget(self.prog_eta)
        lay.addStretch(1)
        self.btn_cancel = button("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_worker)
        lay.addWidget(self.btn_cancel, 0, Qt.AlignmentFlag.AlignCenter)
        return w

    def _page_result(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 12, 16, 12)
        self.res_image = QLabel()
        self.res_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.res_image.setMinimumSize(320, 320)
        self.res_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.res_image, 1)
        self.res_meta = label("", "hint")
        lay.addWidget(self.res_meta)
        row = QHBoxLayout()
        self.btn_save = button("Save", "primary", 120)
        self.btn_save.clicked.connect(self._save)
        row.addWidget(self.btn_save)
        b_copy = button("Copy")
        b_copy.clicked.connect(self._copy)
        row.addWidget(b_copy)
        b_regen = button("Regenerate")
        b_regen.clicked.connect(self._regen)
        row.addWidget(b_regen)
        b_edit = button("Edit prompt")
        b_edit.clicked.connect(lambda: self.go(PAGE_MAIN))
        row.addWidget(b_edit)
        lay.addLayout(row)
        return w

    def _page_blocked(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.addStretch(1)
        lay.addWidget(label("Picture blocked", "big"))
        self.blk_msg = label("", "subtitle", True)
        lay.addWidget(self.blk_msg)
        lay.addStretch(1)
        self.btn_blk_edit = button("Edit prompt", "primary", 200)
        self.btn_blk_edit.clicked.connect(lambda: self.go(PAGE_MAIN))
        lay.addWidget(self.btn_blk_edit, 0, Qt.AlignmentFlag.AlignCenter)
        return w

    def _save(self):
        if not self.result or self.result.image is None:
            return
        from PIL import Image
        d = pictures_dir()
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = d / f"imagine_{ts}_{self.result.seed}.png"
        Image.fromarray(self.result.image).save(path)
        self.saved_path = path
        self.res_meta.setText(self.res_meta.text() + f" · saved {path.name}")
        QMessageBox.information(self, "Saved", f"Saved to:\n{path}")

    def _copy(self):
        if not self.result or self.result.image is None:
            return
        QGuiApplication.clipboard().setImage(qimage(self.result.image))
        self.status_hint.setText("Copied to clipboard.")

    def _regen(self):
        if self.last_params is None:
            return
        self.o_random.setChecked(True)
        self._create()

    def closeEvent(self, e):
        self._save_options()
        self._cancel_worker()
        super().closeEvent(e)


def run(args) -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("AI Imagine")
    app.setOrganizationName("vanu")
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    store = M.ModelStore(Path(args.models) if getattr(args, "models", None) else None)
    win = MainWindow(store, device=getattr(args, "device", "auto") or "auto")
    win.resize(1280, 800)
    win.show()
    if getattr(args, "gui_smoke", None):
        marker = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AIImagine" / "gui_smoke.json"
        marker.parent.mkdir(parents=True, exist_ok=True)

        def _smoke():
            marker.write_text(json.dumps({"ok": True, "version": __version__, "t": time.time()}))
            win.close()
            app.quit()

        QTimer.singleShot(int(float(args.gui_smoke) * 1000), _smoke)
    if getattr(args, "prompt", None):
        win.prompt.setPlainText(args.prompt)
        win._refresh_main()
    return app.exec()


def screenshots(args) -> int:
    """Render UI states offscreen for docs."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    store = M.ModelStore(Path(args.models) if args.models else None)
    win = MainWindow(store, device=getattr(args, "device", "cpu") or "cpu")
    win.resize(1280, 800)
    out = Path(args.screenshots)
    out.mkdir(parents=True, exist_ok=True)

    def grab(name, page=None):
        if page is not None:
            win.go(page)
        app.processEvents()
        win.grab().save(str(out / name))

    win.prompt.setPlainText(args.prompt or "a cozy cabin in the snowy woods at dusk, cinematic lighting")
    win._refresh_main()
    grab("01_setup.png", PAGE_SETUP)
    grab("02_main.png", PAGE_MAIN)
    win.btn_opts.setChecked(True)
    grab("03_options.png", PAGE_MAIN)
    win.prog_label.setText("Creating… step 2 of 4")
    win.prog_bar.setValue(500)
    win.prog_eta.setText("About 8 s left")
    grab("04_progress.png", PAGE_PROGRESS)
    # fake result if we have a generated sample
    sample = out / "_sample.png"
    if getattr(args, "out", None) and Path(args.out).is_file():
        from PIL import Image
        rgb = np.asarray(Image.open(args.out).convert("RGB"))
    else:
        rgb = np.zeros((512, 512, 3), np.uint8)
        rgb[:, :] = (40, 44, 60)
    win.result = GenResult(image=rgb, blocked=False, nsfw=0.01, threshold=0.85, prompt="demo",
                           negative="", seed=1234, seconds=12.0, timesteps=[999, 749, 499, 249],
                           size=(512, 512), provider="CPU", safety_checked=True)
    win.res_image.setPixmap(pix(rgb, 600, 600))
    win.res_meta.setText("512×512 · seed 1234 · 12.0s · CPU · safety 0.010")
    grab("05_result.png", PAGE_RESULT)
    win.blk_msg.setText("This picture was blocked by the safety filter (score 0.92 > 0.85). Nothing was saved.")
    grab("06_blocked.png", PAGE_BLOCKED)
    print(f"screenshots -> {out}")
    return 0
