"""Command-line modes of AIImagine(.exe) / AIImagine_cli.exe."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__

log = logging.getLogger("aii")


def app_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "AIImagine"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _setup_logging(verbose=True):
    handlers = [logging.FileHandler(app_data_dir() / "aiimagine.log", encoding="utf-8")]
    if verbose and sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def versions():
    import onnxruntime as ort
    from PIL import __version__ as pilv
    return dict(
        app=__version__, python=sys.version.split()[0], platform=platform.platform(),
        machine=platform.machine(), processor=platform.processor(), cpu_count=os.cpu_count(),
        onnxruntime=ort.__version__, providers=ort.get_available_providers(),
        numpy=np.__version__, pillow=pilv, frozen=bool(getattr(sys, "frozen", False)),
    )


def _store(args):
    from .models import ModelStore
    return ModelStore(Path(args.models) if args.models else (
        Path(os.environ["AII_MODELS"]) if os.environ.get("AII_MODELS") else None))


def _install(store, import_dir=None):
    last = [0.0]

    def cb(p):
        now = time.time()
        if now - last[0] > 3 or p.stage == "done":
            last[0] = now
            log.info("models: %s %s %.1f/%.1f MB %.1f MB/s", p.stage, p.file, p.done / 1e6, p.total / 1e6, p.bps / 1e6)

    if import_dir:
        got = store.import_from(Path(import_dir), cb)
        log.info("imported %s", got)
    if not store.is_installed():
        store.ensure(cb)
    assert store.is_installed(), "models missing"
    return store


def _pipeline(store, device="cpu"):
    from .engine import Engine
    from .pipeline import Pipeline
    return Pipeline(Engine(store, device))


def _printer():
    last = [0.0]

    def cb(label, frac, eta):
        now = time.time()
        if now - last[0] > 1.5 or frac >= 1.0:
            last[0] = now
            log.info("  %3.0f%%  %s%s", frac * 100, label, f"  (about {eta:.0f}s left)" if eta else "")
    return cb


def _save_png(img, path):
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


PRIVATE_PROMPTS = [
    "a cozy wooden cabin in snowy pine woods at dusk, warm window light, cinematic",
    "a red bicycle leaning on a brick wall under pink cherry blossoms, watercolor painting",
]


def selftest(args) -> int:
    from .pipeline import GenParams
    from . import sdxl as S
    out = Path(args.out or (app_data_dir() / "selftest"))
    out.mkdir(parents=True, exist_ok=True)
    rep = {"versions": versions(), "ok": False, "steps": {}}
    t0 = time.time()
    try:
        store = _store(args)
        rep["models_dir"] = str(store.dir)
        _install(store, args.import_dir)
        rep["steps"]["models"] = "ok"
        pipe = _pipeline(store, "cpu")
        from .pipeline import load_photo
        from PIL import Image as _Image
        prompts = [args.prompt] if args.prompt else PRIVATE_PROMPTS
        results = []
        # --- 1) text-to-image ---
        p = GenParams(prompt=prompts[0], seed=args.seed, quality="fast", aspect="1:1", size="standard")
        r = pipe.generate(p, progress=_printer())
        row = {
            "mode": "txt2img", "prompt": prompts[0], "summary": r.summary(),
            "nsfw_filter": {
                "ran": bool(r.safety_checked and "safety" in r.timings),
                "score": r.nsfw, "threshold": r.threshold, "blocked": r.blocked,
                "nsfw_terms_in_negative": S.NSFW_NEGATIVE in r.negative,
                "safety_model_seconds": r.timings.get("safety"),
            },
        }
        assert row["nsfw_filter"]["ran"], "NSFW classifier did not run"
        assert row["nsfw_filter"]["nsfw_terms_in_negative"], "NSFW terms missing from negative"
        if r.blocked:
            raise AssertionError(f"selftest txt2img blocked (score {r.nsfw:.3f})")
        assert r.image is not None and float(r.image.std()) > 5
        of = out / "selftest_1_txt2img.png"
        _save_png(r.image, of)
        row["file"] = str(of); row["seconds"] = round(r.seconds, 1)
        results.append(row)
        log.info("txt2img ok in %.1fs nsfw=%.4f", r.seconds, r.nsfw)
        # --- 2) photo remake (img2img) with a private synthetic / local photo ---
        photo_path = Path(args.photo) if getattr(args, "photo", None) else None
        if photo_path and photo_path.is_file():
            photo = load_photo(photo_path)
        else:
            # synthetic private subject: gradient "portrait" — no public figures
            yy, xx = np.mgrid[0:512, 0:512]
            photo = np.stack([
                np.clip(80 + xx // 3, 0, 255),
                np.clip(60 + yy // 4, 0, 255),
                np.clip(100 + (xx + yy) // 6, 0, 255),
            ], axis=-1).astype(np.uint8)
            # soft oval "face" region
            cy, cx, a, b = 220, 256, 90, 70
            mask = ((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1
            photo[mask] = (210, 170, 145)
            _save_png(photo, out / "selftest_input_synthetic.png")
        pr2 = prompts[1] if len(prompts) > 1 else "kissing on a sunny beach, romantic, cinematic lighting"
        p2 = GenParams(prompt=pr2, seed=args.seed + 1, quality="fast", aspect="1:1", size="standard",
                       strength=0.65, keep_likeness=True)
        r2 = pipe.generate(p2, photo=photo, progress=_printer())
        row2 = {
            "mode": "img2img", "prompt": pr2, "summary": r2.summary(),
            "nsfw_filter": {
                "ran": bool(r2.safety_checked and "safety" in r2.timings),
                "score": r2.nsfw, "threshold": r2.threshold, "blocked": r2.blocked,
                "nsfw_terms_in_negative": S.NSFW_NEGATIVE in r2.negative,
                "safety_model_seconds": r2.timings.get("safety"),
            },
        }
        assert row2["nsfw_filter"]["ran"], "NSFW classifier did not run (img2img)"
        assert row2["nsfw_filter"]["nsfw_terms_in_negative"]
        if r2.blocked:
            raise AssertionError(f"selftest img2img blocked (score {r2.nsfw:.3f})")
        assert r2.image is not None and float(r2.image.std()) > 5
        assert r2.mode == "img2img"
        of2 = out / "selftest_2_img2img.png"
        _save_png(r2.image, of2)
        row2["file"] = str(of2); row2["seconds"] = round(r2.seconds, 1)
        results.append(row2)
        log.info("img2img ok in %.1fs nsfw=%.4f strength=%.2f", r2.seconds, r2.nsfw, r2.strength)
        rep["results"] = results
        rep["nsfw_filter"] = results[0]["nsfw_filter"]
        rep["seconds_generate"] = results[0]["seconds"]
        rep["provider"] = results[0]["summary"]["provider"]
        if args.dml_smoke:
            rep["dml_smoke"] = dml_smoke(store)
        rep["steps"]["generate_cpu"] = "ok"
        rep["ok"] = True
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
        log.error("selftest failed: %s", rep["traceback"])
    rep["seconds_total"] = round(time.time() - t0, 1)
    (out / "selftest_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log.info("SELFTEST %s -> %s", "PASS" if rep["ok"] else "FAIL", out / "selftest_report.json")
    return 0 if rep["ok"] else 1


def dml_smoke(store) -> dict:
    import onnxruntime as ort
    from .engine import Engine, gpu_adapter_name
    d = {"available_providers": ort.get_available_providers(), "adapters": gpu_adapter_name()}
    if "DmlExecutionProvider" not in d["available_providers"]:
        d["status"] = "DirectML provider not in this onnxruntime build"
        return d
    path = str(store.path("vae_decoder") / "model.onnx")
    try:
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.log_severity_level = 3
        s = ort.InferenceSession(path, so, providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
        d["vae_providers"] = s.get_providers()
        t = time.time()
        y = s.run(None, {"latent_sample": np.random.randn(1, 4, 32, 32).astype(np.float32)})[0]
        d["vae_run"] = {"out_shape": list(y.shape), "seconds": round(time.time() - t, 3), "finite": bool(np.isfinite(y).all())}
    except Exception as e:  # noqa: BLE001
        d["vae_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    e = Engine(store, "auto")
    try:
        e.session("vae_decoder")
    except Exception as ex:  # noqa: BLE001
        d["engine_error"] = str(ex)
    d["engine_active"] = e.info.active
    d["engine_label"] = e.info.label()
    used = d.get("vae_providers") or []
    d["status"] = "DirectML used" if used[:1] == ["DmlExecutionProvider"] else f"DirectML not used (got {used})"
    return d


def generate(args) -> int:
    from .pipeline import GenParams
    store = _store(args)
    _install(store, args.import_dir)
    pipe = _pipeline(store, args.device)
    p = GenParams(
        prompt=args.prompt or PRIVATE_PROMPTS[0],
        negative=args.negative or "",
        aspect=args.aspect, quality=args.quality, size=args.size,
        seed=args.seed, strictness=args.strictness if args.strictness in ("standard", "relaxed") else "relaxed",
    )
    r = pipe.generate(p, progress=_printer())
    info = r.summary()
    info["device"] = pipe.e.info.label()
    if r.blocked:
        log.warning("Safety filter blocked this picture (score %.3f > %.2f). Nothing saved.", r.nsfw, r.threshold)
    else:
        _save_png(r.image, args.generate)
    Path(str(args.generate) + ".json").write_text(json.dumps(info, indent=1, default=str))
    log.info("%s", json.dumps(info, default=str))
    return 2 if r.blocked else 0


SAMPLE_PROMPTS = [
    "a cozy wooden cabin in snowy pine woods at dusk, warm window light, cinematic",
    "a red bicycle leaning on a brick wall under pink cherry blossoms, watercolor",
    "a steaming cup of coffee on a wooden table by a rainy window, soft morning light",
    "an origami paper crane on a bookshelf, shallow depth of field, photo",
    "a small sailboat on calm turquoise water at sunrise, impressionist painting",
]


def samples(args) -> int:
    from PIL import Image, ImageDraw, ImageFont
    from .pipeline import GenParams
    store = _store(args)
    _install(store, args.import_dir)
    pipe = _pipeline(store, args.device)
    out = Path(args.samples)
    tiles = []
    meta = []
    prompts = args.prompts or SAMPLE_PROMPTS
    for i, pr in enumerate(prompts):
        r = pipe.generate(GenParams(prompt=pr, seed=args.seed + i, quality="fast", aspect="1:1", size="standard"),
                          progress=_printer())
        meta.append(dict(prompt=pr, nsfw=r.nsfw, blocked=r.blocked, seconds=round(r.seconds, 1), provider=r.provider))
        if r.blocked:
            ph = np.full((512, 512, 3), 40, np.uint8)
            tiles.append((f'"{pr[:40]}…" — blocked', ph))
        else:
            tiles.append((f'"{pr[:48]}"', r.image))
            Image.fromarray(r.image).save(out.parent / f"sample_{i + 1}.jpg", quality=92)
    tw = 400
    th = 400
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    pad, lab = 12, 48
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * (th + lab) + (rows + 1) * pad + 50), (24, 26, 32))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("segoeui.ttf", 16)
        tfont = ImageFont.truetype("segoeuib.ttf", 22)
    except OSError:
        font = tfont = ImageFont.load_default()
    d.text((pad, 12), f"AI Imagine {__version__} — SDXL-Lightning 4-step, text-to-image (seed {args.seed}) — {pipe.e.info.label()}",
           fill=(235, 235, 240), font=tfont)
    for i, (labtxt, img) in enumerate(tiles):
        x = pad + (i % cols) * (tw + pad)
        y = 50 + pad + (i // cols) * (th + lab + pad)
        sheet.paste(Image.fromarray(img).resize((tw, th), Image.LANCZOS), (x, y + lab))
        d.text((x + 4, y + 6), labtxt[:55], fill=(235, 235, 240), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    (out.parent / "samples.json").write_text(json.dumps(meta, indent=1))
    log.info("samples -> %s (%d bytes)", out, out.stat().st_size)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="AIImagine", description="AI Imagine for Windows " + __version__)
    ap.add_argument("--models", help=r"model folder (default %LOCALAPPDATA%\AIImagine\models)")
    ap.add_argument("--import", dest="import_dir", help="install models from a folder with the release files")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dml-smoke", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--photo", help="optional photo for img2img selftest")
    ap.add_argument("--generate", metavar="OUT.png")
    ap.add_argument("--prompt")
    ap.add_argument("--negative", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--aspect", choices=list(__import__("aii.sdxl", fromlist=["ASPECTS"]).ASPECTS), default="1:1")
    ap.add_argument("--quality", choices=["best", "fast"], default="best")
    ap.add_argument("--size", choices=["standard", "large"], default="standard")
    ap.add_argument("--strictness", choices=["relaxed", "standard"], default="relaxed")
    ap.add_argument("--device", choices=["auto", "dml", "cpu"], default="auto")
    ap.add_argument("--samples", metavar="OUT.png")
    ap.add_argument("--prompts", nargs="*")
    ap.add_argument("--screenshots", metavar="DIR")
    ap.add_argument("--gui-smoke", type=float, metavar="SECONDS")
    args = ap.parse_args(argv)
    gui = not any([args.selftest, args.generate, args.samples, args.download, args.screenshots])
    _setup_logging(verbose=not gui)
    try:
        if gui:
            from .gui.app import run
            return run(args)
        log.info("AI Imagine %s %s", __version__, json.dumps(versions()))
        if args.screenshots:
            from .gui.app import screenshots
            return screenshots(args)
        if args.selftest:
            return selftest(args)
        if args.generate:
            return generate(args)
        if args.samples:
            return samples(args)
        if args.download:
            _install(_store(args), args.import_dir)
            log.info("models installed")
            return 0
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        log.error("fatal: %s", traceback.format_exc())
        return 1
    return 0


def entry():
    code = main()
    try:
        sys.stdout and sys.stdout.flush()
        sys.stderr and sys.stderr.flush()
    finally:
        os._exit(code or 0)
