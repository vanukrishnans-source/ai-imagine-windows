"""Text-to-image + optional photo img2img: SDXL-Lightning 4-step (Euler) + always-on NSFW safety filter."""
from __future__ import annotations

import math
import random
import threading
import time as time_mod
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image, ImageOps

from . import sdxl as S
from .engine import Engine

COST = dict(TEXT=0.4, ENC=0.3, DENOISE=1.0, DECODE=1.5, SAFETY=0.2)


class Cancelled(Exception):
    pass


@dataclass
class GenParams:
    prompt: str = ""
    negative: str = ""
    aspect: str = "1:1"
    quality: str = "best"       # best (CFG 2.0) | fast (CFG 1.2)
    size: str = "standard"
    seed: int = 1234
    strictness: str = "relaxed"
    steps: int = 4
    # Photo remake (img2img) — optional. strength = how much to change (0.3 keep close … 0.95 follow prompt)
    strength: float = 0.65
    keep_likeness: bool = True  # when True and photo set, cap strength a bit lower


@dataclass
class GenResult:
    image: np.ndarray | None
    blocked: bool
    nsfw: float
    threshold: float
    prompt: str
    negative: str
    seed: int
    seconds: float
    timesteps: list
    size: tuple
    timings: dict = field(default_factory=dict)
    provider: str = "CPU"
    safety_checked: bool = False
    cfg: float = 1.0
    mode: str = "txt2img"  # txt2img | img2img
    strength: float = 0.0

    def summary(self):
        d = {k: v for k, v in asdict(self).items() if k not in ("image",)}
        d["negative"] = "(hidden)"
        return d


def new_seed() -> int:
    return random.SystemRandom().randrange(1, 2 ** 31)


def load_photo(path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im)


def prepare_photo(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Cover-crop to WxH (never stretch)."""
    ih, iw = rgb.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    im = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return np.asarray(im.crop((left, top, left + w, top + h)))


class Pipeline:
    def __init__(self, engine: Engine):
        self.e = engine
        self.tok = S.ClipTokenizer()
        self.sched = S.EulerDiscreteScheduler()

    def _encode_prompt(self, text: str):
        ids = np.array([self.tok.encode(text)], dtype=np.int64)
        te1 = self.e.run(
            "text_encoder",
            {"input_ids": ids.astype(np.int32)},
            output_names=["hidden_states.11"],
        )[0].astype(np.float32)
        outs = self.e.run(
            "text_encoder_2",
            {"input_ids": ids.astype(np.int64)},
            output_names=["hidden_states.31", "text_embeds"],
        )
        te2_hidden = outs[0].astype(np.float32)
        pooled = outs[1].astype(np.float32)
        if pooled.ndim == 1:
            pooled = pooled[None, :]
        if pooled.ndim == 3:
            pooled = pooled[:, -1, :]
        if pooled.shape[-1] != 1280:
            pooled = te2_hidden[:, -1, :1280]
        hidden = np.concatenate([te1, te2_hidden], axis=-1)
        return hidden, pooled

    def _encode_image(self, rgb_u8: np.ndarray) -> np.ndarray:
        """VAE encode RGB uint8 HxWx3 -> latents [1,4,h,w] already scaled (* scaling_factor)."""
        x = (rgb_u8.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        lat = self.e.run("vae_encoder", {"sample": x.astype(np.float32)})[0]
        return lat.astype(np.float32)

    def nsfw_score(self, img_u8: np.ndarray) -> float:
        im = Image.fromarray(img_u8).resize((224, 224), Image.BILINEAR)
        sm = np.asarray(im).astype(np.float32) / 255.0
        px = ((sm - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
        out = self.e.run("safety", {"pixel_values": px})[0]
        return float(out[0, 1])

    def generate(self, p: GenParams, photo: np.ndarray | None = None,
                 progress=None, cancel: threading.Event | None = None,
                 sec_per_eval_hint: float | None = None) -> GenResult:
        progress = progress or (lambda label, frac, eta: None)
        cancel = cancel or threading.Event()

        def check():
            if cancel.is_set():
                raise Cancelled()

        t0 = time_mod.time()
        self.e.timing = {}
        W, H = S.target_size(p.aspect, p.size)
        mode = "img2img" if photo is not None else "txt2img"
        strength = float(p.strength)
        if mode == "img2img" and p.keep_likeness:
            strength = min(strength, 0.72)
        strength = float(min(0.95, max(0.25, strength)))

        negative = S.full_negative(p.negative)
        if S.NSFW_NEGATIVE not in negative:
            negative = ", ".join(x for x in (negative, S.NSFW_NEGATIVE) if x)
        cfg = 2.0 if p.quality == "best" else 1.2
        steps = int(p.steps) or 4

        # timesteps (possibly truncated for img2img)
        self.sched.set_timesteps(steps)
        all_ts = list(int(x) for x in self.sched.timesteps)
        if mode == "img2img":
            init_timestep = min(int(steps * strength), steps)
            t_start = max(steps - init_timestep, 0)
            timesteps = all_ts[t_start:]
            sigma_offset = t_start  # index into self.sched.sigmas
        else:
            timesteps = all_ts
            sigma_offset = 0
            strength = 1.0

        evals = len(timesteps) * (2 if cfg > 1.0 else 1)
        total = COST["TEXT"] * 2 + (COST["ENC"] if mode == "img2img" else 0) + evals * COST["DENOISE"] + COST["DECODE"] + COST["SAFETY"]
        st = {"units": 0.0, "spu": sec_per_eval_hint}

        def step(label, add):
            st["units"] += add
            progress(label, min(1.0, st["units"] / total),
                     (total - st["units"]) * st["spu"] if st["spu"] else None)

        progress("Reading your prompt…", 0.0, total * sec_per_eval_hint if sec_per_eval_hint else None)
        cond_h, cond_p = self._encode_prompt(p.prompt.strip() or "a beautiful landscape")
        check()
        if cfg > 1.0:
            unc_h, unc_p = self._encode_prompt(negative)
        else:
            unc_h = unc_p = None
        step("Preparing…", COST["TEXT"] * 2)
        check()

        lh, lw = H // 8, W // 8
        n = 1 * 4 * lh * lw
        noise = S.gaussian(int(p.seed), n).reshape(1, 4, lh, lw)

        if mode == "img2img":
            progress("Encoding your photo…", min(1.0, st["units"] / total), None)
            rgb = prepare_photo(photo, W, H)
            init_latents = self._encode_image(rgb)
            check()
            # add noise at the first (highest) timestep we will use
            first_sigma = float(self.sched.sigmas[sigma_offset])
            latents = (init_latents + noise * np.float32(first_sigma)).astype(np.float32)
            step("Encoding your photo…", COST["ENC"])
        else:
            latents = (noise * np.float32(self.sched.init_noise_sigma)).astype(np.float32)

        time_ids = S.make_time_ids(H, W)
        if "unet" not in self.e.sessions:
            progress("Loading the image generator…", min(1.0, st["units"] / total), None)
            self.e.session("unet")
            check()

        def unet_run(lat, t_scalar, enc_h, txt_emb, step_index):
            te = time_mod.time()
            feed = {
                "sample": lat.astype(np.float32),
                "timestep": np.array([t_scalar], dtype=np.int64),
                "encoder_hidden_states": enc_h.astype(np.float32),
                "text_embeds": txt_emb.astype(np.float32),
                "time_ids": time_ids.astype(np.float32),
            }
            o = self.e.run("unet", feed)[0]
            sec = time_mod.time() - te
            st["spu"] = sec if st["spu"] is None else 0.5 * st["spu"] + 0.5 * sec
            return o.astype(np.float32)

        for i, t in enumerate(timesteps):
            check()
            si = sigma_offset + i
            latent_in = self.sched.scale_model_input(latents, si)
            noise_cond = unet_run(latent_in, t, cond_h, cond_p, si)
            if cfg > 1.0 and unc_h is not None:
                noise_uncond = unet_run(latent_in, t, unc_h, unc_p, si)
                noise_pred = noise_uncond + np.float32(cfg) * (noise_cond - noise_uncond)
                step(f"Creating… step {i + 1} of {len(timesteps)}", COST["DENOISE"] * 2)
            else:
                noise_pred = noise_cond
                step(f"Creating… step {i + 1} of {len(timesteps)}", COST["DENOISE"])
            latents = self.sched.step(noise_pred, si, latents)
            check()

        progress("Finishing the picture…", min(1.0, st["units"] / total),
                 (total - st["units"]) * st["spu"] if st["spu"] else None)
        latents_dec = (latents / np.float32(S.VAE_SCALE)).astype(np.float32)
        img = self.e.run("vae_decoder", {"latent_sample": latents_dec})[0][0]
        img = np.clip(img.transpose(1, 2, 0) / 2 + 0.5, 0, 1)
        out = np.clip(np.floor(img * 255.0 + 0.5), 0, 255).astype(np.uint8)
        step("Safety check…", COST["DECODE"])
        check()

        # ---- NSFW classifier: always runs, cannot be switched off ----
        nsfw = self.nsfw_score(out)
        thr = S.nsfw_threshold(p.strictness)
        blocked = not (nsfw <= thr)
        final = None if blocked else out
        if blocked:
            out = None
        progress("Done", 1.0, 0.0)
        per = self.e.info.per_model
        prov = "DirectML" if per.get("unet") == "DirectML" else "CPU"
        timings = dict(self.e.timing)
        timings["sec_per_unet_eval"] = st["spu"]
        return GenResult(
            image=final, blocked=blocked, nsfw=float(nsfw), threshold=thr,
            prompt=p.prompt, negative=negative, seed=int(p.seed),
            seconds=time_mod.time() - t0, timesteps=timesteps, size=(W, H),
            timings=timings, provider=prov, safety_checked=True, cfg=cfg,
            mode=mode, strength=strength if mode == "img2img" else 0.0,
        )
