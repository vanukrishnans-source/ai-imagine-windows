"""Text-to-image pipeline: SDXL-Lightning 4-step (Euler) + always-on NSFW safety filter."""
from __future__ import annotations

import math
import random
import threading
time_mod = __import__("time")
from dataclasses import dataclass, field, asdict

import numpy as np

from . import sdxl as S
from .engine import Engine

COST = dict(TEXT=0.4, DENOISE=1.0, DECODE=1.5, SAFETY=0.2)


class Cancelled(Exception):
    pass


@dataclass
class GenParams:
    prompt: str = ""
    negative: str = ""          # user negative (NSFW terms always appended, never shown)
    aspect: str = "1:1"         # 1:1 | 16:9 | 9:16 | 4:3 | 3:4
    quality: str = "best"       # best (CFG 2.0) | fast (CFG 1.2)
    size: str = "standard"      # standard | large
    seed: int = 1234
    strictness: str = "relaxed" # relaxed | standard
    steps: int = 4              # Lightning 4-step


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

    def summary(self):
        d = {k: v for k, v in asdict(self).items() if k not in ("image",)}
        d["negative"] = "(hidden)"
        return d


def new_seed() -> int:
    return random.SystemRandom().randrange(1, 2 ** 31)


class Pipeline:
    def __init__(self, engine: Engine):
        self.e = engine
        self.tok = S.ClipTokenizer()
        self.sched = S.EulerDiscreteScheduler()

    def _encode_prompt(self, text: str):
        """Return (encoder_hidden_states [1,77,2048], pooled text_embeds [1,1280])."""
        ids = np.array([self.tok.encode(text)], dtype=np.int64)
        # text_encoder wants int32
        te1 = self.e.run(
            "text_encoder",
            {"input_ids": ids.astype(np.int32)},
            output_names=["hidden_states.11"],  # penultimate
        )[0].astype(np.float32)
        # text_encoder_2: penultimate hidden + pooled text_embeds
        outs = self.e.run(
            "text_encoder_2",
            {"input_ids": ids.astype(np.int64)},
            output_names=["hidden_states.31", "text_embeds"],
        )
        te2_hidden = outs[0].astype(np.float32)
        pooled = outs[1].astype(np.float32)
        if pooled.ndim == 3:
            # some exports return [B, seq, dim] by mistake — take EOS pooled alternative
            pooled = pooled[:, -1, :]
        if pooled.shape[-1] != 1280 and pooled.shape[1] == 1280:
            pass
        # Ensure [B, 1280]
        if pooled.ndim == 1:
            pooled = pooled[None, :]
        if pooled.shape[-1] != 1280:
            # fallback: mean-pool last hidden then project isn't available — use first 1280 of flattened
            # Prefer taking the actual text_embeds if shape is [B, proj]
            if pooled.shape[-1] == te2_hidden.shape[1]:  # [B, seq] nonsense — use last token of hidden via a dummy
                pooled = te2_hidden[:, -1, :1280]
        hidden = np.concatenate([te1, te2_hidden], axis=-1)  # [1,77,2048]
        return hidden, pooled

    def nsfw_score(self, img_u8: np.ndarray) -> float:
        # CLIP safety checker: 224x224, normalize (x-0.5)/0.5
        from PIL import Image
        im = Image.fromarray(img_u8).resize((224, 224), Image.BILINEAR)
        sm = np.asarray(im).astype(np.float32) / 255.0
        px = ((sm - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
        # safety model expects external weights via Location — we use rebuilt fp32 through ORT initializer?
        # Our safety.onnx from AI Image Create uses external weight file via the .wts rebuild path —
        # the graph expects weights as EXTERNAL or as separate. Looking at Android/Windows AIC:
        # they use onnx with external data pointing at .fp32.bin via a custom loader?
        # Check: in AI Image Create, how are weights bound?
        out = self.e.run("safety", {"pixel_values": px})[0]
        return float(out[0, 1])

    def generate(self, p: GenParams, progress=None, cancel: threading.Event | None = None,
                 sec_per_eval_hint: float | None = None) -> GenResult:
        progress = progress or (lambda label, frac, eta: None)
        cancel = cancel or threading.Event()

        def check():
            if cancel.is_set():
                raise Cancelled()

        t0 = time_mod.time()
        self.e.timing = {}
        W, H = S.target_size(p.aspect, p.size)
        negative = S.full_negative(p.negative)
        if S.NSFW_NEGATIVE not in negative:
            negative = ", ".join(x for x in (negative, S.NSFW_NEGATIVE) if x)
        cfg = 2.0 if p.quality == "best" else 1.2
        steps = int(p.steps) or 4
        # cost model
        evals = steps * (2 if cfg > 1.0 else 1)
        total = COST["TEXT"] * 2 + evals * COST["DENOISE"] + COST["DECODE"] + COST["SAFETY"]
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

        self.sched.set_timesteps(steps)
        timesteps = list(int(x) for x in self.sched.timesteps)
        lh, lw = H // 8, W // 8
        n = 1 * 4 * lh * lw
        latents = S.gaussian(int(p.seed), n).reshape(1, 4, lh, lw) * np.float32(self.sched.init_noise_sigma)
        time_ids = S.make_time_ids(H, W)
        # load unet early
        if "unet" not in self.e.sessions:
            progress("Loading the image generator…", min(1.0, st["units"] / total), None)
            self.e.session("unet")
            check()

        def unet_run(lat, t_scalar, enc_h, txt_emb):
            te = time_mod.time()
            # scale model input for Euler
            # step_index from current sigma position
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
            # Euler scale
            latent_in = self.sched.scale_model_input(latents, i)
            noise_cond = unet_run(latent_in, t, cond_h, cond_p)
            if cfg > 1.0 and unc_h is not None:
                noise_uncond = unet_run(latent_in, t, unc_h, unc_p)
                noise = noise_uncond + np.float32(cfg) * (noise_cond - noise_uncond)
                step(f"Creating… step {i + 1} of {steps}", COST["DENOISE"] * 2)
            else:
                noise = noise_cond
                step(f"Creating… step {i + 1} of {steps}", COST["DENOISE"])
            latents = self.sched.step(noise, i, latents)
            check()

        progress("Finishing the picture…", min(1.0, st["units"] / total),
                 (total - st["units"]) * st["spu"] if st["spu"] else None)
        # VAE decode
        latents_dec = (latents / np.float32(S.VAE_SCALE)).astype(np.float32)
        img = self.e.run("vae_decoder", {"latent_sample": latents_dec})[0][0]  # CHW
        img = np.clip(img.transpose(1, 2, 0) / 2 + 0.5, 0, 1)
        out = np.clip(np.floor(img * 255.0 + 0.5), 0, 255).astype(np.uint8)
        step("Safety check…", COST["DECODE"])
        check()

        # ---- NSFW classifier: always runs, cannot be switched off ----
        nsfw = self.nsfw_score(out)
        thr = S.nsfw_threshold(p.strictness)
        blocked = not (nsfw <= thr)  # NaN -> blocked
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
        )
