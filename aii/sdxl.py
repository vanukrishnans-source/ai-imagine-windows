"""SDXL helpers: CLIP tokenizer, Euler Discrete scheduler, prompt encoding, size presets."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

from .resources_path import res

NSFW_NEGATIVE = "nude, naked, nsfw, nipples, genitals, sexual"
NSFW_THRESHOLDS = {"standard": 0.5, "relaxed": 0.85}
DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, watermark, text, logo"

# Aspect presets -> (width, height) for "standard" (~0.25 MP) and "large" (~0.6 MP)
ASPECTS = {
    "1:1":  {"standard": (512, 512), "large": (768, 768)},
    "16:9": {"standard": (768, 432), "large": (1024, 576)},
    "9:16": {"standard": (432, 768), "large": (576, 1024)},
    "4:3":  {"standard": (576, 448), "large": (768, 576)},
    "3:4":  {"standard": (448, 576), "large": (576, 768)},
}

VAE_SCALE = 0.13025


def nsfw_threshold(strictness: str) -> float:
    return NSFW_THRESHOLDS.get(str(strictness).lower(), NSFW_THRESHOLDS["standard"])


def full_negative(user_negative: str | None = None) -> str:
    parts = [
        (user_negative or "").strip() or DEFAULT_NEGATIVE,
        NSFW_NEGATIVE,
    ]
    # dedupe while preserving order
    seen = set()
    out = []
    for p in parts:
        for term in p.split(","):
            t = term.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return ", ".join(out)


def target_size(aspect: str, quality_size: str) -> tuple[int, int]:
    a = ASPECTS.get(aspect, ASPECTS["1:1"])
    w, h = a.get(quality_size, a["standard"])
    # SDXL UNet wants multiples of 8 (actually 64 nicer)
    w = max(64, (w // 8) * 8)
    h = max(64, (h // 8) * 8)
    return w, h


# ---------------- CLIP BPE tokenizer (same as SD / SDXL) ----------------
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class ClipTokenizer:
    BOS, EOS, MAXLEN = 49406, 49407, 77

    def __init__(self, vocab_path=None, merges_path=None):
        import regex
        self._regex = regex
        self.PAT = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )
        vocab_path = vocab_path or res("tokenizer", "vocab.json")
        merges_path = merges_path or res("tokenizer", "merges.txt")
        self.enc = json.load(open(vocab_path, encoding="utf-8"))
        lines = open(merges_path, encoding="utf-8").read().split("\n")[1:49152 - 256 - 2 + 1]
        self.ranks = {tuple(l.split()): i for i, l in enumerate(lines)}
        self.b2u = _bytes_to_unicode()
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            a, b = best
            out = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        self.cache[token] = word
        return word

    def encode(self, text) -> list[int]:
        text = html.unescape(html.unescape(text))
        text = self._regex.sub(r"\s+", " ", text).strip().lower()
        ids = []
        for tok in self.PAT.findall(text):
            tok = "".join(self.b2u[b] for b in tok.encode("utf-8"))
            for bpe_tok in self.bpe(tok):
                ids.append(self.enc.get(bpe_tok, self.enc.get("<|endoftext|>", 0)))
        ids = [self.BOS] + ids[: self.MAXLEN - 2] + [self.EOS]
        if len(ids) < self.MAXLEN:
            ids += [self.EOS] * (self.MAXLEN - len(ids))
        return ids[: self.MAXLEN]


# ---------------- Euler Discrete scheduler (trailing, Lightning 4-step) ----------------
class EulerDiscreteScheduler:
    """Minimal EulerDiscreteScheduler matching diffusers trailing + steps_offset behaviour."""

    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                 beta_schedule="scaled_linear", steps_offset=1, timestep_spacing="trailing",
                 final_sigmas_type="zero"):
        if beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float64) ** 2
        else:
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, dtype=np.float64)
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.timestep_spacing = timestep_spacing
        self.final_sigmas_type = final_sigmas_type
        self.alphas_cumprod = alphas_cumprod
        self.sigmas_full = np.concatenate([sigmas[::-1].copy(), [0.0]]).astype(np.float64)  # unused helper
        self._sigmas = sigmas  # indexed by timestep 0..999
        self.timesteps = None
        self.sigmas = None
        self.init_noise_sigma = 1.0

    def set_timesteps(self, num_inference_steps: int):
        num_inference_steps = int(num_inference_steps)
        if self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            # diffusers: arange(num_train, 0, -step_ratio).round() - 1
            timesteps = np.arange(self.num_train_timesteps, 0, -step_ratio).round().astype(np.float64).copy()
            timesteps -= 1
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float64)
            timesteps += self.steps_offset
        else:  # linspace
            timesteps = np.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=np.float64)
        self.timesteps = timesteps.astype(np.int64)
        # sigmas for each timestep + final 0
        sigmas = np.array([self._sigmas[int(t)] for t in self.timesteps], dtype=np.float64)
        if self.final_sigmas_type == "zero":
            sigmas = np.concatenate([sigmas, [0.0]])
        else:
            sigmas = np.concatenate([sigmas, [sigmas[-1]]])
        self.sigmas = sigmas.astype(np.float32)
        self.init_noise_sigma = float(self.sigmas[0])
        return self.timesteps

    def scale_model_input(self, sample: np.ndarray, step_index: int) -> np.ndarray:
        sigma = float(self.sigmas[step_index])
        return sample / math.sqrt(sigma * sigma + 1)

    def step(self, model_output: np.ndarray, step_index: int, sample: np.ndarray) -> np.ndarray:
        """Euler step; model_output is epsilon prediction."""
        sigma = float(self.sigmas[step_index])
        sigma_next = float(self.sigmas[step_index + 1])
        # pred_original_sample
        pred_original = sample - sigma * model_output
        # derivative
        derivative = (sample - pred_original) / sigma if sigma != 0 else model_output
        dt = sigma_next - sigma
        return (sample + derivative * dt).astype(np.float32)


def gaussian(seed: int, n: int) -> np.ndarray:
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    return rng.randn(n).astype(np.float32)


def make_time_ids(h: int, w: int) -> np.ndarray:
    # original_size, crops_coords_top_left, target_size
    return np.array([[h, w, 0, 0, h, w]], dtype=np.float32)
