# AI Imagine for Windows

Offline **text-to-image** for the **ASUS ROG Ally X** (7″ 1080p touch, Windows 11, Radeon 780M, 24 GB) — and any Windows 10/11 x64 PC with automatic CPU fallback.

Type a prompt, tap **Create**, get a new picture. Nothing is uploaded.

| | |
|---|---|
| **Download** | [Latest release](https://github.com/vanukrishnans-source/ai-imagine-windows/releases/latest) — portable zip |
| **Models** | [ai-imagine-models / models-v1](https://github.com/vanukrishnans-source/ai-imagine-models/releases/tag/models-v1) (~6.5 GB download, ~7.2 GB on disk) |

## Screenshots

Screenshots and a sample sheet are produced by CI and attached under `docs/` / release artifacts after the first successful build.

## Install (first run)

1. Download `AIImagine-1.0.0-win64.zip` from the latest release and unzip it somewhere with ~10 GB free (e.g. `C:\Apps\`).
2. Open the `AIImagine` folder and run **`AIImagine.exe`**.
3. **Windows SmartScreen** (the app isn’t code-signed): click **More info**, then **Run anyway**.
4. Tap **Download** for the one-time AI model download (~6.5 GB from `ai-imagine-models`, SHA-256 checked, resumable). Unpacks to about **7.2 GB** under `%LOCALAPPDATA%\AIImagine\models`.
5. Accept the model licences (CreativeML Open RAIL++-M / OpenRAIL-M). The safety filter is always on.

You can also copy the release files to another PC and use **Import from folder…**.

## How to use

### Mode 1 — text-to-image (no photo)
1. Type what you want to see in the big prompt box.
2. Pick an **aspect ratio** (1:1, 16:9, 9:16, 4:3, 3:4) and **Best / Fast**.
3. Tap **Create**.

### Mode 2 — photo remake (optional)
1. Attach a photo (**Open…**, drag-and-drop, or Paste).
2. Type a detailed prompt (e.g. `kissing on the beach at sunset, cinematic`).
3. Set **How much to change** (default 0.65). Enable **Keep likeness** to bias toward the photo.
4. Tap **Create**.

**Options** (collapsed by default): optional negative prompt, Standard/Large size, seed, safety strictness (Relaxed/Standard), processor Auto/CPU.

**Result page:** **Save** (to `Pictures\AIImagine`), **Copy**, **Regenerate** (new seed), **Edit prompt**.

### Safety filter (always on)

- Every picture goes through the Stable Diffusion safety checker. There is **no** setting, config file, registry key, environment variable, CLI flag or code path that skips it.
- Nudity terms (`nude, naked, nsfw, nipples, genitals, sexual`) are always appended (hidden) to the negative prompt.
- Blocked results show a friendly message and are **not** saved.

## Model

**ByteDance SDXL-Lightning 4-step** (ONNX fp16) — few-step SDXL text-to-image. Stronger and more “prompt-faithful” than SD 1.5 LCM used in the phone img2img app.

- **Not** FLUX.1-schnell: official FLUX ONNX is ~70 GB+ and impractical on a 24 GB handheld; SDXL-Lightning is the practical “Imagine-style” offline choice here.
- Quality is good for an offline 4-step model, but it is **not** identical to cloud Grok Imagine (different model family, fewer steps, no cloud-scale refiners).

### Ally X speed (estimates)

GitHub Actions runners have no usable DirectML GPU, so CI times are **CPU-only** and much slower. On a ROG Ally X (Radeon 780M, DirectML, 24 GB shared):

| Setting | Estimate per image |
|---|---|
| Fast, Standard 512² | ~8–20 s (warm) |
| Best, Standard 512² | ~12–30 s (warm) |
| Best, Large ~768–1024 | ~25–60 s (warm) |

First picture is slower (model load). Treat these as estimates until measured on device.

## Processor badge

Auto uses **DirectML** when `onnxruntime-directml` can open the UNet/VAE on the GPU; otherwise **CPU**. Text encoders and the safety classifier always run on CPU.

## Build from source

```bat
pip install -r requirements-win.txt
pyinstaller AIImagine.spec --noconfirm
powershell -ExecutionPolicy Bypass -File packaging\make_zip.ps1
```

CI: `.github/workflows/build.yml` builds, selftests (2 private prompts, NSFW check), GUI smoke, screenshots, sample sheet, and publishes the zip on tag `v*`.

## Licence

App code: MIT (see `LICENSE`). Models: see `THIRD_PARTY.md` (Open RAIL++-M / OpenRAIL-M / MIT). Not affiliated with xAI.
