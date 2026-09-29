"""Hash local model files and write aii/resources/manifest.json + release/ staging folder."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDXL = Path("/workspace/ai-imagine-models/sdxl-lightning-4step")
SAFETY = Path("/workspace/ai-imagine-models/safety")
STAGE = Path("/workspace/ai-imagine-models/release-v1")
MANIFEST_OUT = ROOT / "aii" / "resources" / "manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(4 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def zip_dir(src: Path, dest_zip: Path, arc_prefix: str = ""):
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and ".cache" not in p.parts:
                arc = str(Path(arc_prefix) / p.relative_to(src)) if arc_prefix else str(p.relative_to(src))
                z.write(p, arc.replace("\\", "/"))


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    assets = []

    # text_encoder zip
    z = STAGE / "text_encoder.zip"
    zip_dir(SDXL / "text_encoder", z, "text_encoder")
    assets.append(dict(name="text_encoder.zip", label="Text encoder (CLIP-L)", bytes=z.stat().st_size,
                       sha256=sha256(z), kind="zip", dest="."))

    z = STAGE / "text_encoder_2.zip"
    zip_dir(SDXL / "text_encoder_2", z, "text_encoder_2")
    assets.append(dict(name="text_encoder_2.zip", label="Text encoder 2 (OpenCLIP-G)", bytes=z.stat().st_size,
                       sha256=sha256(z), kind="zip", dest="."))

    z = STAGE / "vae_decoder.zip"
    zip_dir(SDXL / "vae_decoder", z, "vae_decoder")
    assets.append(dict(name="vae_decoder.zip", label="VAE decoder", bytes=z.stat().st_size,
                       sha256=sha256(z), kind="zip", dest="."))

    # tokenizers (both identical vocab — ship both folders)
    z = STAGE / "tokenizers.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("tokenizer", "tokenizer_2"):
            for p in (SDXL / sub).rglob("*"):
                if p.is_file():
                    zf.write(p, f"{sub}/{p.name}")
        zf.write(SDXL / "scheduler" / "scheduler_config.json", "scheduler/scheduler_config.json")
        zf.write(SDXL / "model_index.json", "model_index.json")
    assets.append(dict(name="tokenizers.zip", label="Tokenizers + scheduler", bytes=z.stat().st_size,
                       sha256=sha256(z), kind="zip", dest="."))

    # UNet: upload raw files (too big to zip usefully; shards already ~2GB)
    unet_dir = STAGE / "unet_files"
    unet_dir.mkdir()
    for name in ("model.onnx", "model.onnx_data", "model.onnx_data_1", "model.onnx_data_2", "config.json"):
        src = SDXL / "unet" / name
        # release asset names
        asset_name = f"unet-{name}"
        dest = STAGE / asset_name
        shutil.copyfile(src, dest)
        assets.append(dict(
            name=asset_name, label=f"UNet {name}", bytes=dest.stat().st_size,
            sha256=sha256(dest), kind="file", dest=f"unet/{name}",
        ))

    # safety pack
    z = STAGE / "safety.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        zf.write(SAFETY / "safety.wts", "safety.wts")
        zf.write(SAFETY / "safety.onnx", "safety.onnx")
        zf.write(SAFETY / "safety.json", "safety.json")
    assets.append(dict(name="safety.zip", label="Safety filter (NSFW)", bytes=z.stat().st_size,
                       sha256=sha256(z), kind="zip", dest="."))

    installed = 0
    for sub in ("text_encoder", "text_encoder_2", "unet", "vae_decoder"):
        for p in (SDXL / sub).rglob("*"):
            if p.is_file() and ".cache" not in p.parts:
                installed += p.stat().st_size
    installed += 344_752_128  # safety.fp32.bin
    installed += 6_000_000  # tokenizers etc.

    man = {
        "version": "models-v1",
        "model": "ByteDance/SDXL-Lightning 4-step (ONNX fp16) + SDXL VAE + safety checker",
        "licence": "CreativeML Open RAIL++-M (SDXL-Lightning / SDXL base); MIT (VAE fp16-fix); CreativeML OpenRAIL-M (safety)",
        "download_bytes": sum(a["bytes"] for a in assets),
        "installed_bytes": installed,
        "assets": assets,
    }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(man, indent=2), encoding="utf-8")
    (STAGE / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    lines = [f"{a['sha256']}  {a['name']}" for a in assets]
    (STAGE / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: man[k] for k in ("download_bytes", "installed_bytes")}, indent=2))
    for a in assets:
        print(f"{a['bytes']:12,}  {a['name']}")
    print("staged ->", STAGE)
    print("manifest ->", MANIFEST_OUT)


if __name__ == "__main__":
    main()
