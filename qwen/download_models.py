#!/usr/bin/env python3
"""Download the Qwen-Image-Edit 2511 model files needed by this app.

Files (same URLs used by the repo's Dockerfile):
  - diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors   (~20 GB)
  - loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
  - text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
  - vae/qwen_image_vae.safetensors

Usage:  python download_models.py [--dest models]
Resume-safe: partial downloads use .part files and are resumed automatically.
"""
import argparse, sys, time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests first")

BASE = "https://huggingface.co"
FILES = [
    ("diffusion_models", f"{BASE}/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"),
    ("loras",            f"{BASE}/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    ("text_encoders",    f"{BASE}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("vae",              f"{BASE}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"),
]

def human(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024: return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"

def download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True) as r:
        if r.status_code == 416 and done:          # already fully downloaded
            part.rename(dest); print(f"[skip] {dest.name} already complete"); return
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + done
        mode = "ab" if done else "wb"
        print(f"[get ] {dest.name}  ({human(total)}), resuming at {human(done)}" if done else f"[get ] {dest.name}  ({human(total)})")
        t0, last = time.time(), done
        with open(part, mode) as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk); done += len(chunk)
                if done - last >= 200 * 1024 * 1024:
                    pct = done / total * 100 if total else 0
                    print(f"       ... {human(done)} / {human(total)}  ({pct:.1f}%)", flush=True)
                    last = done
    part.rename(dest)
    print(f"[ok  ] {dest}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).parent / "models"))
    a = ap.parse_args()
    root = Path(a.dest)
    for sub, url in FILES:
        target = root / sub / Path(url).name
        if target.exists():
            print(f"[skip] {target.name} already exists"); continue
        download(url, target)
    print("\nAll model files present in:", root)
