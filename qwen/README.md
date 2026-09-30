# Qwen-Image-Edit 2511 — Building Front Elevation Generator (GPU)

Redesign the **front elevation** of any building photo using wood, paint, marble,
glass & stone materials — while **preserving the building structure exactly**
(same floors, windows, doors, roofline, camera angle).

## Files
| File | Purpose |
|---|---|
| `setup_and_run.sh` | **One-shot helper**: installs deps, checks GPU, downloads models, then generates (pass a photo as argument) |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |
| `download_models.py` | Downloads Qwen-Image-Edit-**2511** + text encoder + VAE + Lightning LoRA into `models/` (~27 GB total, resume-safe) |
| `generate.py` | Runs the model on your **GPU** via ComfyUI and generates the elevation image |

## Easiest way (one command)
```bash
cd qwen
./setup_and_run.sh my_building.jpg     # installs everything, downloads 2511, generates design
# or just setup without an image yet:
./setup_and_run.sh
```

## Quick Start (3 steps)

### 1. Download models (one-time)
```bash
pip install requests
python download_models.py
```

### 2. Install & start ComfyUI with GPU
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # NVIDIA GPU
python main.py --listen 127.0.0.1 --port 8188
```
Copy the downloaded model files into ComfyUI's folders:
```
models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors  -> ComfyUI/models/diffusion_models/
models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors         -> ComfyUI/models/text_encoders/
models/vae/qwen_image_vae.safetensors                              -> ComfyUI/models/vae/
models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors -> ComfyUI/models/loras/
```

### 3. Generate your elevation design
```bash
pip install websocket-client
python generate.py --image your_building_photo.jpg
```
That uses the built-in "really good" prompt:
> *Create the best, absolutely stunning modern front elevation design for this existing building… wooden louvers, designer paint, polished marble cladding, stone veneer, glass windows… STRICTLY PRESERVE the building structure…*

Output saved as `elevation_<timestamp>.png`.

## Useful options
```bash
# Best quality (20 steps instead of the fast 4-step LoRA mode)
python generate.py --image building.jpg --steps 20 --megapixels 1.5

# Try different design ideas
python generate.py --image building.jpg --extra "Mediterranean style with terracotta and white stucco"

# Repeatable result
python generate.py --image building.jpg --seed 42 --out my_design.png

# ComfyUI running on another machine
python generate.py --image building.jpg --comfy http://192.168.1.50:8188
```

Tips: use a clear, straight-on photo of the facade; VRAM ≥ 12 GB recommended
(fp8 model); `--steps 4` is ~5× faster, `--steps 20` gives maximum detail.
