#!/usr/bin/env python3
"""
Qwen-Image-Edit 2511 — Building Front Elevation Generator (GPU)

Takes a photo of an existing building and generates a stunning NEW front
elevation design on it (wood, paint, marble, glass, stone...) while keeping
the building's structure completely intact.

Runs fully on your NVIDIA GPU via a local ComfyUI server.

USAGE
-----
1) Download the models once:
       python download_models.py

2) Start ComfyUI (any machine/PC with the GPU), pointing its model folders
   at ./models (i.e. put/symlink the downloaded subfolders into ComfyUI's
   models/diffusion_models, models/loras, models/text_encoders, models/vae):
       python main.py --listen 127.0.0.1 --port 8188 --use-cpu do_not_use  (or just default GPU run)

3) Generate:
       python generate.py --image building_photo.jpg
       python generate.py --image building_photo.jpg --comfy http://127.0.0.1:8188
       python generate.py --image building_photo.jpg --prompt "custom idea..." --steps 20
       python generate.py --image building_photo.jpg --seed 42 --out result.png --megapixels 1.5

Options:
  --image PATH        input photo of the building (required)
  --prompt TEXT       override the built-in elevation prompt
  --extra TEXT        append extra style hints to the built-in prompt
  --steps N           4  = fast (Lightning LoRA, default) | 20 = best quality
  --cfg FLOAT         guidance (default 1.0 for Lightning, 2.5-3.5 for full model)
  --seed N            reproducibility (default: random)
  --megapixels F      output resolution in megapixels (default 1.0)
  --shift F           timestep shift (default 3.0)
  --comfy URL         ComfyUI server address (default http://127.0.0.1:8188)
  --out PATH          output file (default elevation_<timestamp>.png)
"""
import argparse, json, sys, time, uuid, urllib.request, urllib.parse
from pathlib import Path

try:
    import websocket  # pip install websocket-client
except ImportError:
    sys.exit("Missing dependency. Run:  pip install websocket-client requests")

# ---------------------------------------------------------------------------
# The "really good" front-elevation prompt — redesigns the facade with mixed
# materials but strictly preserves the building's structure & surroundings.
# ---------------------------------------------------------------------------
ELEVATION_PROMPT = (
    "Create the best, absolutely stunning modern front elevation design for this "
    "existing building exactly as it is. Redesign only the facade finishes using a "
    "luxurious mix of premium materials: warm natural wooden louvers and wood-panel "
    "accents, fresh designer exterior paint in elegant neutral tones, polished white "
    "marble cladding on pillars and the ground floor, textured stone and grey stone "
    "veneer feature walls, and sleek glass windows with dark aluminium framing. Add a "
    "stylish entrance porch with a cantilevered canopy, subtle facade lighting, balanced "
    "proportions and contemporary architectural details. STRICTLY PRESERVE the building "
    "structure: keep the exact same footprint, number of floors, floor heights, window "
    "and door positions and sizes, roofline, walls, columns, balcony layout, orientation, "
    "camera angle and all surroundings unchanged. Photorealistic professional architectural "
    "visualization, golden-hour daylight, crisp shadows, ultra high detail, 8k."
)


def load_graph():
    """Build the ComfyUI API graph for Qwen-Image-Edit-2511 (single image)."""
    return {
        "37": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "89": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["37", 0], "lora_name": "__LORA__", "strength_model": 1.0}},
        "66": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["89", 0], "shift": 3.0}},
        "38": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "78": {"class_type": "LoadImage", "inputs": {"image": "__IMAGE__"}},
        "93": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["78", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 1}},
        "88": {"class_type": "VAEEncode", "inputs": {"pixels": ["93", 0], "vae": ["39", 0]}},
        "111": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0], "image2": None, "image3": None,
            "prompt": "__PROMPT__"}},
        "110": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0], "image2": None, "image3": None,
            "prompt": ""}},
        "3": {"class_type": "KSampler", "inputs": {
            "model": ["66", 0], "positive": ["111", 0], "negative": ["110", 0], "latent_image": ["88", 0],
            "seed": 0, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "60": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "elevation"}},
    }


def upload_image(comfy: str, path: Path) -> str:
    """Upload the input image to ComfyUI, return the server-side name."""
    import mimetypes
    boundary = "----qwen" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{path.name}\"\r\nContent-Type: {ctype}\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{comfy}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["name"]


def wait_server(comfy: str, seconds=120):
    for i in range(seconds):
        try:
            urllib.request.urlopen(comfy + "/system_stats", timeout=3); return
        except Exception:
            time.sleep(1)
    sys.exit(f"ComfyUI not reachable at {comfy}. Start it first (see README.md).")


def run_graph(comfy: str, graph: dict, timeout=1800):
    cid = str(uuid.uuid4())
    req = urllib.request.Request(f"{comfy}/prompt",
                                 data=json.dumps({"prompt": graph, "client_id": cid}).encode(),
                                 headers={"Content-Type": "application/json"})
    prompt_id = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]

    ws = websocket.WebSocket()
    ws.settimeout(60)
    ws.connect(comfy.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={cid}")
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout:
            sys.exit("Timed out waiting for generation.")
        try:
            msg = ws.recv()
        except TimeoutError:
            continue
        if isinstance(msg, bytes):
            continue  # preview image, skip
        m = json.loads(msg)
        if m.get("type") == "executing" and m["data"].get("node") is None \
                and m["data"].get("prompt_id") == prompt_id:
            break
        if m.get("type") == "execution_error":
            sys.exit(f"ComfyUI execution error: {json.dumps(m['data'], indent=2)[:1500]}")
        if m.get("type") == "status":
            print("\r  generating...", end="", flush=True)
    ws.close()
    print()

    hist = json.loads(urllib.request.urlopen(f"{comfy}/history/{prompt_id}", timeout=30).read())[prompt_id]
    for node_out in hist["outputs"].values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
            data = urllib.request.urlopen(f"{comfy}/view?{q}", timeout=60).read()
            return img["filename"], data
    sys.exit("No output image produced.")


def main():
    ap = argparse.ArgumentParser(description="Qwen-Image-Edit 2511 building elevation generator")
    ap.add_argument("--image", required=True, help="input photo of the building")
    ap.add_argument("--prompt", default=None, help="override the built-in elevation prompt")
    ap.add_argument("--extra", default="", help="append extra hints to the built-in prompt")
    ap.add_argument("--steps", type=int, default=4, help="4=fast Lightning LoRA (default), 20=max quality")
    ap.add_argument("--cfg", type=float, default=None, help="default: 1.0 if steps<=8 else 3.0")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--megapixels", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--no-lora", action="store_true", help="disable the 4-step Lightning LoRA")
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    img_path = Path(a.image).expanduser()
    if not img_path.exists():
        sys.exit(f"Input image not found: {img_path}")

    comfy = a.comfy.rstrip("/")
    cfg = a.cfg if a.cfg is not None else (1.0 if a.steps <= 8 else 3.0)
    seed = a.seed if a.seed is not None else int(time.time() * 1000) % (2**62)
    prompt = a.prompt if a.prompt else (ELEVATION_PROMPT + (" " + a.extra.strip() if a.extra else ""))

    use_lora = (not a.no_lora) and a.steps <= 8
    graph = load_graph()
    graph["89"]["inputs"]["lora_name"] = (
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" if use_lora else "None")
    if not use_lora:  # bypass LoRA node cleanly by wiring model directly
        graph["66"]["inputs"]["model"] = ["37", 0]
    graph["66"]["inputs"]["shift"] = a.shift
    graph["93"]["inputs"]["megapixels"] = a.megapixels
    graph["111"]["inputs"]["prompt"] = prompt
    graph["3"]["inputs"].update(seed=seed, steps=a.steps, cfg=cfg)

    print(f"[1/3] Checking ComfyUI at {comfy} ...")
    wait_server(comfy)
    print(f"[2/3] Uploading '{img_path.name}' and generating ({a.steps} steps, cfg {cfg}, seed {seed}) ...")
    remote_name = upload_image(comfy, img_path)
    graph["78"]["inputs"]["image"] = remote_name
    fname, data = run_graph(comfy, graph)

    out = Path(a.out) if a.out else Path(f"elevation_{time.strftime('%Y%m%d_%H%M%S')}.png")
    out.write_bytes(data)
    print(f"[3/3] Done ✅  Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
