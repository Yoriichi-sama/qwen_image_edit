#!/usr/bin/env bash
# One-shot setup + run helper for the Qwen-Image-Edit 2511 elevation generator.
# Usage:
#   ./setup_and_run.sh                      # just install deps & check GPU
#   ./setup_and_run.sh path/to/building.jpg # install deps AND generate design
set -e
cd "$(dirname "$0")"

echo "==> [1/4] Installing Python dependencies..."
python3 -m pip install -r requirements.txt

echo "==> [2/4] Checking NVIDIA GPU (CUDA)..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "   ! nvidia-smi not found - make sure NVIDIA drivers are installed."
fi

echo "==> [3/4] Downloading model 2511 into ./models (skips if already present, resume-safe)..."
python3 download_models.py

echo "==> [4/4] ComfyUI check..."
if ! curl -s http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    cat <<'EOF'
   ------------------------------------------------------------------
   ComfyUI server is NOT running on http://127.0.0.1:8188 yet.
   Start it (one-time setup if needed):

     git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
     python main.py --listen 127.0.0.1 --port 8188

   Then copy files from qwen/models/ into ComfyUI/models/ matching folders
   (diffusion_models/, text_encoders/, vae/, loras/) - see README.md.
   ------------------------------------------------------------------
EOF
fi

if [ -n "$1" ]; then
    echo "==> Generating front elevation design for: $1"
    python3 generate.py --image "$1"
    echo "Done! Output file listed above."
else
    echo ""
    echo "Setup complete. To generate your design run:"
    echo "  python3 generate.py --image your_building_photo.jpg"
fi
