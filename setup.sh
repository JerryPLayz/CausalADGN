#!/usr/bin/env bash
set -euo pipefail

echo "=== CA-DGN Environment Setup ==="
echo "Target: Python 3.11 | PyTorch 2.7.0 | CUDA 12.8 | WSL2"
echo ""

# 1. Verify Python 3.11 is available
if ! command -v python3.11 &>/dev/null; then
    echo "Python 3.11 not found. Installing via deadsnakes PPA..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi
echo ">>>   ✓ Python 3.11: $(python3.11 --version)"

# 2. Create venv
python3.11 -m venv venv
source venv/bin/activate
echo ">>>   ✓ venv created and activated: $(which python)"

# 3. Upgrade pip
pip install --upgrade pip wheel setuptools

# 4. Install PyTorch (CUDA 12.8)
echo ""
echo ">>>   Installing PyTorch 2.7.0+cu128..."
pip install \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 5. Install PyG base
echo ""
echo ">>>   Installing PyG 2.7.0..."
pip install torch_geometric==2.7.0

# 6. Install PyG accelerators (torch 2.7.0 + cu128 wheels)
echo ""
echo ">>>   Installing PyG accelerators..."
pip install \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html


# 7. Install System-level prerequisites
echo ""
echo ">>>  Installing System Prerequisites for other Pip Packages..."
sudo apt-get install -y \
    libpango1.0-dev \
    libpangocairo-1.0-0 \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0
	
pkg-config --modversion pangocairo

# 8. Install remaining requirements
echo ""
echo ">>>  Installing remaining requirements..."
pip install \
    torchdyn==1.0.6 \
    "transformers>=4.51.0" \
    "accelerate>=1.0.0" \
    "tokenizers>=0.21.0" \
    "huggingface_hub>=0.30.0" \
    "dill>=0.3.4,<0.3.5" \
    matplotlib \
    numpy \
    seaborn \
    tqdm \
    pandas \
    scikit-learn \
    ogb \
    openpyxl \
    "ray[tune]>=2.0.0" \
    manim

# 9. Verify installation
echo ""
echo "=== Verification ==="
python -c "
import torch
import torch_geometric
import transformers
import torchdyn

print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU:            {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'PyG:            {torch_geometric.__version__}')
print(f'Transformers:   {transformers.__version__}')
print(f'Torchdyn:       {torchdyn.__version__}')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source venv/bin/activate"