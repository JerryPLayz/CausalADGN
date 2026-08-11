from pathlib import Path
import os

# Set cache folders
base = Path.cwd() / "hf"
HF_HOME_DIR = base / "home"
HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_DIR = base / "hub"
HF_HUB_DIR.mkdir(parents=True, exist_ok=True)

TRANS_CACHE_DIR = base / "transformers"
TRANS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(TRANS_CACHE_DIR)

# Make a checkpoint dir and other save dir
CHECKPOINT_DIR = Path.cwd() / "cp"   # used for in-progress training
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SAVES_DIR = Path.cwd() / "saves"  # to save the parts of the full pipeline
SAVES_DIR.mkdir(parents=True, exist_ok=True)
