"""Shared helpers: paths, seeding, label space. Imported by every script."""
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
META = DATA / "meta" / "eccv_18_annotation_files"
IMG = DATA / "img"            # resized (max side 640 px) sampled images
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 42

# Public LILA BC bucket (Google Cloud mirror). Image names are relative to this.
LILA_BASE = "https://storage.googleapis.com/public-datasets-lila"
IMG_URL = LILA_BASE + "/caltech-unzipped/cct_images/{}"


def set_seed(seed: int = SEED) -> None:
    """Fix every RNG we use so a re-run reproduces the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # CPU inference is deterministic anyway
    except ImportError:
        pass


def save_json(obj, name: str) -> Path:
    """Write results/<name>.json. Every number quoted in a .md file must come from here."""
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{name}.json"
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return p


def load_json(name: str):
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)
