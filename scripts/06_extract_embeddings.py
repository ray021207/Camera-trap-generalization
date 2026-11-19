"""Phase 4a / 6a: turn every sampled image into a 768-number embedding with a FROZEN DINOv2-small.

    python scripts/06_extract_embeddings.py --mode full     # whole frame, 224x308
    python scripts/06_extract_embeddings.py --mode crop     # MegaDetector crop, padded, 224x224

Embedding = [CLS token (384) , mean of all patch tokens (384)] from the last layer, the same
recipe the DINOv2 paper uses for linear probing. Output rows follow data/meta/splits.csv order.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))   # keep model weights inside the project
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import preprocess as P
from src.common import DATA, IMG, SEED, save_json, set_seed

MODEL_ID = "facebook/dinov2-small"
MD_CONF = 0.2          # MegaDetector v5a "typical" detection threshold, from the results file's own metadata


class Frames(Dataset):
    def __init__(self, df, mode, md):
        self.files, self.mode, self.md = df.file_name.tolist(), mode, md

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        img = P.load_rgb(IMG / fn)
        if self.mode == "full":
            return P.to_tensor_normalized(P.resize_full(img)), 1
        dets = [d for d in self.md.get(fn, []) if d["conf"] >= MD_CONF]
        if dets:                                   # highest-confidence box of any type (animal / vehicle / person)
            best = max(dets, key=lambda d: d["conf"])
            return P.to_tensor_normalized(P.resize_crop(P.crop_box(img, best["bbox"]))), 1
        # nothing detected: feed the whole frame, letterboxed, so the classifier can learn "background only"
        return P.to_tensor_normalized(P.letterbox_pad_square(img, P.CROP_SIZE)), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "crop"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (speed test)")
    a = ap.parse_args()
    set_seed(SEED)
    torch.set_num_threads(8)

    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    df = pd.read_csv(DATA / "meta" / "splits.csv")
    if a.limit:
        df = df.head(a.limit)
    md = {}
    if a.mode == "crop":
        mdj = json.load(open(next((DATA / "md").glob("*.json"))))
        md = {r["file"]: r["detections"] for r in mdj["images"]}
    dl = DataLoader(Frames(df, a.mode, md), batch_size=32, num_workers=4, shuffle=False)

    embs, det = [], []
    t0 = time.time()
    with torch.inference_mode():
        for x, d in tqdm(dl, mininterval=30):
            h = model(pixel_values=x).last_hidden_state
            embs.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy().astype(np.float32))
            det.append(d.numpy())
    E = np.concatenate(embs)
    out_dir = DATA / "emb"; out_dir.mkdir(exist_ok=True)
    if not a.limit:
        np.save(out_dir / f"dinov2s_{a.mode}.npy", E.astype(np.float16))
        if a.mode == "crop":
            np.save(out_dir / "crop_has_detection.npy", np.concatenate(det))
    secs = time.time() - t0
    info = dict(model=MODEL_ID, mode=a.mode, n_images=len(E), dim=E.shape[1], seconds=round(secs),
                images_per_second=round(len(E) / secs, 1), md_conf_threshold=MD_CONF if a.mode == "crop" else None)
    print(info)
    if not a.limit:
        save_json(info, f"phase4_extract_{a.mode}")


if __name__ == "__main__":
    main()
