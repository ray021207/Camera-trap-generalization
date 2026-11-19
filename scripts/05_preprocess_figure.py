"""Phase 3: show what every preprocessing step does to the pixels of one day and one night frame,
and verify the normalisation constants against our own data. Writes results/phase3.json,
figures/p3_preprocessing.png, figures/p3_normalization.png."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import preprocess as P
from src.common import DATA, FIGURES, IMG, SEED, save_json, set_seed

MD_JSON = next((DATA / "md").glob("*.json"))


def load_md():
    j = json.load(open(MD_JSON))
    return {r["file"]: r["detections"] for r in j["images"]}


def pick(df, md, ir):
    """First animal frame of a mid-sized species with one confident, reasonably large MegaDetector box."""
    g = df[(df.pool == "seen") & (df.is_ir == ir) & (df.label.isin(["coyote", "bobcat", "raccoon"]))]
    for r in g.sample(frac=1, random_state=SEED).itertuples():
        d = [x for x in md.get(r.file_name, []) if x["category"] == "1" and x["conf"] > 0.9]
        if len(d) == 1 and 0.04 < d[0]["bbox"][2] * d[0]["bbox"][3] < 0.25:
            return r, d[0]["bbox"]
    raise RuntimeError("no demo image found")


def main():
    set_seed(SEED)
    df = pd.read_csv(DATA / "meta" / "image_stats.csv")
    md = load_md()
    out = {}
    rows = {"day": pick(df, md, False), "night (IR)": pick(df, md, True)}
    cols = ["1 original\n(stored copy)", "alt A: centre crop\n(rejected)", "alt B: pad to square\n(rejected)",
            "CHOSEN: resize\n224x308", "aug: h-flip", "aug: random\nresized crop", "aug: colour\njitter",
            "aug: greyscale", "mitigation: MD crop\n+ pad, 224x224"]
    fig, axes = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 5.4))
    for r, (name, (row, bbox)) in enumerate(rows.items()):
        img = P.load_rgb(IMG / row.file_name)
        steps = [img, P.center_crop_square(img), P.letterbox_pad_square(img), P.resize_full(img)]
        base = steps[3]
        steps += [P.aug_hflip(base), P.aug_random_resized_crop(base, SEED), P.aug_color_jitter(base, SEED),
                  P.aug_grayscale(base), P.resize_crop(P.crop_box(img, bbox))]
        for c, s in enumerate(steps):
            ax = axes[r, c]; ax.imshow(s); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{s.size[0]}x{s.size[1]} px", fontsize=8)
            if r == 0: ax.set_title(cols[c], fontsize=8.5)
        axes[r, 0].set_ylabel(f"{name}\n{row.label}", fontsize=10)
        out[f"demo_{name.split()[0]}"] = dict(file=row.file_name, label=row.label, location=int(row.location),
                                              md_bbox_xywh_norm=bbox, size_stored=list(img.size))
    fig.suptitle("Preprocessing, step by step (same frame, before and after each operation)", y=1.0)
    fig.tight_layout(); fig.savefig(FIGURES / "p3_preprocessing.png", dpi=130); plt.close(fig)

    # geometry facts
    W, H = out["demo_day"]["size_stored"]
    out["centre_crop_keeps_share_of_pixels"] = float(min(W, H) ** 2 / (W * H))
    out["centre_crop_loses_share_of_width"] = float(1 - min(W, H) / W)
    out["pad_square_share_of_canvas_that_is_padding"] = float(1 - (H / W))
    out["chosen_input_hw"] = [P.FULL_H, P.FULL_W]
    out["chosen_aspect_distortion_vs_4_3_pct"] = float(abs(P.FULL_W / P.FULL_H / (4 / 3) - 1) * 100)
    out["chosen_tokens_14px_patches"] = [P.FULL_H // 14, P.FULL_W // 14, (P.FULL_H // 14) * (P.FULL_W // 14)]
    out["crop_input_hw"] = [P.CROP_SIZE, P.CROP_SIZE]
    out["crop_context_margin"] = P.CROP_MARGIN

    # normalisation: raw vs normalised statistics, per image, plus dataset-wide check on 600 images
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
    for r, (name, (row, bbox)) in enumerate(rows.items()):
        img = P.resize_full(P.load_rgb(IMG / row.file_name))
        raw = np.asarray(img).astype(np.float32)
        x = P.to_tensor_normalized(img)
        for ch, col in enumerate("rgb"):
            axes[r, 0].hist(raw[..., ch].ravel(), bins=64, range=(0, 255), histtype="step", color=col)
            axes[r, 1].hist(x[ch].numpy().ravel(), bins=64, range=(-2.2, 2.7), histtype="step", color=col)
        axes[r, 0].set_title(f"{name}: raw pixel values (0..255)"); axes[r, 1].set_title(f"{name}: after (x/255 - mean)/std")
        out[f"norm_{name.split()[0]}"] = dict(raw_mean_per_channel=raw.mean((0, 1)).tolist(), raw_std_per_channel=raw.std((0, 1)).tolist(),
                                              normalised_mean_per_channel=x.mean((1, 2)).tolist(), normalised_std_per_channel=x.std((1, 2)).tolist(),
                                              tensor_shape=list(x.shape))
    fig.tight_layout(); fig.savefig(FIGURES / "p3_normalization.png", dpi=130); plt.close(fig)

    rs = df.sample(600, random_state=SEED)
    for tag, g in (("colour_frames", rs[~rs.is_ir]), ("ir_frames", rs[rs.is_ir]), ("all", rs)):
        arr = np.stack([np.asarray(P.resize_full(P.load_rgb(IMG / f))) for f in g.file_name]).astype(np.float32) / 255
        out[f"dataset_mean_std_{tag}"] = dict(n=int(len(g)), mean=arr.mean((0, 1, 2)).tolist(), std=arr.std((0, 1, 2)).tolist())
    out["imagenet_mean"], out["imagenet_std"] = P.IMAGENET_MEAN, P.IMAGENET_STD
    save_json(out, "phase3")
    print(json.dumps({k: v for k, v in out.items() if "norm" not in k}, indent=1))


if __name__ == "__main__":
    main()
