"""Phase 1c: data audit. Counts from FULL metadata (57,864 images); pixel-based stats
(IR/night, brightness, blur) from the downloaded SAMPLE. Writes results/phase1_audit.json,
figures/p1_*.png and data/meta/image_stats.csv (reused by later phases)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGURES, IMG, SEED, save_json, set_seed

SEEN_C, UNSEEN_C = "#2a6f97", "#d1495b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def pixel_stats(fn):
    im = Image.open(IMG / fn).convert("RGB")
    im.thumbnail((320, 320))
    a = np.asarray(im).astype(np.float32)
    # Colour-ness: for an IR/greyscale frame R == G == B, so channel differences are ~0.
    colour = float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())
    g = a.mean(-1)
    blur = float(ndimage.laplace(g).var())  # low variance of the Laplacian = few sharp edges = blurry
    return colour, float(g.mean()), blur


def main():
    set_seed(SEED)
    FIGURES.mkdir(exist_ok=True)
    full = pd.read_csv(DATA / "meta" / "table_full.csv")
    sam = pd.read_csv(DATA / "meta" / "table_sampled.csv")
    out = {}
    pool_of = full.groupby("location").pool.first().to_dict()

    # ---------- per class ----------
    kept = full[full.keep]
    cls = kept.groupby(["label", "pool"]).size().unstack(fill_value=0)
    cls["total"] = cls.sum(1)
    cls = cls.sort_values("total", ascending=False)
    out["images_per_class_full_metadata"] = cls.to_dict("index")
    out["share_empty_overall"] = float((kept.label == "empty").mean())
    out["share_empty_seen_pool"] = float((kept[kept.pool == "seen"].label == "empty").mean())
    out["share_empty_unseen_pool"] = float((kept[kept.pool == "unseen"].label == "empty").mean())
    out["empty_images_per_source"] = {s: int(((kept.source == s) & (kept.label == "empty")).sum())
                                      for s in kept.source.unique()}
    vc = kept.label.value_counts()
    out["imbalance_ratio_max_over_min_class"] = float(vc.max() / vc.min())
    out["largest_class"], out["smallest_class"] = vc.idxmax(), vc.idxmin()
    non_empty = vc.drop("empty")
    out["imbalance_ratio_excluding_empty"] = float(non_empty.max() / non_empty.min())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    idx = np.arange(len(cls)); w = 0.4
    ax.bar(idx - w / 2, cls["seen"], w, color=SEEN_C, label="seen locations (10 cameras)")
    ax.bar(idx + w / 2, cls["unseen"], w, color=UNSEEN_C, label="unseen locations (10 cameras)")
    ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xticklabels(cls.index, rotation=40, ha="right")
    ax.set_ylabel("images (log scale)"); ax.set_title("Images per class (full CCT20 metadata)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES / "p1_class_counts.png", dpi=150); plt.close(fig)

    # ---------- per location ----------
    loc = kept.groupby(["location", "pool"]).size().reset_index(name="n").sort_values("n", ascending=False)
    out["images_per_location"] = {int(r.location): dict(pool=r.pool, n=int(r.n)) for r in loc.itertuples()}
    out["n_locations"] = int(kept.location.nunique())
    out["largest_location_share"] = float(loc.n.max() / loc.n.sum())
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(range(len(loc)), loc.n, color=[SEEN_C if p == "seen" else UNSEEN_C for p in loc.pool])
    ax.set_xticks(range(len(loc))); ax.set_xticklabels(loc.location, rotation=90)
    ax.set_xlabel("camera location id"); ax.set_ylabel("images")
    ax.set_title("Images per camera location (blue = seen, red = unseen)")
    fig.tight_layout(); fig.savefig(FIGURES / "p1_location_counts.png", dpi=150); plt.close(fig)

    # class mix differs per camera -> location is a (spurious) predictor of class
    mix = kept.groupby(["location", "label"]).size().unstack(fill_value=0)
    mix = mix.div(mix.sum(1), axis=0)
    order = list(loc.location)
    mix = mix.loc[order][cls.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mix.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(mix.shape[1])); ax.set_xticklabels(mix.columns, rotation=40, ha="right")
    ax.set_yticks(range(mix.shape[0])); ax.set_yticklabels([f"{l} ({pool_of[l]})" for l in mix.index])
    fig.colorbar(im, label="share of that camera's images"); ax.set_title("Class mix differs strongly between cameras")
    fig.tight_layout(); fig.savefig(FIGURES / "p1_class_by_location.png", dpi=150); plt.close(fig)
    out["class_mix_per_location_max_share"] = {int(l): dict(label=mix.loc[l].idxmax(), share=float(mix.loc[l].max())) for l in mix.index}

    # ---------- per sequence (burst) ----------
    seq = kept.groupby("seq_id").agg(n=("id", "size"), nlab=("label", "nunique"), nloc=("location", "nunique"),
                                      declared=("seq_num_frames", "first"), pool=("pool", "first"))
    out["n_sequences"] = int(len(seq))
    out["frames_per_sequence_distribution"] = {int(k): int(v) for k, v in seq.n.value_counts().sort_index().items()}
    out["mean_frames_per_sequence"] = float(seq.n.mean())
    out["share_images_in_multiframe_sequences"] = float(seq[seq.n > 1].n.sum() / seq.n.sum())
    out["share_sequences_with_mixed_labels"] = float((seq.nlab > 1).mean())
    out["sequences_spanning_multiple_locations"] = int((seq.nloc > 1).sum())
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals = seq.n.value_counts().sort_index()
    ax.bar(vals.index.astype(str), vals.values, color="#555")
    ax.set_xlabel("frames in the burst"); ax.set_ylabel("sequences"); ax.set_title("Burst length")
    fig.tight_layout(); fig.savefig(FIGURES / "p1_sequence_lengths.png", dpi=150); plt.close(fig)

    # ---------- time of day (metadata) ----------
    full["hour"] = pd.to_datetime(full.date_captured, errors="coerce").dt.hour
    hh = full[full.keep].groupby(["hour", "pool"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(hh.index, hh["seen"] / hh["seen"].sum(), color=SEEN_C, label="seen")
    ax.plot(hh.index, hh["unseen"] / hh["unseen"].sum(), color=UNSEEN_C, label="unseen")
    ax.set_xlabel("hour of day (camera clock)"); ax.set_ylabel("share of images"); ax.legend(frameon=False)
    ax.set_title("Capture time"); fig.tight_layout(); fig.savefig(FIGURES / "p1_hour_of_day.png", dpi=150); plt.close(fig)
    night_h = [h for h in hh.index if h >= 22 or h <= 5]
    out["share_captured_22h_to_5h"] = {p: float(hh.loc[night_h, p].sum() / hh[p].sum()) for p in hh.columns}

    # ---------- pixel statistics on the downloaded sample ----------
    stats = [pixel_stats(f) for f in tqdm(sam.file_name, mininterval=20)]
    sam["colourness"], sam["brightness"], sam["sharpness"] = zip(*stats)
    sam["is_ir"] = sam.colourness < 3.0
    sam.to_csv(DATA / "meta" / "image_stats.csv", index=False)
    out["sample_n_images"] = int(len(sam))
    out["sample_share_ir_greyscale"] = float(sam.is_ir.mean())
    out["sample_share_ir_by_pool"] = sam.groupby("pool").is_ir.mean().to_dict()
    out["sample_share_ir_by_class"] = sam.groupby("label").is_ir.mean().to_dict()
    out["sample_share_empty"] = float((sam.label == "empty").mean())
    out["ir_threshold_colourness"] = 3.0
    out["sample_ir_share_per_location"] = {int(k): float(v) for k, v in sam.groupby("location").is_ir.mean().items()}
    out["sample_brightness_mean_by_pool"] = sam.groupby("pool").brightness.mean().to_dict()
    out["sample_sharpness_median_by_pool"] = sam.groupby("pool").sharpness.median().to_dict()
    out["sample_class_counts_by_pool"] = {p: g.label.value_counts().to_dict() for p, g in sam.groupby("pool")}
    out["sample_images_per_location"] = {int(k): int(v) for k, v in sam.groupby("location").size().items()}

    fig, ax = plt.subplots(figsize=(9, 3.6))
    pl = sam.groupby("location").is_ir.mean().reindex(order)
    ax.bar(range(len(pl)), pl.values, color=[SEEN_C if pool_of[l] == "seen" else UNSEEN_C for l in pl.index])
    ax.set_xticks(range(len(pl))); ax.set_xticklabels(pl.index, rotation=90)
    ax.set_ylabel("share of IR / greyscale frames"); ax.set_xlabel("camera location id")
    ax.set_title("Night/IR frames per camera (sample)"); fig.tight_layout(); fig.savefig(FIGURES / "p1_ir_by_location.png", dpi=150); plt.close(fig)

    # ---------- sample grid: backgrounds differ ----------
    rng = np.random.default_rng(SEED)
    seen_l = rng.choice(sorted(sam[sam.pool == "seen"].location.unique()), 3, replace=False)
    uns_l = rng.choice(sorted(sam[sam.pool == "unseen"].location.unique()), 3, replace=False)
    locs = list(seen_l) + list(uns_l)
    out["grid_locations"] = {"seen": [int(x) for x in seen_l], "unseen": [int(x) for x in uns_l]}
    fig, axes = plt.subplots(6, 4, figsize=(11, 12.5))
    for r, l in enumerate(locs):
        pick = sam[sam.location == l].sample(4, random_state=SEED)
        for c, row in enumerate(pick.itertuples()):
            ax = axes[r, c]; ax.imshow(Image.open(IMG / row.file_name)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(SEEN_C if row.pool == "seen" else UNSEEN_C); s.set_linewidth(3)
            ax.set_title(row.label, fontsize=9)
        axes[r, 0].set_ylabel(f"loc {l}\n({'seen' if r < 3 else 'unseen'})", fontsize=10)
    fig.suptitle("Same task, different backgrounds (random frames; border blue = seen, red = unseen)", y=0.995)
    fig.tight_layout(); fig.savefig(FIGURES / "p1_location_grid.png", dpi=110); plt.close(fig)

    save_json(out, "phase1_audit")
    print({k: v for k, v in out.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
