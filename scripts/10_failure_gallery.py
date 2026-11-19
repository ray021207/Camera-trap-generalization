"""Phase 5c: failure gallery for the baseline on UNSEEN cameras (+ error categories with counts).

    python scripts/10_failure_gallery.py

We use saved image grids instead of FiftyOne: FiftyOne needs a MongoDB binary and a web app
(hundreds of MB, fragile on Windows) and the 6 GB budget is tight. Grids are static, diffable, and
viewable straight from the repo.

Categories are assigned by RULES (first match wins, order below) from signals we can compute:
the true/predicted label, the IR flag, and the precomputed MegaDetector box. They are heuristics.
The same flags are also computed for ALL unseen-test images so you can see whether a flag is
over-represented among errors (lift = share among errors / share among everything).
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGURES, IMG, SEED, save_json, set_seed

MD_CONF = 0.2
TINY_AREA = 0.02        # box covers < 2% of the frame
BORDER = 0.01           # box edge within 1% of the frame edge = "touches the border"
BLUR_PCTL = 20          # in-box sharpness in the lowest 20% of unseen-test animal frames = "soft"
GROUPS = {"felid": {"bobcat", "cat"}, "canid": {"coyote", "dog"}, "small mammal": {"squirrel", "rodent", "rabbit"},
          "mid-size nocturnal": {"raccoon", "opossum", "skunk"}}
ORDER = ["empty frame confusion", "vehicle (car) confusion", "night / IR frame", "animal tiny, partial or undetected",
         "motion blur / soft", "look-alike species", "other / unexplained"]


def top_box(dets):
    d = [x for x in dets if x["conf"] >= MD_CONF]
    return max(d, key=lambda x: x["conf"]) if d else None


def box_sharpness(fn, bbox):
    g = np.asarray(Image.open(IMG / fn).convert("L"), dtype=np.float32)
    H, W = g.shape
    x, y, w, h = bbox
    r = g[int(y * H):max(int((y + h) * H), int(y * H) + 3), int(x * W):max(int((x + w) * W), int(x * W) + 3)]
    return float(ndimage.laplace(r).var()) if r.size > 9 else np.nan


def same_group(a, b):
    return any(a in g and b in g for g in GROUPS.values())


def main():
    set_seed(SEED)
    df = pd.read_csv(DATA / "meta" / "splits.csv")
    st = pd.read_csv(DATA / "meta" / "image_stats.csv")[["id", "is_ir", "sharpness", "brightness"]]
    pr = pd.read_csv(DATA / "preds_full.csv")
    df = df.merge(st, on="id").merge(pr, on="id")
    md = {r["file"]: r["detections"] for r in json.load(open(next((DATA / "md").glob("*.json"))))["images"]}

    u = df[df.unseen_role == "unseen_test"].copy().reset_index(drop=True)
    u["error"] = u.label != u.pred_b
    boxes = [top_box(md.get(f, [])) for f in u.file_name]
    u["has_box"] = [b is not None for b in boxes]
    u["box_area"] = [b["bbox"][2] * b["bbox"][3] if b else np.nan for b in boxes]
    u["touches_border"] = [bool(b and (b["bbox"][0] < BORDER or b["bbox"][1] < BORDER or b["bbox"][0] + b["bbox"][2] > 1 - BORDER
                                       or b["bbox"][1] + b["bbox"][3] > 1 - BORDER)) for b in boxes]
    u["box_sharp"] = [box_sharpness(f, b["bbox"]) if b else np.nan for f, b in zip(u.file_name, boxes)]
    animal = (u.label != "empty") & (u.label != "car") & u.has_box
    thr = float(np.nanpercentile(u.box_sharp[animal], BLUR_PCTL))

    u["f_empty"] = (u.label == "empty") | (u.pred_b == "empty")
    u["f_vehicle"] = (u.label == "car") | (u.pred_b == "car")
    u["f_ir"] = u.is_ir
    u["f_tiny_partial"] = (~u.f_empty) & (~u.has_box | (u.box_area < TINY_AREA) | u.touches_border)
    u["f_blur"] = (u.box_sharp < thr)
    u["f_lookalike"] = [same_group(a, b) for a, b in zip(u.label, u.pred_b)]

    def cat(r):
        if r.f_empty: return ORDER[0]
        if r.f_vehicle: return ORDER[1]
        if r.f_ir: return ORDER[2]
        if r.f_tiny_partial: return ORDER[3]
        if r.f_blur: return ORDER[4]
        if r.f_lookalike: return ORDER[5]
        return ORDER[6]
    u["category"] = [cat(r) if r.error else "" for r in u.itertuples()]
    u["wrong_conf"] = np.where(u.error, u.conf_b, np.nan)
    u[["id", "file_name", "label", "pred_b", "conf_b", "error", "category", "location"] + [c for c in u if c.startswith("f_")]].to_csv(
        DATA / "failure_table.csv", index=False)

    E = u[u.error]
    out = {"n_unseen_test": int(len(u)), "n_errors": int(len(E)), "error_rate": float(len(E) / len(u)),
           "thresholds": dict(md_conf=MD_CONF, tiny_box_area=TINY_AREA, border=BORDER, blur_percentile=BLUR_PCTL, blur_laplacian_var=thr),
           "category_priority_order": ORDER,
           "category_counts": {c: int((E.category == c).sum()) for c in ORDER},
           "category_share_of_errors": {c: float((E.category == c).mean()) for c in ORDER}}
    emp = E[E.category == ORDER[0]]
    out["empty_confusion_breakdown"] = {"animal_or_car_predicted_empty": int((emp.pred_b == "empty").sum()),
                                        "empty_predicted_as_animal_or_car": int((emp.label == "empty").sum())}
    out["top_confusions_true_to_pred"] = {f"{a} -> {b}": int(n) for (a, b), n in E.groupby(["label", "pred_b"]).size().sort_values(ascending=False).head(12).items()}
    flags = [c for c in u if c.startswith("f_")]
    out["flag_rates_non_exclusive"] = {f: dict(share_of_errors=float(E[f].mean()), share_of_all_unseen_test=float(u[f].mean()),
                                               lift=float(E[f].mean() / max(u[f].mean(), 1e-9))) for f in flags}
    out["errors_per_location"] = {int(l): dict(n=int(len(g)), errors=int(g.error.sum()), error_rate=float(g.error.mean())) for l, g in u.groupby("location")}
    out["unseen_test_share_with_no_md_box"] = float((~u.has_box).mean())
    out["unseen_test_animal_frames_with_no_md_box"] = float((~u.has_box[(u.label != "empty")]).mean())
    save_json(out, "phase5_failures")

    # --- gallery 1: worst (most confident) mistakes
    W = E.sort_values("conf_b", ascending=False).head(36)
    fig, axes = plt.subplots(6, 6, figsize=(16, 13))
    for ax, r in zip(axes.ravel(), W.itertuples()):
        ax.imshow(Image.open(IMG / r.file_name)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"true {r.label} -> {r.pred_b} ({r.conf_b:.2f})\nloc {r.location} | {textwrap.shorten(r.category, 28)}", fontsize=7.5)
    fig.suptitle("Worst mistakes on UNSEEN cameras: most confident wrong predictions (baseline, full frame)", y=0.995)
    fig.tight_layout(); fig.savefig(FIGURES / "p5_failure_gallery_worst.png", dpi=100); plt.close(fig)

    # --- gallery 2: random examples per category (to sanity-check the rules)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(ORDER), 6, figsize=(15, 2.5 * len(ORDER)))
    for i, c in enumerate(ORDER):
        g = E[E.category == c]
        pick = g.sample(min(6, len(g)), random_state=SEED) if len(g) else g
        for j in range(6):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            if j < len(pick):
                r = pick.iloc[j]; ax.imshow(Image.open(IMG / r.file_name)); ax.set_title(f"{r.label} -> {r.pred_b}", fontsize=8)
            else:
                ax.axis("off")
        axes[i, 0].set_ylabel(f"{textwrap.fill(c, 18)}\n(n={len(g)})", fontsize=8.5)
    fig.suptitle("Random examples per error category (unseen cameras)", y=0.998)
    fig.tight_layout(); fig.savefig(FIGURES / "p5_failure_categories.png", dpi=90); plt.close(fig)
    print(json.dumps({k: out[k] for k in ["n_errors", "error_rate", "thresholds", "category_counts", "empty_confusion_breakdown", "top_confusions_true_to_pred"]}, indent=1))
    print({f: (round(v["share_of_errors"], 3), round(v["share_of_all_unseen_test"], 3), round(v["lift"], 2)) for f, v in out["flag_rates_non_exclusive"].items()})


if __name__ == "__main__":
    main()
