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

    # ---- image properties that do NOT depend on the prediction (used for base-rate / error-rate tables)
    u["f_ir"] = u.is_ir
    u["f_no_md_box"] = ~u.has_box
    u["f_tiny_box"] = u.has_box & (u.box_area < TINY_AREA)
    u["f_box_touches_border"] = u.touches_border
    u["f_soft_box"] = (u.box_sharp < thr)
    u["f_true_empty"] = u.label == "empty"
    u["f_true_car"] = u.label == "car"
    u["f_tiny_partial_or_undetected"] = u.f_no_md_box | u.f_tiny_box | u.f_box_touches_border
    # ---- prediction-dependent flags (only meaningful for errors)
    u["c_empty"] = (u.label == "empty") | (u.pred_b == "empty")
    u["c_vehicle"] = (u.label == "car") | (u.pred_b == "car")
    u["c_lookalike"] = [(a != b) and same_group(a, b) for a, b in zip(u.label, u.pred_b)]

    def cat(r, order_ir_first):
        if r.c_empty: return ORDER[0]
        if r.c_vehicle: return ORDER[1]
        if order_ir_first and r.f_ir: return ORDER[2]
        if r.f_tiny_partial_or_undetected: return ORDER[3]
        if r.f_soft_box: return ORDER[4]
        if r.c_lookalike: return ORDER[5]
        if (not order_ir_first) and r.f_ir: return ORDER[2] + " (no other flag)"
        return ORDER[6]
    u["category"] = [cat(r, True) if r.error else "" for r in u.itertuples()]
    u["category_ir_demoted"] = [cat(r, False) if r.error else "" for r in u.itertuples()]
    u["wrong_conf"] = np.where(u.error, u.conf_b, np.nan)
    keep = ["id", "file_name", "label", "pred_b", "conf_b", "error", "category", "category_ir_demoted", "location"]
    u[keep + [c for c in u if c.startswith(("f_", "c_"))]].to_csv(DATA / "failure_table.csv", index=False)

    E = u[u.error]
    ORDER2 = [ORDER[0], ORDER[1], ORDER[3], ORDER[4], ORDER[5], ORDER[2] + " (no other flag)", ORDER[6]]
    out = {"n_unseen_test": int(len(u)), "n_errors": int(len(E)), "error_rate": float(len(E) / len(u)),
           "thresholds": dict(md_conf=MD_CONF, tiny_box_area=TINY_AREA, border=BORDER, blur_percentile=BLUR_PCTL, blur_laplacian_var=thr),
           "category_priority_order_primary": ORDER,
           "category_counts": {c: int((E.category == c).sum()) for c in ORDER},
           "category_share_of_errors": {c: float((E.category == c).mean()) for c in ORDER},
           "category_priority_order_ir_demoted": ORDER2,
           "category_counts_ir_demoted": {c: int((E.category_ir_demoted == c).sum()) for c in ORDER2},
           "category_share_of_errors_ir_demoted": {c: float((E.category_ir_demoted == c).mean()) for c in ORDER2}}
    emp = E[E.category == ORDER[0]]
    out["empty_confusion_breakdown"] = {"animal_or_car_predicted_empty": int((emp.pred_b == "empty").sum()),
                                        "empty_predicted_as_animal_or_car": int((emp.label == "empty").sum())}
    out["top_confusions_true_to_pred"] = {f"{a} -> {b}": int(n) for (a, b), n in E.groupby(["label", "pred_b"]).size().sort_values(ascending=False).head(12).items()}
    out["lookalike_confusions_share_of_errors"] = float(E.c_lookalike.mean())

    def table(d):
        res = {}
        for f in [c for c in u if c.startswith("f_")]:
            m = d[f].astype(bool)
            res[f] = dict(share_of_images=float(m.mean()), share_of_errors=float(d.error[m].sum() / max(d.error.sum(), 1)),
                          error_rate_if_flag=float(d.error[m].mean()) if m.any() else None,
                          error_rate_if_not_flag=float(d.error[~m].mean()) if (~m).any() else None)
        return res
    out["property_vs_error_all_unseen_test_images"] = table(u)
    out["property_vs_error_animal_frames_only"] = table(u[~u.label.isin(["empty", "car"])])
    out["share_unseen_test_images_ir_by_true_class_error_rate"] = {c: dict(ir_share=float(g.is_ir.mean()), error_rate_ir=float(g.error[g.is_ir].mean()) if g.is_ir.any() else None,
                                                                           error_rate_colour=float(g.error[~g.is_ir].mean()) if (~g.is_ir).any() else None)
                                                                  for c, g in u.groupby("label")}
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
    print(json.dumps({k: out[k] for k in ["n_errors", "error_rate", "category_counts", "category_counts_ir_demoted", "empty_confusion_breakdown", "lookalike_confusions_share_of_errors"]}, indent=1))
    print({f: (round(v["share_of_images"], 3), round(v["share_of_errors"], 3), v["error_rate_if_flag"] and round(v["error_rate_if_flag"], 3), v["error_rate_if_not_flag"] and round(v["error_rate_if_not_flag"], 3)) for f, v in out["property_vs_error_animal_frames_only"].items()})
    print({f: (round(v["share_of_images"], 3), round(v["share_of_errors"], 3), v["error_rate_if_flag"] and round(v["error_rate_if_flag"], 3), v["error_rate_if_not_flag"] and round(v["error_rate_if_not_flag"], 3)) for f, v in out["property_vs_error_all_unseen_test_images"].items()})


if __name__ == "__main__":
    main()
