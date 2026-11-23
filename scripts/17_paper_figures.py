"""Figures for the paper that show the inputs step by step.

    python scripts/17_paper_figures.py

figures/p6_input_variants.png  The three classifier inputs compared in Phase 6, for the same frames:
    original frame with the MegaDetector box -> whole frame (baseline) -> background greyed out at the same
    resolution (control) -> MegaDetector crop (mitigation), each with probe b's prediction for that input.
    Rows: the day and night frames used in Fig. p3_preprocessing, plus one unseen-camera frame with a small
    animal, chosen by a fixed rule to illustrate magnification (tiny box, whole frame wrong, crop right).
figures/p8_video_pipeline.png  The video stage on one public-domain clip: sampled frames with the MegaDetector v6
    box and label, the 224x224 crop the classifier sees, and the clip-level vote.
results/paper_figures.json     Which frames were used and the predictions shown (every number in the figures).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import preprocess as P
from src.common import DATA, FIGURES, IMG, load_json, save_json

MD_CONF = 0.2
VIDEO_CLIP = "v048"          # USGS bobcat, public domain
VIDEO_TIMES = [2.0, 5.0, 6.0, 8.0, 9.0]


def best_det(md, fn):
    d = [x for x in md.get(fn, []) if x["conf"] >= MD_CONF]
    return max(d, key=lambda x: x["conf"]) if d else None


def draw_box(ax, img, bbox, color="#39d353", lw=2.5):
    W, H = img.size
    x, y, w, h = bbox
    ax.add_patch(mpatches.Rectangle((x * W, y * H), w * W, h * H, fill=False, ec=color, lw=lw))


def panel(ax, im, title, sub=None, sub_color="black"):
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#888"); s.set_linewidth(0.6)
    ax.set_title(title, fontsize=10)
    if sub:
        ax.set_xlabel(sub, fontsize=10, color=sub_color)


def input_variants(out):
    df = pd.read_csv(DATA / "meta" / "splits.csv")
    mdj = json.load(open(next((DATA / "md").glob("*.json"))))
    md = {r["file"]: r["detections"] for r in mdj["images"]}
    preds = {k: pd.read_csv(DATA / f"preds_{k}.csv").set_index("id") for k in ["full", "mask", "crop"]}
    p3 = load_json("phase3")
    rows = [("day frame", p3["demo_day"]["file"]), ("night (IR) frame", p3["demo_night"]["file"])]

    # small animal on an unseen camera: smallest box among frames where the whole-frame model is wrong and the
    # crop model is right (fixed rule, so the choice is reproducible; it is an illustration, not a statistic)
    t = df[df.unseen_role == "unseen_test"].copy()
    t["pf"] = t.id.map(preds["full"].pred_b); t["pc"] = t.id.map(preds["crop"].pred_b)
    t["pm"] = t.id.map(preds["mask"].pred_b)
    cand = []
    for r in t[(t.pf != t.label) & (t.pc == t.label) & (t.pm != t.label) & (t.label != "empty")].itertuples():
        d = best_det(md, r.file_name)
        if d and d["category"] == "1" and d["conf"] > 0.8:
            cand.append((d["bbox"][2] * d["bbox"][3], r.file_name))
    small = sorted(cand)[len(cand) // 10][1]             # a small but not the very smallest box (still visible)
    rows.append(("small animal, unseen camera", small))

    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.6 * len(rows)),
                             gridspec_kw=dict(width_ratios=[1.37, 1.37, 1.37, 1]))
    rec = []
    for i, (name, fn) in enumerate(rows):
        r = df[df.file_name == fn].iloc[0]
        img = P.load_rgb(IMG / fn)
        d = best_det(md, fn)
        pr = {k: (preds[k].loc[r.id, "pred_b"], float(preds[k].loc[r.id, "conf_b"])) for k in preds}
        col = lambda k: "#1a7f37" if pr[k][0] == r.label else "#b42318"
        fmt = lambda k: f"prediction: {pr[k][0]} ({pr[k][1]:.0%})"
        a = axes[i]
        panel(a[0], img, "1  original + MegaDetector box" if i == 0 else "", f"true: {r.label} · detector {d['conf']:.0%}")
        draw_box(a[0], img, d["bbox"])
        a[0].set_ylabel(name, fontsize=11)
        panel(a[1], P.resize_full(img), "2  whole frame (baseline)\n224×308" if i == 0 else "", fmt("full"), col("full"))
        panel(a[2], P.resize_full(P.mask_outside_box(img, d["bbox"])),
              "3  background greyed out (control)\n224×308, animal keeps its size" if i == 0 else "", fmt("mask"), col("mask"))
        panel(a[3], P.resize_crop(P.crop_box(img, d["bbox"])), "4  MegaDetector crop\n+10% margin, padded, 224×224" if i == 0 else "",
              fmt("crop"), col("crop"))
        rec.append(dict(row=name, file=fn, label=r.label, location=int(r.location), detector_conf=d["conf"],
                        box_area_frac=d["bbox"][2] * d["bbox"][3], predictions={k: dict(pred=v[0], conf=v[1]) for k, v in pr.items()}))
    fig.suptitle("The three classifier inputs compared in the mitigation and control experiments (probe b predictions; green = correct)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)
    return dict(rows=rec, small_animal_rule="unseen_test, whole-frame and mask wrong, crop right, animal box conf > 0.8; "
                                            "10th percentile of box area among candidates", n_candidates=len(cand))


def video_pipeline(out):
    from src import video as V
    clips = pd.read_csv(DATA / "video_test" / "clips.csv", dtype=str).set_index("clip_id")
    c = clips.loc[VIDEO_CLIP]
    det, cls = V.Detector("mdv6"), V.Classifier("torch")
    frames = []
    for t, img in V.iter_frames(DATA / "video_test" / c.file, fps=1.0):
        f = V.process_frame(img, det, cls); f["t"] = t; f["img"] = img
        frames.append(f)
    v = V.vote(frames, cls.classes)
    show = [min(frames, key=lambda f: abs(f["t"] - t)) for t in VIDEO_TIMES]
    fig, axes = plt.subplots(2, len(show), figsize=(3.3 * len(show), 5.6), gridspec_kw=dict(height_ratios=[1, 1.05]))
    rec = []
    for j, f in enumerate(show):
        img, a_top, a_bot = f["img"], axes[0, j], axes[1, j]
        animal = V.is_animal_frame(f, cls.classes)
        k = int(np.argmax(f["proba"])) if f["proba"] is not None else None
        panel(a_top, img, f"t = {f['t']:.0f} s")
        if f["proba"] is not None:
            draw_box(a_top, img, f["bbox"], "#39d353" if animal else "#f79009")
            a_top.set_xlabel(f"detector {f['best_conf']:.0%}", fontsize=10)
            crop = P.resize_crop(P.crop_box(img, f["bbox"]))
            lab = cls.classes[k]
            panel(a_bot, crop, "", f"{lab} {f['proba'][k]:.0%}" + ("" if animal else "\n(counted as no animal)"),
                  "#1a7f37" if lab == c.model_class else ("#667085" if not animal else "#b42318"))
        else:
            a_top.set_xlabel("no detection", fontsize=10)
            a_bot.axis("off")
        rec.append(dict(t=round(f["t"], 2), detector_conf=round(f["best_conf"], 4),
                        top=None if k is None else cls.classes[k], top_prob=None if k is None else round(float(f["proba"][k]), 4),
                        counted=bool(animal)))
    axes[0, 0].set_ylabel("sampled frame\n+ MegaDetector v6 box", fontsize=10)
    axes[1, 0].set_ylabel("crop seen by the classifier", fontsize=10)
    fig.suptitle(f"Video stage on a public-domain USGS clip ({c.true_species}): 1 frame/s, crops classified, frames vote.  "
                 f"Clip verdict: {v['pred']} ({v['conf']:.0%}) from {v['n_animal_frames']} of {v['n_frames']} frames", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)
    return dict(clip=VIDEO_CLIP, source=c.source_page, license=c.license, verdict=v["pred"], verdict_conf=v["conf"],
                n_frames=v["n_frames"], n_animal_frames=v["n_animal_frames"], shown=rec)


def main():
    res = dict(input_variants=input_variants(FIGURES / "p6_input_variants.png"),
               video_pipeline=video_pipeline(FIGURES / "p8_video_pipeline.png"))
    save_json(res, "paper_figures")
    print(json.dumps(res, indent=1, default=str)[:2500])


if __name__ == "__main__":
    main()
