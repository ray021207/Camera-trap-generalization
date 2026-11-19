"""Phase 6e: did cropping the animal (MegaDetector) shrink the seen-vs-unseen gap?

    python scripts/11_mitigation_compare.py

Everything is PAIRED: the same test images and the same bootstrap resamples of sequences are used
for the full-frame model and the crop model, so differences are not drowned by sampling noise.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as M
from src.common import DATA, FIGURES, SEED, load_json, save_json, set_seed

N_BOOT = 2000
SEEN_C, UNSEEN_C = "#2a6f97", "#d1495b"


def main():
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA / "meta" / "splits.csv")
    pf, pc = pd.read_csv(DATA / "preds_full.csv"), pd.read_csv(DATA / "preds_crop.csv")
    df["full"], df["crop"] = pf.pred_b, pc.pred_b
    df["has_det"] = np.load(DATA / "emb" / "crop_has_detection.npy").astype(bool)
    classes = sorted(df.label.unique()); K = len(classes); cid = {c: i for i, c in enumerate(classes)}
    sets = {"seen": df[df.split_b == "test"], "unseen": df[df.unseen_role == "unseen_test"]}
    T, present, W = {}, {}, {}
    for s, d in sets.items():
        y = d.label.map(cid).values
        for f in ("full", "crop"):
            T[(s, f)] = M.conf_tensor(y, d[f].map(cid).values, d.seq_id.values, K)
        present[s] = T[(s, "full")].sum((0, 1)) > 0
        present[s] = T[(s, "full")].sum(0).sum(1) > 0
        W[s] = np.stack([M.draw_weights(T[(s, "full")].shape[0], rng) for _ in range(N_BOOT)])

    def boot(s, f):
        return M.bootstrap(T[(s, f)], present[s], N_BOOT, rng, W[s])
    B = {(s, f): boot(s, f) for s in sets for f in ("full", "crop")}
    out = {"n_boot": N_BOOT, "paired": True, "headline": {}, "deltas": {}}
    for s in sets:
        for f in ("full", "crop"):
            cm = T[(s, f)].sum(0); m = M.from_cm(cm, present[s])
            out["headline"][f"{s}_{f}"] = dict(accuracy=float(m["accuracy"]), accuracy_ci95=M.ci(B[(s, f)][0]),
                                               macro_f1=float(m["macro_f1"]), macro_f1_ci95=M.ci(B[(s, f)][1]))
    H = out["headline"]
    for i, key in enumerate(["accuracy", "macro_f1"]):
        gap = {f: (H[f"seen_{f}"][key] - H[f"unseen_{f}"][key]) for f in ("full", "crop")}
        gb = {f: B[("seen", f)][i] - B[("unseen", f)][i] for f in ("full", "crop")}
        out["deltas"][f"{key}_gap_full"] = dict(value=gap["full"], ci95=M.ci(gb["full"]))
        out["deltas"][f"{key}_gap_crop"] = dict(value=gap["crop"], ci95=M.ci(gb["crop"]))
        out["deltas"][f"{key}_gap_change_crop_minus_full"] = dict(value=gap["crop"] - gap["full"], ci95=M.ci(gb["crop"] - gb["full"]))
        for s in sets:
            d = B[(s, "crop")][i] - B[(s, "full")][i]
            out["deltas"][f"{key}_{s}_change_crop_minus_full"] = dict(value=H[f"{s}_crop"][key] - H[f"{s}_full"][key], ci95=M.ci(d))

    # per-class recall change, paired CIs
    pcr = {}
    for s in sets:
        rec = {}
        for f in ("full", "crop"):
            flat = T[(s, f)].reshape(T[(s, f)].shape[0], -1).astype(float)
            cms = np.stack([(w @ flat).reshape(K, K) for w in W[s]])
            rec[f] = cms.diagonal(axis1=1, axis2=2) / np.maximum(cms.sum(2), 1)
        pt = {f: M.from_cm(T[(s, f)].sum(0))["recall"] for f in ("full", "crop")}
        pcr[s] = {c: dict(support=int(T[(s, "full")].sum(0).sum(1)[i]), recall_full=float(pt["full"][i]), recall_crop=float(pt["crop"][i]),
                          change=float(pt["crop"][i] - pt["full"][i]), change_ci95=M.ci(rec["crop"][:, i] - rec["full"][:, i]))
                  for i, c in enumerate(classes)}
    out["per_class_recall_change"] = pcr

    # what MegaDetector did
    for s, d in sets.items():
        out[f"md_{s}_share_images_with_detection"] = float(d.has_det.mean())
        out[f"md_{s}_share_animal_frames_with_detection"] = float(d.has_det[d.label != "empty"].mean())
        out[f"md_{s}_share_empty_frames_with_detection"] = float(d.has_det[d.label == "empty"].mean())

    # fixed vs broken on unseen cameras (uses the failure categories of the baseline)
    ft = pd.read_csv(DATA / "failure_table.csv")[["id", "category"]]
    u = sets["unseen"].merge(ft, on="id")
    u["err_full"], u["err_crop"] = u.full != u.label, u.crop != u.label
    out["unseen_errors"] = dict(full=int(u.err_full.sum()), crop=int(u.err_crop.sum()),
                                fixed_by_crop=int((u.err_full & ~u.err_crop).sum()), broken_by_crop=int((~u.err_full & u.err_crop).sum()),
                                wrong_both=int((u.err_full & u.err_crop).sum()))
    cats = [c for c in u.category.dropna().unique() if c]
    out["unseen_errors_by_baseline_category"] = {c: dict(baseline_errors=int(((u.category == c)).sum()),
                                                         still_wrong_with_crop=int(((u.category == c) & u.err_crop).sum()))
                                                 for c in cats}
    out["unseen_errors_by_md_detection"] = {k: dict(n=int(len(g)), err_full=float(g.err_full.mean()), err_crop=float(g.err_crop.mean()))
                                            for k, g in u.groupby(u.has_det.map({True: "md_box_found", False: "no_md_box"}))}

    # does the crop embedding still encode the camera?  (from phase4 umap jsons)
    for k in ("full", "crop"):
        j = load_json(f"phase4_umap_{k}")
        out[f"camera_decodability_{k}"] = {x: j[x] for x in ["knn10_same_location_share", "knn10_same_location_chance",
                                                             "knn10_same_location_given_same_species",
                                                             "camera_probe_empty_frames_only", "camera_probe_animal_frames_only",
                                                             "knn10_same_species_share"]}
    save_json(out, "phase6_mitigation")

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, key, lab in zip(ax, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]):
        for k, (s, f) in enumerate([("seen", "full"), ("unseen", "full"), ("seen", "crop"), ("unseen", "crop")]):
            h = H[f"{s}_{f}"]; v = h[key]; lo, hi = h[key + "_ci95"]
            a.bar(k + (0.4 if k >= 2 else 0), v, yerr=[[v - lo], [hi - v]], color=SEEN_C if s == "seen" else UNSEEN_C, capsize=4, alpha=1 if f == "crop" else 0.55)
            a.text(k + (0.4 if k >= 2 else 0), v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
        a.set_xticks([0.5, 2.9]); a.set_xticklabels(["full frame\n(baseline)", "MegaDetector crop\n(mitigation)"]); a.set_ylim(0, 1.05); a.set_title(lab)
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=SEEN_C, label="seen cameras"), Patch(color=UNSEEN_C, label="unseen cameras")], frameon=False, loc="lower right")
    fig.suptitle("Seen vs unseen cameras, before and after cropping (95% CI)"); fig.tight_layout()
    fig.savefig(FIGURES / "p6_gap_before_after.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ch = [pcr["unseen"][c]["change"] for c in classes]
    lo = [pcr["unseen"][c]["change"] - pcr["unseen"][c]["change_ci95"][0] for c in classes]; hi = [pcr["unseen"][c]["change_ci95"][1] - pcr["unseen"][c]["change"] for c in classes]
    ax.bar(range(K), ch, yerr=[lo, hi], color=[("#3b8f4a" if x > 0 else "#b5483a") for x in ch], capsize=3)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(K)); ax.set_xticklabels([f"{c}\n(n={pcr['unseen'][c]['support']})" for c in classes], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("recall change (crop - full)"); ax.set_title("Unseen cameras: which classes gained or lost with cropping")
    fig.tight_layout(); fig.savefig(FIGURES / "p6_class_change.png", dpi=150); plt.close(fig)
    print(json.dumps({k: out[k] for k in ["headline", "deltas", "unseen_errors"]}, indent=1))


if __name__ == "__main__":
    main()
