"""Phase 6f: control experiment. Is the crop's benefit from REMOVING BACKGROUND, or just from giving the
animal MORE PIXELS?

    full  : whole frame, 224x308                     (animal small, background present)
    mask  : whole frame 224x308, outside-box greyed  (animal small, background removed)
    crop  : box cropped, padded, 224x224             (animal magnified, background mostly removed)

full -> mask isolates background removal; mask -> crop isolates magnification. All paired bootstrap
(same test bursts, same resamples).   python scripts/11b_control_mask.py   (needs preds_mask.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as M
from src.common import DATA, SEED, load_json, save_json, set_seed

N_BOOT = 2000


def main():
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA / "meta" / "splits.csv")
    for v in ("full", "crop", "mask"):
        df[v] = pd.read_csv(DATA / f"preds_{v}.csv").pred_b
    classes = sorted(df.label.unique()); K = len(classes); cid = {c: i for i, c in enumerate(classes)}
    sets = {"seen": df[df.split_b == "test"], "unseen": df[df.unseen_role == "unseen_test"]}
    T, W, present = {}, {}, {}
    for s, d in sets.items():
        y = d.label.map(cid).values
        for v in ("full", "crop", "mask"):
            T[(s, v)] = M.conf_tensor(y, d[v].map(cid).values, d.seq_id.values, K)
        present[s] = T[(s, "full")].sum(0).sum(1) > 0
        W[s] = np.stack([M.draw_weights(T[(s, "full")].shape[0], rng) for _ in range(N_BOOT)])
    B = {(s, v): M.bootstrap(T[(s, v)], present[s], N_BOOT, rng, W[s]) for s in sets for v in ("full", "crop", "mask")}
    out = {"n_boot": N_BOOT, "variants": {}, "paired_differences": {}}
    for v in ("full", "mask", "crop"):
        row = {}
        for s in sets:
            m = M.from_cm(T[(s, v)].sum(0), present[s])
            row[s] = dict(accuracy=float(m["accuracy"]), accuracy_ci95=M.ci(B[(s, v)][0]), macro_f1=float(m["macro_f1"]), macro_f1_ci95=M.ci(B[(s, v)][1]))
        for i, k in enumerate(["accuracy", "macro_f1"]):
            g = B[("seen", v)][i] - B[("unseen", v)][i]
            row[f"{k}_gap_seen_minus_unseen"] = dict(value=row["seen"][k] - row["unseen"][k], ci95=M.ci(g))
        out["variants"][v] = row
    for a, b in (("full", "mask"), ("mask", "crop"), ("full", "crop")):
        d = {}
        for i, k in enumerate(["accuracy", "macro_f1"]):
            for s in sets:
                diff = B[(s, b)][i] - B[(s, a)][i]
                d[f"{k}_{s}_change"] = dict(value=out["variants"][b][s][k] - out["variants"][a][s][k], ci95=M.ci(diff))
            gdiff = (B[("seen", b)][i] - B[("unseen", b)][i]) - (B[("seen", a)][i] - B[("unseen", a)][i])
            d[f"{k}_gap_change"] = dict(value=out["variants"][b][f"{k}_gap_seen_minus_unseen"]["value"] - out["variants"][a][f"{k}_gap_seen_minus_unseen"]["value"], ci95=M.ci(gdiff))
        out["paired_differences"][f"{a}_to_{b}"] = d
    for v in ("full", "crop", "mask"):
        try:
            j = load_json(f"phase4_umap_{v}")
            out["variants"][v]["camera_decodability"] = {x: j[x] for x in ["knn10_same_location_share", "camera_probe_all_frames_10_seen_cameras", "camera_probe_empty_frames_only"]}
        except FileNotFoundError:
            pass
    save_json(out, "phase6_control_mask")
    for v, r in out["variants"].items():
        print(v, {s: (round(r[s]["accuracy"], 3), round(r[s]["macro_f1"], 3)) for s in sets}, "gap acc", round(r["accuracy_gap_seen_minus_unseen"]["value"], 3))
    for k, d in out["paired_differences"].items():
        print(k, {kk: (round(vv["value"], 3), [round(x, 3) for x in vv["ci95"]]) for kk, vv in d.items() if "unseen" in kk or "gap" in kk})


if __name__ == "__main__":
    main()
