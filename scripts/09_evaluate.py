"""Phase 5 / 6d: evaluate on all setups, with cluster-bootstrap 95% CIs.

    python scripts/09_evaluate.py --feature full     # baseline
    python scripts/09_evaluate.py --feature crop     # MegaDetector-crop mitigation

Per setup: accuracy, macro-F1 (+95% CI), per-class recall, confusion matrix, per-location accuracy.
Gaps (seen minus unseen) get their own CI from independent resamples of the two test sets.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as M
from src.common import DATA, FIGURES, SEED, save_json, set_seed

N_BOOT = 2000
SEEN_C, UNSEEN_C, A_C = "#2a6f97", "#d1495b", "#8a8a8a"


def setups(df):
    """name -> (mask over df, prediction column, description)"""
    return {
        "a_random_image": (df.split_a == "test", "pred_a", "(a) random IMAGE split, seen cameras"),
        "b_random_sequence": (df.split_b == "test", "pred_b", "(b) random SEQUENCE split, seen cameras"),
        "c_unseen_locations": (df.unseen_role == "unseen_test", "pred_b", "(c) held-out cameras (CCT20 trans_test), same model as (b)"),
        "c_extra_unseen_camera": (df.unseen_role == "unseen_val", "pred_b", "extra unseen camera 125 (CCT20 trans_val)"),
        "official_cis_test": (df.source == "cis_test", "pred_official", "official CCT20 cis_test (model: official train+cis_val)"),
        "official_trans_test": (df.source == "trans_test", "pred_official", "official CCT20 trans_test (model: official train+cis_val)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["full", "crop"], required=True)
    feat = ap.parse_args().feature
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA / "meta" / "splits.csv")
    pr = pd.read_csv(DATA / f"preds_{feat}.csv")
    assert (pr.id == df.id).all()
    df = pd.concat([df, pr.drop(columns="id")], axis=1)
    classes = sorted(df.label.unique()); K = len(classes); cid = {c: i for i, c in enumerate(classes)}
    out = {"feature": feat, "classes": classes, "n_boot": N_BOOT, "ci_method": "cluster (sequence) bootstrap, percentile 95%", "setups": {}}
    cache = {}
    for name, (mask, col, desc) in setups(df).items():
        d = df[mask.values]
        y, p = d.label.map(cid).values, d[col].map(cid).values
        T = M.conf_tensor(y, p, d.seq_id.values, K)
        cm = T.sum(0)
        present = cm.sum(1) > 0
        m = M.from_cm(cm, present)
        acc_b, f1_b = M.bootstrap(T, present, N_BOOT, rng)
        out["setups"][name] = dict(
            description=desc, n_images=int(len(d)), n_sequences=int(T.shape[0]),
            accuracy=float(m["accuracy"]), accuracy_ci95=M.ci(acc_b),
            macro_f1=float(m["macro_f1"]), macro_f1_ci95=M.ci(f1_b),
            per_class_recall={c: (None if np.isnan(m["recall"][i]) else float(m["recall"][i])) for i, c in enumerate(classes)},
            per_class_f1={c: float(m["f1"][i]) for i, c in enumerate(classes)},
            per_class_support={c: int(m["support"][i]) for i, c in enumerate(classes)},
            confusion_matrix_rows_true_cols_pred=cm.tolist(),
            per_location_accuracy={int(l): dict(n=int(len(g)), accuracy=float((g.label == g[col]).mean()))
                                   for l, g in d.groupby("location")},
            per_location_macro_f1_note="not computed: too few images of some classes per camera",
        )
        cache[name] = (T, present, acc_b, f1_b)

    # ---- gaps: seen minus unseen, CI from independent resamples ----
    def gap(a, b):
        Ta, pa, _, _ = cache[a]; Tb, pb, _, _ = cache[b]
        wa = np.stack([M.draw_weights(Ta.shape[0], rng) for _ in range(N_BOOT)])
        wb = np.stack([M.draw_weights(Tb.shape[0], rng) for _ in range(N_BOOT)])
        aa, fa = M.bootstrap(Ta, pa, N_BOOT, rng, wa); ab, fb = M.bootstrap(Tb, pb, N_BOOT, rng, wb)
        S = out["setups"]
        return dict(
            accuracy_gap=S[a]["accuracy"] - S[b]["accuracy"], accuracy_gap_ci95=M.ci(aa - ab),
            macro_f1_gap=S[a]["macro_f1"] - S[b]["macro_f1"], macro_f1_gap_ci95=M.ci(fa - fb))
    out["gaps"] = {
        "b_seen_minus_c_unseen": gap("b_random_sequence", "c_unseen_locations"),
        "a_random_image_minus_c_unseen": gap("a_random_image", "c_unseen_locations"),
        "a_random_image_minus_b_random_sequence": gap("a_random_image", "b_random_sequence"),
        "official_cis_minus_official_trans": gap("official_cis_test", "official_trans_test"),
    }
    save_json(out, f"phase5_eval_{feat}")

    # ---- figures ----
    S = out["setups"]
    show = [("a_random_image", "(a) random\nimage split", A_C), ("b_random_sequence", "(b) random\nsequence split", SEEN_C),
            ("c_unseen_locations", "(c) unseen\ncameras", UNSEEN_C)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, lab in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]):
        v = [S[k][key] for k, _, _ in show]
        err = np.array([[S[k][key] - S[k][key + "_ci95"][0], S[k][key + "_ci95"][1] - S[k][key]] for k, _, _ in show]).T
        ax.bar(range(3), v, yerr=err, color=[c for _, _, c in show], capsize=5)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _ in show]); ax.set_ylim(0, 1.0); ax.set_title(lab)
        for i, x in enumerate(v): ax.text(i, x + 0.03, f"{x:.3f}", ha="center", fontsize=9)
    fig.suptitle(f"Same frozen backbone + linear probe ({feat}), three ways to test (95% CI, sequence bootstrap)")
    fig.tight_layout(); fig.savefig(FIGURES / f"p5_split_comparison_{feat}.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, key, title in zip(axes, ["b_random_sequence", "c_unseen_locations"], ["seen cameras (b)", "unseen cameras (c)"]):
        cm = np.array(S[key]["confusion_matrix_rows_true_cols_pred"], float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(K)); ax.set_yticklabels(classes, fontsize=8)
        for i in range(K):
            for j in range(K):
                if cmn[i, j] >= 0.05: ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title}: % of each true class")
    fig.tight_layout(); fig.savefig(FIGURES / f"p5_confusion_{feat}.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.38
    rb = [S["b_random_sequence"]["per_class_recall"][c] or 0 for c in classes]
    rc = [S["c_unseen_locations"]["per_class_recall"][c] or 0 for c in classes]
    ax.bar(np.arange(K) - w / 2, rb, w, color=SEEN_C, label="seen cameras (b)"); ax.bar(np.arange(K) + w / 2, rc, w, color=UNSEEN_C, label="unseen cameras (c)")
    ax.set_xticks(range(K)); ax.set_xticklabels(classes, rotation=40, ha="right"); ax.set_ylabel("recall"); ax.legend(frameon=False); ax.set_title(f"Per-class recall ({feat})")
    fig.tight_layout(); fig.savefig(FIGURES / f"p5_per_class_recall_{feat}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    items = [(l, v["accuracy"], v["n"], "seen") for l, v in S["b_random_sequence"]["per_location_accuracy"].items()] + \
            [(l, v["accuracy"], v["n"], "unseen") for k in ("c_unseen_locations", "c_extra_unseen_camera") for l, v in S[k]["per_location_accuracy"].items()]
    items.sort(key=lambda t: (t[3] != "seen", -t[1]))
    ax.bar(range(len(items)), [i[1] for i in items], color=[SEEN_C if i[3] == "seen" else UNSEEN_C for i in items])
    ax.set_xticks(range(len(items))); ax.set_xticklabels([f"{i[0]}\n(n={i[2]})" for i in items], fontsize=7)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.set_title(f"Accuracy per camera ({feat}); blue = seen, red = unseen")
    fig.tight_layout(); fig.savefig(FIGURES / f"p5_per_location_{feat}.png", dpi=150); plt.close(fig)

    print({k: (round(v["accuracy"], 4), [round(x, 3) for x in v["accuracy_ci95"]], round(v["macro_f1"], 4), [round(x, 3) for x in v["macro_f1_ci95"]]) for k, v in S.items()})
    print({k: {kk: (round(vv, 4) if isinstance(vv, float) else [round(x, 3) for x in vv]) for kk, vv in g.items()} for k, g in out["gaps"].items()})


if __name__ == "__main__":
    main()
