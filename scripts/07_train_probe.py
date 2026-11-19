"""Phase 4b / 6b: linear probe = multinomial logistic regression on frozen embeddings.

    python scripts/07_train_probe.py --feature full
    python scripts/07_train_probe.py --feature crop

Three models, all with the same recipe (standardise -> L2 logistic regression, class-balanced,
C picked on that model's own SEEN-camera validation set; unseen cameras are never used to tune):
  a        trained on the random-IMAGE split's train part
  b        trained on the random-SEQUENCE split's train part  (also evaluated on unseen cameras = setup c)
  official trained on CCT20's own train + cis_val files (sanity row; note it has almost no empties)
Writes predictions for every image to data/preds_<feature>.csv and results/phase4_probe_<feature>.json.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, SEED, save_json, set_seed

C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0]


def fit(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr)
    best, curve = None, {}
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED)
        clf.fit(sc.transform(Xtr), ytr)
        f1 = f1_score(yva, clf.predict(sc.transform(Xva)), average="macro")
        curve[C] = float(f1)
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    return sc, best[2], best[1], curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["full", "crop"], required=True)
    feat = ap.parse_args().feature
    set_seed(SEED)
    df = pd.read_csv(DATA / "meta" / "splits.csv")
    E = np.load(DATA / "emb" / f"dinov2s_{feat}.npy").astype(np.float32)
    assert len(E) == len(df)
    classes = sorted(df.label.unique())
    y = df.label.map({c: i for i, c in enumerate(classes)}).values
    out = {"feature": feat, "classes": classes, "n_images": len(df), "models": {}}
    pred = pd.DataFrame({"id": df.id})

    specs = {
        "a": (df.split_a == "train", df.split_a == "val"),
        "b": (df.split_b == "train", df.split_b == "val"),
        "official": (df.source.isin(["train", "cis_val"]), df.split_b == "val"),  # val = b-val (seen cameras only)
    }
    for name, (tr, va) in specs.items():
        tr, va = tr.values, va.values
        if name == "official":      # b-val frames could overlap official-train frames; drop them from training
            tr = tr & ~va
        sc, clf, C, curve = fit(E[tr], y[tr], E[va], y[va])
        proba = clf.predict_proba(sc.transform(E))
        pred[f"pred_{name}"] = [classes[i] for i in proba.argmax(1)]
        pred[f"conf_{name}"] = proba.max(1)
        if name == "b":
            np.save(DATA / "emb" / f"proba_b_{feat}.npy", proba.astype(np.float32))
        out["models"][name] = dict(C=C, val_macro_f1_by_C=curve, n_train=int(tr.sum()), n_val=int(va.sum()))

    # quick-look metrics (the full evaluation with CIs is Phase 5)
    evalsets = {
        "a_random_image_test": (df.split_a == "test", "pred_a"),
        "b_random_sequence_test": (df.split_b == "test", "pred_b"),
        "c_unseen_locations_trans_test": (df.unseen_role == "unseen_test", "pred_b"),
        "c_unseen_extra_location_trans_val": (df.unseen_role == "unseen_val", "pred_b"),
        "official_cis_test": (df.source == "cis_test", "pred_official"),
        "official_trans_test": (df.source == "trans_test", "pred_official"),
    }
    out["quicklook"] = {}
    for k, (m, col) in evalsets.items():
        m = m.values
        out["quicklook"][k] = dict(n=int(m.sum()), accuracy=float(accuracy_score(df.label[m], pred[col][m])),
                                   macro_f1=float(f1_score(df.label[m], pred[col][m], average="macro", labels=sorted(df.label[m].unique()))))  # classes present in the ground truth
    pred.to_csv(DATA / f"preds_{feat}.csv", index=False)
    save_json(out, f"phase4_probe_{feat}")
    print({k: (v["n"], round(v["accuracy"], 4), round(v["macro_f1"], 4)) for k, v in out["quicklook"].items()})
    print({k: m["C"] for k, m in out["models"].items()})


if __name__ == "__main__":
    main()
