"""Phase 4c / 6c: 2-D UMAP of the embeddings, coloured by species and by camera location,
plus numbers that back up what the picture suggests.

    python scripts/08_umap.py --feature full
    python scripts/08_umap.py --feature crop

Numbers (results/phase4_umap_<feature>.json):
  * kNN purity: of a point's 10 nearest neighbours (cosine, in the full 768-d space, NOT the 2-D map),
    what share has the same species / the same camera? Compared with the share expected by chance.
  * camera decodability: a logistic regression that tries to name the camera from the embedding.
    Also run on EMPTY frames only, where the only information left is background.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGURES, SEED, save_json, set_seed

PER_LOC = 250


def knn_purity(Z, labels, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    lab = np.asarray(labels)
    same = (lab[idx] == lab[:, None]).mean()
    p = pd.Series(lab).value_counts(normalize=True).values
    return float(same), float((p ** 2).sum())   # observed, chance (prob. two random points share a label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["full", "crop", "mask"], required=True)
    feat = ap.parse_args().feature
    set_seed(SEED)
    df = pd.read_csv(DATA / "meta" / "splits.csv")
    E = np.load(DATA / "emb" / f"dinov2s_{feat}.npy").astype(np.float32)
    out = {"feature": feat}

    # equal number of points per camera so every camera is visible
    rng = np.random.default_rng(SEED)
    sel = np.concatenate([rng.choice(g.index.values, min(PER_LOC, len(g)), replace=False) for _, g in df.groupby("location")])
    S, Z = df.loc[sel].reset_index(drop=True), E[sel]
    out["n_points"] = int(len(S)); out["points_per_location_max"] = PER_LOC

    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=SEED).fit_transform(Z)

    # ---- plots ----
    species = sorted(S.label.unique())
    cmap_s = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    ax = axes[0]
    for i, c in enumerate(species):
        m = (S.label == c).values
        ax.scatter(xy[m, 0], xy[m, 1], s=5, color="#b8b8b8" if c == "empty" else cmap_s(i % 20), label=c, alpha=0.8)
    ax.legend(markerscale=3, fontsize=8, frameon=False, ncol=2, loc="best"); ax.set_title(f"UMAP of embeddings ({feat}) coloured by SPECIES")
    ax = axes[1]
    locs = sorted(S.location.unique())
    for i, l in enumerate(locs):
        m = (S.location == l).values
        seen = S.pool[m].iloc[0] == "seen"
        ax.scatter(xy[m, 0], xy[m, 1], s=6 if seen else 9, color=cmap_s(i % 20), marker="o" if seen else "x", label=f"{l}{'' if seen else ' (unseen)'}", alpha=0.8)
    ax.legend(markerscale=2.5, fontsize=7, frameon=False, ncol=2, loc="best", title="camera (o seen, x unseen)")
    ax.set_title(f"same map coloured by CAMERA LOCATION")
    for a in axes: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(FIGURES / f"p4_umap_{feat}.png", dpi=130); plt.close(fig)

    # ---- numbers ----
    o, c = knn_purity(Z, S.label.values)
    out["knn10_same_species_share"], out["knn10_same_species_chance"] = o, c
    o, c = knn_purity(Z, S.location.values)
    out["knn10_same_location_share"], out["knn10_same_location_chance"] = o, c
    ne = (S.label != "empty").values
    o, c = knn_purity(Z[ne], S.location.values[ne])
    out["knn10_same_location_share_animal_frames_only"], out["knn10_same_location_chance_animal_frames_only"] = o, c
    # same-species neighbours: of those neighbours that share the species, how many share the camera?
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    lab, loc = S.label.to_numpy(dtype=object), S.location.to_numpy()
    sp = lab[idx] == lab[:, None]
    out["knn10_same_location_given_same_species"] = float(((loc[idx] == loc[:, None]) & sp).sum() / sp.sum())

    # camera decodability, sequence-disjoint, seen cameras
    full = df[df.pool == "seen"]
    Ef = E[full.index.values]
    def probe(mask_fn):
        tr = ((full.split_b == "train") & mask_fn(full)).values
        te = ((full.split_b == "test") & mask_fn(full)).values
        sc = StandardScaler().fit(Ef[tr])
        clf = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED).fit(sc.transform(Ef[tr]), full.location[tr])
        acc = float((clf.predict(sc.transform(Ef[te])) == full.location[te].values).mean())
        maj = float(full.location[te].value_counts(normalize=True).max())
        return dict(accuracy=acc, majority_baseline=maj, n_test=int(te.sum()), n_train=int(tr.sum()))
    out["camera_probe_all_frames_10_seen_cameras"] = probe(lambda f: f.label == f.label)
    out["camera_probe_empty_frames_only"] = probe(lambda f: f.label == "empty")
    out["camera_probe_animal_frames_only"] = probe(lambda f: f.label != "empty")
    out["chance_uniform_10_way"] = 0.1
    save_json(out, f"phase4_umap_{feat}")
    print({k: v for k, v in out.items()})


if __name__ == "__main__":
    main()
