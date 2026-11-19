"""Phase 2: build the three evaluation setups and MEASURE burst leakage.

All three use the same 10 "seen" cameras for training data and share the same label space.
  (a) random IMAGE split of the seen pool            -> frames of one burst can straddle train/test
  (b) random SEQUENCE split of the seen pool         -> a burst is entirely train, val or test
  (c) held-out LOCATION: model trained on (b)'s train, tested on the 10 cameras it never saw
      (CCT20 "trans" = trans_test; trans_val = 1 extra unseen camera, reported separately)
The seen-vs-unseen gap is then measured on ONE model, so the only thing that changes between the
two test sets is the camera.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, IMG, SEED, save_json, set_seed

FRACS = dict(train=0.7, val=0.1, test=0.2)


def assign(units, rng):
    units = np.array(sorted(units))
    rng.shuffle(units)
    n = len(units)
    n_tr, n_va = int(round(FRACS["train"] * n)), int(round(FRACS["val"] * n))
    m = {}
    for i, u in enumerate(units):
        m[u] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return m


def tiny(fn):
    """32x32 greyscale thumbnail: a crude but honest 'how similar are these two frames' fingerprint."""
    im = Image.open(IMG / fn).convert("L").resize((32, 32), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def main():
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA / "meta" / "table_sampled.csv")
    seen = df[df.pool == "seen"].copy()

    # (a) by image, (b) by sequence
    a = dict(zip(seen.id, [None] * len(seen)))
    ids = seen.id.tolist()
    am = assign(ids, rng)
    df["split_a"] = df.id.map(am)
    bm = assign(seen.seq_id.unique(), rng)
    df["split_b"] = df.seq_id.map(bm).where(df.pool == "seen")
    df["unseen_role"] = np.where(df.source == "trans_test", "unseen_test",
                                 np.where(df.source == "trans_val", "unseen_val", None))
    df["split_official"] = df.source  # model for the official-files sanity row
    df.to_csv(DATA / "meta" / "splits.csv", index=False)

    out = {"fractions": FRACS, "seed": SEED}
    s = df[df.pool == "seen"]
    for k in ("split_a", "split_b"):
        out[f"{k}_image_counts"] = s[k].value_counts().to_dict()
        out[f"{k}_class_counts_test"] = s[s[k] == "test"].label.value_counts().to_dict()
    out["unseen_image_counts"] = df[df.pool == "unseen"].unseen_role.value_counts().to_dict()

    # ---- hard checks that the splits mean what we claim ----
    tr_b, te_b = s[s.split_b == "train"], s[s.split_b == "test"]
    out["check_b_sequences_shared_train_test"] = int(len(set(tr_b.seq_id) & set(te_b.seq_id)))
    out["check_c_locations_shared_train_unseen"] = int(len(set(s.location) & set(df[df.pool == "unseen"].location)))
    assert out["check_b_sequences_shared_train_test"] == 0 and out["check_c_locations_shared_train_unseen"] == 0

    # ---- how often do burst-mates straddle the boundary? ----
    for name, col in (("a", "split_a"), ("b", "split_b")):
        tr, te = s[s[col] == "train"], s[s[col] == "test"]
        train_seqs = set(tr.seq_id)
        has_mate = te.seq_id.isin(train_seqs)
        out[f"{name}_test_images_with_burst_mate_in_train"] = int(has_mate.sum())
        out[f"{name}_share_test_images_with_burst_mate_in_train"] = float(has_mate.mean())
        out[f"{name}_n_test_images"] = int(len(te))
        out[f"{name}_share_test_sequences_split_across_train_test"] = float(te.seq_id.drop_duplicates().isin(train_seqs).mean())

    # ---- how near-identical are leaked frames? pixel distance, burst mate vs. other-sequence same camera ----
    te = s[s.split_a == "test"]; tr = s[s.split_a == "train"]
    mate_rows = te[te.seq_id.isin(set(tr.seq_id))].sample(n=min(1500, int(te.seq_id.isin(set(tr.seq_id)).sum())), random_state=SEED)
    cache = {}
    def T(fn):
        if fn not in cache:
            cache[fn] = tiny(fn)
        return cache[fn]
    d_mate, d_other, same_label_mate = [], [], []
    by_seq = tr.groupby("seq_id").file_name.apply(list).to_dict()
    by_loc = tr.groupby("location").file_name.apply(list).to_dict()
    seq_of = dict(zip(tr.file_name, tr.seq_id))
    for r in tqdm(mate_rows.itertuples(), total=len(mate_rows), mininterval=20):
        x = T(r.file_name)
        mate_fn = by_seq[r.seq_id][0]
        d_mate.append(float(np.abs(x - T(mate_fn)).mean()))
        pool = [f for f in rng.choice(by_loc[r.location], size=min(20, len(by_loc[r.location])), replace=False) if seq_of[f] != r.seq_id]
        d_other.append(float(np.mean([np.abs(x - T(f)).mean() for f in pool])))
        same_label_mate.append(bool(tr[tr.file_name == mate_fn].label.iloc[0] == r.label))
    out["pixel_distance_test_vs_burst_mate_in_train_mean_abs_diff_0to255"] = float(np.mean(d_mate))
    out["pixel_distance_test_vs_other_sequence_same_camera_mean_abs_diff_0to255"] = float(np.mean(d_other))
    out["share_test_images_closer_to_burst_mate_than_to_other_sequence"] = float(np.mean(np.array(d_mate) < np.array(d_other)))
    out["share_burst_mates_with_same_label"] = float(np.mean(same_label_mate))
    out["n_leakage_pairs_measured"] = len(d_mate)

    # ---- residual leakage in (b): same camera, nearby in time ----
    s2 = s.copy(); s2["t"] = pd.to_datetime(s2.date_captured)
    seqt = s2.groupby("seq_id").agg(t=("t", "min"), loc=("location", "first"), split=("split_b", "first"))
    gaps = []
    for l, g in seqt.groupby("loc"):
        trt = np.sort(g[g.split == "train"].t.values.astype("datetime64[s]").astype(np.int64))
        for t in g[g.split == "test"].t.values.astype("datetime64[s]").astype(np.int64):
            i = np.searchsorted(trt, t)
            c = [abs(t - trt[j]) for j in (i - 1, i) if 0 <= j < len(trt)]
            if c:
                gaps.append(min(c))
    gaps = np.array(gaps) / 60.0
    out["b_test_sequence_gap_minutes_to_nearest_train_sequence_same_camera_median"] = float(np.median(gaps))
    out["b_share_test_sequences_within_60min_of_a_train_sequence"] = float((gaps <= 60).mean())
    out["b_share_test_sequences_within_24h_of_a_train_sequence"] = float((gaps <= 1440).mean())

    save_json(out, "phase2_splits")
    print({k: v for k, v in out.items()})


if __name__ == "__main__":
    main()
