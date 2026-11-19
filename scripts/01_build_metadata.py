"""Phase 1a: merge the five CCT20 annotation files into one table and draw the image sample.

Sampling unit = SEQUENCE (burst of 1-5 frames), never a single frame, so burst structure
survives. Sequences are drawn uniformly at random inside each (source file x location)
stratum, so every camera keeps the same share of the data and no location disappears.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import META, DATA, SEED, save_json, set_seed

SAMPLE_FRACTION = 0.5
SOURCES = {  # annotation file -> (pool, role)
    "train": "seen", "cis_val": "seen", "cis_test": "seen",
    "trans_val": "unseen", "trans_test": "unseen",
}
# Classes with too few examples to train or test on (rule printed in results json).
MIN_SEEN, MIN_UNSEEN_TEST = 50, 20


def load():
    rows, names = [], {}
    for src, pool in SOURCES.items():
        fn = "train_annotations.json" if src == "train" else f"{src}_annotations.json"
        j = json.load(open(META / fn))
        names.update({c["id"]: c["name"] for c in j["categories"]})
        per = collections.defaultdict(set)
        for a in j["annotations"]:
            per[a["image_id"]].add(a["category_id"])
        for im in j["images"]:
            cats = sorted(per.get(im["id"], []))
            rows.append(dict(
                id=im["id"], file_name=im["file_name"], source=src, pool=pool,
                location=im["location"], seq_id=im["seq_id"], frame_num=im["frame_num"],
                seq_num_frames=im["seq_num_frames"], date_captured=im["date_captured"],
                width=im["width"], height=im["height"], n_labels=len(cats),
                label_id=cats[0] if len(cats) == 1 else -1))
    df = pd.DataFrame(rows)
    df["label"] = df.label_id.map(names).fillna("multi-label")
    return df, names


def main():
    set_seed(SEED)
    df, names = load()
    out = {"n_images_total": len(df), "images_per_source": df.source.value_counts().to_dict()}

    # Do official files share sequences? (matters for leakage reasoning in Phase 2)
    seq_src = df.groupby("seq_id").source.nunique()
    out["sequences_spanning_more_than_one_official_file"] = int((seq_src > 1).sum())
    out["n_sequences_total"] = int(df.seq_id.nunique())

    # label-space rule: enough data in the seen pool AND present in unseen test
    seen_n = df[df.pool == "seen"].label.value_counts()
    uns_n = df[df.source == "trans_test"].label.value_counts()
    keep = [c for c in seen_n.index if c != "multi-label"
            and seen_n[c] >= MIN_SEEN and uns_n.get(c, 0) >= MIN_UNSEEN_TEST]
    dropped = {c: dict(seen=int(seen_n.get(c, 0)), unseen_test=int(uns_n.get(c, 0)))
               for c in df.label.unique() if c not in keep and c != "multi-label"}
    out["label_rule"] = f"keep class if >= {MIN_SEEN} seen-pool images and >= {MIN_UNSEEN_TEST} trans_test images"
    out["classes_kept"] = sorted(keep)
    out["classes_dropped"] = dropped
    out["multi_label_images_dropped"] = int((df.label == "multi-label").sum())
    df["keep"] = df.label.isin(keep)
    out["images_after_filter"] = int(df.keep.sum())
    out["images_dropped_by_filter"] = int((~df.keep).sum())

    # sequence-level stratified sample. A sequence is dropped if ANY frame is filtered out? No:
    # filtered frames are removed first, then whole remaining sequences are sampled.
    rng = np.random.default_rng(SEED)
    cand = df[df.keep]
    picked = []
    for (src, loc), g in cand.groupby(["source", "location"]):
        seqs = np.array(sorted(g.seq_id.unique()))
        n = int(round(len(seqs) * SAMPLE_FRACTION))
        picked.extend(rng.choice(seqs, size=n, replace=False).tolist())
    picked = set(picked)
    sam = cand[cand.seq_id.isin(picked)].copy()
    out["sample_fraction_of_sequences"] = SAMPLE_FRACTION
    out["images_sampled"] = len(sam)
    out["sequences_sampled"] = int(sam.seq_id.nunique())
    out["sampled_images_per_source"] = sam.source.value_counts().to_dict()
    out["sampled_locations_seen"] = sorted(sam[sam.pool == "seen"].location.unique().tolist())
    out["sampled_locations_unseen"] = sorted(sam[sam.pool == "unseen"].location.unique().tolist())
    out["why_sample"] = ("Full CCT20 image archive is 6 GB (does not fit a 6 GB total budget) and "
                         "CPU embedding time scales with image count; half of all sequences keeps "
                         "every location and class while halving download and compute.")

    df.to_csv(DATA / "meta" / "table_full.csv", index=False)
    sam.to_csv(DATA / "meta" / "table_sampled.csv", index=False)
    save_json(out, "phase1_sampling")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
