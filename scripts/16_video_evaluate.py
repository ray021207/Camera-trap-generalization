"""Video stage: score the per-clip (per-visit) predictions from 15_video_predict.py.

    python scripts/16_video_evaluate.py                     # 1 fps, MegaDetector v6 predictions (fills the README)
    python scripts/16_video_evaluate.py --detector mdv5a    # legacy v5a predictions, for comparison (README untouched)

Each clip is one visit. Clips from the same site are not independent (same camera or same study), so the
95% intervals come from a CLUSTER bootstrap that resamples whole sites (the `region` column), the same idea
as resampling whole bursts in Phase 5.

Reports
  - known species (tier 1 = a training class on a new camera, tier 2 = a related species that maps to a class):
    accuracy overall, by tier, and on clips with low risk of sharing a camera network with the training data;
  - unknown species (tier 3): how often the forced answer is a sensible one, and how confident it is;
  - a confidence cut-off sweep: what an "unsure, send to a human" rule would flag at each threshold.
Writes results/video_eval_<fps>fps.json, results/video_tables.md, results/video_clips.csv (provenance log,
no video files), and the VIDEO_TABLE block in README.md.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.common import DATA, RESULTS, SEED, save_json

VT = DATA / "video_test"
N_BOOT = 10000
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def cluster_ci(values, clusters, rng, n_boot=N_BOOT):
    """Mean of `values` with a 95% interval from resampling whole clusters."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if len(values) == 0:
        return dict(n=0, mean=None, ci=[None, None], n_clusters=0)
    u, inv = np.unique(clusters, return_inverse=True)
    s, c = np.bincount(inv, weights=values), np.bincount(inv)
    draws = rng.integers(0, len(u), size=(n_boot, len(u)))
    boot = s[draws].sum(1) / c[draws].sum(1)
    return dict(n=int(len(values)), mean=float(values.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                n_clusters=int(len(u)))


def fmt(r):
    if r["mean"] is None:
        return "-"
    return f"{r['mean']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--detector", default="mdv6", choices=["mdv6", "mdv5a"])
    a = ap.parse_args()
    tag = f"{a.fps:g}fps_{a.detector}"
    rng = np.random.default_rng(SEED)

    clips = pd.read_csv(VT / "clips.csv", dtype=str).fillna("")
    pred = pd.read_csv(VT / f"preds_clips_{tag}.csv", dtype={"clip_id": str})
    df = clips.merge(pred[["clip_id", "pred", "conf", "second", "n_frames", "n_det_frames"]], on="clip_id", how="inner")
    df["tier"] = df.tier.astype(int)
    df["conf"] = df.conf.fillna(0.0)
    df["correct"] = (df.pred == df.model_class).astype(float)
    df["sensible"] = [float(p in s.split("|")) if s else 0.0 for p, s in zip(df.pred, df.sensible_classes)]
    known, unknown = df[df.tier <= 2], df[df.tier == 3]

    out = dict(fps=tag, n_clips=len(df), n_sites=int(df.region.nunique()), n_boot=N_BOOT, seed=SEED,
               cluster="region (site)", tiers={int(k): int(v) for k, v in df.tier.value_counts().sort_index().items()},
               overlap_risk=df.cct_overlap_risk.value_counts().to_dict())
    out["known"] = dict(
        all=cluster_ci(known.correct, known.region, rng),
        tier1=cluster_ci(known[known.tier == 1].correct, known[known.tier == 1].region, rng),
        tier2=cluster_ci(known[known.tier == 2].correct, known[known.tier == 2].region, rng),
        low_overlap_risk=cluster_ci(known[known.cct_overlap_risk == "low"].correct, known[known.cct_overlap_risk == "low"].region, rng),
        median_conf_correct=float(known[known.correct == 1].conf.median()) if (known.correct == 1).any() else None,
        median_conf_wrong=float(known[known.correct == 0].conf.median()) if (known.correct == 0).any() else None,
        per_class={c: dict(n=int(len(g)), correct=int(g.correct.sum())) for c, g in known.groupby("model_class")},
        errors=known[known.correct == 0][["clip_id", "true_species", "pred", "conf"]].to_dict("records"))
    out["unknown"] = dict(
        n=int(len(unknown)),
        detected=int((unknown.n_det_frames > 0).sum()),
        sensible=cluster_ci(unknown.sensible, unknown.region, rng),
        median_conf=float(unknown.conf.median()) if len(unknown) else None,
        answers=unknown[["clip_id", "true_species", "pred", "conf", "sensible_classes"]].to_dict("records"))
    sweep = []
    for t in THRESHOLDS:
        flag = df.conf < t
        k, u = known.index, unknown.index
        sweep.append(dict(threshold=t,
                          known_correct_flagged=float(flag[k][known.correct == 1].mean()),
                          known_wrong_flagged=float(flag[k][known.correct == 0].mean()) if (known.correct == 0).any() else None,
                          unknown_flagged=float(flag[u].mean()) if len(u) else None))
    out["confidence_cutoff_sweep"] = sweep
    save_json(out, f"video_eval_{tag}")

    # provenance log for the repo (licences, sources, checksums); the videos themselves stay out of git
    clips.drop(columns=["file"]).to_csv(RESULTS / "video_clips.csv", index=False)

    K = out["known"]
    lines = [f"Video test, {out['n_clips']} clips from {out['n_sites']} sites, one visit per clip, {a.fps:g} frame/s, detector {a.detector} "
             f"(`results/video_eval_{tag}.json`). Accuracy with 95% site-cluster bootstrap intervals.", "",
             "| Clips | n | Accuracy [95% CI] |", "|---|---|---|",
             f"| Known species, all (tier 1 + 2) | {K['all']['n']} | {fmt(K['all'])} |",
             f"| Tier 1: training species, new camera | {K['tier1']['n']} | {fmt(K['tier1'])} |",
             f"| Tier 2: related species mapped to a class | {K['tier2']['n']} | {fmt(K['tier2'])} |",
             f"| Known species, low risk of sharing the training camera network | {K['low_overlap_risk']['n']} | {fmt(K['low_overlap_risk'])} |"]
    other = "mdv5a" if a.detector == "mdv6" else "mdv6"
    op = RESULTS / f"video_eval_{a.fps:g}fps_{other}.json"
    if op.exists():                     # same clips, other detector: side-by-side row
        import json
        O = json.load(open(op))["known"]
        name = "legacy MegaDetector v5a (the detector behind the training crops)" if other == "mdv5a" else "MegaDetector v6"
        lines.append(f"| Known species, all, with the {name} | {O['all']['n']} | {fmt(O['all'])} |")
    lines += ["", f"Unknown species (tier 3, n = {out['unknown']['n']}): the forced answer was a sensible relative in "
             f"{fmt(out['unknown']['sensible'])} of clips; median confidence {out['unknown']['median_conf']:.2f} "
             f"(known species: {K['median_conf_correct']:.2f} when right, {K['median_conf_wrong']:.2f} when wrong).", "",
             "| Confidence cut-off | Known, correct, flagged | Known, wrong, flagged | Unknown species flagged |", "|---|---|---|---|"]
    out_tables = RESULTS / ("video_tables.md" if a.detector == "mdv6" else f"video_tables_{a.detector}.md")
    pct = lambda v: "-" if v is None else format(v, ".0%")
    for s in sweep:
        lines.append(f"| < {s['threshold']:.1f} | {pct(s['known_correct_flagged'])} | "
                     f"{pct(s['known_wrong_flagged'])} | {pct(s['unknown_flagged'])} |")
    block = "\n".join(lines)
    out_tables.write_text(block + "\n", encoding="utf-8")
    rd = (ROOT / "README.md").read_text(encoding="utf-8")
    if "<!-- VIDEO_TABLE:START -->" in rd and a.detector == "mdv6" and a.fps == 1.0:
        rd = re.sub(r"(<!-- VIDEO_TABLE:START -->)(.*?)(<!-- VIDEO_TABLE:END -->)",
                    lambda m: m.group(1) + "\n" + block + "\n" + m.group(3), rd, flags=re.S)
        (ROOT / "README.md").write_text(rd, encoding="utf-8")
    print(block)


if __name__ == "__main__":
    main()
