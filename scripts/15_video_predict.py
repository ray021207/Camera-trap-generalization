"""Video stage: run the crop pipeline on every clip in data/video_test/clips.csv.

    python scripts/15_video_predict.py                 # all clips not yet processed, 1 frame per second
    python scripts/15_video_predict.py --fps 2 --force # re-run everything at 2 fps
    python scripts/15_video_predict.py --clips v046    # just these clip ids
    python scripts/15_video_predict.py --source 0 --seconds 30 --show   # live webcam demo in a window
    python scripts/15_video_predict.py --source data/video_test/raw/v046.mp4 --show   # watch one clip

Per-clip frame results are cached in data/video_test/frames/<clip_id>_<fps>fps.json, so adding clips
later only processes the new ones. Writes data/video_test/preds_clips.csv (one row per clip = one visit)
and results/video_predict.json (counts and speed; accuracy is computed by 16_video_evaluate.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src import video as V
from src.common import DATA, SEED, save_json, set_seed

VT = DATA / "video_test"


def run_source(source, fps, det, cls, max_seconds=None, live_print=False, show=False):
    frames = []
    if show:
        import cv2
        win = "CameraTrap - press q to stop"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for t, img in V.iter_frames(source, fps=fps, max_seconds=max_seconds):
        f = V.process_frame(img, det, cls)
        f["t"] = round(t, 3)
        if show:
            cv2.imshow(win, V.draw(img, f, cls.classes, t))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        f["proba"] = None if f["proba"] is None else [round(float(x), 5) for x in f["proba"]]
        frames.append(f)
        if live_print:
            top = "-" if f["proba"] is None else cls.classes[int(np.argmax(f["proba"]))]
            print(f"t={t:6.1f}s  dets={f['n_dets']}  md={f['best_conf']:.2f}  -> {top:9s}"
                  f"  ({f['md_seconds'] + f['cls_seconds']:.2f} s/frame)", flush=True)
    if show:
        cv2.waitKey(1500)                  # keep the last frame up briefly
        cv2.destroyAllWindows()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--clips", nargs="*", help="clip ids to run (default: all)")
    ap.add_argument("--force", action="store_true", help="ignore cached frame results")
    ap.add_argument("--source", help="live/demo mode: webcam index (0) or a video file path, instead of the clip list")
    ap.add_argument("--seconds", type=float, default=30, help="live test length (webcam only)")
    ap.add_argument("--show", action="store_true", help="open a window with boxes and labels drawn on each frame")
    a = ap.parse_args()
    set_seed(SEED)
    torch.set_num_threads(8)

    t0 = time.time()
    det, cls = V.Detector(), V.Classifier()
    load_s = time.time() - t0
    print(f"models loaded in {load_s:.0f} s")

    if a.source is not None:
        cam = a.source.isdigit()
        src = int(a.source) if cam else a.source
        frames = run_source(src, a.fps, det, cls, max_seconds=a.seconds if cam else None,
                            live_print=True, show=a.show)
        v = V.vote(frames, cls.classes)
        print(f"visit vote: {v['pred']} (conf {v['conf']}) from {v['n_det_frames']}/{v['n_frames']} frames")
        return

    clips = pd.read_csv(VT / "clips.csv", dtype=str)
    if a.clips:
        clips = clips[clips.clip_id.isin(a.clips)]
    cache = VT / "frames"; cache.mkdir(exist_ok=True)
    tag = f"{a.fps:g}fps"

    for _, c in clips.iterrows():
        cf = cache / f"{c.clip_id}_{tag}.json"
        if cf.exists() and not a.force:
            continue
        path = VT / c.file
        if not path.exists():
            print(f"{c.clip_id}: file missing ({c.file}), skipped"); continue
        ts = time.time()
        frames = run_source(path, a.fps, det, cls)
        json.dump(dict(clip_id=c.clip_id, fps=a.fps, classes=cls.classes, frames=frames,
                       wall_seconds=round(time.time() - ts, 2)), open(cf, "w"))
        v = V.vote(frames, cls.classes)
        print(f"{c.clip_id}: {len(frames)} frames, {v['n_det_frames']} with a detection -> {v['pred']}"
              f" ({v['conf'] if v['conf'] is None else round(v['conf'], 3)}); truth {c.true_species} / {c.model_class}"
              f"  [{time.time() - ts:.0f} s]", flush=True)

    # rebuild the clip table from every cached result at this fps
    allclips = pd.read_csv(VT / "clips.csv", dtype=str).set_index("clip_id")
    rows, md_s, cls_s, n_frames = [], [], [], 0
    for cf in sorted(cache.glob(f"*_{tag}.json")):
        r = json.load(open(cf))
        if r["clip_id"] not in allclips.index:
            continue
        frames = r["frames"]
        for f in frames:
            if f["proba"] is not None:
                f["proba"] = np.array(f["proba"])
            md_s.append(f["md_seconds"]); cls_s.append(f["cls_seconds"])
        n_frames += len(frames)
        v = V.vote(frames, r["classes"])
        meta = allclips.loc[r["clip_id"]]
        row = dict(clip_id=r["clip_id"], tier=meta.tier, true_species=meta.true_species, model_class=meta.model_class,
                   pred=v["pred"], conf=v["conf"], second=v.get("second"), second_conf=v.get("second_conf"),
                   n_frames=v["n_frames"], n_det_frames=v["n_det_frames"],
                   max_md_conf=max((f["best_conf"] for f in frames), default=0.0))
        if v["proba"] is not None:
            row.update({f"p_{k}": float(p) for k, p in zip(r["classes"], v["proba"])})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(VT / f"preds_clips_{tag}.csv", index=False)

    cls_nonzero = [s for s in cls_s if s > 0]
    save_json(dict(fps=a.fps, n_clips=len(out), n_frames=n_frames, md_conf=V.MD_CONF, md_weights=V.MD_WEIGHTS.name,
                   probe_C=V.load_probe()["C"], probe_check_max_abs_diff=V.load_probe()["check_max_abs_diff"],
                   model_load_seconds=round(load_s, 1),
                   md_seconds_per_frame_median=float(np.median(md_s)) if md_s else None,
                   classifier_seconds_per_crop_median=float(np.median(cls_nonzero)) if cls_nonzero else None,
                   torch_threads=torch.get_num_threads()),
              f"video_predict_{tag}")
    print(out[["clip_id", "tier", "model_class", "pred", "conf", "n_det_frames", "n_frames"]].to_string(index=False))


if __name__ == "__main__":
    main()
