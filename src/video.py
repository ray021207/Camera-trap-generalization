"""Video pipeline: frames -> MegaDetector -> crop -> DINOv2-small -> linear probe -> per-visit vote.

The steps are the same ones the still-image crop model used (Phase 6), so a video score can be
compared directly with the unseen-camera benchmark:

    frame (sampled at `fps`)
      -> MegaDetector v5a, keep boxes with conf >= 0.2 (the same threshold as training)
      -> highest-confidence box, +10% margin, padded to a square, 224x224   (src/preprocess.py)
      -> DINOv2-small embedding [CLS, mean patch]                          (same as 06_extract_embeddings)
      -> probe "b" refit on the crop embeddings with its saved C           (same as 07_train_probe)
    visit = mean of the per-frame class probabilities over frames that have a detection;
            no detection in any frame -> "empty".

The input source is swappable: `iter_frames` takes a video file path or a webcam index.
"""
import os
import time
from pathlib import Path

import numpy as np

from src.common import DATA, ROOT, SEED, load_json

os.environ.setdefault("HF_HOME", str(DATA / "hf"))   # keep DINOv2 weights inside the project

MD_WEIGHTS = DATA / "models" / "md_v5a.0.0.pt"
MD_CONF = 0.2
DINO_ID = "facebook/dinov2-small"
PROBE_CACHE = DATA / "models" / "probe_b_crop.joblib"


# ---- input ------------------------------------------------------------------------------
def iter_frames(source, fps: float = 1.0, max_seconds: float | None = None):
    """Yield (t_seconds, PIL RGB image) sampled at about `fps` frames per second.

    source: path to a video file, or an int webcam index (0 = default camera).
    Files are sampled by their own timestamps; a live camera is sampled by wall-clock time."""
    import cv2
    from PIL import Image

    live = isinstance(source, int)
    cap = cv2.VideoCapture(source if live else str(source))
    if not cap.isOpened():
        raise IOError(f"cannot open video source {source!r}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1.0 / fps
    next_t, i, t_start = 0.0, 0, time.time()
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = (time.time() - t_start) if live else i / native
            i += 1
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-6 < next_t:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                break
            next_t += step
            yield t, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


# ---- detector ---------------------------------------------------------------------------
class Detector:
    def __init__(self, weights: Path = MD_WEIGHTS):
        from megadetector.detection.run_detector import load_detector
        self.model = load_detector(str(weights))

    def __call__(self, img):
        """Detections with conf >= MD_CONF, MegaDetector format: bbox = [x, y, w, h] as fractions."""
        r = self.model.generate_detections_one_image(img, detection_threshold=MD_CONF)
        return r.get("detections") or []


# ---- classifier -------------------------------------------------------------------------
def fit_probe_b_crop():
    """Refit probe 'b' on the stored crop embeddings, exactly as 07_train_probe.py did, with the C it chose."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    info = load_json("phase4_probe_crop")
    classes, C = info["classes"], info["models"]["b"]["C"]
    df = pd.read_csv(DATA / "meta" / "splits.csv")
    E = np.load(DATA / "emb" / "dinov2s_crop.npy").astype(np.float32)
    y = df.label.map({c: i for i, c in enumerate(classes)}).values
    tr = (df.split_b == "train").values
    sc = StandardScaler().fit(E[tr])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED)
    clf.fit(sc.transform(E[tr]), y[tr])
    return dict(scaler=sc, clf=clf, classes=classes, C=C, E=E, df=df)


def load_probe():
    """Load the cached probe, or refit it and check it reproduces the saved Phase 4 probabilities."""
    import joblib
    # joblib unpickles, so only load the cache this function wrote itself (data/ is local and gitignored).
    if PROBE_CACHE.exists():
        return joblib.load(PROBE_CACHE)
    p = fit_probe_b_crop()
    ref = np.load(DATA / "emb" / "proba_b_crop.npy")
    got = p["clf"].predict_proba(p["scaler"].transform(p["E"]))
    max_diff = float(np.abs(got - ref).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"refit probe does not match Phase 4 (max |dp| = {max_diff:.4g})")
    probe = dict(scaler=p["scaler"], clf=p["clf"], classes=p["classes"], C=p["C"], check_max_abs_diff=max_diff)
    PROBE_CACHE.parent.mkdir(exist_ok=True)
    joblib.dump(probe, PROBE_CACHE)
    return probe


class Classifier:
    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(DINO_ID).eval()
        probe = load_probe()
        self.scaler, self.clf, self.classes = probe["scaler"], probe["clf"], probe["classes"]

    def embed(self, crops):
        from src import preprocess as P
        x = self.torch.stack([P.to_tensor_normalized(P.resize_crop(c)) for c in crops])
        with self.torch.inference_mode():
            h = self.model(pixel_values=x).last_hidden_state
        return self.torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy().astype(np.float32)

    def proba(self, crops):
        E = self.embed(crops).astype(np.float16).astype(np.float32)   # training embeddings were stored as float16
        return self.clf.predict_proba(self.scaler.transform(E))


# ---- per-frame and per-visit ------------------------------------------------------------
def process_frame(img, det: Detector, cls: Classifier):
    """Return a dict for one frame: best detection (if any) and the 13 class probabilities for its crop."""
    from src import preprocess as P
    t0 = time.perf_counter()
    dets = det(img)
    t1 = time.perf_counter()
    out = dict(n_dets=len(dets), md_seconds=t1 - t0)
    if not dets:
        out.update(best_conf=0.0, best_category=None, bbox=None, proba=None, cls_seconds=0.0)
        return out
    best = max(dets, key=lambda d: d["conf"])
    p = cls.proba([P.crop_box(img, best["bbox"])])[0]
    out.update(best_conf=float(best["conf"]), best_category=best["category"], bbox=best["bbox"],
               proba=p, cls_seconds=time.perf_counter() - t1)
    return out


def draw(img, f, classes, t=None):
    """Return a BGR numpy frame with the detection box and predicted label drawn on it (for the live window)."""
    import cv2
    return draw_bgr(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), f, classes, t)


def draw_bgr(bgr, f, classes, t=None):
    """Draw in place on a BGR numpy frame and return it."""
    import cv2
    H, W = bgr.shape[:2]
    s = max(1.0, W / 960)                                  # scale text and lines with the frame size
    if f["proba"] is not None:
        p = np.asarray(f["proba"])
        k = int(p.argmax())
        x, y, w, h = f["bbox"]
        x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), int(2 * s))
        label = f"{classes[k]} {p[k]:.2f}  (md {f['best_conf']:.2f})"
        cv2.putText(bgr, label, (x0, max(int(25 * s), y0 - int(8 * s))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * s, (0, 255, 0), int(2 * s))
    else:
        cv2.putText(bgr, "no animal detected", (int(10 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * s, (0, 200, 255), int(2 * s))
    if t is not None:
        secs = f["md_seconds"] + f["cls_seconds"]
        cv2.putText(bgr, f"t={t:.1f}s  {secs:.2f} s/frame", (int(10 * s), H - int(45 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 255), int(2 * s))
    return bgr


def annotate_video(in_path, out_path, det: Detector, cls: Classifier, fps: float = 2.0,
                   max_seconds: float | None = None, progress=None):
    """Write a copy of a video with boxes and labels drawn on every frame.

    Pass 1 runs the pipeline on frames sampled at `fps`. Pass 2 re-reads the video and draws, on every
    frame, the result of the NEAREST sampled frame (so a box lags or leads by at most half a sampling
    interval). Output plays at the original frame rate as H.264 MP4 (plays in browsers).
    Returns (per-frame results of the sampled frames, visit vote)."""
    import cv2
    import imageio_ffmpeg

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {in_path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    dur = total / native if total else None
    if max_seconds is not None and dur:
        dur = min(dur, max_seconds)

    # pass 1: analyse sampled frames
    frames = []
    for t, img in iter_frames(in_path, fps=fps, max_seconds=max_seconds):
        f = process_frame(img, det, cls)
        f["t"] = round(t, 3)
        frames.append(f)
        if progress is not None and dur:
            progress(min(1.0, t / dur))
    times = np.array([f["t"] for f in frames]) if frames else np.zeros(0)

    # pass 2: draw the nearest result on every frame and encode
    cap = cv2.VideoCapture(str(in_path))
    writer = imageio_ffmpeg.write_frames(str(out_path), (W, H), fps=native, codec="libx264",
                                         pix_fmt_out="yuv420p", macro_block_size=2, quality=7)
    writer.send(None)
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = i / native
            if max_seconds is not None and t > max_seconds:
                break
            if len(times):
                draw_bgr(bgr, frames[int(np.abs(times - t).argmin())], cls.classes, t)
            writer.send(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            i += 1
    finally:
        cap.release()
        writer.close()
    return frames, vote(frames, cls.classes)


def vote(frames, classes):
    """Visit-level decision: average the probabilities of frames with a detection."""
    P = [f["proba"] for f in frames if f["proba"] is not None]
    if not P:
        return dict(pred="empty", conf=None, n_frames=len(frames), n_det_frames=0, proba=None)
    m = np.mean(P, axis=0)
    order = np.argsort(m)[::-1]
    return dict(pred=classes[order[0]], conf=float(m[order[0]]), second=classes[order[1]],
                second_conf=float(m[order[1]]), n_frames=len(frames), n_det_frames=len(P), proba=m)
