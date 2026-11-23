"""Video pipeline: frames -> MegaDetector -> crop -> DINOv2-small -> linear probe -> per-visit vote.

The steps are the same ones the still-image crop model used (Phase 6), so a video score can be
compared directly with the unseen-camera benchmark:

    frame (sampled at `fps`)
      -> animal detector, keep boxes with conf >= 0.2
      -> highest-confidence box, +10% margin, padded to a square, 224x224   (src/preprocess.py)
      -> DINOv2-small embedding [CLS, mean patch]                          (same as 06_extract_embeddings)
      -> probe "b" trained on the crop embeddings with its saved C         (same as 07_train_probe)
    visit = mean of the per-frame class probabilities over frames that have a detection;
            no detection in any frame -> "empty".

Detectors:
  "mdv6"  (default) MegaDetector v6 MDV6-apa-rtdetr-c, Apache-2.0 / CC BY 4.0, run with the vendored RT-DETRv2
          code in src/third_party (Apache-2.0). Needs only torch + torchvision.
  "mdv5a" (legacy)  MegaDetector v5a, the detector whose boxes the classifier was trained on (via LILA's published
          results). Needs `pip install megadetector`, which pulls in GPL-3.0 YOLOv5 code; weight licence unclear.
Pick with the CAMERATRAP_DETECTOR environment variable or Detector(name=...).

The input source is swappable: `iter_frames` takes a video file path or a webcam index.
"""
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from src.common import DATA, ROOT, SEED, load_json

os.environ.setdefault("HF_HOME", str(DATA / "hf"))   # keep DINOv2 weights inside the project

MODELS = DATA / "models"
DETECTOR = os.environ.get("CAMERATRAP_DETECTOR", "mdv6")
MD_CONF = 0.2
MDV5A_WEIGHTS = MODELS / "md_v5a.0.0.pt"
MDV6 = dict(file="MDV6-apa-rtdetr-c.pth", md5="3992888e26f627ee5a4ff9f116be97c3",
            url="https://zenodo.org/records/15398270/files/MDV6-apa-rtdetr-c.pth?download=1",
            config=ROOT / "src" / "third_party" / "rtdetrv2_pytorch" / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_megadetector.yml")
DINO_ID = "facebook/dinov2-small"
PROBE_NPZ = ROOT / "models" / "probe_b_crop.npz"   # committed: the trained classifier as plain arrays (no pickle)


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
def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_mdv6(dest_dir: Path = MODELS) -> Path:
    """Download the MDv6 RT-DETR-c weights once (322 MB) and check their md5."""
    p = Path(dest_dir) / MDV6["file"]
    if not p.exists():
        import requests
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        with requests.get(MDV6["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for b in r.iter_content(1 << 20):
                    f.write(b)
        tmp.rename(p)
    if _md5(p) != MDV6["md5"]:
        raise RuntimeError(f"{p} has the wrong checksum; delete it and run again")
    return p


def mdv6_state_dict(dest_dir: Path = MODELS) -> Path:
    """Tensors-only copy of the MDv6 EMA weights (~80 MB instead of 322 MB), so normal loading never has to
    unpickle. Built once from the md5-checked Zenodo checkpoint. That one unpickling step is the only place
    weights_only=False is used, and only on the file whose checksum was just verified."""
    import torch
    out = Path(dest_dir) / "MDV6-apa-rtdetr-c.ema.pt"
    if not out.exists():
        ck = torch.load(fetch_mdv6(dest_dir), map_location="cpu", weights_only=False)
        torch.save(ck["ema"]["module"] if "ema" in ck else ck["model"], out)
    return out


class MDv6Detector:
    """MegaDetector v6, RT-DETRv2 compact (MDV6-apa-rtdetr-c). Loader follows Pytorch-Wildlife's
    rtdetr_apache_base.py (MIT): resize to 640x640, ToTensor (no normalisation), model + post-processor."""
    name = "mdv6"

    def __init__(self, weights: Path | None = None):
        import torch
        import torchvision.transforms as T
        from src.third_party.rtdetrv2_pytorch.src.core import YAMLConfig   # importing the package registers the modules
        weights = Path(weights) if weights else mdv6_state_dict()
        # pretrained=False: skip downloading ImageNet backbone weights, the checkpoint below replaces them anyway
        cfg = YAMLConfig(str(MDV6["config"]), PResNet={"pretrained": False})
        cfg.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model, self.post = cfg.model.deploy().eval(), cfg.postprocessor.deploy().eval()
        self.tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        self.torch = torch

    def __call__(self, img):
        """Detections with conf >= MD_CONF in MegaDetector format: category '1' animal, '2' person, '3' vehicle;
        bbox = [x, y, w, h] as fractions of the image."""
        W, H = img.size
        with self.torch.inference_mode():
            out = self.post(self.model(self.tf(img)[None]), self.torch.tensor([[W, H]]))
        labels, boxes, scores = out[0] if isinstance(out, (list, tuple)) and isinstance(out[0], (list, tuple)) else out
        labels, boxes, scores = labels[0].numpy(), boxes[0].numpy(), scores[0].numpy()
        dets = []
        for c, (x0, y0, x1, y1), s in zip(labels, boxes, scores):
            if s < MD_CONF:
                continue
            x0, y0, x1, y1 = max(0.0, float(x0)), max(0.0, float(y0)), min(W, float(x1)), min(H, float(y1))
            if x1 <= x0 or y1 <= y0:
                continue
            dets.append(dict(category=str(int(c) + 1), conf=float(s),
                             bbox=[x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]))
        return dets


class MDv5aDetector:
    """Legacy: MegaDetector v5a through the `megadetector` package (not in requirements; GPL-3.0 YOLOv5 code)."""
    name = "mdv5a"

    def __init__(self, weights: Path = MDV5A_WEIGHTS):
        from megadetector.detection.run_detector import load_detector
        self.model = load_detector(str(weights))

    def __call__(self, img):
        r = self.model.generate_detections_one_image(img, detection_threshold=MD_CONF)
        return r.get("detections") or []


ONNX_DIR = DATA / "onnx"
ONNX_FILES = dict(detector="mdv6_rtdetr_c_fp16w.onnx", classifier="classifier_fp16w.onnx")   # the files the web demo loads


class OnnxDetector(MDv6Detector):
    """MDv6 through the exported ONNX graph (scripts/export_onnx.py), i.e. exactly what the web demo runs."""
    name = "mdv6onnx"

    def __init__(self, path: Path | None = None):
        import onnxruntime as ort
        import torchvision.transforms as T
        self.sess = ort.InferenceSession(str(path or ONNX_DIR / ONNX_FILES["detector"]), providers=["CPUExecutionProvider"])
        self.tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    def __call__(self, img):
        W, H = img.size
        labels, boxes, scores = self.sess.run(None, {"images": self.tf(img)[None].numpy(),
                                                     "orig_target_sizes": np.array([[W, H]], dtype=np.int64)})
        dets = []
        for c, (x0, y0, x1, y1), s in zip(labels[0], boxes[0], scores[0]):
            if s < MD_CONF:
                continue
            x0, y0, x1, y1 = max(0.0, float(x0)), max(0.0, float(y0)), min(W, float(x1)), min(H, float(y1))
            if x1 <= x0 or y1 <= y0:
                continue
            dets.append(dict(category=str(int(c) + 1), conf=float(s), bbox=[x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]))
        return dets


def Detector(name: str = DETECTOR):
    return {"mdv6": MDv6Detector, "mdv5a": MDv5aDetector, "mdv6onnx": OnnxDetector}[name]()


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


def probe_proba(probe, E):
    """Multinomial logistic regression on standardised embeddings: softmax((E - mean) / scale @ W.T + b)."""
    z = (np.asarray(E, np.float64) - probe["mean"]) / probe["scale"]
    logits = z @ probe["coef"].T + probe["intercept"]
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def export_probe(path: Path = PROBE_NPZ):
    """Refit probe 'b' from the stored Phase 4 crop embeddings, check it reproduces the saved Phase 4
    probabilities, and write it as plain arrays. Needs data/ (the training embeddings); the result is committed."""
    p = fit_probe_b_crop()
    probe = dict(mean=p["scaler"].mean_, scale=p["scaler"].scale_, coef=p["clf"].coef_, intercept=p["clf"].intercept_)
    ref = np.load(DATA / "emb" / "proba_b_crop.npy")
    max_diff = float(np.abs(probe_proba(probe, p["E"]) - ref).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"exported probe does not match Phase 4 (max |dp| = {max_diff:.4g})")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, classes=np.array(p["classes"]), C=p["C"], check_max_abs_diff=max_diff,
             **{k: v.astype(np.float64) for k, v in probe.items()})
    return path


def load_probe(path: Path = PROBE_NPZ):
    """The trained classifier from models/probe_b_crop.npz (plain arrays, loaded without pickle)."""
    if not path.exists():
        export_probe(path)
    z = np.load(path, allow_pickle=False)
    return dict(mean=z["mean"], scale=z["scale"], coef=z["coef"], intercept=z["intercept"],
                classes=[str(c) for c in z["classes"]], C=float(z["C"]), check_max_abs_diff=float(z["check_max_abs_diff"]))


class OnnxClassifier:
    """DINOv2 + probe through the exported (8-bit weight) ONNX graph, i.e. exactly what the web demo runs."""

    def __init__(self, path: Path | None = None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path or ONNX_DIR / ONNX_FILES["classifier"]), providers=["CPUExecutionProvider"])
        self.classes = load_probe()["classes"]

    def proba(self, crops):
        from src import preprocess as P
        x = np.stack([P.to_tensor_normalized(P.resize_crop(c)).numpy() for c in crops])
        return self.sess.run(None, {"pixel_values": x})[0]


def Classifier(name: str | None = None):
    """'torch' (default) or 'onnx'; defaults to 'onnx' when the ONNX detector is selected."""
    name = name or ("onnx" if DETECTOR == "mdv6onnx" else "torch")
    return OnnxClassifier() if name == "onnx" else TorchClassifier()


class TorchClassifier:
    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(DINO_ID).eval()
        self.probe = load_probe()
        self.classes = self.probe["classes"]

    def embed(self, crops):
        from src import preprocess as P
        x = self.torch.stack([P.to_tensor_normalized(P.resize_crop(c)) for c in crops])
        with self.torch.inference_mode():
            h = self.model(pixel_values=x).last_hidden_state
        return self.torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy().astype(np.float32)

    def proba(self, crops):
        E = self.embed(crops).astype(np.float16).astype(np.float32)   # training embeddings were stored as float16
        return probe_proba(self.probe, E)


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
    s = max(0.6, min(W, H * 16 / 9) / 1280)                # scale text and lines with the frame size
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, max(1, int(2 * s))

    def tag(text, x, y, fg, bg):
        """Text on a filled box; (x, y) is the box's bottom-left corner, kept inside the frame."""
        (tw, tht), base = cv2.getTextSize(text, font, fs, th)
        pad = int(6 * s)
        x = min(max(0, x), max(0, W - tw - 2 * pad))
        y = min(max(tht + base + 2 * pad, y), H)
        cv2.rectangle(bgr, (x, y - tht - base - 2 * pad), (x + tw + 2 * pad, y), bg, -1)
        cv2.putText(bgr, text, (x + pad, y - base - pad), font, fs, fg, th, cv2.LINE_AA)

    green = (60, 200, 60)
    if f["proba"] is not None and is_animal_frame(f, classes):
        p = np.asarray(f["proba"])
        k = int(p.argmax())
        x, y, w, h = f["bbox"]
        x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), green, max(2, int(3 * s)))
        tag(f"{classes[k]} {p[k]:.0%}", x0, y0, (255, 255, 255), green)
        below = y1 + int(34 * s)
        if below > H - int(80 * s):            # box reaches the bottom: put the tag just inside its top edge
            below = max(y0, int(44 * s)) + int(34 * s)
        tag(f"detector {f['best_conf']:.0%}", x0, below, (255, 255, 255), (70, 70, 70))
    else:
        tag("no animal", int(10 * s), int(44 * s), (255, 255, 255), (70, 70, 70))
    if t is not None:
        tag(f"t = {t:.1f} s", int(10 * s), H - int(40 * s), (255, 255, 255), (40, 40, 40))
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


def is_animal_frame(f, classes):
    """A frame counts as showing an animal when the detector boxed something AND the classifier's top class
    for that crop is not 'empty'. The 'empty' class was trained on crops of false detections, so it acts as
    a filter for detector false positives (e.g. MDv6 boxing title cards or logos)."""
    return f["proba"] is not None and classes[int(np.argmax(f["proba"]))] != "empty"


def vote(frames, classes):
    """Visit-level decision: average the probabilities of the animal frames (see is_animal_frame).
    No animal frame -> 'empty'. (The 'empty' filter was added after MDv6 produced false boxes on clip title
    cards; it is applied to both detectors and reported in results/video_eval_*.json.)"""
    det = [f for f in frames if f["proba"] is not None]
    P = [f["proba"] for f in det if is_animal_frame(f, classes)]
    if not P:
        conf = float(np.mean([f["proba"][classes.index("empty")] for f in det])) if det else None
        return dict(pred="empty", conf=conf, n_frames=len(frames), n_det_frames=len(det), n_animal_frames=0, proba=None)
    m = np.mean(P, axis=0)
    order = np.argsort(m)[::-1]
    return dict(pred=classes[order[0]], conf=float(m[order[0]]), second=classes[order[1]],
                second_conf=float(m[order[1]]), n_frames=len(frames), n_det_frames=len(det), n_animal_frames=len(P), proba=m)
