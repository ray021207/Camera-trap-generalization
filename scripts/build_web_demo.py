"""Assemble the files the in-browser demo loads, in build/web/ (gitignored), ready to upload to a Hugging Face
model repo (model repos are free; the page itself is static and lives on GitHub Pages).

    python scripts/export_onnx.py        # first: writes data/onnx/*.onnx
    python scripts/build_web_demo.py     # then: build/web/{models,examples,config.json,README.md}
    hf upload <user>/camera-trap-demo-models build/web . --repo-type model

Example results are computed with the same ONNX files the browser runs (src/video.py OnnxDetector /
OnnxClassifier), so clicking an example shows exactly what an upload of that clip would give.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import preprocess as P
from src import video as V
from src.common import DATA

OUT = ROOT / "build" / "web"
EXAMPLES = [("v046", "Skunk"), ("v048", "Bobcat"), ("v047", "Coyote"), ("v055", "Raccoon"),
            ("v049", "Gray fox (unknown species)"), ("v061", "Wolf family (unknown species)")]
FPS = 1.0

MODEL_CARD = """---
license: apache-2.0
tags: [camera-traps, wildlife, megadetector, dinov2, onnx]
---
# Camera-trap demo models (ONNX)

Files loaded by the in-browser camera-trap demo (onnxruntime-web). Project and code:
https://github.com/ray021207/Camera-trap-generalization

| File | What it is |
|---|---|
| `models/mdv6_rtdetr_c_fp16w.onnx` | MegaDetector v6 `MDV6-apa-rtdetr-c` with its post-processor. Input: frame stretched to 640x640, values in [0,1], plus the original (W, H) as int64. |
| `models/classifier_fp16w.onnx` | Frozen DINOv2-small -> [CLS, mean patch] embedding -> linear probe (13 classes) -> softmax. Input: 224x224 ImageNet-normalised animal crop. |

Both graphs store their large weights as float16 and cast them back to float32 on load, so all arithmetic is float32
and only standard ONNX operators are used.

| `config.json` | Class names and preprocessing constants. |
| `examples/` | Public-domain example clips (USGS, NPS) and the demo's results on them. |

Checks against the PyTorch pipeline (23 test clips): same clip verdict on all 23, clip confidences within 0.005,
identical accuracy.

## Credits and licences
- **MegaDetector v6** (`MDV6-apa-rtdetr-c`), Microsoft AI for Good Lab / Pytorch-Wildlife. Weights from
  https://doi.org/10.5281/zenodo.15398270 (CC BY 4.0; Apache-2.0 per the MegaDetector model zoo). RT-DETRv2 architecture by
  lyuwenyu (Apache-2.0).
- **DINOv2-small**, Meta AI, https://huggingface.co/facebook/dinov2-small (Apache-2.0).
- **Linear probe** trained on Caltech Camera Traps CCT20 (CDLA-Permissive-1.0); Beery, Van Horn, Perona, ECCV 2018.
- **Example clips**: public domain; sources in `examples/examples.json`.
"""


def frame_record(f, classes):
    r = dict(t=round(f["t"], 3), det_conf=round(f["best_conf"], 4))
    if f["proba"] is not None:
        r["bbox"] = [round(float(b), 5) for b in f["bbox"]]
        r["probs"] = [round(float(p), 5) for p in f["proba"]]
    return r


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    for k, f in V.ONNX_FILES.items():
        shutil.copy(V.ONNX_DIR / f, OUT / "models" / f)
    classes = V.load_probe()["classes"]
    config = dict(classes=classes, detector=f"models/{V.ONNX_FILES['detector']}", classifier=f"models/{V.ONNX_FILES['classifier']}",
                  det_size=640, det_conf=V.MD_CONF, crop_margin=P.CROP_MARGIN, crop_size=P.CROP_SIZE, crop_fill=[124, 116, 104],
                  mean=list(P.IMAGENET_MEAN), std=list(P.IMAGENET_STD), fps=FPS, low_conf=0.5,
                  categories={"0": "animal", "1": "person", "2": "vehicle"})
    json.dump(config, open(OUT / "config.json", "w"), indent=1)
    (OUT / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    clips = pd.read_csv(DATA / "video_test" / "clips.csv", dtype=str).fillna("").set_index("clip_id")
    det, cls = V.OnnxDetector(), V.OnnxClassifier()
    (OUT / "examples").mkdir()
    meta = []
    for cid, label in EXAMPLES:
        c = clips.loc[cid]
        assert c.license.lower().startswith("public domain"), f"{cid} is not public domain"
        shutil.copy(DATA / "video_test" / c.file, OUT / "examples" / f"{cid}.mp4")
        frames = []
        for t, img in V.iter_frames(DATA / "video_test" / c.file, fps=FPS):
            f = V.process_frame(img, det, cls)
            f["t"] = t
            frames.append(f)
        v = V.vote(frames, classes)
        json.dump(dict(fps=FPS, frames=[frame_record(f, classes) for f in frames]), open(OUT / "examples" / f"{cid}.json", "w"))
        meta.append(dict(id=cid, label=label, video=f"examples/{cid}.mp4", result=f"examples/{cid}.json",
                         title=c.title.split(": ", 1)[-1], credit=c.credit or c.author, license=c.license, source=c.source_page,
                         python_verdict=v["pred"], python_conf=None if v["conf"] is None else round(v["conf"], 4)))
        print(f"{cid} {label:30s} -> {v['pred']} ({v['conf']:.2f})")
    json.dump(meta, open(OUT / "examples" / "examples.json", "w"), indent=1)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"built {OUT} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
