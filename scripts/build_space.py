"""Assemble the Hugging Face Space folder for the portfolio demo in build/space/ (gitignored).

    python scripts/build_space.py              # build (pre-computes the example results, ~1 min)
    python scripts/build_space.py --serve      # build, then run the built app locally to check it

Copies the app (space/), the pipeline code (src/video.py, preprocess.py, common.py, third_party/), the trained
probe (models/probe_b_crop.npz), the MegaDetector v6 EMA weights, and a few public-domain example clips with
their annotated results, so clicking an example on the website is instant. Upload the folder with
`hf upload <user>/<space> build/space . --repo-type space` (or huggingface_hub.upload_folder).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.common import DATA

OUT = ROOT / "build" / "space"
# (clip_id, button label) - public-domain clips only, so they can be redistributed with the Space
EXAMPLES = [("v046", "Skunk"), ("v048", "Bobcat"), ("v047", "Coyote"), ("v055", "Raccoon"),
            ("v049", "Gray fox (unknown species)"), ("v061", "Wolf family (unknown species)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    from src import video as V

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src").mkdir(parents=True)
    for f in ["app.py", "README.md", "requirements.txt"]:
        shutil.copy(ROOT / "space" / f, OUT / f)
    for f in ["video.py", "preprocess.py", "common.py"]:
        shutil.copy(ROOT / "src" / f, OUT / "src" / f)
    shutil.copytree(ROOT / "src" / "third_party", OUT / "src" / "third_party",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (OUT / "models").mkdir()
    shutil.copy(V.PROBE_NPZ, OUT / "models" / V.PROBE_NPZ.name)
    (OUT / "licenses").mkdir()
    shutil.copy(ROOT / "licenses" / "Apache-2.0.txt", OUT / "licenses" / "Apache-2.0.txt")
    (OUT / "data" / "models").mkdir(parents=True)
    shutil.copy(V.mdv6_state_dict(), OUT / "data" / "models" / "MDV6-apa-rtdetr-c.ema.pt")

    clips = pd.read_csv(DATA / "video_test" / "clips.csv", dtype=str).fillna("").set_index("clip_id")
    ex_dir = OUT / "examples"; ex_dir.mkdir()
    det, cls = V.Detector("mdv6"), V.Classifier()
    sys.path.insert(0, str(ROOT / "space"))
    meta = []
    for cid, label in EXAMPLES:
        c = clips.loc[cid]
        assert c.license.lower().startswith("public domain"), f"{cid} is not public domain"
        src = DATA / "video_test" / c.file
        vid, ann, res = f"{cid}.mp4", f"{cid}_annotated.mp4", f"{cid}.json"
        shutil.copy(src, ex_dir / vid)
        frames, v = V.annotate_video(src, ex_dir / ann, det, cls, fps=1.0)
        import app as A                     # reuse the app's summary/table formatting (loads models once more)
        json.dump(dict(summary=A.summary_md(v), rows=A.rows_for(frames)), open(ex_dir / res, "w"))
        credit = f"{c.title.split(': ', 1)[-1]} ({c.credit or c.author}, {c.license}), {c.source_page}"
        meta.append(dict(label=label, video=vid, annotated=ann, result=res, credit=credit, clip_id=cid))
        print(f"{cid} {label:30s} -> {v['pred']} ({v['conf']:.2f})" if v["conf"] else f"{cid} -> {v['pred']}")
    json.dump(meta, open(ex_dir / "examples.json", "w"), indent=1)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"built {OUT} ({size:.0f} MB)")
    if a.serve:
        subprocess.run([sys.executable, "app.py"], cwd=OUT)


if __name__ == "__main__":
    main()
