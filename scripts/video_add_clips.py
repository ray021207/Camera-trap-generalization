"""Add test clips to data/video_test: download, (optionally) shrink, hash, and log each one in clips.csv.

    python scripts/video_add_clips.py batch.json

batch.json is a list of objects with the clips.csv fields (file_url, source_page, title, tier, true_species,
model_class, sensible_classes, region, license, license_url, author, credit, notes). clip_id and file are
assigned here. An entry may give "local_path" for a file already on disk (e.g. pulled out of a dataset zip);
file_url then records where it came from. Tiers: 1 = a training class on new cameras, 2 = related species that maps to a class,
3 = species outside the 13 classes (model_class = "none"; sensible_classes = forced choices that would
still be reasonable, "|"-separated, or empty).

Files above --max-mb are re-encoded with ffmpeg to at most 1280 px wide (H.264, CRF 26) to respect the
6 GB project budget; the original's size and SHA-256 are kept in the notes. Clips already in clips.csv
(same file_url) are skipped. Never add Caltech Camera Traps footage: it is the training data.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from src.common import DATA

VT = DATA / "video_test"
RAW = VT / "raw"
COLS = ["clip_id", "file", "tier", "true_species", "model_class", "sensible_classes", "region", "title",
        "source_page", "file_url", "license", "license_url", "author", "credit", "duration_s", "bytes",
        "sha256", "fixed_camera_verified", "cct_overlap_risk", "notes"]
# cct_overlap_risk: CCT images were collected by NPS (Santa Monica Mountains) and USGS in southern California.
#   high   = NPS Santa Monica Mountains footage (could be the same camera network as training)
#   medium = other USGS / NPS footage from southern California
#   low    = anywhere else
# The footage itself is never in CCT (CCT is stills only); the risk is that the CAMERA was seen in training.
UA = {"User-Agent": "CameraTrapVideoTest/0.1 (research script)"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def duration(p):
    import cv2
    c = cv2.VideoCapture(str(p))
    n, fps = c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FPS)
    c.release()
    return round(n / fps, 1) if fps and n > 0 else None


def shrink(src, dst):
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-crf", "26", "-preset", "medium",
                    "-an", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--max-mb", type=float, default=20)
    ap.add_argument("--pause", type=float, default=5, help="seconds between downloads")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    log = VT / "clips.csv"
    df = pd.read_csv(log, dtype=str) if log.exists() else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df:
            df[c] = ""
    nxt = max([int(x[1:]) for x in df.clip_id if str(x).startswith("v")] + [0]) + 1

    for e in json.load(open(a.batch)):
        if e["file_url"] in set(df.file_url) and not e.get("local_path"):
            print("skip (already logged):", e["title"]); continue
        if e.get("local_path") and e["title"] in set(df.title):
            print("skip (already logged):", e["title"]); continue
        cid = f"v{nxt:03d}"; nxt += 1
        ext = Path(e["file_url"].split("?")[0]).suffix.lower() or ".mp4"
        tmp = RAW / f"{cid}.download{ext}"
        if e.get("local_path"):                      # already downloaded (e.g. pulled out of a dataset zip)
            ext = Path(e["local_path"]).suffix.lower()
            tmp = RAW / f"{cid}.download{ext}"
            import shutil
            shutil.copyfile(e["local_path"], tmp)
        for attempt in range(0 if e.get("local_path") else 6):   # Wikimedia answers 429 when downloads come too fast
            with requests.get(e["file_url"], headers=UA, stream=True, timeout=120) as r:
                if r.status_code == 429 and attempt < 5:
                    wait = int(r.headers.get("Retry-After", 0) or 0) or 30 * (attempt + 1)
                    print(f"  rate-limited, waiting {wait} s"); time.sleep(wait); continue
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for b in r.iter_content(1 << 20):
                        f.write(b)
                break
        if not e.get("local_path"):
            time.sleep(a.pause)
        notes = e.get("notes", "")
        if tmp.stat().st_size > a.max_mb * 1e6 or ext not in (".mp4", ".webm", ".ogv", ".ogg"):
            out = RAW / f"{cid}.mp4"
            orig = (tmp.stat().st_size, sha256(tmp))
            shrink(tmp, out)
            tmp.unlink()
            notes = (notes + "; " if notes else "") + f"re-encoded to <=1280px H.264 CRF26 from original ({orig[0]} bytes, sha256 {orig[1]})"
        else:
            out = RAW / f"{cid}{ext}"
            tmp.rename(out)
        row = {**{c: "" for c in COLS}, **{k: v for k, v in e.items() if k in COLS}}
        row.update(clip_id=cid, file=f"raw/{out.name}", duration_s=duration(out), bytes=out.stat().st_size,
                   sha256=sha256(out), notes=notes)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df[COLS].to_csv(log, index=False)
        print(f"{cid}: tier {row['tier']} {row['true_species']:28s} {row['duration_s']}s {out.stat().st_size / 1e6:.1f} MB")
    print(f"total in log: {len(df)} clips, {sum(int(b) for b in df.bytes if str(b).isdigit()) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
