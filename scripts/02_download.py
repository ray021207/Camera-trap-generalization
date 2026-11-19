"""Phase 1b: download the sampled images from LILA's public bucket and keep a resized copy.

Original frames are 2048x1494 (~430 KB). We keep max-side-640 JPEGs (quality 88), roughly
10x smaller, which is plenty for a 224 px model input and for cropping animals later.
Re-runnable: files that already exist are skipped.
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, IMG, IMG_URL, save_json

MAX_SIDE, QUALITY, WORKERS = 640, 88, 24
sess = requests.Session()
sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS))


def fetch(fn):
    dst = IMG / fn
    if dst.exists():
        return fn, "cached", 0
    for attempt in range(4):
        try:
            r = sess.get(IMG_URL.format(fn), timeout=60)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content))
            im.load()
            nbytes = len(r.content)
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            tmp = dst.with_suffix(".tmp")
            im.convert("RGB").save(tmp, format="JPEG", quality=QUALITY)
            tmp.replace(dst)
            return fn, "ok", nbytes
        except Exception as e:  # network blip, truncated file ...
            err = repr(e)
            time.sleep(1 + attempt * 2)
    return fn, "FAILED " + err, 0


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "meta" / "table_sampled.csv")
    files = df.file_name.tolist()
    t0, stats, fails, orig_bytes = time.time(), {"ok": 0, "cached": 0}, [], 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch, f) for f in files]
        for fu in tqdm(as_completed(futs), total=len(futs), mininterval=30):
            fn, st, nb = fu.result()
            orig_bytes += nb
            if st.startswith("FAILED"):
                fails.append((fn, st))
            else:
                stats[st] += 1
    disk = sum(p.stat().st_size for p in IMG.glob("*.jpg"))
    out = dict(requested=len(files), downloaded=stats["ok"], already_present=stats["cached"],
               failed=len(fails), failed_examples=fails[:10], seconds=round(time.time() - t0),
               original_gb_transferred=round(orig_bytes / 1e9, 2), resized_dir_gb=round(disk / 1e9, 3),
               max_side=MAX_SIDE, jpeg_quality=QUALITY, source=IMG_URL.format("<file_name>"))
    save_json(out, "phase1_download")
    print(out)


if __name__ == "__main__":
    main()
