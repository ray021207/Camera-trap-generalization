"""Phase 1d: how does the CCT20 benchmark differ from the full Caltech Camera Traps dataset?
Downloads the 9 MB full image-level annotation file (no images) and counts empties, so the
the README can say how unrepresentative the benchmark's class balance is. Writes results/phase1_full_cct_context.json"""
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, LILA_BASE, save_json

URL = LILA_BASE + "/caltechcameratraps/labels/caltech_camera_traps.json.zip"


def main():
    dst = DATA / "meta" / "caltech_camera_traps.json"
    if not dst.exists():
        z = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=120).content))
        dst.write_bytes(z.read(z.namelist()[0]))
    j = json.load(open(dst))
    names = {c["id"]: c["name"] for c in j["categories"]}
    per = {}
    for a in j["annotations"]:
        per.setdefault(a["image_id"], set()).add(names[a["category_id"]])
    n = len(j["images"])
    empty = sum(1 for im in j["images"] if per.get(im["id"], set()) == {"empty"})
    out = dict(source=URL, n_images_full_dataset=n, n_empty_images_full_dataset=empty, share_empty_full_dataset=empty / n,
               n_locations_full_dataset=len({im.get("location") for im in j["images"]}))
    save_json(out, "phase1_full_cct_context")
    print(out)


if __name__ == "__main__":
    main()
