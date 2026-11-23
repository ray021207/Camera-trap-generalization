# Licences and attribution

What each third-party component is, under what terms it is used here, what this repo does and does not redistribute, and what it asks you to cite. Verified on 2026-11-19 from the sources linked below. This is a good-faith engineering record, not legal advice.

Nothing here required signing up, accepting a click-through, or requesting access: the dataset, the model weights and every Python package are published under open licences that apply by use. The full licence texts are in `licenses/`.

## 1. Data: Caltech Camera Traps (CCT20 benchmark split)
| | |
|---|---|
| Licence | **Community Data License Agreement – Permissive, Version 1.0** (CDLA-Permissive-1.0). Stated on the dataset page https://lila.science/datasets/caltech-camera-traps . Full text: `licenses/CDLA-Permissive-1.0.md` (also https://cdla.dev/permissive-1-0/ ). |
| Hosted by | LILA BC (Labeled Information Library of Alexandria: Biology and Conservation), https://lila.science |
| Data creators | Images collected from the U.S. National Park Service (NPS) and the U.S. Geological Survey (USGS) "with help from Justin Brown and Erin Boydston" (the dataset's own `info.description`). The per-image `rights_holder` field names Justin Brown (52,912 images) and Erin Boydston (4,952 images) (`results/licenses.json`). |
| Cite | Sara Beery, Grant Van Horn, Pietro Perona. *Recognition in Terra Incognita.* ECCV 2018. |
| Contact listed by LILA | caltechcameratraps@gmail.com |

What CDLA-Permissive-1.0 asks, and how this repo complies:
- *If you publish the Data, include the agreement text or a link, mark changed files, and keep attribution.* The raw images and annotation files are **not** in this repo (`data/` is gitignored). The `figures/` folder does show example frames (and the Phase 3 figure shows deliberately altered ones). Those figures are **modified excerpts** of CCT20 images: resized, cropped, colour-/contrast-altered, annotated and arranged in grids. They are provided under the same CDLA-Permissive-1.0 terms with the attribution above; this file is the notice.
- *Results are not subject to the agreement.* The numbers in `results/`, the trained models and the written analysis are our own results.
- *Privacy.* A few frames show people walking past a camera. The data provider states it took reasonable care to have the right to publish the data; we use the public images unchanged in content and do not attempt to identify anyone.
- The CCT20 split metadata and images are fetched by script from LILA's public bucket at run time; nothing is re-hosted.

## 2. Animal detections: MegaDetector v5a results published by LILA
The Phase 6 mitigation uses the bounding boxes in LILA's published file `caltech-camera-traps_mdv5a.0.0_results.filtered_rde_0.150_0.850_10_0.300.json.zip` (https://lila.science/megadetector-results-for-camera-trap-datasets). **MegaDetector was not downloaded or run in this project**; we only read the box coordinates and confidences that LILA published.

| | |
|---|---|
| MegaDetector code | MIT licence (https://github.com/agentmorris/MegaDetector ; code written on or before 2023-04-28 is copyright Microsoft). Text: `licenses/MegaDetector-repo-MIT.txt`. We do not use or ship this code. |
| MegaDetector **v5a weights** | **Unclear / not MIT.** Microsoft's model zoo lists MegaDetectorV5a under **AGPL-3.0**; the Hugging Face model card for `agentmorris/megadetector` lists a custom "other" licence and notes the weights were trained with a GPL-licensed YOLOv5 framework. The sources disagree and we did not resolve it. We neither download, run nor redistribute the weights, so no weight licence is exercised here. |
| The LILA results JSON | The results page states **no licence or terms** for these files (checked 2026-11-19). We read it locally (`data/md/`, gitignored) and do **not redistribute** it or commit it. Only derived metrics (`results/*.json`) and figures showing crops are published. If you want to redistribute the detections, or run the MDv5 weights yourself, check the terms first; the newer MegaDetector v6 "MIT" variants are a licence-clear alternative. |
| Cite | Sara Beery, Dan Morris, Siyu Yang. *Efficient Pipeline for Camera Trap Image Review.* arXiv:1907.06772, 2019. And credit LILA BC for hosting and running the detector on this dataset. |

**Video stage (added 2026-11-22): MegaDetector v5a is run locally.** The image results above still use only LILA's published boxes. The video test (`src/video.py`, `scripts/15_video_predict.py`, `scripts/demo_app.py`) downloads `md_v5a.0.0.pt` from the MegaDetector GitHub release (md5 `ec1d7603ec8cf642d6e0cd008ba2be8c`, verified) into `data/models/` (gitignored) and runs it on CPU through the `megadetector` package (MIT). This uses the weights whose licence is unclear (see the row above). They are used unmodified for non-commercial research inference and are **not redistributed**. Before redistributing them, offering the demo as a public service, or any commercial use, resolve the weight licence or switch to a MegaDetector v6 "MIT" model. `megadetector` loads the weights through the `ultralytics-yolov5` package, which ships a **GPL-3.0** licence file (see section 4).

## 3. Pretrained model: DINOv2-small
| | |
|---|---|
| Weights | `facebook/dinov2-small` on Hugging Face, downloaded at run time into `data/hf/` (gitignored, not redistributed). |
| Licence | **Apache License 2.0** (as shown in the model card metadata, https://huggingface.co/facebook/dinov2-small). Full text: `licenses/Apache-2.0.txt`. The model card states no further usage restrictions. |
| Cite | Maxime Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193, 2023. |

We rely on the Apache-2.0 licence stated on that model card. Weights are used frozen and unmodified.

## 4. Python packages
122 pinned packages (`requirements.txt`; 64 before the video stage added `megadetector`, `opencv-python`, `gradio`, `imageio-ffmpeg` and their dependencies). `scripts/13_collect_licenses.py` reads each package's declared licence, copies the licence files it ships into `licenses/third_party/` (regenerate locally; that folder is not committed), and also checks the title of each shipped licence file, because declared metadata can be wrong. Table: `THIRD_PARTY_LICENSES.md`; machine-readable: `results/licenses.json`.
- The packages used by the image pipeline (Phases 0–7) are all permissive (MIT, BSD, Apache-2.0, PSF, ISC, MIT-CMU/HPND-style, Zlib and combinations of these).
- **Copyleft in the video stage:**
  - `ultralytics-yolov5` declares "Apache" in its metadata but ships a **GPL-3.0** licence file (upstream YOLOv5 is AGPL-3.0 in recent versions). It is pulled in by `megadetector` to run the v5a weights. It is not modified or redistributed here.
  - `imageio-ffmpeg` (BSD-2-Clause) bundles an **ffmpeg executable built with `--enable-gpl --enable-version3` (GPLv3)**. We call it as a separate program to encode MP4s and do not redistribute it.
  - Whoever picks a licence for this repo's own code (section 5) should note that the video scripts import GPL code at run time. Keeping the image pipeline permissive and treating the video stage as optional, or moving to a MegaDetector v6 MIT model, avoids the question.
- Two packages are under MPL-2.0 (file-level copyleft): `certifi` and `tqdm` (dual MPL-2.0 AND MIT). We use them unmodified as installed dependencies and do not redistribute them.
- `tokenizers` (Apache-2.0) ships no licence file in its metadata; the Apache-2.0 text in `licenses/` applies.
- The packages themselves are not redistributed in this repo; only the version pins are.
- Key scientific libraries worth citing if you build on this: PyTorch, scikit-learn (Pedregosa et al., JMLR 2011), UMAP (McInnes, Healy, Melville, arXiv:1802.03426).

## 4b. Video test clips
The video test uses public video clips, listed one per row in [results/video_clips.csv](results/video_clips.csv) with source page, file URL, author, credit, licence, licence URL and SHA-256. Licences used: public domain (USGS and NPS footage, including NPS clips mirrored on Wikimedia Commons and the Internet Archive) and Creative Commons (CC BY 4.0, CC BY-SA 4.0). The video files are kept locally in `data/video_test/` (gitignored) and are **not redistributed**. Some were re-encoded to save space (noted per clip, with the original's checksum). Nothing comes from Caltech Camera Traps. Clips from NPS Santa Monica Mountains or USGS southern California are flagged in `cct_overlap_risk`, because those programmes supplied the training images and could share cameras with them.

Attribution for the CC-licensed clips is the `author` / `credit` column of that file, each with its licence link. Clips shown with YouTube's standard licence were not downloaded.

## 5. This repository's own code
No licence has been chosen for the code and writing in this repo yet, so by default all rights are reserved by the author. Adding a licence (for example MIT) is a decision for the repository owner; none of the components above requires a particular one. Third-party licences above remain unchanged by that choice.
