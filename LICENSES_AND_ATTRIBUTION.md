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

## 3. Pretrained model: DINOv2-small
| | |
|---|---|
| Weights | `facebook/dinov2-small` on Hugging Face, downloaded at run time into `data/hf/` (gitignored, not redistributed). |
| Licence | **Apache License 2.0** (as shown in the model card metadata, https://huggingface.co/facebook/dinov2-small). Full text: `licenses/Apache-2.0.txt`. The model card states no further usage restrictions. |
| Cite | Maxime Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193, 2023. |

We rely on the Apache-2.0 licence stated on that model card. Weights are used frozen and unmodified.

## 4. Python packages
64 pinned packages (`requirements.txt`). `scripts/13_collect_licenses.py` read each package's declared licence and copied the licence files it ships into `licenses/third_party/` (regenerate locally; that folder is not committed). Table: `THIRD_PARTY_LICENSES.md`; machine-readable: `results/licenses.json`.
- All are permissive (MIT, BSD, Apache-2.0, PSF, ISC, MIT-CMU/HPND-style, Zlib and combinations of these). **No GPL, AGPL, LGPL or non-commercial package** was found (the script flags those; none flagged).
- Two packages are under MPL-2.0 (file-level copyleft): `certifi` and `tqdm` (dual MPL-2.0 AND MIT). We use them unmodified as installed dependencies and do not redistribute them.
- `tokenizers` (Apache-2.0) ships no licence file in its metadata; the Apache-2.0 text in `licenses/` applies.
- The packages themselves are not redistributed in this repo; only the version pins are.
- Key scientific libraries worth citing if you build on this: PyTorch, scikit-learn (Pedregosa et al., JMLR 2011), UMAP (McInnes, Healy, Melville, arXiv:1802.03426).

## 5. This repository's own code
No licence has been chosen for the code and writing in this repo yet, so by default all rights are reserved by the author. Adding a licence (for example MIT) is a decision for the repository owner; none of the components above requires a particular one. Third-party licences above remain unchanged by that choice.
