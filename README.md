# Do camera-trap species classifiers survive a new camera?

(Work in progress — completed in Phase 7. See PROGRESS.md.)

<!-- RESULTS_TABLE:START -->
<!-- RESULTS_TABLE:END -->

## Dataset
Caltech Camera Traps, "CCT20" ECCV-2018 benchmark split (Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018),
hosted by LILA BC: https://lila.science/datasets/caltech-camera-traps. The benchmark split has 57,864 images from 20 camera
locations (10 "cis"/seen, 10 "trans"/unseen); the full dataset has 243,100 images.
- Split metadata: https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz
- Images (sampled, fetched by URL): https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/FILE_NAME
- Precomputed MegaDetector v5a results (RDE-filtered), published by LILA: https://lila.science/public/lila-md-results/caltech-camera-traps_mdv5a.0.0_results.filtered_rde_0.150_0.850_10_0.300.json.zip
- License: Community Data License Agreement, Permissive, Version 1.0 (https://cdla.io/permissive-1-0/).
- Version string inside the files: "Caltech Camera Traps - ECCV18". Files fetched 2026-11-19.
- Used here: a seeded 50% sample of bursts (28,943 images, 2.17 GB after shrinking to max side 640 px). Details: `results/phase1_sampling.json`.

## Reproduce
Windows, Python 3.12, CPU only. Run from the repo root, in this order. `data/` is not in git.
```
uv venv --python 3.12 .venv ; .venv\Scripts\activate
uv pip install -r requirements.txt
# one-off: metadata (see PROGRESS.md "Resume instructions" for the exact downloads) into data/meta and data/md
python scripts/00_env_check.py              # phase 0: machine + budget
python scripts/01_build_metadata.py         # phase 1: merge annotation files, draw the burst-level sample
python scripts/02_download.py               # phase 1: download + shrink the sampled images (~15 min)
python scripts/03_audit.py                  # phase 1: audit, figures, per-image IR/sharpness stats
python scripts/03b_full_cct_context.py      # phase 1: empty share of the FULL dataset (metadata only)
python scripts/04_splits.py                 # phase 2: random-image / random-sequence / held-out-camera splits + leakage measurement
python scripts/05_preprocess_figure.py      # phase 3: preprocessing figures + normalisation statistics
python scripts/06_extract_embeddings.py --mode full   # phase 4: frozen DINOv2-small embeddings (~26 min on CPU)\npython scripts/07_train_probe.py --feature full      # phase 4: linear probes for splits a, b, official\npython scripts/08_umap.py --feature full             # phase 4: UMAP + camera-decodability numbers\npython scripts/09_evaluate.py --feature full           # phase 5: metrics, bootstrap CIs, confusion matrices\npython scripts/10_failure_gallery.py                 # phase 5: failure gallery + error categories\n```
