# PROGRESS

## Status
- [x] Phase 0 — environment (results/phase0.json)
- [x] Phase 1 — data download + audit (results/phase1_*.json, figures/p1_*.png)
- [x] Phase 2 — splits (results/phase2_splits.json)
- [x] Phase 3 — preprocessing (results/phase3.json, figures/p3_*.png)
- [x] Phase 4 — embeddings, linear probe, UMAP (results/phase4_*.json)
- [x] Phase 5 — evaluation + failure gallery (results/phase5_*.json)
- [ ] Phase 6 — MegaDetector crops: 06 --mode crop, 07/08/09 --feature crop, 11_mitigation_compare
- [ ] Phase 7 — write-up (12_make_tables.py fills the README table)

## Next
Phase 5: 09 and 10 have been run for --feature full; notes pending commit. Then Phase 6 (06 --mode crop, 07/08/09 --feature crop, 11, 12).

## Blockers / decisions
- Pushing: remote `origin` = https://github.com/ray021207/Camera-trap-generalization.git is configured locally and NOT pushed (brief says ask before pushing).
- MegaDetector is not run locally: LILA's precomputed MDv5a detections are used (CPU inference on ~29k full-size images is impractical). State this wherever Phase 6 is described.

## Storage used (budget 6 GB)
venv ~1.2 GB + data/img 2.17 GB + small files, about 3.5 GB. Re-check with `du -sh .` after Phase 4.

## Resume instructions
Activate the venv (`.venv\Scripts\activate`), read "Reproduce" in README.md, run the next unchecked script.
data/ is gitignored. If data/ is missing, re-fetch the metadata, then re-run scripts 01, 02, 03:
- `curl -O https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz` and extract into data/meta/
- MegaDetector JSON: URL in README "Dataset"; unzip into data/md/
