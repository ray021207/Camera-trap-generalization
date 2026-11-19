# PROGRESS

## Status
- [x] Phase 0 — environment (results/phase0.json)
- [x] Phase 1 — data download + audit (results/phase1_*.json, figures/p1_*.png)
- [ ] Phase 2 — splits (scripts/04_splits.py written, not yet run)
- [ ] Phase 3 — preprocessing figure (scripts/05_preprocess_figure.py written, not run)
- [ ] Phase 4 — embeddings (06), linear probe (07), UMAP (08): written, not run
- [ ] Phase 5 — evaluation (09) + failure gallery (10): written, not run
- [ ] Phase 6 — MegaDetector crops: 06 --mode crop, 07/08/09 --feature crop, 11_mitigation_compare
- [ ] Phase 7 — write-up (12_make_tables.py fills the README table)

## Next
Run scripts/04_splits.py, then 05, then 06 --mode full (about 20-30 min on CPU).

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
