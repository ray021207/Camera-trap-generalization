# PROGRESS

## Status
- [x] Phase 0 — environment (results/phase0.json)
- [x] Phase 1 — data download + audit (results/phase1_*.json, figures/p1_*.png)
- [x] Phase 2 — splits (results/phase2_splits.json)
- [x] Phase 3 — preprocessing (results/phase3.json, figures/p3_*.png)
- [x] Phase 4 — embeddings, linear probe, UMAP (results/phase4_*.json)
- [x] Phase 5 — evaluation + failure gallery (results/phase5_*.json)
- [x] Phase 6 — MegaDetector crops + mask control (results/phase6_*.json)
- [x] Phase 7 — write-up, licences, final checks (results/phase7_final_checks.json)

## Next
All 7 phases done. Remaining, needs the owner: (1) decide whether to push to the GitHub remote (configured as `origin`, never pushed); (2) pick a licence for the repo's own code; (3) optional follow-ups listed in README 'What I would do next'.

## Licences (done)
LICENSES_AND_ATTRIBUTION.md + licenses/ + THIRD_PARTY_LICENSES.md written. Open points: MegaDetector v5a WEIGHT licence is unclear (we only use LILA's precomputed boxes; results JSON has no stated licence, not redistributed); the repo's own code licence is not chosen yet (owner's decision).

## Blockers / decisions
- Pushing: remote `origin` = https://github.com/ray021207/Camera-trap-generalization.git is configured locally and NOT pushed (brief says ask before pushing).
- MegaDetector is not run locally: LILA's precomputed MDv5a detections are used (CPU inference on ~29k full-size images is impractical). State this wherever Phase 6 is described.

## Storage used (budget 6 GB)
3.798 GB excluding .git (data/ 2.646 GB, .venv 1.141 GB) per results/phase7_final_checks.json.

## Resume instructions
Activate the venv (`.venv\Scripts\activate`), read "Reproduce" in README.md, run the next unchecked script.
data/ is gitignored. If data/ is missing, re-fetch the metadata, then re-run scripts 01, 02, 03:
- `curl -O https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz` and extract into data/meta/
- MegaDetector JSON: URL in README "Dataset"; unzip into data/md/
