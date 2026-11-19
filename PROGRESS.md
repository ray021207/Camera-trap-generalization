# PROGRESS

## Status
- [x] Phase 0 — environment (results/phase0.json)
- [ ] Phase 1 — data audit
- [ ] Phase 2 — splits
- [ ] Phase 3 — preprocessing
- [ ] Phase 4 — baseline (embeddings + linear probe + UMAP)
- [ ] Phase 5 — evaluation + failure gallery
- [ ] Phase 6 — MegaDetector crop mitigation
- [ ] Phase 7 — write-up

## Next
Phase 1: download the sampled image subset (scripts/01_download.py), then audit it.

## Blockers
None.

## Storage used (budget 6 GB)
~1.2 GB (venv only; see results/phase0.json for the exact figure).

## Resume instructions
Activate the venv (`.venv\Scripts\activate`), read the "Reproduce" section of README.md, and run
the next unchecked script. `data/` is gitignored; scripts re-create it.
