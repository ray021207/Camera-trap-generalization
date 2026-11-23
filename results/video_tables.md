Video test, 23 clips from 13 sites, one visit per clip, 1 frame/s, detector mdv6 (`results/video_eval_1fps_mdv6.json`). Accuracy with 95% site-cluster bootstrap intervals.

| Clips | n | Accuracy [95% CI] |
|---|---|---|
| Known species, all (tier 1 + 2) | 16 | 0.69 [0.38, 0.87] |
| Tier 1: training species, new camera | 11 | 0.64 [0.17, 0.86] |
| Tier 2: related species mapped to a class | 5 | 0.80 [0.57, 1.00] |
| Known species, low risk of sharing the training camera network | 12 | 0.67 [0.38, 0.86] |
| Known species, all, with the legacy MegaDetector v5a (the detector behind the training crops) | 16 | 0.88 [0.71, 1.00] |

Unknown species (tier 3, n = 7): the forced answer was a sensible relative in 0.71 [0.40, 1.00] of clips; median confidence 0.52 (known species: 0.80 when right, 0.33 when wrong).

| Confidence cut-off | Known, correct, flagged | Known, wrong, flagged | Unknown species flagged |
|---|---|---|---|
| < 0.3 | 0% | 40% | 14% |
| < 0.4 | 9% | 100% | 29% |
| < 0.5 | 18% | 100% | 43% |
| < 0.6 | 27% | 100% | 71% |
| < 0.7 | 45% | 100% | 71% |
| < 0.8 | 55% | 100% | 100% |
