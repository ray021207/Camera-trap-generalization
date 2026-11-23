Video test, 23 clips from 13 sites, one visit per clip, 1 frame/s (`results/video_eval_1fps.json`). Accuracy with 95% site-cluster bootstrap intervals.

| Clips | n | Accuracy [95% CI] |
|---|---|---|
| Known species, all (tier 1 + 2) | 16 | 0.88 [0.71, 1.00] |
| Tier 1: training species, new camera | 11 | 0.82 [0.57, 1.00] |
| Tier 2: related species mapped to a class | 5 | 1.00 [1.00, 1.00] |
| Known species, low risk of sharing the training camera network | 12 | 0.83 [0.62, 1.00] |

Unknown species (tier 3, n = 7): the forced answer was a sensible relative in 0.71 [0.40, 1.00] of clips; median confidence 0.48 (known species: 0.60 when right, 0.31 when wrong).

| Confidence cut-off | Known, correct, flagged | Known, wrong, flagged | Unknown species flagged |
|---|---|---|---|
| < 0.3 | 14% | 0% | 14% |
| < 0.4 | 29% | 100% | 29% |
| < 0.5 | 29% | 100% | 57% |
| < 0.6 | 50% | 100% | 57% |
| < 0.7 | 64% | 100% | 57% |
| < 0.8 | 64% | 100% | 100% |
