| Split | Accuracy, baseline | Macro-F1, baseline | Accuracy, MegaDetector crop | Macro-F1, MegaDetector crop |
|---|---|---|---|---|
| (a) random image split (seen cameras) | 0.860 [0.847, 0.873] | 0.854 [0.838, 0.868] | 0.876 [0.864, 0.889] | 0.867 [0.851, 0.881] |
| (b) random sequence split (seen cameras) | 0.766 [0.744, 0.786] | 0.749 [0.721, 0.774] | 0.845 [0.828, 0.861] | 0.829 [0.806, 0.847] |
| (c) held-out cameras (CCT20 trans_test) | 0.509 [0.496, 0.523] | 0.409 [0.396, 0.422] | 0.704 [0.692, 0.715] | 0.599 [0.583, 0.613] |
| **Gap (b) minus (c): the camera-shift penalty** | +0.256 [+0.230, +0.281] | +0.341 [+0.310, +0.367] | +0.142 [+0.120, +0.162] | +0.229 [+0.201, +0.253] |
| **Gap (a) minus (c): leakage + camera shift** | +0.351 [+0.333, +0.370] | +0.445 [+0.426, +0.465] | +0.173 [+0.155, +0.190] | +0.267 [+0.246, +0.288] |

Paired change from cropping (crop minus baseline), 95% CI from the same bootstrap resamples:

| Quantity | Accuracy | Macro-F1 |
|---|---|---|
| Unseen-camera score | +0.194 [+0.181, +0.207] | +0.191 [+0.175, +0.205] |
| Seen-camera score | +0.080 [+0.060, +0.100] | +0.079 [+0.056, +0.106] |
| Seen-minus-unseen gap | -0.115 [-0.138, -0.091] | -0.111 [-0.138, -0.081] |

Control: is it the background, or the zoom? (paired 95% CI; same frozen backbone and probe)

| Input to the model | Unseen-camera accuracy | Unseen-camera macro-F1 | Seen-minus-unseen accuracy gap |
|---|---|---|---|
| whole frame (baseline) | 0.509 [0.496, 0.523] | 0.409 [0.395, 0.421] | 0.256 [0.231, 0.281] |
| whole frame, background greyed out (same resolution) | 0.536 [0.524, 0.549] | 0.431 [0.418, 0.442] | 0.208 [0.185, 0.231] |
| MegaDetector crop | 0.704 [0.692, 0.715] | 0.599 [0.584, 0.613] | 0.142 [0.120, 0.163] |

Test sizes: (a) 3298 images, (b) 3332 images, (c) 11616 images in 4488 bursts. CIs: cluster (sequence) bootstrap, percentile 95%, 2000 resamples.
