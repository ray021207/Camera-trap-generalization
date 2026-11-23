---
title: Camera-trap species classifier
emoji: 🦝
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Find and name the animal in trail-camera videos
---

# Camera-trap species classifier

Upload a trail-camera video, or pick an example, and get it back with a box, species label and confidence on
each analysed frame, plus one answer for the whole clip.

Pipeline: MegaDetector v6 (`MDV6-apa-rtdetr-c`) finds the animal, the crop goes through a frozen DINOv2-small,
and a linear probe trained on Caltech Camera Traps (CCT20) names one of 13 labels. The frames then vote.

Full project (how much camera-trap classifiers lose on cameras they have never seen, and what helps):
https://github.com/ray021207/Camera-trap-generalization

## Credits and licences
- **MegaDetector v6** (`MDV6-apa-rtdetr-c`), Microsoft AI for Good Lab / Pytorch-Wildlife. Weights:
  https://doi.org/10.5281/zenodo.15398270 (CC BY 4.0; listed as Apache-2.0 in the MegaDetector model zoo).
  Included here as `data/models/MDV6-apa-rtdetr-c.ema.pt` (the EMA weights extracted from that checkpoint, unchanged).
- **RT-DETRv2** code by lyuwenyu (Apache-2.0), via Pytorch-Wildlife (MIT): `src/third_party/` (see `NOTICE.md`).
- **DINOv2-small**, Meta AI (Apache-2.0), downloaded from the Hugging Face Hub at start-up.
- **Classifier** trained on Caltech Camera Traps (CDLA-Permissive-1.0); Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018.
- **Example clips**: public domain (USGS, NPS). Each clip's source is listed in the app under "Example clip credits".
