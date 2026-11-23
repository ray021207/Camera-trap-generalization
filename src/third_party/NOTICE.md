# Third-party code in this folder

## rtdetrv2_pytorch/
RT-DETRv2 model code, used to run the MegaDetector v6 `MDV6-apa-rtdetr-c` detector in the video stage.

- Original author: lyuwenyu, https://github.com/lyuwenyu/RT-DETR (file headers: "Copyright(c) 2023 lyuwenyu").
  Licensed under the **Apache License 2.0**; full text in [`../../licenses/Apache-2.0.txt`](../../licenses/Apache-2.0.txt).
- This copy, including the MegaDetector configs (`configs/dataset/megadetector_detection.yml`,
  `configs/rtdetrv2/*_megadetector.yml`), was taken from Microsoft's Pytorch-Wildlife package
  (`PytorchWildlife` 1.3.1 on PyPI, `PytorchWildlife/models/detection/rtdetr_apache/rtdetrv2_pytorch/`),
  https://github.com/microsoft/Pytorch-Wildlife, MIT licence, Copyright (c) Microsoft Corporation.
- **Changes made here** (Apache-2.0 section 4(b)):
  - `src/core/_config.py`: the `SummaryWriter` import is only used as a type hint, so it is now imported under
    `typing.TYPE_CHECKING`. Inference no longer needs `tensorboard`.
  - `__pycache__` folders removed. No other file is modified.

The loader that builds the model from this code is `src/video.py` (`MDv6Detector`). It follows
`PytorchWildlife/models/detection/rtdetr_apache/rtdetr_apache_base.py` (MIT, Microsoft): resize to 640x640,
`ToTensor`, then the model's own post-processor.

## Model weights (not in this repo)
`MDV6-apa-rtdetr-c.pth` from the Pytorch-Wildlife model-weights record on Zenodo,
https://doi.org/10.5281/zenodo.15398270 (record licence **CC BY 4.0**; Microsoft's MegaDetector model zoo lists
this variant as **Apache-2.0**). Credit: Microsoft AI for Good Lab, Pytorch-Wildlife team (Zhongqi Miao et al.).
Downloaded at run time into `data/models/`; not redistributed here.
