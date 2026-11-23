"""Export the video pipeline's two networks to ONNX for the in-browser demo, and check them against PyTorch.

    python scripts/export_onnx.py              # fp32 export + int8 (weight-only) versions + numeric checks

Writes to data/onnx/ (gitignored; the files are uploaded to a Hugging Face model repo for the web page):
  mdv6_rtdetr_c.onnx       MegaDetector v6 (MDV6-apa-rtdetr-c) incl. its post-processor.
                           inputs: images [1,3,640,640] float in [0,1] (frame stretched to 640x640),
                                   orig_target_sizes [1,2] int64 (W, H of the original frame)
                           outputs: labels [1,300] (0 animal, 1 person, 2 vehicle), boxes [1,300,4] (x0,y0,x1,y1 px), scores [1,300]
  classifier.onnx          DINOv2-small -> [CLS, mean patch] -> float16 rounding (as in training) -> standardise
                           -> linear probe -> softmax.   input: pixel_values [N,3,224,224] ImageNet-normalised crop
                           output: probs [N,13] in the order of models/probe_b_crop.npz "classes"
  *_int8.onnx              same graphs with MatMul/Gemm weights quantised to int8 (onnxruntime quantize_dynamic)
  export_check.json        max differences against PyTorch on real frames from the test clips
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src import preprocess as P
from src import video as V
from src.common import DATA

OUT = DATA / "onnx"


class DetectorGraph(torch.nn.Module):
    def __init__(self, det):
        super().__init__()
        self.model, self.post = det.model, det.post

    def forward(self, images, orig_target_sizes):
        return self.post(self.model(images), orig_target_sizes)


class ClassifierGraph(torch.nn.Module):
    def __init__(self, dino, probe):
        super().__init__()
        self.dino = dino
        for k in ["mean", "scale", "coef", "intercept"]:
            self.register_buffer(k, torch.tensor(probe[k], dtype=torch.float32))

    def forward(self, pixel_values):
        h = self.dino(pixel_values=pixel_values).last_hidden_state
        e = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        e = e.to(torch.float16).to(torch.float32)          # training embeddings were stored as float16
        z = (e - self.mean) / self.scale
        return torch.softmax(z @ self.coef.T + self.intercept, dim=1)


def sample_frames(n=12):
    """A few real frames (with and without animals) from the test clips, for the numeric checks."""
    import pandas as pd
    clips = pd.read_csv(DATA / "video_test" / "clips.csv", dtype=str)
    frames = []
    for f in clips.file.tolist()[:: max(1, len(clips) // 6)][:6]:
        for t, img in V.iter_frames(DATA / "video_test" / f, fps=0.5):
            frames.append(img)
            if len(frames) % 2 == 0:
                break
    return frames[:n]


def fp16_weights(src, dst, min_elems=1024):
    """Store large float32 weights as float16 and add a Cast back to float32 in front of their users.
    The arithmetic stays float32 (only the stored weights are rounded), the file is about half the size,
    and it uses only standard operators, so every onnxruntime-web build can run it."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(str(src))
    g = m.graph
    casts, keep = [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= min_elems:
            a = numpy_helper.to_array(init).astype(np.float16)
            half = numpy_helper.from_array(a, init.name + "__fp16")
            keep.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def export(model, args, path, input_names, output_names, dynamic=None):
    torch.onnx.export(model, args, str(path), input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic, opset_version=17, dynamo=False)


def main():
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    det, cls = V.MDv6Detector(), V.Classifier()
    frames = sample_frames()
    check = {}

    # --- detector ---
    dg = DetectorGraph(det).eval()
    x = det.tf(frames[0])[None]
    sz = torch.tensor([[frames[0].size[0], frames[0].size[1]]])
    export(dg, (x, sz), OUT / "mdv6_rtdetr_c.onnx", ["images", "orig_target_sizes"], ["labels", "boxes", "scores"])

    # --- classifier ---
    cg = ClassifierGraph(cls.model, cls.probe).eval()
    crops = []
    for img in frames:
        d = det(img)
        crops.append(P.crop_box(img, d[0]["bbox"]) if d else img)
    xc = torch.stack([P.to_tensor_normalized(P.resize_crop(c)) for c in crops])
    export(cg, (xc[:1],), OUT / "classifier.onnx", ["pixel_values"], ["probs"], {"pixel_values": {0: "n"}, "probs": {0: "n"}})

    # --- smaller files for the web page ---
    # int8 dynamic quantisation is kept only as a documented negative result: it broke the classifier (DINOv2's
    # activations have outliers). 8-bit weight-only MatMulNBits kept accuracy but does not run in onnxruntime-web
    # 1.22's WASM build. fp16-stored weights (cast back to float32) work everywhere and are what the page loads.
    for name in ["mdv6_rtdetr_c", "classifier"]:
        quantize_dynamic(str(OUT / f"{name}.onnx"), str(OUT / f"{name}_int8.onnx"), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        fp16_weights(OUT / f"{name}.onnx", OUT / f"{name}_fp16w.onnx")

    # --- checks against PyTorch on the same inputs ---
    for variant in ["", "_int8", "_fp16w"]:
        sd = ort.InferenceSession(str(OUT / f"mdv6_rtdetr_c{variant}.onnx"), providers=["CPUExecutionProvider"])
        sc = ort.InferenceSession(str(OUT / f"classifier{variant}.onnx"), providers=["CPUExecutionProvider"])
        dmax, top_score_diff, top_box_diff = 0.0, [], []
        for img in frames:
            xi = det.tf(img)[None]
            s = torch.tensor([[img.size[0], img.size[1]]])
            lt, bt, st = dg(xi, s)
            lo, bo, so = sd.run(None, {"images": xi.numpy(), "orig_target_sizes": s.numpy()})
            it, io = int(st[0].argmax()), int(so[0].argmax())
            top_score_diff.append(abs(float(st[0, it]) - float(so[0, io])))
            top_box_diff.append(float(np.abs(bt[0, it].numpy() - bo[0, io]).max()) / max(img.size))
        pt = cg(xc).numpy()
        po = sc.run(None, {"pixel_values": xc.numpy()})[0]
        check[f"detector{variant}"] = dict(max_top_score_diff=float(max(top_score_diff)),
                                           max_top_box_diff_frac_of_image=float(max(top_box_diff)))
        check[f"classifier{variant}"] = dict(max_prob_diff=float(np.abs(pt - po).max()),
                                             top1_agreement=float((pt.argmax(1) == po.argmax(1)).mean()))
    sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in sorted(OUT.glob("*.onnx"))}
    check["sizes_mb"] = sizes
    check["classes"] = cls.classes
    check["n_check_frames"] = len(frames)
    json.dump(check, open(OUT / "export_check.json", "w"), indent=1)
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
