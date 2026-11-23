"""Local upload demo: drop in a video, get it back with MegaDetector boxes and species labels drawn on.

    python scripts/demo_app.py            # opens http://127.0.0.1:7860 in your browser
    python scripts/demo_app.py --share    # also prints a temporary public gradio.live link (anyone with it can upload)

Same pipeline as scripts/15_video_predict.py (src/video.py). Uploaded videos are processed in temporary
files and are NOT added to the video test set.
"""
import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch

from src import video as V

MAX_SECONDS = 120          # longer uploads are cut here: about 1.4 s of CPU per analysed frame
LOW_CONF = 0.5             # below this the summary flags the answer as uncertain (rule of thumb, not yet validated)

torch.set_num_threads(8)
print("loading MegaDetector + DINOv2 ...")
DET, CLS = V.Detector(), V.Classifier()
print("ready")


def analyse(video_path, fps, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    progress(0, desc="analysing frames")
    frames, v = V.annotate_video(video_path, out, DET, CLS, fps=fps, max_seconds=MAX_SECONDS,
                                 progress=lambda x: progress(x, desc="analysing frames"))

    n, nd = v["n_frames"], v["n_det_frames"]
    if v["conf"] is None:
        summary = f"### No animal detected\nMegaDetector found nothing in any of the {n} analysed frames."
    else:
        summary = (f"### Verdict: **{v['pred']}** ({v['conf']:.0%})\n"
                   f"Runner-up: {v['second']} ({v['second_conf']:.0%}). "
                   f"Animal found in {nd} of {n} analysed frames.")
        if v["conf"] < LOW_CONF:
            summary += ("\n\n⚠️ **Low confidence.** This may be a species the model was not trained on; "
                        "it can only answer with one of its 13 classes.")
    summary += ("\n\n<sub>Classes: bird, bobcat, car, cat, coyote, dog, empty, opossum, rabbit, raccoon, rodent, "
                f"skunk, squirrel. Videos longer than {MAX_SECONDS} s are cut.</sub>")

    rows = []
    for f in frames:
        if f["proba"] is None:
            rows.append([f["t"], 0.0, "-", None])
        else:
            k = int(f["proba"].argmax())
            rows.append([f["t"], round(f["best_conf"], 2), CLS.classes[k], round(float(f["proba"][k]), 2)])
    return out, summary, rows


with gr.Blocks(title="CameraTrap video demo") as demo:
    gr.Markdown("# CameraTrap video demo\nUpload a camera-trap video. Each analysed frame goes through "
                "MegaDetector → animal crop → DINOv2-small → linear probe, and the frames vote on the species.")
    with gr.Row():
        with gr.Column():
            inp = gr.Video(label="Upload video", sources=["upload"])
            fps = gr.Slider(0.5, 4, value=2, step=0.5, label="Frames analysed per second",
                            info="More = smoother boxes but slower (≈1.4 s per frame on this CPU)")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            outv = gr.Video(label="Annotated video")
            summ = gr.Markdown()
    table = gr.Dataframe(headers=["time (s)", "detector conf", "top class", "class prob"],
                         label="Per-frame results", wrap=True)
    btn.click(analyse, [inp, fps], [outv, summ, table])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    a = ap.parse_args()
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", inbrowser=True, share=a.share)
