"""Camera-trap video demo (portfolio / Hugging Face Space).

Pick an example clip or upload your own; the app returns the video with a box, species label and confidence
drawn on each analysed frame, plus one answer for the whole clip.

Runs from the repo (python space/app.py) or from the folder built by scripts/build_space.py (python app.py).
Pipeline: src/video.py (MegaDetector v6 -> animal crop -> DINOv2-small -> linear probe -> per-visit vote).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src" / "video.py").exists() else HERE.parent      # built Space folder, or the repo
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch

from src import video as V

REPO_URL = "https://github.com/ray021207/Camera-trap-generalization"
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", 30))    # uploads are cut here to keep the free CPU responsive
LOW_CONF = 0.5
EXAMPLES = json.loads((HERE / "examples" / "examples.json").read_text()) if (HERE / "examples" / "examples.json").exists() else []

torch.set_num_threads(max(1, os.cpu_count() or 1))
DET, CLS = V.Detector("mdv6"), V.Classifier()


def summary_md(v, n_seconds_cut=False):
    n, na = v["n_frames"], v.get("n_animal_frames", v["n_det_frames"])
    if v["pred"] == "empty" or v["conf"] is None:
        s = f"### No animal found\nNo animal in any of the {n} analysed frames."
    else:
        s = (f"### Verdict: **{v['pred']}** ({v['conf']:.0%})\n"
             f"Runner-up: {v['second']} ({v['second_conf']:.0%}). Animal seen in {na} of {n} analysed frames.")
        if v["conf"] < LOW_CONF:
            s += ("\n\n⚠️ **Low confidence.** This may be a species the model was never trained on. "
                  "It can only answer with one of its 13 labels.")
    if n_seconds_cut:
        s += f"\n\n<sub>Only the first {MAX_SECONDS:.0f} s were analysed.</sub>"
    return s


def rows_for(frames):
    rows = []
    for f in frames:
        if f["proba"] is None or not V.is_animal_frame(f, CLS.classes):
            rows.append([round(f["t"], 1), round(f.get("best_conf", 0.0), 2), "no animal", None])
        else:
            k = int(max(range(len(f["proba"])), key=lambda i: f["proba"][i]))
            rows.append([round(f["t"], 1), round(f["best_conf"], 2), CLS.classes[k], round(float(f["proba"][k]), 2)])
    return rows


def analyse(video_path, fps, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video or pick an example first.")
    import cv2
    c = cv2.VideoCapture(str(video_path))
    dur = (c.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (c.get(cv2.CAP_PROP_FPS) or 30)
    c.release()
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    progress(0, desc="analysing frames")
    frames, v = V.annotate_video(video_path, out, DET, CLS, fps=fps, max_seconds=MAX_SECONDS,
                                 progress=lambda x: progress(x, desc="analysing frames"))
    return out, summary_md(v, dur > MAX_SECONDS + 0.5), rows_for(frames)


def show_example(i):
    e = EXAMPLES[i]
    r = json.loads((HERE / "examples" / e["result"]).read_text())
    return (str(HERE / "examples" / e["video"]), str(HERE / "examples" / e["annotated"]),
            r["summary"] + f"\n\n<sub>Clip: {e['credit']}</sub>", r["rows"])


ABOUT = f"""
**How it works.** One frame per second goes through
[MegaDetector v6](https://github.com/microsoft/MegaDetector), which finds animals of any kind and draws a box.
The box is cropped, turned into features by a frozen [DINOv2-small](https://huggingface.co/facebook/dinov2-small),
and classified by a linear probe trained on the Caltech Camera Traps (CCT20) benchmark. The frames then vote on
the species for the whole clip.

**What it knows.** 13 labels from southwestern-US camera traps: bird, bobcat, car, cat, coyote, dog, empty,
opossum, rabbit, raccoon, rodent, skunk, squirrel. Any other animal is forced into the closest label. The
examples marked *unknown species* show that: a gray fox or a wolf gets a sensible relative, often with lower confidence.

**Why it exists.** The project measures how much a camera-trap classifier loses on cameras it has never seen,
and what helps (cropping to the animal recovers about half of the gap). Code, results and limitations:
[{REPO_URL.split('/')[-1]}]({REPO_URL}).

<sub>Runs on a free CPU, so uploads take about a second per analysed frame and are cut at {MAX_SECONDS:.0f} s.
Uploaded videos are processed in temporary files and not stored.</sub>
"""

with gr.Blocks(title="Camera-trap species classifier") as demo:
    gr.Markdown("# Camera-trap species classifier\nFind the animal in a trail-camera video and name the species: "
                "a box, label and confidence on every analysed frame, and one answer for the whole clip.")
    if EXAMPLES:
        gr.Markdown("### Try an example")
        with gr.Row():
            ex_buttons = [gr.Button(e["label"], size="sm") for e in EXAMPLES]
    with gr.Row():
        with gr.Column():
            inp = gr.Video(label="Your video (or pick an example above)", sources=["upload"])
            fps = gr.Slider(0.5, 2, value=1, step=0.5, label="Frames analysed per second",
                            info="More = smoother boxes, but slower")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            outv = gr.Video(label="Result: box, species and confidence")
            summ = gr.Markdown()
    table = gr.Dataframe(headers=["time (s)", "detector confidence", "species", "species confidence"],
                         label="Per-frame results", wrap=True)
    gr.Markdown(ABOUT)
    if EXAMPLES:
        credits = "\n".join(f"- **{e['label']}**: {e['credit']}" for e in EXAMPLES)
        gr.Markdown(f"<details><summary>Example clip credits</summary>\n\n{credits}\n</details>")
        for i, b in enumerate(ex_buttons):
            b.click(lambda i=i: show_example(i), None, [inp, outv, summ, table])
    btn.click(analyse, [inp, fps], [outv, summ, table])


if __name__ == "__main__":
    share = "--share" in sys.argv
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1",
        inbrowser=not os.environ.get("SPACE_ID"), share=share)
