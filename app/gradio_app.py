"""Gradio test interface: upload / record a muted clip (or pick a demo crowd clip), see the
tracked faces, the mouth ROI the model read, and baseline vs distilled transcripts.

python -m app.gradio_app --baseline WORK/runs/R2/best.pt --distilled WORK/runs/R3/best.pt [--share]
"""
import argparse
import html
import json
import os

import gradio as gr

from app import infer
from data.crowd import WORK

DEMO_DIR = os.path.join(WORK, "crowd", "demo")
NOTE = ("**Vocabulary = GRID grammar** (command · colour · preposition · letter · digit · adverb), e.g. "
        "*\"bin blue at f two now\"*, *\"place red by d three again\"*. The model reads lips only — audio is ignored. "
        "Clips are resampled to 25 fps; the first 9 s are used and read in 3 s chunks.")
CSS = """
.chip{display:inline-block;margin:2px 3px;padding:3px 8px;border-radius:6px;font-family:monospace;font-size:15px;color:#0b0b0b}
.chip small{color:#52514e;font-size:11px;margin-left:4px}
.model{margin:10px 0 2px;font-weight:600}
.alt{color:#52514e;font-size:13px;margin:2px 0 4px}
"""


DEC = {"grammar": "grammar-constrained", "greedy": "raw greedy CTC"}


def chips(words):
    out = []
    for w, c in words:
        # single-hue (blue) ramp: higher confidence = stronger fill; the number is printed too
        alpha = 0.15 + 0.6 * max(0.0, min(1.0, c))
        out.append(f'<span class="chip" style="background:rgba(42,120,214,{alpha:.2f})">{html.escape(w)}'
                   f'<small>{c:.2f}</small></span>')
    return "".join(out) or "<i>(nothing read)</i>"


def load_demos():
    p = os.path.join(DEMO_DIR, "index.json")
    return json.load(open(p)) if os.path.exists(p) else []


def build(readers):
    demos = load_demos()
    names = list(readers)
    choices = names + (["both"] if len(names) > 1 else [])

    def pick_demo(label):
        for d in demos:
            if f"{d['kind']} · {d['file']}" == label:
                return os.path.join(DEMO_DIR, d["file"]), d["text"]
        return None, ""

    def go(video, which, reference, grammar):
        if not video:
            return None, None, "<b>Upload, record or pick a clip first.</b>"
        use = readers if which == "both" else {which: readers[which]}
        try:
            r = infer.run(video, use, reference or "")
        except Exception as e:  # surface pipeline errors in the UI
            return None, None, f"<b>Error:</b> {html.escape(str(e))}"
        if not r["reads"]:
            return r["overlay"], None, f"<b>No face found</b> in {r['n_frames']} frames."
        parts = [f"<div>faces tracked: <b>{r['n_faces']}</b> · speaker mouth ROI: <b>{r['mouth_px']:.0f} px</b> · "
                 f"frames: {r['n_frames']} · localisation {r['t_localise']:.1f} s</div>"]
        main, other = ("grammar", "greedy") if grammar else ("greedy", "grammar")
        for name, o in r["reads"].items():
            w = "" if o["wer"][main] is None else f" · <b>WER {100 * o['wer'][main]:.1f}%</b>"
            ow = "" if o["wer"][other] is None else f" (WER {100 * o['wer'][other]:.1f}%)"
            parts.append(f'<div class="model">{html.escape(name)} · {DEC[main]}{w} <small>({o["t_read"] * 1000:.0f} ms)</small></div>'
                         f"<div>{chips(o[main][1])}</div>"
                         f"<div class='alt'>{DEC[other]}: <code>{html.escape(o[other][0]) or '(empty)'}</code>{ow}</div>")
        if reference.strip():
            parts.append(f"<div style='margin-top:6px;color:#52514e'>reference: <code>{html.escape(reference.lower())}</code></div>")
        return r["overlay"], r["strip"], "".join(parts)

    with gr.Blocks(title="LipSync2LipRead demo") as app:
        gr.Markdown("# LipSync2LipRead — lip reading with distilled lip-sync knowledge\n"
                    "IE 643 course project · Team PieTorch\n\n" + NOTE)
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Muted video (upload or record)", sources=["upload", "webcam"])
                demo = gr.Dropdown([f"{d['kind']} · {d['file']}" for d in demos], label="…or pick a demo clip")
                which = gr.Radio(choices, value=choices[-1], label="Model")
                grammar = gr.Checkbox(value=True, label="Grammar-constrained decoding (GRID grammar); raw greedy text is shown below it")
                ref = gr.Textbox(label="Reference transcript (optional, enables WER)")
                btn = gr.Button("Read lips", variant="primary")
            with gr.Column():
                out_video = gr.Video(label="Tracked faces · chosen speaker's mouth ROI")
                strip = gr.Image(label="Mouth crops the model read (8 of the frames)", type="numpy")
                result = gr.HTML()
        demo.change(pick_demo, demo, [video, ref])
        btn.click(go, [video, which, ref, grammar], [out_video, strip, result])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="checkpoint of the no-KD baseline")
    ap.add_argument("--distilled", help="checkpoint of the distilled student")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    readers = {}
    if a.baseline:
        readers["baseline"] = infer.Reader(a.baseline)
    if a.distilled:
        readers["distilled"] = infer.Reader(a.distilled)
    assert readers, "pass --baseline and/or --distilled"
    build(readers).launch(share=a.share, server_port=a.port, css=CSS, allowed_paths=[DEMO_DIR])


if __name__ == "__main__":
    main()
