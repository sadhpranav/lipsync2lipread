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
.chip{display:inline-block;margin:2px 3px;padding:3px 8px;border-radius:6px;font-family:monospace;font-size:15px;
      color:var(--body-text-color)}
.chip small{color:inherit;opacity:.7;font-size:11px;margin-left:5px}
.model{margin:10px 0 2px;font-weight:600}
.alt,.ref{color:var(--body-text-color-subdued);font-size:13px;margin:2px 0 4px}
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

    def demo_label(d):
        # "crowd 16px" / "clean GRID" -> readable label; the reference text is filled in separately
        return f"{d['kind'].replace('px', ' px mouth')} · {os.path.splitext(d['file'])[0]}"

    # easy -> hard: clean GRID first, then crowd scenes with shrinking mouths
    rank = {"clean GRID": 0, "crowd 32px": 1, "crowd 24px": 2, "crowd 16px": 3}
    labels = {demo_label(d): d for d in sorted(demos, key=lambda d: (rank.get(d["kind"], 9), d["file"]))}

    def pick_demo(label):
        d = labels.get(label)
        return (os.path.join(DEMO_DIR, d["file"]), d["text"]) if d else (None, "")

    first = next(iter(labels), None)
    first_video, first_ref = pick_demo(first)

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
        parts = []
        if r["n_frames"] < 50:
            parts.append(f"<div><b>Note:</b> only {r['n_frames']} frames ({r['n_frames'] / 25:.1f} s); a GRID sentence "
                         "takes about 3 s, so the reading below is unreliable.</div>")
        parts += [f"<div>faces tracked: <b>{r['n_faces']}</b> · speaker mouth ROI: <b>{r['mouth_px']:.0f} px</b> · "
                 f"frames: {r['n_frames']} · localisation {r['t_localise']:.1f} s</div>"]
        main, other = ("grammar", "greedy") if grammar else ("greedy", "grammar")
        for name, o in r["reads"].items():
            w = "" if o["wer"][main] is None else f" · <b>WER {100 * o['wer'][main]:.1f}%</b>"
            ow = "" if o["wer"][other] is None else f" (WER {100 * o['wer'][other]:.1f}%)"
            parts.append(f'<div class="model">{html.escape(name)} · {DEC[main]}{w} <small>({o["t_read"] * 1000:.0f} ms)</small></div>'
                         f"<div>{chips(o[main][1])}</div>"
                         f"<div class='alt'>{DEC[other]}: <code>{html.escape(o[other][0]) or '(empty)'}</code>{ow}</div>")
        if (reference or "").strip():
            parts.append(f"<div class='ref' style='margin-top:6px'>reference: <code>{html.escape(reference.lower())}</code></div>")
        return r["overlay"], r["strip"], "".join(parts)

    with gr.Blocks(title="LipSync2LipRead demo") as app:
        gr.Markdown("# LipSync2LipRead — lip reading with distilled lip-sync knowledge\n"
                    "IE 643 course project · Team PieTorch\n\n" + NOTE)
        with gr.Row():
            with gr.Column():
                video = gr.Video(value=first_video, label="Muted video (upload or record)", sources=["upload", "webcam"])
                demo = gr.Dropdown(list(labels), value=first, label="…or pick a demo clip")
                which = gr.Radio(choices, value=choices[-1], label="Model")
                grammar = gr.Checkbox(value=True, label="Grammar-constrained decoding (GRID grammar); raw greedy text is shown below it")
                ref = gr.Textbox(value=first_ref, label="Reference transcript (optional, enables WER)")
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
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if usable)")
    a = ap.parse_args()
    readers = {}
    if a.baseline:
        readers["baseline"] = infer.Reader(a.baseline, a.device)
    if a.distilled:
        readers["distilled"] = infer.Reader(a.distilled, a.device)
    assert readers, "pass --baseline and/or --distilled"
    build(readers).launch(share=a.share, server_port=a.port, css=CSS, allowed_paths=[DEMO_DIR])


if __name__ == "__main__":
    main()
