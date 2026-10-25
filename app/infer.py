"""Inference shared by the Gradio app and crowd evaluation: video -> localise -> student -> text."""
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch

from data import localise as L
from data.dataset import to_input
from eval.decode import grammar_decode
from eval.metrics import ctc_greedy_with_conf, wer
from student.model import load_student

read_video = L.read_video
localise = L.localise


class Reader:
    """Wraps a trained student checkpoint."""

    def __init__(self, ckpt, device=None):
        # is_available() can be True with no usable device (e.g. CUDA_VISIBLE_DEVICES=""), so also check the count
        self.device = device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
        self.model, self.cfg = load_student(ckpt, self.device)
        self.ckpt = ckpt

    @torch.no_grad()
    def logits(self, clips_u8, lengths):
        """clips_u8 uint8 (B,75,88,88), lengths list -> float logits (B,75,C) on cpu."""
        x = to_input(torch.as_tensor(np.ascontiguousarray(clips_u8)), self.device)
        lens = torch.as_tensor(lengths, device=self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            out = self.model(x, lens)
        return out["logits"].float().cpu()

    def read(self, mouth88):
        """mouth88 uint8 (T,88,88), any T: read in 75-frame chunks with both raw greedy CTC and
        GRID-grammar-constrained decoding. Returns dict(greedy=(text, words), grammar=(text, words), logits)."""
        out = {"greedy": ([], []), "grammar": ([], [])}
        all_logits = []
        for s in range(0, max(len(mouth88), 1), L.CLIP_LEN):
            chunk = mouth88[s:s + L.CLIP_LEN]
            if len(chunk) < 10 and s > 0:
                break
            clip, n = L.pad_clip(chunk)
            lg = self.logits(clip[None], [n])[0, :n]
            for key, (t, w) in (("greedy", ctc_greedy_with_conf(lg)), ("grammar", grammar_decode(lg, return_scores=True))):
                out[key][0].append(t)
                out[key][1].extend(w)
            all_logits.append(lg)
        res = {k: (" ".join(t for t in ts if t), ws) for k, (ts, ws) in out.items()}
        res["logits"] = torch.cat(all_logits)
        return res


def overlay_video(frames, loc, out_path=None, max_w=960):
    """Draw all face tracks (thin) and the chosen speaker's mouth ROI (bold) -> H.264 mp4 path."""
    h, w = frames[0].shape[:2]
    k = min(1.0, max_w / w) if w > max_w else max(1.0, 640 / w)
    ow, oh = int(round(w * k / 2) * 2), int(round(h * k / 2) * 2)
    out_path = out_path or tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", "25",
           "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-an", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t, f in enumerate(frames):
        img = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_LINEAR)
        for i, tr in enumerate(loc["tracks"]):
            if i != loc["chosen"]:
                b = (tr["boxes"][t] * k).astype(int)
                cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (200, 200, 200), 1)
        if loc["chosen"] >= 0:
            b = (loc["tracks"][loc["chosen"]]["boxes"][t] * k).astype(int)
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (60, 200, 60), 2)
            mb = (L.ft.mouth_box(loc["geom"][t]) * k).astype(int)
            cv2.rectangle(img, tuple(mb[:2]), tuple(mb[2:]), (40, 40, 240), 2)
            # label on a filled tag above the speaker's face box so it stays readable on any background
            txt = f"speaker - mouth ROI {loc['mouth_px']:.0f} px"
            (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x0 = int(np.clip(b[0], 0, max(0, ow - tw - 8)))
            y1 = int(b[1]) - 4 if b[1] - th - base - 8 >= 0 else int(b[3]) + th + base + 8
            cv2.rectangle(img, (x0, y1 - th - base - 4), (x0 + tw + 8, y1), (60, 200, 60), -1)
            cv2.putText(img, txt, (x0 + 4, y1 - base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_path


def mouth_strip(mouth88, n=8, scale=1.5):
    """n evenly spaced mouth crops side by side (RGB uint8)."""
    idx = np.linspace(0, len(mouth88) - 1, n).astype(int)
    strip = np.hstack([np.pad(mouth88[i], 2, constant_values=255) for i in idx])
    strip = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(strip, cv2.COLOR_GRAY2RGB)


def run(path, readers, reference="", max_frames=225):
    """Full pipeline on a video file. readers: {label: Reader}. Returns dict for the UI."""
    t0 = time.time()
    frames = read_video(path, max_frames=max_frames)
    if not frames:
        raise ValueError("could not decode any frames")
    loc = localise(frames)
    t_loc = time.time() - t0
    res = {"n_frames": len(frames), "n_faces": len(loc["tracks"]), "mouth_px": loc["mouth_px"], "t_localise": t_loc,
           "overlay": overlay_video(frames, loc), "strip": None, "reads": {}}
    if loc["chosen"] < 0:
        return res
    res["strip"] = mouth_strip(loc["mouth88"])
    for name, r in readers.items():
        t1 = time.time()
        out = r.read(loc["mouth88"])
        out["t_read"] = time.time() - t1
        ref = " ".join(reference.lower().split())
        out["wer"] = {k: wer([ref], [out[k][0]]) if ref else None for k in ("greedy", "grammar")}
        res["reads"][name] = out
    res["t_total"] = time.time() - t0
    return res
