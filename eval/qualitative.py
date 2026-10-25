"""Qualitative panel on GRID-Crowd: crowd frame + chosen ROI, the low-res mouth crops the model read,
and GT / baseline / distilled transcripts with word errors marked.

python -m eval.qualitative --baseline R2_ft_noKD --distilled R3_ft_m2m3 [--mode detected]
"""
import argparse
import difflib
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.crowd import WORK
from eval.figures import FIG, INK, INK2, RUNS, SURFACE, save
from eval.metrics import word_errors

ERR = "#d03b3b"  # status "critical": always paired with underline + legend text, never colour alone


def correct_mask(ref, hyp):
    r, h = ref.split(), hyp.split()
    ok = [False] * len(h)
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if tag == "equal":
            ok[j1:j2] = [True] * (j2 - j1)
    return h, ok


def draw_words(ax, x, y, prefix, words, ok, fig):
    r = fig.canvas.get_renderer()
    t = ax.text(x, y, prefix, transform=ax.transAxes, fontsize=10, color=INK2, family="monospace", va="center")
    x = ax.transAxes.inverted().transform(t.get_window_extent(r))[1, 0]
    for w, good in zip(words, ok):
        t = ax.text(x, y, w + " ", transform=ax.transAxes, fontsize=10, family="monospace", va="center",
                    color=INK if good else ERR, fontweight="normal" if good else "bold")
        bb = ax.transAxes.inverted().transform(t.get_window_extent(r))
        if not good:
            ax.plot([bb[0, 0], bb[1, 0] - 0.01], [y - 0.09] * 2, color=ERR, lw=1.2, transform=ax.transAxes)
        x = bb[1, 0]
    if not words:
        ax.text(x, y, "(empty)", transform=ax.transAxes, fontsize=10, color=ERR, va="center", family="monospace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--distilled", required=True)
    ap.add_argument("--mode", default="detected", choices=["detected", "oracle"])
    ap.add_argument("--decode", default="greedy", choices=["greedy", "grammar"])
    ap.add_argument("--n_good", type=int, default=4)
    ap.add_argument("--n_bad", type=int, default=2)
    a = ap.parse_args()
    meta = {m["cid"]: m for m in json.load(open(os.path.join(WORK, "crowd", "meta.json")))}
    pb = {p["cid"]: p for p in json.load(open(os.path.join(RUNS, a.baseline, "eval_crowd.json")))["preds"]}
    pd = json.load(open(os.path.join(RUNS, a.distilled, "eval_crowd.json")))["preds"]
    hk = f"hyp_{a.mode}" + ("_grammar" if a.decode == "grammar" else "")
    rows = []
    for p in pd:
        b = pb[p["cid"]]
        rows.append((p["cid"], word_errors(p["ref"], b[hk])[0], word_errors(p["ref"], p[hk])[0], p["target_px"]))
    # successes: distilled right where the baseline was wrong, smallest faces first; failures: worst distilled
    good = sorted([r for r in rows if r[2] == 0], key=lambda r: (-r[1], r[3]))[:a.n_good]
    bad = sorted(rows, key=lambda r: (-r[2], r[3]))[:a.n_bad]
    pick = [(r, "success") for r in good] + [(r, "failure") for r in bad]
    fig, axes = plt.subplots(len(pick), 3, figsize=(15, 2.25 * len(pick)), gridspec_kw={"width_ratios": [1.6, 2.2, 3.2]})
    axes = np.atleast_2d(axes)
    for (cid, eb, ed, px), kind in pick:
        i = pick.index(((cid, eb, ed, px), kind))
        m, loc = meta[cid], np.load(os.path.join(WORK, "crowd", "loc", cid + ".npz"))
        cap = cv2.VideoCapture(os.path.join(WORK, "crowd", "videos", cid + ".mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 37)
        _, frame = cap.read()
        mb = np.array(m["target_mouth"][37]).astype(int)
        fb = np.array(m["target_boxes"][37]).astype(int)
        cv2.rectangle(frame, tuple(fb[:2]), tuple(fb[2:]), (80, 220, 80), 1)
        cv2.rectangle(frame, tuple(mb[:2]), tuple(mb[2:]), (40, 40, 240), 2)
        axes[i, 0].imshow(frame[:, :, ::-1])
        axes[i, 0].set_title(f"clip {cid} · mouth ROI {px} px · {kind}", fontsize=9, loc="left", color=INK)
        crops = loc[a.mode]
        idx = np.linspace(8, 66, 6).astype(int)
        axes[i, 1].imshow(np.hstack([np.pad(crops[j], 2, constant_values=255) for j in idx]), cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title(f"what the model read ({a.mode} crops, frames {idx[0]}–{idx[-1]})", fontsize=9, loc="left")
        ax = axes[i, 2]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ref = m["text"]
        draw_words(ax, 0.0, 0.78, "truth     ", ref.split(), [True] * len(ref.split()), fig)
        for y, name, pred in ((0.48, "baseline  ", pb[cid][hk]), (0.18, "distilled ", [p for p in pd if p["cid"] == cid][0][hk])):
            draw_words(ax, 0.0, y, name, *correct_mask(ref, pred), fig)
        for k in range(3):
            axes[i, k].set_axis_off()
    fig.text(0.01, -0.01, "Bold red underlined = word error (substitution or insertion) against the ground-truth transcript. "
             f"Green box = speaker face, red box = mouth ROI (ground truth). Decoding: {a.decode}.", fontsize=9, color=INK2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    save(fig, f"qualitative_crowd_{a.mode}_{a.decode}")


if __name__ == "__main__":
    main()
