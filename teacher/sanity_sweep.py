"""Does the lip-sync teacher's knowledge transfer to GRID, and how does it degrade with face size?

For speech windows: sync accuracy = P[cos(face_t, audio_t) > cos(face_t, audio_{t+10})] (0.4 s offset),
and cos(face_emb at full res, face_emb at reduced res). Writes WORK/runs/teacher_sanity.json.

Usage: python -m teacher.sanity_sweep --spk s1 s2 --clips 200
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from teacher.precompute import load_teacher  # noqa: E402

WORK = os.environ.get("L2L_WORK", os.path.join(os.path.dirname(__file__), "..", "..", "work"))
RES = [96, 64, 48, 32, 24, 16]
SHIFT = 10


def downscale(tface, res):
    """Simulate a smaller face: the 96-px face crop shrunk to res px, then upsampled back."""
    if res == 96:
        return tface
    out = []
    for f in tface:  # (48, 96, 3) lower half -> scale both dims by res/96
        h, w = max(1, round(48 * res / 96)), res
        small = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        out.append(cv2.resize(small, (96, 48), interpolation=cv2.INTER_LINEAR))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", nargs="+", default=["s1", "s2"])
    ap.add_argument("--clips", type=int, default=200)
    a = ap.parse_args()
    teacher = load_teacher()
    rng = np.random.default_rng(0)
    acc = {r: [] for r in RES}
    cos_hl = {r: [] for r in RES}
    pos_all, neg_all = [], []
    for spk in a.spk:
        d = os.path.join(WORK, "proc", spk)
        tface = np.load(os.path.join(d, "tface.npy"), mmap_mode="r")
        ea = torch.from_numpy(np.load(os.path.join(d, "eaud.npy")).astype(np.float32))
        sm = np.load(os.path.join(d, "smask.npy"))
        for i in rng.choice(len(tface), min(a.clips, len(tface)), replace=False):
            m = sm[i][:-SHIFT]
            if m.sum() == 0:
                continue
            hi = None
            for r in RES:
                ef = teacher.embed_face(teacher.face_windows(downscale(tface[i], r))).cpu()
                if r == 96:
                    hi = ef
                    pos_all += (ef[:-SHIFT] * ea[i][:-SHIFT]).sum(-1)[m].tolist()
                    neg_all += (ef[:-SHIFT] * ea[i][SHIFT:]).sum(-1)[m].tolist()
                pos = (ef[:-SHIFT] * ea[i][:-SHIFT]).sum(-1)
                neg = (ef[:-SHIFT] * ea[i][SHIFT:]).sum(-1)
                acc[r] += (pos > neg)[m].float().tolist()
                cos_hl[r] += (ef * hi).sum(-1)[sm[i]].tolist()
    out = dict(res_list=RES, sync_acc=[float(np.mean(acc[r])) for r in RES],
               cos_hi_lo=[float(np.mean(cos_hl[r])) for r in RES],
               pos_cos=np.histogram(pos_all, bins=40, range=(-0.2, 1))[0].tolist(),
               neg_cos=np.histogram(neg_all, bins=40, range=(-0.2, 1))[0].tolist(),
               hist_range=[-0.2, 1.0], n_windows=len(pos_all), speakers=a.spk,
               pos_mean=float(np.mean(pos_all)), neg_mean=float(np.mean(neg_all)))
    os.makedirs(os.path.join(WORK, "runs"), exist_ok=True)
    json.dump(out, open(os.path.join(WORK, "runs", "teacher_sanity.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("pos_cos", "neg_cos")}, indent=1))


if __name__ == "__main__":
    main()
