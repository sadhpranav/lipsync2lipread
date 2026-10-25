"""Extract ground-truth-box mouth crops from a GRID-Crowd folder into one memory-mappable array, for training.

python -m data.crowd_crops --root crowd_train      -> WORK/crowd_train/crops.npy (N,75,88,88) uint8 + crops.json
Only built from training speakers (data.crowd --spk <train speakers> --out crowd_train); the test set is never used.
"""
import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from data import facetrack as ft
from data import localise as L
from data.crowd import WORK

T = L.CLIP_LEN


def crop_one(args):
    root, m = args
    frames = L.read_video(os.path.join(root, "videos", m["cid"] + ".mp4"))
    n = min(len(frames), T)
    frames = (frames + [frames[-1]] * T)[:T]
    crops = np.stack([ft.crop_square(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), b, ft.MOUTH_SIZE)
                      for f, b in zip(frames, np.array(m["target_mouth"]))])
    return crops, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="crowd_train")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    root = os.path.join(WORK, a.root)
    meta = json.load(open(os.path.join(root, "meta.json")))
    out = np.lib.format.open_memmap(os.path.join(root, "crops.npy"), "w+", np.uint8, (len(meta), T, ft.MOUTH_SIZE, ft.MOUTH_SIZE))
    nframes = []
    with Pool(a.workers) as pool:
        for i, (c, n) in enumerate(pool.imap(crop_one, [(root, m) for m in meta], chunksize=8)):
            out[i] = c
            nframes.append(n)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(meta)}", flush=True)
    out.flush()
    info = {"texts": [m["text"] for m in meta], "nframes": nframes, "spk": [m["spk"] for m in meta],
            "target_px": [m["target_px"] for m in meta]}
    json.dump(info, open(os.path.join(root, "crops.json"), "w"))
    print(f"wrote {len(meta)} clips to {root}/crops.npy; speakers: {sorted(set(info['spk']), key=lambda s: int(s[1:]))}")


if __name__ == "__main__":
    main()
