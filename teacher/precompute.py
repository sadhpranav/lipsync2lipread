"""Precompute frozen-teacher targets for every 5-frame window: eface, eaud, tconf (fp16).

Usage: python -m teacher.precompute WORK/proc/s1 [WORK/proc/s2 ...]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from teacher.syncnet_teacher import SyncNetTeacher  # noqa: E402


def load_teacher():
    return SyncNetTeacher(device="cuda" if torch.cuda.is_available() else "cpu")


def process_speaker(d, teacher, batch=16):
    if os.path.exists(os.path.join(d, "eface.npy")):
        return
    tface = np.load(os.path.join(d, "tface.npy"), mmap_mode="r")
    mel = np.load(os.path.join(d, "mel.npy"), mmap_mode="r")
    ef, ea = [], []
    for i in range(0, len(tface), batch):
        fw = torch.cat([teacher.face_windows(tface[j]) for j in range(i, min(i + batch, len(tface)))])
        mw = torch.cat([teacher.mel_windows(mel[j].astype(np.float32), tface.shape[1] - 4) for j in range(i, min(i + batch, len(tface)))])
        ef.append(teacher.embed_face(fw).half().cpu())
        ea.append(teacher.embed_audio(mw).half().cpu())
    n, w = len(tface), tface.shape[1] - 4
    ef = torch.cat(ef).view(n, w, 512)
    ea = torch.cat(ea).view(n, w, 512)
    np.save(os.path.join(d, "tconf.npy"), (ef.float() * ea.float()).sum(-1).half().numpy())
    np.save(os.path.join(d, "eaud.npy"), ea.numpy())
    np.save(os.path.join(d, "eface.npy"), ef.numpy())  # written last: marks completion
    print(f"teacher done: {d}", flush=True)


if __name__ == "__main__":
    t = load_teacher()
    for d in sys.argv[1:]:
        process_speaker(d, t)
