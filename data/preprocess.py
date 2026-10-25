"""GRID preprocessing: extract zips -> track face -> mouth ROI, teacher face crop, mel, transcript.

Writes WORK/proc/<spk>/{mouth,geom,nframes,smask,tface,mel}.npy + meta.json, then runs the
frozen SyncNet teacher (teacher/precompute.py) to add eface/eaud/tconf.

Usage: python -m data.preprocess --spk s1 s2 ... [--watch] [--workers 8]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import facetrack as ft  # noqa: E402

WORK = os.environ.get("L2L_WORK", os.path.join(os.path.dirname(__file__), "..", "..", "work"))
RAW, PROC = os.path.join(WORK, "raw"), os.path.join(WORK, "proc")
T = 75
W = T - 4
SR = 16000
NONSPEECH = {"sil", "sp"}

_lm = None


def extract(spk):
    """Unzip a downloaded speaker archive (mpg files only). Returns True if videos are available."""
    vdir = os.path.join(RAW, "videos", spk)
    flag = os.path.join(RAW, "videos", f".{spk}.extracted")
    if os.path.exists(flag):
        return True
    for zp in glob.glob(os.path.join(RAW, "*", f"{spk}.zip")):
        if not os.path.exists(zp + ".done"):
            continue
        os.makedirs(vdir, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.endswith(".mpg") and not n.startswith("__MACOSX"):
                    with open(os.path.join(vdir, os.path.basename(n)), "wb") as f:
                        f.write(z.read(n))
        open(flag, "w").close()
        return True
    return False


def read_align(path):
    words, speech = [], np.zeros(T, bool)
    for line in open(path):
        s, e, w = line.split()
        if w in NONSPEECH:
            continue
        words.append(w)
        speech[int(s) // 1000:max(int(s) // 1000 + 1, int(e) // 1000)] = True
    return " ".join(words), speech


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames


def read_audio(path):
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


def _init():
    global _lm
    import logging
    logging.disable(logging.WARNING)
    _lm = ft.Landmarker()


def process_clip(args):
    vpath, apath = args
    from teacher.syncnet_teacher import melspectrogram
    try:
        frames = read_frames(vpath)
        if len(frames) < 40:
            return None
        frames = frames[:T]
        n = len(frames)
        geom = ft.track_video(frames, _lm)
        if geom is None:
            return None
        mouth = np.stack([ft.mouth_crop(f, g) for f, g in zip(frames, geom)])
        tface = np.stack([ft.teacher_face_crop(f, g) for f, g in zip(frames, geom)])
        if n < T:  # edge-pad short clips
            idx = np.minimum(np.arange(T), n - 1)
            mouth, tface, geom = mouth[idx], tface[idx], geom[idx]
        mel = melspectrogram(read_audio(vpath))
        need = int(3.2 * (W - 1)) + 16
        mel = np.pad(mel, ((0, 0), (0, max(0, need - mel.shape[1]))), mode="edge")[:, :need]
        text, speech = read_align(apath)
        smask = np.array([speech[t:t + 5].any() for t in range(W)])
        return dict(id=os.path.basename(vpath)[:-4], text=text, mouth=mouth, tface=tface, geom=geom,
                    mel=mel.astype(np.float16), nframes=n, smask=smask)
    except Exception as e:  # corrupt clip
        print("fail", vpath, e, flush=True)
        return None


def align_dir_for(spk, vids):
    """Zenodo's alignments/sN folders are permuted w.r.t. the sN.zip videos (e.g. videos s5 <-> align s6),
    so pick the alignment folder whose file names overlap the speaker's video names."""
    names = {os.path.basename(v)[:-4] for v in vids}
    best = max(glob.glob(os.path.join(RAW, "align", "s*")),
               key=lambda d: len(names & {f[:-6] for f in os.listdir(d)}))
    if os.path.basename(best) != spk:
        print(f"{spk}: using alignments from {os.path.basename(best)}", flush=True)
    return best


def process_speaker(spk, workers):
    out = os.path.join(PROC, spk)
    if os.path.exists(os.path.join(out, "meta.json")):
        return True
    vids = sorted(glob.glob(os.path.join(RAW, "videos", spk, "*.mpg")))
    adir = align_dir_for(spk, vids)
    jobs = [(v, os.path.join(adir, os.path.basename(v)[:-4] + ".align")) for v in vids]
    jobs = [j for j in jobs if os.path.exists(j[1])]
    if len(jobs) < 0.5 * len(vids):  # e.g. Zenodo s5: video names do not match alignments
        print(f"{spk}: only {len(jobs)}/{len(vids)} videos have alignments, skipping", flush=True)
        return False
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        res = [r for r in pool.imap(process_clip, jobs, chunksize=4) if r is not None]
    os.makedirs(out, exist_ok=True)
    for k in ["mouth", "tface", "geom", "mel", "smask"]:
        np.save(os.path.join(out, f"{k}.npy"), np.stack([r[k] for r in res]))
    np.save(os.path.join(out, "nframes.npy"), np.array([r["nframes"] for r in res], np.int16))
    json.dump({"ids": [r["id"] for r in res], "texts": [r["text"] for r in res]}, open(os.path.join(out, "meta.json"), "w"))
    print(f"{spk}: {len(res)}/{len(jobs)} clips in {time.time() - t0:.0f}s", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--watch", action="store_true", help="wait for downloads to finish")
    ap.add_argument("--no_teacher", action="store_true")
    a = ap.parse_args()
    teacher = None
    todo = list(a.spk)
    while todo:
        progressed = False
        for spk in list(todo):
            if not extract(spk):
                continue
            if process_speaker(spk, a.workers) and not a.no_teacher:
                from teacher import precompute
                teacher = teacher or precompute.load_teacher()
                precompute.process_speaker(os.path.join(PROC, spk), teacher)
            todo.remove(spk)
            progressed = True
        if todo and not progressed:
            if not a.watch:
                print("not downloaded:", todo)
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
