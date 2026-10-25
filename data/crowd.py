"""GRID-Crowd: muted 640x360 composites with a small talking target face and
static distractor faces on a procedural crowd-like background, H.264 low bitrate.

python -m data.crowd --n 300            -> WORK/crowd/videos/*.mp4 + meta.json + demo/
"""
import argparse
import json
import os
import shutil
import subprocess
from multiprocessing import Pool

import cv2
import numpy as np

from data import facetrack as ft

WORK = os.environ.get("L2L_WORK", os.path.join(os.path.dirname(__file__), "..", "..", "work"))
TEST_SPK = ["s1", "s2", "s20", "s22"]
H, W, T = 360, 640, 75
PX_CHOICES = [16, 24, 32]
FACE_MARGIN = 1.4        # patch side = FACE_MARGIN * max(face w, h)


def proc_meta(spk):
    """{id: text} from WORK/proc/<spk>/meta.json (transcripts re-matched to videos by preprocess.py;
    never read WORK/raw/align directly: Zenodo's alignment folders are permuted vs the speaker zips)."""
    p = os.path.join(WORK, "proc", spk, "meta.json")
    if not os.path.exists(p):
        return {}
    m = json.load(open(p))
    return dict(zip(m["ids"], m["texts"]))


def read_frames(spk, cid):
    cap = cv2.VideoCapture(os.path.join(WORK, "raw", "videos", spk, cid + ".mpg"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames


def background(rng):
    top, bot = rng.integers(40, 200, 3), rng.integers(30, 160, 3)
    a = np.linspace(0, 1, H)[:, None, None]
    img = (top * (1 - a) + bot * a) * np.ones((1, W, 1))
    for _ in range(rng.integers(15, 30)):
        c = rng.integers(20, 230, 3).tolist()
        x, y = rng.integers(0, W), rng.integers(0, H)
        if rng.random() < 0.5:
            cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(20, 200)), int(y + rng.integers(20, 150))), c, -1)
        else:
            cv2.ellipse(img, (int(x), int(y)), (int(rng.integers(10, 120)), int(rng.integers(10, 80))), 0, 0, 360, c, -1)
    img = cv2.GaussianBlur(img, (0, 0), 12)
    for _ in range(rng.integers(4, 9)):  # blurred person-like blobs
        x, y, r = int(rng.integers(0, W)), int(rng.integers(H // 4, H)), int(rng.integers(8, 25))
        c = rng.integers(20, 140, 3).tolist()
        cv2.circle(img, (x, y), r, c, -1)
        cv2.ellipse(img, (x, y + 3 * r), (int(1.8 * r), int(2.2 * r)), 0, 0, 360, c, -1)
    img = cv2.GaussianBlur(img, (0, 0), 3) + rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


def face_patch_box(geom):
    fb = np.median(geom[:, 3:7], 0)
    cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
    side = FACE_MARGIN * max(fb[2] - fb[0], fb[3] - fb[1])
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], np.float32)


def soft_ellipse(p):
    m = np.zeros((p, p), np.float32)
    cv2.ellipse(m, (p // 2, int(p * 0.5)), (max(1, int(p * 0.27)), max(1, int(p * 0.38))), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), max(0.8, p * 0.025))[..., None]


def paste(canvas, patch, mask, x, y):
    """Alpha-blend patch at float position (x, y) (sub-pixel)."""
    p = patch.shape[0]
    ix, iy = int(np.floor(x)), int(np.floor(y))
    M = np.float32([[1, 0, x - ix], [0, 1, y - iy]])
    sz = (p + 2, p + 2)
    pw = cv2.warpAffine(patch.astype(np.float32), M, sz)
    mw = cv2.warpAffine(mask[..., 0], M, sz)[..., None]
    y0, y1, x0, x1 = max(iy, 0), min(iy + p + 2, H), max(ix, 0), min(ix + p + 2, W)
    sub = (slice(y0 - iy, y1 - iy), slice(x0 - ix, x1 - ix))
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - mw[sub]) + pw[sub] * mw[sub]


def place(rng, size, taken, tries=200):
    for _ in range(tries):
        x, y = rng.uniform(4, W - size - 4), rng.uniform(4, H - size - 4)
        box = np.array([x, y, x + size, y + size])
        if all(box[0] > b[2] + 4 or box[2] < b[0] - 4 or box[1] > b[3] + 4 or box[3] < b[1] - 4 for b in taken):
            taken.append(box)
            return x, y
    x, y = rng.uniform(4, W - size - 4), rng.uniform(4, H - size - 4)
    taken.append(np.array([x, y, x + size, y + size]))
    return x, y


def load_geom(spk, cid, frames, lm):
    """Geometry from WORK/proc if available, else track now."""
    d = os.path.join(WORK, "proc", spk)
    if os.path.exists(os.path.join(d, "geom.npy")) and os.path.exists(os.path.join(d, "meta.json")):
        ids = json.load(open(os.path.join(d, "meta.json")))["ids"]
        if cid in ids:
            return np.load(os.path.join(d, "geom.npy"), mmap_mode="r")[ids.index(cid)].astype(np.float32)
    return ft.track_video(frames, lm)


def transform_box(b, pbox, s, x, y):
    return np.array([(b[0] - pbox[0]) * s + x, (b[1] - pbox[1]) * s + y,
                     (b[2] - pbox[0]) * s + x, (b[3] - pbox[1]) * s + y], np.float32)


_LM, TEXTS = None, {}


def _init(texts):
    global _LM, TEXTS
    _LM, TEXTS = ft.Landmarker(), texts


def build_one(job):
    k, spk, cid, px, distractors, seed, out_dir, bitrate = job
    rng = np.random.default_rng(seed)
    frames = read_frames(spk, cid)
    if len(frames) < 10:
        return None
    frames = (frames + [frames[-1]] * T)[:T]
    geom = load_geom(spk, cid, frames, _LM)
    if geom is None:
        return None
    canvas0 = background(rng)
    taken = []
    # target
    pbox = face_patch_box(geom)
    s = px / (ft.MOUTH_SCALE * float(np.median(geom[:, 2])))
    p = max(8, int(round((pbox[2] - pbox[0]) * s)))
    s = p / (pbox[2] - pbox[0])
    tx, ty = place(rng, p, taken)
    tmask = soft_ellipse(p)
    tpatches = [ft.crop_square(f, pbox, p) for f in frames]
    # distractors: single frame, slow sub-pixel drift
    dlist, dmeta = [], []
    for dspk, dcid in distractors:
        dfr = read_frames(dspk, dcid)
        if not dfr:
            continue
        f = dfr[int(rng.integers(0, len(dfr)))]
        faces = _LM(f)
        if not faces:
            continue
        g = ft.geometry_from_landmarks(faces[0])[None]
        dbox = face_patch_box(g)
        dpx = float(rng.choice(PX_CHOICES))
        dp = max(8, int(round((dbox[2] - dbox[0]) * dpx / (ft.MOUTH_SCALE * g[0, 2]))))
        ds = dp / (dbox[2] - dbox[0])
        dx, dy = place(rng, dp, taken)
        amp, ph, fr = rng.uniform(0.3, 1.0, 2), rng.uniform(0, 2 * np.pi, 2), rng.uniform(0.2, 0.6)
        drift = np.stack([amp[j] * np.sin(2 * np.pi * fr * np.arange(T) / 25 + ph[j]) for j in range(2)], 1)
        dlist.append((ft.crop_square(f, dbox, dp), soft_ellipse(dp), dx, dy, drift))
        fb = transform_box(g[0, 3:7], dbox, ds, dx, dy)
        dmeta.append({"spk": dspk, "idx": dcid, "px": dpx,
                      "boxes": [(fb + np.tile(drift[t], 2)).round(1).tolist() for t in range(T)]})
    # render + encode
    path = os.path.join(out_dir, f"{k:04d}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25",
           "-i", "-", "-c:v", "libx264", "-b:v", bitrate, "-pix_fmt", "yuv420p", "-an", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    face_boxes, mouth_boxes = [], []
    for t in range(T):
        canvas = canvas0.copy()
        for dpatch, dmask, dx, dy, drift in dlist:
            paste(canvas, dpatch, dmask, dx + drift[t, 0], dy + drift[t, 1])
        paste(canvas, tpatches[t], tmask, tx, ty)
        proc.stdin.write(np.clip(canvas, 0, 255).astype(np.uint8).tobytes())
        face_boxes.append(transform_box(geom[t, 3:7], pbox, s, tx, ty).round(1).tolist())
        mouth_boxes.append(transform_box(ft.mouth_box(geom[t]), pbox, s, tx, ty).round(1).tolist())
    proc.stdin.close()
    proc.wait()
    return {"cid": f"{k:04d}", "spk": spk, "idx": cid, "text": TEXTS[spk][cid], "target_px": px,
            "actual_px": round(float(np.median(ft.MOUTH_SCALE * geom[:, 2] * s)), 2), "frame_size": [W, H],
            "target_boxes": face_boxes, "target_mouth": mouth_boxes, "distractors": dmeta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--spk", nargs="+", default=TEST_SPK)
    ap.add_argument("--n_demo", type=int, default=8)
    ap.add_argument("--distractor_spk", nargs="*", default=None, help="distractor pool (default: the other --spk)")
    ap.add_argument("--bitrate", default="250k", help="H.264 bitrate of the 640x360 composite (low = CCTV-like)")
    ap.add_argument("--out", default="crowd", help="output folder under WORK (the test set is WORK/crowd)")
    args = ap.parse_args()
    root = os.path.join(WORK, args.out)
    rng = np.random.default_rng(args.seed)
    texts = {s: proc_meta(s) for s in set(args.spk) | set(args.distractor_spk or [])}
    spks = [s for s in args.spk if texts[s]]
    missing = sorted(set(args.spk) - set(spks))
    if missing:
        print("WARNING: not preprocessed yet, skipped:", missing)
    pool_ids = {s: sorted(i for i in texts[s] if os.path.exists(os.path.join(WORK, "raw", "videos", s, i + ".mpg")))
                for s in texts if texts[s]}
    print("speakers:", {s: len(v) for s, v in pool_ids.items() if s in spks})
    out_dir = os.path.join(root, "videos")
    for stale in ("videos", "loc", "demo"):  # clip ids are reused across rebuilds
        shutil.rmtree(os.path.join(root, stale), ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    per = int(np.ceil(args.n / len(spks)))
    for si, spk in enumerate(spks):
        for j, cid in enumerate(rng.choice(pool_ids[spk], min(per, len(pool_ids[spk])), replace=False)):
            k = len(jobs)
            if k >= args.n:
                break
            nd = int(rng.integers(2, 6))
            others = args.distractor_spk or [s for s in spks if s != spk] or [spk]
            dis = []
            for _ in range(nd):
                ds = str(rng.choice(others))
                dis.append((ds, str(rng.choice(pool_ids[ds]))))
            jobs.append((k, spk, str(cid), PX_CHOICES[k % 3], dis, args.seed * 100003 + k, out_dir, args.bitrate))
    with Pool(args.workers, initializer=_init, initargs=(texts,)) as pool:
        meta = [m for m in pool.imap(build_one, jobs, chunksize=4) if m is not None]
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"built {len(meta)} / {len(jobs)} clips")
    # demo set: a few crowd clips at each size + clean GRID test clips (muted mp4)
    demo = os.path.join(root, "demo")
    os.makedirs(demo, exist_ok=True)
    index = []
    for px in PX_CHOICES:
        for m in [m for m in meta if m["target_px"] == px][:max(1, args.n_demo // 3)]:
            fn = f"crowd_{px}px_{m['cid']}.mp4"
            shutil.copy(os.path.join(out_dir, m["cid"] + ".mp4"), os.path.join(demo, fn))
            index.append({"file": fn, "text": m["text"], "kind": f"crowd {px}px"})
    for spk in spks[:2]:
        cid = pool_ids[spk][0]
        fn = f"clean_{spk}_{cid}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(WORK, "raw", "videos", spk, cid + ".mpg"),
                        "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", os.path.join(demo, fn)], check=True)
        index.append({"file": fn, "text": texts[spk][cid], "kind": "clean GRID"})
    with open(os.path.join(demo, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"demo: {len(index)} clips")


if __name__ == "__main__":
    main()
