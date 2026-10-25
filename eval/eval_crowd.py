"""GRID-Crowd evaluation: oracle (GT mouth boxes) vs detected (data/localise.py) mouth crops.

Localisation is model-independent, so it runs once per clip and is cached in WORK/crowd/loc/.
python -m eval.eval_crowd --ckpt WORK/runs/R2/best.pt WORK/runs/R3/best.pt
Writes WORK/runs/<run>/eval_crowd.json (keys oracle, detected = raw greedy CTC; *_grammar = GRID-grammar
decoding) and WORK/crowd/localisation.json.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from data import facetrack as ft
from data import localise as L
from data.crowd import WORK
from eval.metrics import bootstrap_ci, cer, char_errors, slot_acc, wer, word_errors

LOC_DIR = os.path.join(WORK, "crowd", "loc")
CHECK_FRAMES = [10, 37, 64]


def localise_one(m):
    out = os.path.join(LOC_DIR, m["cid"] + ".npz")
    if os.path.exists(out):
        return out
    frames = L.read_video(os.path.join(WORK, "crowd", "videos", m["cid"] + ".mp4"))
    frames = (frames + [frames[-1]] * L.CLIP_LEN)[:L.CLIP_LEN]
    oracle = np.stack([ft.crop_square(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), b, ft.MOUTH_SIZE)
                       for f, b in zip(frames, np.array(m["target_mouth"]))])
    t = time.time()
    loc = L.localise(frames)
    dt = time.time() - t
    gts = [np.array(m["target_boxes"])] + [np.array(d["boxes"]) for d in m["distractors"]]
    hits = [np.mean([any(L.iou(g[i:i + 1], tr["boxes"][i:i + 1])[0, 0] > 0.5 for tr in loc["tracks"]) for i in CHECK_FRAMES])
            for g in gts]
    chosen_ok = loc["chosen"] >= 0 and np.mean(
        [L.iou(gts[0][i:i + 1], loc["tracks"][loc["chosen"]]["boxes"][i:i + 1])[0, 0] > 0.5 for i in CHECK_FRAMES]) > 0.5
    detected = loc["mouth88"] if loc["chosen"] >= 0 else np.zeros_like(oracle)
    np.savez_compressed(out, oracle=oracle, detected=detected, hits=np.array(hits), chosen_ok=chosen_ok,
                        n_tracks=len(loc["tracks"]), mouth_px=loc["mouth_px"] or 0.0, sec=dt)
    return out


def run_localisation(meta, workers):
    os.makedirs(LOC_DIR, exist_ok=True)
    todo = [m for m in meta if not os.path.exists(os.path.join(LOC_DIR, m["cid"] + ".npz"))]
    if todo:
        print(f"localising {len(todo)} clips with {workers} workers ...", flush=True)
        with Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(localise_one, todo, chunksize=2)):
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    rows = [dict(np.load(os.path.join(LOC_DIR, m["cid"] + ".npz"))) for m in meta]
    summ = {"n_clips": len(meta)}
    for px in ["all", 16, 24, 32]:
        sel = [r for r, m in zip(rows, meta) if px == "all" or m["target_px"] == px]
        summ[str(px)] = {"face_recall": float(np.mean(np.concatenate([r["hits"] for r in sel]))),
                         "target_recall": float(np.mean([r["hits"][0] for r in sel])),
                         "speaker_select_acc": float(np.mean([bool(r["chosen_ok"]) for r in sel])),
                         "sec_per_clip": float(np.mean([float(r["sec"]) for r in sel]))}
    with open(os.path.join(WORK, "crowd", "localisation.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print("localisation:", json.dumps(summ["all"]))
    return rows, summ


def score(refs, hyps):
    we = [word_errors(r, h) for r, h in zip(refs, hyps)]
    ce = [char_errors(r, h) for r, h in zip(refs, hyps)]
    return {"n_clips": len(refs), "wer": wer(refs, hyps), "cer": cer(refs, hyps),
            "wer_ci95": bootstrap_ci([e for e, _ in we], [n for _, n in we]),
            "cer_ci95": bootstrap_ci([e for e, _ in ce], [n for _, n in ce]),
            **{f"acc_{k}": v for k, v in slot_acc(refs, hyps).items()}}


def eval_model(ckpt, meta, rows, loc_summ, bs=16):
    from app.infer import Reader
    from eval.decode import grammar_decode
    from eval.metrics import ctc_greedy
    r = Reader(ckpt)
    out = {"ckpt": os.path.abspath(ckpt), "localisation": loc_summ}
    preds = [{"cid": m["cid"], "spk": m["spk"], "idx": m["idx"], "target_px": m["target_px"], "ref": m["text"]} for m in meta]
    for mode in ["oracle", "detected"]:
        clips = np.stack([row[mode] for row in rows])
        for s in range(0, len(clips), bs):
            lg = r.logits(clips[s:s + bs], [L.CLIP_LEN] * len(clips[s:s + bs]))
            for p, h, g in zip(preds[s:s + bs], ctc_greedy(lg, [L.CLIP_LEN] * len(lg)), grammar_decode(lg)):
                p[f"hyp_{mode}"], p[f"hyp_{mode}_grammar"] = h, g
        for key in (mode, f"{mode}_grammar"):
            out[key] = {}
            for px in ["all", 16, 24, 32]:
                sel = [p for p in preds if px == "all" or p["target_px"] == px]
                out[key][str(px)] = score([p["ref"] for p in sel], [p[f"hyp_{key}"] for p in sel])
            print(f"{os.path.basename(os.path.dirname(os.path.abspath(ckpt)))} {key:16s} " +
                  " ".join(f"{px}px WER {100 * out[key][str(px)]['wer']:.1f}" for px in ["all", 16, 24, 32]), flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "eval_crowd.json")
    with open(path, "w") as f:
        json.dump({"summary": out, "preds": preds}, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(WORK, "crowd", "meta.json")))[:a.limit]
    rows, loc_summ = run_localisation(meta, a.workers)
    for c in a.ckpt:
        eval_model(c, meta, rows, loc_summ)


if __name__ == "__main__":
    main()
