"""Train the student lip reader with optional SyncNet distillation.

  --kd none|m2|m3|m2m3|nce|m2m3nce   which teacher losses to add to CTC
  --m1                                init the 3D stem from SyncNet's first conv
  --degrade                           M4: student sees degraded crops, teacher targets stay clean
  --init ckpt                         warm start (weights only), e.g. fine-tune from the clean baseline
Runs are sized by wall clock (--max_minutes) and/or optimizer steps (--max_steps); the cosine schedule
follows whichever budget runs out first. Validation (raw greedy WER on a fixed val subset) every --val_every steps.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset import CrowdCrops, GridDataset, collate, to_input, worker_init
from data.splits import TEST, VAL, VAL_CLIPS, available_train, work_dir
from distill.losses import cos_kd, ctc_loss, info_nce, window_mask
from eval.metrics import ctc_greedy, wer
from student.model import LipReader, init_stem_from_syncnet, load_student, save_ckpt

SYNCNET = os.environ.get("L2L_SYNCNET", os.path.join(work_dir(), "ckpt", "lipsync_expert.pth"))
LOSSES = ("ctc", "m2", "m3", "nce")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--kd", default="none", choices=["none", "m2", "m3", "m2m3", "nce", "m2m3nce"])
    p.add_argument("--m1", action="store_true")
    p.add_argument("--degrade", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--codec", type=float, default=0.0,
                   help="with --degrade: probability of an H.264 round trip (low-bitrate video) instead of JPEG")
    p.add_argument("--init", default=None)
    p.add_argument("--backend", default="conformer", choices=["conformer", "bigru"])
    p.add_argument("--max_minutes", type=float, default=60)
    p.add_argument("--max_steps", type=int, default=0, help="optimizer-step budget (0: wall clock only)")
    p.add_argument("--epochs", type=int, default=1000, help="upper bound only; budgets above normally stop first")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None, help="optimizer steps; default 1000 (scratch) / 200 (fine-tune)")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda_v", type=float, default=1.0)
    p.add_argument("--lambda_a", type=float, default=1.0)
    p.add_argument("--lambda_nce", type=float, default=0.1)
    p.add_argument("--train_spk", nargs="*", default=None)
    p.add_argument("--max_per_spk", type=int, default=None)
    p.add_argument("--crowd_train", default=None,
                   help="folder under WORK with crops.npy from data/crowd_crops.py (training speakers only), mixed in")
    p.add_argument("--crowd_rep", type=int, default=1, help="how many times the crowd clips are repeated per epoch")
    p.add_argument("--labeled_spk", nargs="*", default=None,
                   help="low-label runs: only these speakers' transcripts are used; the rest feed KD only")
    p.add_argument("--val_spk", nargs="*", default=VAL)
    p.add_argument("--val_clips", type=int, default=VAL_CLIPS)
    return p.parse_args()


def load_val(spks, res, max_per_spk):
    """Fixed validation set, degraded once and kept in memory (no worker spawning at each validation)."""
    ds = GridDataset(spks, res=res, kd=False, max_per_spk=max_per_spk)
    items = [ds[i] for i in range(len(ds))]
    return collate(items)


@torch.no_grad()
def validate(model, val, device, bs=16):
    model.eval()
    hyps = []
    for i in range(0, len(val["texts"]), bs):
        lengths = val["lengths"][i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(to_input(val["mouth"][i:i + bs], device), lengths.to(device))
        hyps += ctc_greedy(out["logits"].float(), lengths)
    model.train()
    return wer(val["texts"], hyps), list(zip(val["texts"][:3], hyps[:3]))


def main():
    torch.backends.cudnn.benchmark = True
    a = parse()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda"
    run = os.path.join(work_dir(), "runs", a.name)
    os.makedirs(run, exist_ok=True)
    if os.path.exists(os.path.join(run, "DONE")):
        print(f"{a.name}: already done, skipping")
        return
    use = {"m2": "m2" in a.kd, "m3": "m3" in a.kd, "nce": "nce" in a.kd}
    kd = any(use.values())
    finetune = a.init is not None
    lr = a.lr or (1.5e-4 if finetune else 3e-4)
    spks = a.train_spk or available_train()
    assert spks, "no preprocessed training speakers found"

    if a.labeled_spk:
        assert kd, "unlabeled speakers carry no loss without KD; use --train_spk <labeled> for the no-KD control"
        spks = sorted(set(spks) | set(a.labeled_spk), key=lambda s: int(s[1:]))
    train_ds = GridDataset(spks, degrade=a.degrade, kd=kd, max_per_spk=a.max_per_spk, augment=not a.no_augment,
                           labeled=a.labeled_spk, codec_p=a.codec)
    if a.crowd_train:
        assert not kd, "crowd clips carry no teacher targets; use --kd none with --crowd_train"
        crowd = CrowdCrops(os.path.join(work_dir(), a.crowd_train), augment=not a.no_augment)
        leak = sorted(set(crowd.spk) & set(VAL + TEST))
        assert not leak, f"crowd training clips include val/test speakers {leak}"
        train_ds = ConcatDataset([train_ds] + [crowd] * a.crowd_rep)
        print(f"mixing {len(crowd)} x {a.crowd_rep} crowd clips into {len(train_ds) - len(crowd) * a.crowd_rep} GRID clips",
              flush=True)
    train_dl = DataLoader(train_ds, a.bs, shuffle=True, drop_last=True, collate_fn=collate, num_workers=a.workers,
                          worker_init_fn=worker_init, pin_memory=True, persistent_workers=a.workers > 0)
    val_sets = [load_val(a.val_spk, r, a.val_clips) for r in (None, 24)]

    model = LipReader(a.backend).to(device)
    if finetune:
        src, _ = load_student(a.init, device)
        model.load_state_dict(src.state_dict())
        del src
    elif a.m1:
        init_stem_from_syncnet(model, SYNCNET)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler()
    total = a.max_steps or a.epochs * (len(train_dl) // a.accum)
    warmup = a.warmup if a.warmup is not None else (200 if finetune else 1000)
    cfg = {**model.cfg, **vars(a), "lr": lr, "warmup": warmup, "train_spk": spks, "n_train": len(train_ds),
           "n_labeled": len(train_ds) if a.crowd_train else int(sum(train_ds.is_lab))}
    with open(os.path.join(run, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    state = {"epoch": 0, "step": 0, "best": float("inf"), "elapsed": 0.0}
    last = os.path.join(run, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        print(f"resumed from step {state['step']} ({state['elapsed'] / 60:.1f} min)", flush=True)

    budget = a.max_minutes * 60

    def lr_at(step, elapsed):
        if step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        if budget:
            prog = max(prog, elapsed / budget)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    log_path = os.path.join(run, "log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["epoch", "step"] + list(LOSSES) + ["val_wer_clean", "val_wer_24", "lr", "elapsed_min", "clips_per_s"])

    sums, n, seen, tw = dict.fromkeys(LOSSES, 0.0), 0, 0, time.time()

    def checkpoint():
        nonlocal sums, n, seen, tw
        state["elapsed"] = time.time() - t0
        w_clean, ex = validate(model, val_sets[0], device)
        w_24, _ = validate(model, val_sets[1], device)
        score = w_24 if a.degrade else w_clean
        print(f"== step {state['step']} val WER clean {w_clean:.4f} 24px {w_24:.4f} | {ex[0]}", flush=True)
        log.writerow([state["epoch"], state["step"]] + [round(sums[k] / max(n, 1), 4) for k in LOSSES]
                     + [round(w_clean, 4), round(w_24, 4), opt.param_groups[0]["lr"], round(state["elapsed"] / 60, 2),
                        round(seen / max(1e-6, time.time() - tw), 1)])
        logf.flush()
        if score < state["best"]:
            state["best"] = score
            save_ckpt(os.path.join(run, "best.pt"), model, cfg, step=state["step"], val_wer_clean=w_clean, val_wer_24=w_24)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "state": state,
                    "cfg": cfg}, last)
        sums, n, seen, tw = dict.fromkeys(LOSSES, 0.0), 0, 0, time.time()

    model.train()
    t0 = time.time() - state["elapsed"]
    stop = False
    while state["epoch"] < a.epochs and not stop:
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(train_dl):
            lengths = b["lengths"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(to_input(b["mouth"], device), lengths)
            w_lab = b["labeled"].to(device) if a.labeled_spk else None
            losses = {"ctc": ctc_loss(out["logits"], lengths, b["targets"].to(device), b["target_lengths"].to(device),
                                      w_lab)}
            if kd:
                m = window_mask(b["smask"].to(device), lengths)
                if use["m2"]:
                    losses["m2"] = cos_kd(out["h_face"], b["eface"].to(device), m)
                if use["m3"]:
                    losses["m3"] = cos_kd(out["h_audio"], b["eaud"].to(device), m, b["tconf"].to(device).float().clamp(0, 1))
                if use["nce"]:
                    losses["nce"] = info_nce(out["h_audio"], b["eaud"].to(device), m)
            loss = losses["ctc"] + a.lambda_v * losses.get("m2", 0) + a.lambda_a * losses.get("m3", 0) \
                + a.lambda_nce * losses.get("nce", 0)
            scaler.scale(loss / a.accum).backward()
            for k, v in losses.items():
                sums[k] += float(v.detach())
            n += 1
            seen += len(b["texts"])
            if (it + 1) % a.accum:
                continue
            elapsed = time.time() - t0
            for g in opt.param_groups:
                g["lr"] = lr_at(state["step"], elapsed)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            if state["step"] % 100 == 0:
                print(f"step {state['step']} " + " ".join(f"{k} {v / n:.3f}" for k, v in sums.items() if v)
                      + f" lr {opt.param_groups[0]['lr']:.2e} {seen / (time.time() - tw):.1f} clips/s"
                      + f" {elapsed / 60:.1f} min", flush=True)
            if (budget and elapsed > budget) or (a.max_steps and state["step"] >= a.max_steps):
                stop = True
                break
            if state["step"] % a.val_every == 0:
                checkpoint()
        else:
            state["epoch"] += 1
    checkpoint()
    logf.close()
    open(os.path.join(run, "DONE"), "w").close()
    print(f"{a.name}: done, {state['step']} steps, best val {state['best']:.4f}, {state['elapsed'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
