"""Evaluate a student checkpoint on test speakers under clean / degraded conditions.

Two decoders are reported: `raw` (greedy CTC, single pass) and `grammar_tta` (GRID-grammar-constrained
decoding of flip-TTA log-probs, from eval/decode.py if present). Top-level summary keys are `raw`.
Writes WORK/runs/<run>/eval_<cond>.json (summary + per-clip predictions) and eval_summary.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader

from data.dataset import GridDataset, collate, to_input, worker_init
from data.splits import TEST, ready
from distill.losses import window_mask
from eval.metrics import bootstrap_ci, cer, char_errors, ctc_greedy, slot_acc, wer, word_errors
from student.model import load_student

CONDS = {"clean": None, "deg32": 32, "deg24": 24, "deg16": 16}


def _decoder():
    try:
        from eval.decode import grammar_decode, tta_logits
        return grammar_decode, tta_logits
    except ImportError:
        print("eval/decode.py not available: grammar+tta variant skipped")
        return None


def scores(preds, key, spks):
    refs, hyps = [p["ref"] for p in preds], [p[key] for p in preds]
    we = [word_errors(r, h) for r, h in zip(refs, hyps)]
    ce = [char_errors(r, h) for r, h in zip(refs, hyps)]
    return {"wer": wer(refs, hyps), "cer": cer(refs, hyps),
            "wer_ci95": bootstrap_ci([e for e, _ in we], [n for _, n in we]),
            "cer_ci95": bootstrap_ci([e for e, _ in ce], [n for _, n in ce]),
            **{f"acc_{k}": v for k, v in slot_acc(refs, hyps).items()},
            "per_speaker_wer": {s: wer([p["ref"] for p in preds if p["spk"] == s], [p[key] for p in preds if p["spk"] == s])
                                for s in spks}}


@torch.no_grad()
def run_cond(model, spks, res, device, workers, max_per_spk=None, dec=None):
    ds = GridDataset(spks, res=res, kd=True, max_per_spk=max_per_spk)
    dl = DataLoader(ds, 16, collate_fn=collate, num_workers=workers, worker_init_fn=worker_init)
    preds, sync_sum, sync_n = [], 0.0, 0
    for b in dl:
        lengths = b["lengths"].to(device)
        x = to_input(b["mouth"], device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, lengths)
            lp = dec[1](model, x, lengths) if dec else None
        hyps = ctc_greedy(out["logits"].float(), b["lengths"])
        hyps_g = dec[0](lp.float(), b["lengths"].tolist()) if dec else [None] * len(hyps)
        m = window_mask(b["smask"].to(device), lengths)
        cos = (out["h_audio"] * torch.nn.functional.normalize(b["eaud"].to(device).float(), dim=-1)).sum(-1)
        sync_sum += float((cos * m).sum())
        sync_n += int(m.sum())
        for (spk, ci), ref, h, hg in zip(b["keys"], b["texts"], hyps, hyps_g):
            p = {"spk": spk, "idx": int(ci), "ref": ref, "hyp": h}
            p["word_err"], p["n_words"] = word_errors(ref, h)
            if hg is not None:
                p["hyp_grammar_tta"] = hg
            preds.append(p)
    summary = {"n_clips": len(preds), **scores(preds, "hyp", spks), "sync_score": sync_sum / max(1, sync_n)}
    if dec:
        summary["grammar_tta"] = scores(preds, "hyp_grammar_tta", spks)
    return summary, preds


def main():
    torch.backends.cudnn.benchmark = True
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--spk", nargs="*", default=TEST)
    p.add_argument("--conds", nargs="*", default=list(CONDS))
    p.add_argument("--max_per_spk", type=int, default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no_grammar", action="store_true")
    p.add_argument("--out", default=None, help="combined summary json (default: <run>/eval_summary.json)")
    a = p.parse_args()
    device = "cuda"
    model, cfg = load_student(a.ckpt, device)
    dec = None if a.no_grammar else _decoder()
    spks = [s for s in a.spk if ready(s)]
    missing = sorted(set(a.spk) - set(spks))
    if missing:
        print("WARNING: speakers not preprocessed, skipped:", missing)
    run = os.path.dirname(os.path.abspath(a.ckpt))
    allsum = {"ckpt": os.path.abspath(a.ckpt), "speakers": spks,
              "cfg": {k: cfg.get(k) for k in ("name", "kd", "m1", "degrade", "backend", "seed", "init")}}
    for c in a.conds:
        summary, preds = run_cond(model, spks, CONDS[c], device, a.workers, a.max_per_spk, dec)
        with open(os.path.join(run, f"eval_{c}.json"), "w") as f:
            json.dump({"summary": summary, "preds": preds}, f)
        allsum[c] = summary
        line = (f"{c:6s} raw WER {summary['wer']:.4f} [{summary['wer_ci95'][0]:.4f},{summary['wer_ci95'][1]:.4f}] "
                f"CER {summary['cer']:.4f} letter {summary['acc_letter']:.3f} digit {summary['acc_digit']:.3f} "
                f"sync {summary['sync_score']:.3f}")
        if dec:
            g = summary["grammar_tta"]
            line += f" | grammar+tta WER {g['wer']:.4f} letter {g['acc_letter']:.3f} digit {g['acc_digit']:.3f}"
        print(line, flush=True)
    with open(a.out or os.path.join(run, "eval_summary.json"), "w") as f:
        json.dump(allsum, f, indent=1)


if __name__ == "__main__":
    main()
