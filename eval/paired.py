"""Paired comparison of two runs on the identical test clips: WER difference (B - A) with a paired bootstrap CI.

python -m eval.paired R2_ft_noKD R3_ft_m2m3 [R9_ft_codec R10_ft_codec_m2m3 ...]   (pairs: A B A B ...)
Pairing removes the clip-difficulty variance that two independent CIs both carry, so it is the right test for
"does KD change WER on the same inputs". Writes WORK/figures/paired.md.
"""
import json
import os
import sys

import numpy as np

from data.splits import work_dir

CONDS = ["clean", "deg32", "deg24", "deg16"]


def load(run, cond):
    preds = json.load(open(os.path.join(work_dir(), "runs", run, f"eval_{cond}.json")))["preds"]
    return {(p["spk"], p["idx"]): (p["word_err"], p["n_words"]) for p in preds}


def paired(a, b, n_boot=2000, seed=0):
    keys = sorted(set(a) & set(b))
    ea, eb = np.array([a[k][0] for k in keys], float), np.array([b[k][0] for k in keys], float)
    n = np.array([a[k][1] for k in keys], float)
    delta = (eb.sum() - ea.sum()) / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n_boot, len(keys)))
    boot = (eb[idx].sum(1) - ea[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))  # two-sided bootstrap p
    return len(keys), delta, lo, hi, p


def main():
    runs = sys.argv[1:]
    assert runs and len(runs) % 2 == 0, "pass pairs of run names: A B [A B ...]"
    lines = ["| A | B | condition | n clips | WER(B) - WER(A), pt | 95% CI | p |", "|---|---|---|---|---|---|---|"]
    for a, b in zip(runs[::2], runs[1::2]):
        for c in CONDS:
            n, d, lo, hi, p = paired(load(a, c), load(b, c))
            lines.append(f"| {a} | {b} | {c} | {n} | {100 * d:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | {p:.3f} |")
    out = "\n".join(lines)
    print(out)
    os.makedirs(os.path.join(work_dir(), "figures"), exist_ok=True)
    with open(os.path.join(work_dir(), "figures", "paired.md"), "w", encoding="utf-8") as f:
        f.write("Paired bootstrap (2000 resamples of test clips) of the raw greedy WER difference; "
                "negative = B better.\n\n" + out + "\n")


if __name__ == "__main__":
    main()
