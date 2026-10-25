"""Report figures and tables from WORK/runs/*/eval_*.json, log.csv and teacher_sanity.json.

python -m eval.figures [--runs R1_base_clean R2_ft_noKD R3_ft_m2m3 ...]   -> WORK/figures/
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.crowd import WORK

RUNS = os.path.join(WORK, "runs")
FIG = os.path.join(WORK, "figures")
CONDS = [("clean", "clean\n(88 px)"), ("deg32", "32 px"), ("deg24", "24 px"), ("deg16", "16 px")]
# categorical slots in fixed order (validated palette); colour follows the run, never its rank
ORDER = ["R1", "R2", "R3", "R5", "R6", "R8", "R4"]
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
LABELS = {"R1": "Baseline, clean training", "R2": "Baseline + low-res aug (no KD)",
          "R3": "KD face+audio (M2+M3)", "R5": "KD audio only (M3)", "R6": "KD face only (M2)",
          "R8": "KD M2+M3 + sync-contrastive", "R4": "M1 SyncNet stem init (clean)",
          "LL_noKD": "3 labeled spk, no KD", "LL_m2m3": "3 labeled spk + KD on 25 unlabeled spk"}
# low-label pair reuses the no-KD / KD colours so the encoding means the same thing in every figure
COLORS.update(LL_noKD=COLORS["R2"], LL_m2m3=COLORS["R3"])
ORDER += ["LL_noKD", "LL_m2m3"]
BASELINES = {"R1", "R2", "LL_noKD"}
OPT = {"suffix": "", "crowd": ""}  # set from CLI: eval file suffix (e.g. _grammar) and crowd decoding key suffix
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7})


def key(name):
    m = re.match(r"(R\d+)", name)
    return m.group(1) if m else name


def is_seed(name):
    return bool(re.match(r"R\d+s\d", name))


def style(name):
    k = key(name)
    base = k in BASELINES
    return dict(color=COLORS.get(k, INK2), ls="--" if base else "-", marker="o" if base else "s",
                alpha=0.45 if is_seed(name) else 1.0)


def label(name):
    return LABELS.get(key(name), name) + (" (seed 2)" if is_seed(name) else "")


def load_eval(name, cond):
    p = os.path.join(RUNS, name, f"eval_{cond}.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"]
    if OPT["suffix"] == "_grammar":  # evaluate.py nests grammar+TTA metrics under "grammar_tta"
        return {**s["grammar_tta"], "sync_score": s.get("sync_score")} if "grammar_tta" in s else None
    return s


def all_runs():
    names = [os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RUNS, "*", "eval_clean.json"))]
    return sorted(names, key=lambda n: (ORDER.index(key(n)) if key(n) in ORDER else 99, n))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIG, name + ".png"))


def headline(runs, name="headline_wer_vs_resolution", title="WER vs mouth resolution, unseen test speakers"):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(CONDS))
    ends = []
    for n in runs:
        s = [load_eval(n, c) for c, _ in CONDS]
        if any(v is None for v in s):
            continue
        y = np.array([100 * v["wer"] for v in s])
        lo = y - np.array([100 * v["wer_ci95"][0] for v in s])
        hi = np.array([100 * v["wer_ci95"][1] for v in s]) - y
        st = style(n)
        ax.errorbar(x, y, yerr=[lo, hi], capsize=3, elinewidth=1, label=label(n), **st)
        if not is_seed(n):
            ends.append(y[-1])
    # direct end labels, nudged apart so they never collide
    if ends:
        gap = 0.045 * max(max(ends), 1)
        order = np.argsort(ends)
        pos = np.array(ends, float)[order]
        for i in range(1, len(pos)):
            pos[i] = max(pos[i], pos[i - 1] + gap)
        for j, i in enumerate(order):
            ax.annotate(f"{ends[i]:.1f}", (x[-1], ends[i]), xytext=(x[-1] + 0.12, pos[j]), textcoords="data",
                        va="center", fontsize=9, color=INK)
    ax.set_xticks(x, [l for _, l in CONDS])
    ax.set_xlim(-0.3, len(CONDS) - 0.5)
    ax.set_xlabel("mouth ROI resolution (simulated camera distance →)")
    ax.set_ylabel("word error rate, %  (lower is better)")
    ax.set_title(title, loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, name)


def crowd_bars(runs):
    runs = [n for n in runs if os.path.exists(os.path.join(RUNS, n, "eval_crowd.json")) and not is_seed(n)]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True, layout="constrained")
    pxs = ["16", "24", "32"]
    w = 0.8 / len(runs)
    for ax, mode in zip(axes, ["oracle", "detected"]):
        for i, n in enumerate(runs):
            s = json.load(open(os.path.join(RUNS, n, "eval_crowd.json")))["summary"][mode + OPT["crowd"]]
            y = [100 * s[p]["wer"] for p in pxs]
            xs = np.arange(len(pxs)) + (i - (len(runs) - 1) / 2) * w
            ax.bar(xs, y, w * 0.92, color=style(n)["color"], label=label(n),
                   hatch=None if key(n) not in BASELINES else "//", edgecolor=SURFACE, linewidth=0)
        ax.set_xticks(range(len(pxs)), [f"{p} px" for p in pxs])
        ax.set_title(f"GRID-Crowd, {'GT mouth boxes' if mode == 'oracle' else 'detected + selected speaker'}", loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("word error rate, %")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=min(4, len(l)), fontsize=8.5)
    save(fig, "crowd_wer")


def fmt_ci(s):
    return f"{100 * s['wer']:.1f} [{100 * s['wer_ci95'][0]:.1f}, {100 * s['wer_ci95'][1]:.1f}]"


def table(runs):
    head = ["Run", "Clean WER", "32px WER", "24px WER", "16px WER", "16px CER", "16px letter", "16px digit",
            "16px sync", "Crowd WER (GT box)", "Crowd WER (detected)"]
    rows = []
    for n in runs:
        s = {c: load_eval(n, c) for c, _ in CONDS}
        if s["clean"] is None:
            continue
        cr = os.path.join(RUNS, n, "eval_crowd.json")
        cr = json.load(open(cr))["summary"] if os.path.exists(cr) else None
        d16 = s["deg16"]
        rows.append([f"{key(n)}{' s2' if is_seed(n) else ''}: {label(n)}"] +
                    [fmt_ci(s[c]) if s[c] else "n/a" for c, _ in CONDS] +
                    ([f"{100 * d16['cer']:.1f}", f"{100 * d16['acc_letter']:.1f}", f"{100 * d16['acc_digit']:.1f}",
                      f"{d16['sync_score']:.3f}"] if d16 else ["n/a"] * 4) +
                    ([f"{100 * cr['oracle' + OPT['crowd']]['all']['wer']:.1f}", f"{100 * cr['detected' + OPT['crowd']]['all']['wer']:.1f}"] if cr else ["n/a", "n/a"]))
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(r) + " |\n" for r in rows)
    tex = ("\\begin{tabular}{l" + "c" * (len(head) - 1) + "}\n\\hline\n" + " & ".join(head) + " \\\\\n\\hline\n" +
           "".join(" & ".join(c.replace("%", "\\%").replace("_", "\\_") for c in r) + " \\\\\n" for r in rows) +
           "\\hline\n\\end{tabular}\n")
    os.makedirs(FIG, exist_ok=True)
    open(os.path.join(FIG, "ablation.md"), "w", encoding="utf-8").write(
        "WER/CER in %, 95% bootstrap CI over clips; letter/digit = GRID free-slot accuracy %; "
        "sync = mean cos(student h_audio, teacher e_audio) on speech windows.\n\n" + md)
    open(os.path.join(FIG, "ablation.tex"), "w", encoding="utf-8").write(tex)
    print(md)


def curves(runs):
    logs = {n: list(csv.DictReader(open(os.path.join(RUNS, n, "log.csv")))) for n in runs
            if os.path.exists(os.path.join(RUNS, n, "log.csv"))}
    logs = {n: r for n, r in logs.items() if r}
    if not logs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for n, rows in logs.items():
        ep = [float(r["epoch"]) for r in rows]
        st = style(n)
        axes[0].plot(ep, [float(r["ctc"]) for r in rows], label=label(n), **st)
        v = [(float(r["epoch"]), 100 * float(r["val_wer_24"])) for r in rows if r.get("val_wer_24") not in (None, "", "nan")]
        if v:
            axes[1].plot(*zip(*v), **st)
    axes[0].set(title="Training CTC loss", xlabel="epoch", ylabel="CTC loss")
    axes[1].set(title="Validation WER at 24 px (speaker s13)", xlabel="epoch", ylabel="WER, %")
    axes[0].legend(fontsize=8)
    save(fig, "training_curves")


def teacher_sanity():
    p = os.path.join(RUNS, "teacher_sanity.json")
    if not os.path.exists(p):
        print("no teacher_sanity.json yet")
        return
    d = json.load(open(p))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), layout="constrained")
    ax = axes[0]
    for k, lab, c in (("pos_cos", "in-sync audio", COLORS["R1"]), ("neg_cos", "audio shifted 0.4 s", COLORS["R2"])):
        v = d[k]
        if isinstance(v, dict):
            ax.stairs(v["counts"], v["bins"], label=lab, color=c, fill=True, alpha=0.55)
        elif "hist_range" in d:  # v holds counts over equal bins spanning hist_range
            ax.stairs(v, np.linspace(*d["hist_range"], len(v) + 1), label=f"{lab} (mean {d.get(k[:3] + '_mean', np.nan):.2f})",
                      color=c, fill=True, alpha=0.55)
        else:
            ax.hist(v, bins=40, range=(-0.2, 1.0), alpha=0.6, color=c, label=lab, edgecolor=SURFACE, linewidth=0.5)
    sub = f"{d['n_windows']} windows, {', '.join(d['speakers'])}" if "n_windows" in d else ""
    ax.set(title="Teacher on GRID (unseen data)", xlabel="cos(face emb, audio emb)" + ("\n" + sub if sub else ""),
           ylabel="windows")
    ax.legend(fontsize=8)
    res = d["res_list"]
    axes[1].plot(res, 100 * np.asarray(d["sync_acc"]), color=COLORS["R1"], marker="o")
    axes[1].set(title="Sync accuracy vs face size", xlabel="face crop side, px", ylabel="windows correct, %")
    axes[1].axhline(50, color=INK2, lw=1, ls=":")
    axes[1].annotate("chance", (max(res), 50), xytext=(2, 3), textcoords="offset points", fontsize=8, color=INK2)
    axes[2].plot(res, d["cos_hi_lo"], color=COLORS["R3"], marker="s")
    axes[2].set(title="Embedding drift vs face size", xlabel="face crop side, px", ylabel="cos(emb@96 px, emb@res)")
    for ax in axes[1:]:
        ax.invert_xaxis()
    save(fig, "teacher_sanity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="runs for the headline plot (default: R1, R2, R3 and the best other KD run)")
    ap.add_argument("--decode", default="greedy", choices=["greedy", "grammar"],
                    help="grammar: read eval_<cond>_grammar.json and the *_grammar crowd keys")
    a = ap.parse_args()
    if a.decode == "grammar":
        OPT.update(suffix="_grammar", crowd="_grammar")
        global FIG
        FIG = os.path.join(FIG, "grammar")
    runs = all_runs()
    print("runs with eval:", runs)
    head = a.runs
    if not head:
        head = [n for n in runs if key(n) in ("R1", "R2", "R3") and not is_seed(n)]
        kd = [n for n in runs if key(n) in ("R5", "R6", "R8") and load_eval(n, "deg16")]
        if kd:
            head.append(min(kd, key=lambda n: load_eval(n, "deg16")["wer"]))
        head += [n for n in runs if is_seed(n)]
    teacher_sanity()
    if runs:
        headline(head)
        crowd_bars([n for n in head if not is_seed(n)])
        table(runs)
        curves([n for n in runs if not is_seed(n) and n.startswith("R")])
        low = [n for n in runs if n.startswith("LL_")]
        if low:
            headline(low, "lowlabel_wer_vs_resolution", "Low-label regime: 3 transcribed training speakers")


if __name__ == "__main__":
    main()
