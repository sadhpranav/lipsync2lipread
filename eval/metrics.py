"""CTC decoding and lip-reading metrics (WER, CER, GRID letter/digit slot accuracy, bootstrap CI)."""
import numpy as np
import torch

from data.vocab import BLANK, CHARS


def _collapse(ids):
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out


def _clean(s):
    return " ".join(s.split())


def ctc_greedy(logits, lengths):
    best = logits.argmax(-1).cpu().numpy()
    return [_clean("".join(CHARS[i - 1] for i in _collapse(b[:int(n)]))) for b, n in zip(best, lengths)]


def ctc_greedy_with_conf(logits):
    """logits (T,C) for one clip -> (text, [(word, conf)]); conf = mean max-prob over the word's emitting frames."""
    probs = torch.as_tensor(logits).float().softmax(-1)
    p, best = probs.max(-1)
    words, cur, confs, prev = [], "", [], None
    for i, (c, pc) in enumerate(zip(best.tolist(), p.tolist())):
        if c != BLANK and c != prev:
            ch = CHARS[c - 1]
            if ch == " ":
                if cur:
                    words.append((cur, float(np.mean(confs))))
                cur, confs = "", []
            else:
                cur += ch
                confs.append(pc)
        elif c != BLANK and confs:
            confs.append(pc)
        prev = c
    if cur:
        words.append((cur, float(np.mean(confs))))
    return " ".join(w for w, _ in words), words


def _lev(a, b):
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return int(d[-1])


def word_errors(ref, hyp):
    return _lev(ref.split(), hyp.split()), len(ref.split())


def char_errors(ref, hyp):
    return _lev(list(ref), list(hyp)), len(ref)


def wer(refs, hyps):
    e = [word_errors(r, h) for r, h in zip(refs, hyps)]
    return sum(x for x, _ in e) / max(1, sum(n for _, n in e))


def cer(refs, hyps):
    e = [char_errors(r, h) for r, h in zip(refs, hyps)]
    return sum(x for x, _ in e) / max(1, sum(n for _, n in e))


def _align(ref, hyp):
    """Levenshtein alignment; returns for each ref position the aligned hyp word (or None if deleted)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [None] * n, n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out[i - 1] = hyp[j - 1]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def slot_acc(refs, hyps):
    """Accuracy on GRID's two free slots: letter (word 4) and digit (word 5)."""
    hit = {"letter": 0, "digit": 0}
    n = 0
    for r, h in zip(refs, hyps):
        rw = r.split()
        if len(rw) != 6:
            continue
        a = _align(rw, h.split())
        hit["letter"] += a[3] == rw[3]
        hit["digit"] += a[4] == rw[4]
        n += 1
    return {k: v / max(1, n) for k, v in hit.items()}


def bootstrap_ci(errors, counts, n_boot=1000, seed=0):
    """95% CI of sum(errors)/sum(counts) under clip-level resampling."""
    errors, counts = np.asarray(errors, float), np.asarray(counts, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errors), (n_boot, len(errors)))
    stats = errors[idx].sum(1) / np.maximum(counts[idx].sum(1), 1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
