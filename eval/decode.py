"""GRID-grammar-constrained CTC decoding, test-time flip averaging, lexicon snapping.

grammar_decode: beam search over the 6 GRID slots with exact CTC forward (sum-over-alignments)
scores; each prefix is ranked by its exact marginal over all grammar completions (a forward pass
over the time-reversed grammar). beam=0 gives exact Viterbi over the grammar token graph.
"""
import numpy as np
import torch
import torch.nn.functional as F

from data.vocab import BLANK, GRID_SLOTS, decode, encode

SPACE = encode(" ")[0]
_NEG = -1e30
_G = None


def _build_graph(slots=GRID_SLOTS):
    """States: initial blank (slot 0), then per slot word a CTC segment [t0, b1, t1, ..., t_{n-1}, b_n].
    Slot k>1 words start with a space token. Entry node k (index N+k) feeds slot k's first tokens and is
    the max over slot k-1's exit states; entry 0 = start, entry 7 = final."""
    labels, preds, seg, exits = [BLANK], [[0, ("e", 0), None]], [-1], [[0]]
    words = []
    for k, slot in enumerate(slots, start=1):
        ex = []
        for w in slot:
            toks = encode(w if k == 1 else " " + w)
            sid = len(words)
            words.append((k - 1, w))
            prev_t = None
            for j, x in enumerate(toks):
                t_idx = len(labels)
                if j == 0:
                    labels.append(x); preds.append([t_idx, ("e", k), None]); seg.append(sid)
                else:
                    b_idx = t_idx
                    labels.append(BLANK); preds.append([b_idx, prev_t, None]); seg.append(sid)
                    t_idx = b_idx + 1
                    labels.append(x); preds.append([t_idx, b_idx, prev_t if x != toks[j - 1] else None]); seg.append(sid)
                prev_t = t_idx
            b_idx = len(labels)
            labels.append(BLANK); preds.append([b_idx, prev_t, None]); seg.append(sid)
            ex += [prev_t, b_idx]
        exits.append(ex)
    N = len(labels)
    n_entry = len(slots) + 2
    sentinel = N + n_entry
    P = np.array([[sentinel if p is None else (N + p[1] if isinstance(p, tuple) else p) for p in ps] for ps in preds])
    return dict(labels=np.array(labels), preds=P, seg=np.array(seg), exits=[np.array(e) for e in exits],
                words=words, N=N, n_entry=n_entry, sentinel=sentinel)


def _graph():
    global _G
    if _G is None:
        _G = _build_graph()
    return _G


def _viterbi(lp, g=None):
    """lp (T, C) numpy log-probs -> (word list, per-frame state path) or (None, None) if infeasible."""
    g = g or _graph()
    N, ne, labels, preds, exits = g["N"], g["n_entry"], g["labels"], g["preds"], g["exits"]
    T = lp.shape[0]
    V = np.full(N + ne + 1, _NEG)
    V[N + 0] = V[N + 1] = 0.0  # start; slot 1 may begin at t=0 without a leading blank
    Vs = np.empty((T, N + ne + 1))
    bptr = np.empty((T, N), np.int64)
    ar = np.arange(N)
    for t in range(T):
        cand = V[preds]
        a = cand.argmax(1)
        new = np.full_like(V, _NEG)
        new[:N] = cand[ar, a] + lp[t, labels]
        for k in range(1, ne):
            new[N + k] = new[exits[k - 1]].max()
        bptr[t] = preds[ar, a]
        Vs[t] = V = new
    if V[N + ne - 1] <= _NEG / 2:
        return None, None
    path = np.empty(T, np.int64)
    e = exits[ne - 2]
    cur = e[Vs[T - 1, e].argmax()]
    for t in range(T - 1, -1, -1):
        path[t] = cur
        p = bptr[t, cur]
        if t > 0 and p >= N:
            e = exits[p - N - 1]
            p = e[Vs[t - 1, e].argmax()]
        cur = p
    segs = [s for i, s in enumerate(g["seg"][path]) if s >= 0 and (i == 0 or s != g["seg"][path][i - 1])]
    return [g["words"][s][1] for s in segs], path


def _word_scores(lp, path, words, g=None):
    g = g or _graph()
    seg, labels = g["seg"][path], g["labels"][path]
    post = np.exp(lp[np.arange(len(path)), labels])
    keep = (labels != BLANK) & (labels != SPACE) & (seg >= 0)
    order = [s for i, s in enumerate(seg) if s >= 0 and (i == 0 or s != seg[i - 1])]
    out = []
    for w, s in zip(words, order):
        m = keep & (seg == s)
        out.append((w, float(post[m].mean()) if m.any() else 0.0))
    return out


def ctc_logliks(lp_t, sentences):
    """Exact CTC log-likelihood (sum over alignments) of each sentence under lp_t (T, C) torch log-probs."""
    tg = [torch.tensor(encode(s), dtype=torch.long) for s in sentences]
    T, C = lp_t.shape
    n = len(sentences)
    nll = F.ctc_loss(lp_t[:, None].expand(T, n, C), torch.cat(tg), torch.full((n,), T, dtype=torch.long),
                     torch.tensor([len(x) for x in tg]), blank=BLANK, reduction="none", zero_infinity=False)
    return -nll.numpy()


_NINF = float("-inf")


def _seg_forward(lp, ein, toks, tlen):
    """CTC forward through one word segment for many candidates at once (sum semiring).
    lp (B,T,C) log-probs; ein (B,n,T+1) entry: ein[...,0] = start before frame 0, ein[...,t+1] = prefix ended at frame t;
    toks (n,M) padded token ids, tlen (n,) lengths. Segment states [t0, b1, t1, ..., t_{m-1}, b_m].
    Returns end_any (B,n,T) = ends at frame t in last token or trailing blank; end_tok (B,n,T) = in last token."""
    B, T, _ = lp.shape
    n, M = toks.shape
    S = 2 * M
    lab = torch.zeros(n, S, dtype=torch.long)
    lab[:, 0::2] = toks
    valid = torch.arange(S)[None] < 2 * tlen[:, None]
    skip = torch.zeros(n, S, dtype=torch.bool)
    skip[:, 2::2] = toks[:, 1:] != toks[:, :-1]
    skip &= valid
    last_t, last_b = 2 * tlen - 2, 2 * tlen - 1
    alpha = torch.full((B, n, S), _NINF)
    end_any = torch.empty(B, n, T)
    end_tok = torch.empty(B, n, T)
    pad = torch.full((B, n, 1), _NINF)
    for t in range(T):
        s1 = torch.cat([pad, alpha[..., :-1]], -1)
        s2 = torch.where(skip, torch.cat([pad, pad, alpha[..., :-2]], -1), _NINF)
        pre = torch.logsumexp(torch.stack([alpha, s1, s2]), 0)
        pre[..., 0] = torch.logaddexp(alpha[..., 0], ein[..., t])
        alpha = torch.where(valid, pre + lp[:, t][:, lab], _NINF)
        a_t = alpha.gather(-1, last_t.expand(B, n)[..., None])[..., 0]
        end_tok[..., t] = a_t
        end_any[..., t] = torch.logaddexp(a_t, alpha.gather(-1, last_b.expand(B, n)[..., None])[..., 0])
    return end_any, end_tok


def _tok(words, lead_space, reverse=False):
    seqs = [encode((" " if lead_space else "") + w) for w in words]
    seqs = [x[::-1] for x in seqs] if reverse else seqs
    M = max(len(x) for x in seqs)
    return torch.tensor([x + [BLANK] * (M - len(x)) for x in seqs]), torch.tensor([len(x) for x in seqs])


def _blank_start(lp):
    """(B,T+1): index 0 = 0 (start), index t+1 = log P(frames 0..t all blank)."""
    return torch.cat([torch.zeros(lp.shape[0], 1), lp[..., BLANK].cumsum(1)], 1)


def _completions(lp):
    """C[k] (B,T+1): log-sum over all grammar completions (slots k+1..6) of P(frames s..T-1 emit them),
    starting exactly with the leading space token at frame s. Computed by a forward pass over the
    time-reversed, word-reversed grammar (sum over words at every slot is exact in the sum semiring)."""
    B, T, _ = lp.shape
    lpr = lp.flip(1)
    R = _blank_start(lpr)
    C = {}
    for j in range(len(GRID_SLOTS), 1, -1):  # slot index j = 6..2 (1-based)
        tk, tl = _tok(GRID_SLOTS[j - 1], True, reverse=True)
        e_any, e_tok = _seg_forward(lpr, R[:, None].expand(B, len(tl), T + 1), tk, tl)
        c = torch.logsumexp(e_tok, 1).flip(1)  # reversed frame t' -> original start frame T-1-t'
        C[j - 1] = torch.cat([c, torch.full((B, 1), _NINF)], 1)
        R = torch.cat([torch.full((B, 1), _NINF), torch.logsumexp(e_any, 1)], 1)
    return C


def _beam_decode(lp, beam=16):
    """lp (B,T,C) log-probs (padded frames: blank = 0). Returns list of word lists."""
    B, T, _ = lp.shape
    C = _completions(lp)
    E = _blank_start(lp)[:, None]          # (B,1,T+1) beam of prefix end vectors
    hyps = [[[]] for _ in range(B)]
    for k, slot in enumerate(GRID_SLOTS, start=1):
        K = E.shape[1]
        tk, tl = _tok(slot, k > 1)
        nw = len(slot)
        ein = E[:, :, None].expand(B, K, nw, T + 1).reshape(B, K * nw, T + 1)
        e_any, _ = _seg_forward(lp, ein, tk.repeat(K, 1), tl.repeat(K))
        if k < len(GRID_SLOTS):
            score = torch.logsumexp(e_any + C[k][:, None, 1:], -1)
        else:
            score = e_any[..., -1]
        top = score.topk(min(beam, K * nw) if k < len(GRID_SLOTS) else 1, -1).indices  # (B,K')
        E = torch.cat([torch.full((B, top.shape[1], 1), _NINF), e_any.gather(1, top[..., None].expand(-1, -1, T))], -1)
        hyps = [[hyps[b][int(i) // nw] + [slot[int(i) % nw]] for i in top[b]] for b in range(B)]
    return [h[0] for h in hyps]


def _pad_lp(x, lengths):
    """log_softmax and force frames beyond each length to blank (probability 1), which leaves CTC sums unchanged."""
    lp = x.log_softmax(-1)
    if lengths is not None:
        L = torch.as_tensor(lengths).view(-1, 1)
        m = torch.arange(lp.shape[1])[None] >= L
        fill = torch.full((lp.shape[-1],), _NINF)
        fill[BLANK] = 0.0
        lp = torch.where(m[..., None], fill, lp)
    return lp


def grammar_decode(log_probs, lengths=None, return_scores=False, beam=16, chunk=256):
    """GRID-grammar-constrained CTC decoding.
    log_probs (T, C) or (B, T, C), torch or numpy (raw logits fine: log_softmax is applied).
    beam > 0: beam search over slots with exact CTC sum scores, prefixes ranked by their exact marginal
    over all grammar completions; beam = 0: exact Viterbi (best single alignment) over the grammar graph.
    Returns a sentence (or list), or (sentence, [(word, conf)]) with return_scores."""
    x = log_probs.detach().float().cpu() if torch.is_tensor(log_probs) else torch.as_tensor(np.asarray(log_probs), dtype=torch.float32)
    single = x.dim() == 2
    if single:
        x, lengths = x[None], None if lengths is None else [int(lengths)]
    lp = _pad_lp(x, lengths)
    out = []
    for i in range(0, lp.shape[0], chunk):
        sub = lp[i:i + chunk]
        if beam > 0:
            words = _beam_decode(sub, beam)
        else:
            words = [_viterbi(l.double().numpy())[0] for l in sub]
        for l, w in zip(sub, words):
            lpn = l.double().numpy()
            if w is None:  # infeasible: fall back to greedy + lexicon snap
                g = lpn.argmax(1)
                text = lexicon_snap(decode([int(c) for j, c in enumerate(g) if c != BLANK and (j == 0 or c != g[j - 1])]))
                out.append((text, [(t, 0.0) for t in text.split()]) if return_scores else text)
                continue
            text = " ".join(w)
            if not return_scores:
                out.append(text)
                continue
            g = _build_graph([[t] for t in w])  # forced alignment for per-word confidence
            _, path = _viterbi(lpn, g)
            out.append((text, _word_scores(lpn, path, w, g)))
    return out[0] if single else out


@torch.no_grad()
def tta_logits(model, x, lengths):
    """Mean log-softmax over the clip and its horizontal flip."""
    a = model(x, lengths)["logits"].float().log_softmax(-1)
    b = model(torch.flip(x, dims=[-1]), lengths)["logits"].float().log_softmax(-1)
    return (a + b) / 2


def _ed(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def lexicon_snap(text):
    """Snap greedy words to the GRID grammar: monotone DP aligning predicted words to the 6 slots
    (match = edit distance to the slot's nearest word; extra predicted words dropped; missing slots
    filled with the slot's shortest word)."""
    pw = text.split()
    S, n = len(GRID_SLOTS), len(pw)
    best = [[min(((_ed(p, w), w) for w in slot)) for slot in GRID_SLOTS] for p in pw]
    fill = [min(slot, key=len) for slot in GRID_SLOTS]
    INF = float("inf")
    D = [[INF] * (S + 1) for _ in range(n + 1)]
    bt = [[None] * (S + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for i in range(n + 1):
        for j in range(S + 1):
            if D[i][j] == INF:
                continue
            if i < n and j < S and D[i][j] + best[i][j][0] < D[i + 1][j + 1]:
                D[i + 1][j + 1], bt[i + 1][j + 1] = D[i][j] + best[i][j][0], (i, j, best[i][j][1])
            if i < n and D[i][j] + len(pw[i]) < D[i + 1][j]:
                D[i + 1][j], bt[i + 1][j] = D[i][j] + len(pw[i]), (i, j, None)
            if j < S and D[i][j] + len(fill[j]) < D[i][j + 1]:
                D[i][j + 1], bt[i][j + 1] = D[i][j] + len(fill[j]), (i, j, fill[j])
    out, i, j = [], n, S
    while (i, j) != (0, 0):
        pi, pj, w = bt[i][j]
        if pj != j:
            out.append(w)
        i, j = pi, pj
    return " ".join(reversed(out))
