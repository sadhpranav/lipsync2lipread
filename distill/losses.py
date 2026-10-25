"""Task loss (CTC) and distillation losses from the frozen SyncNet teacher."""
import torch
from torch.nn import functional as F

from data.vocab import BLANK


def ctc_loss(logits, lengths, targets, target_lengths, weight=None):
    """weight (B,) masks clips without labels (low-label runs); None is the plain mean over the batch."""
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    if weight is None:
        return F.ctc_loss(lp, targets, lengths, target_lengths, blank=BLANK, reduction="mean", zero_infinity=True)
    per = F.ctc_loss(lp, targets, lengths, target_lengths, blank=BLANK, reduction="none", zero_infinity=True)
    w = weight.float()
    return (per / target_lengths.clamp_min(1) * w).sum() / w.sum().clamp_min(1)


def window_mask(smask, lengths):
    """Speech windows (from alignments) that also lie fully inside the real frames."""
    W = smask.shape[1]
    valid = torch.arange(W, device=smask.device)[None] < (lengths[:, None] - 4)
    return smask.bool() & valid


def cos_kd(h, e, mask, weight=None):
    """Weighted mean of (1 - cos(h, e)) over masked windows. h is already L2-normalised."""
    cos = (h * F.normalize(e.float(), dim=-1)).sum(-1)
    w = mask.float() if weight is None else mask.float() * weight.float()
    return ((1 - cos) * w).sum() / w.sum().clamp_min(1e-6)


def info_nce(h, e, mask, tau=0.1, exclude=2):
    """Sync-contrastive KD: each student window must pick its own teacher audio embedding among the
    clip's other speech windows. Negatives within `exclude` frames overlap the positive heavily and
    are dropped (they are near-duplicates, not true off-sync examples)."""
    e = F.normalize(e.float(), dim=-1)
    total, count = h.new_zeros(()), 0
    for b in range(h.shape[0]):
        idx = mask[b].nonzero().squeeze(1)
        if idx.numel() < 2:
            continue
        logits = h[b, idx] @ e[b, idx].T / tau
        dist = (idx[:, None] - idx[None]).abs()
        logits = logits.masked_fill((dist > 0) & (dist <= exclude), float("-inf"))
        total = total + F.cross_entropy(logits, torch.arange(idx.numel(), device=h.device), reduction="sum")
        count += idx.numel()
    return total / max(count, 1)
