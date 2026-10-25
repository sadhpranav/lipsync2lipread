"""GRID dataset over WORK/proc/<spk>/ arrays (see INTERFACES.md).

Arrays are memory-mapped lazily inside each worker, so nothing large is pickled
when DataLoader workers are spawned on Windows.
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from data import vocab
from data.degrade import SIZE, degrade, test_rng, train_res
from data.splits import proc_dir

MEAN, STD = 0.421, 0.165
KD_KEYS = ("eface", "eaud", "tconf", "smask")


def augment(mouth, length, rng, shift=4, n_masks=2, max_mask=10):
    """Train-time only, identical for every run: random shift (pad + crop), h-flip, time masking with the clip mean.
    Teacher targets are left untouched."""
    p = np.pad(mouth, ((0, 0), (shift, shift), (shift, shift)), mode="edge")
    y, x = rng.integers(0, 2 * shift + 1, 2)
    m = p[:, y:y + SIZE, x:x + SIZE]
    if rng.random() < 0.5:
        m = m[:, :, ::-1]
    m = np.ascontiguousarray(m)
    mean = m[:length].mean(0).astype(np.uint8)
    for _ in range(n_masks):
        w = int(rng.integers(0, max_mask + 1))
        if w:
            s = int(rng.integers(0, max(1, length - w)))
            m[s:s + w] = mean
    return m


class GridDataset(Dataset):
    """res=None: clean (or random degradation if degrade=True); res=int: fixed deterministic degradation."""

    def __init__(self, spks, degrade=False, res=None, kd=True, max_per_spk=None, p_degrade=0.8, augment=False,
                 labeled=None, codec_p=0.0):
        """labeled: speakers whose transcripts may be used (low-label runs). Their clips are oversampled so they
        make up about half of each epoch; the other speakers' clips only feed the KD losses."""
        self.spks, self.do_degrade, self.res, self.kd, self.p_degrade = list(spks), degrade, res, kd, p_degrade
        self.augment, self.codec_p = augment, codec_p
        self.index, self.texts, self.is_lab = [], [], []
        per_spk = []
        for si, s in enumerate(self.spks):
            with open(os.path.join(proc_dir(s), "meta.json")) as f:
                meta = json.load(f)
            n = len(meta["texts"]) if max_per_spk is None else min(max_per_spk, len(meta["texts"]))
            per_spk.append((si, n, meta["texts"][:n], labeled is None or s in labeled))
        n_lab = sum(n for _, n, _, lab in per_spk if lab)
        n_unl = sum(n for _, n, _, lab in per_spk if not lab)
        rep = max(1, round(n_unl / max(n_lab, 1))) if labeled is not None else 1
        for si, n, texts, lab in per_spk:
            for _ in range(rep if lab else 1):
                self.index += [(si, ci) for ci in range(n)]
                self.texts += texts
                self.is_lab += [lab] * n
        self._arrs = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_arrs"] = {}
        return d

    def _arrays(self, si):
        if si not in self._arrs:
            d = proc_dir(self.spks[si])
            keys = ("mouth", "nframes") + (KD_KEYS if self.kd else ())
            self._arrs[si] = {k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r") for k in keys}
        return self._arrs[si]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, ci = self.index[i]
        a = self._arrays(si)
        mouth = np.array(a["mouth"][ci])
        length = int(a["nframes"][ci])
        if self.res is not None:
            mouth = degrade(mouth, self.res, test_rng(self.spks[si], ci, self.res))
        elif self.do_degrade or self.augment:
            rng = np.random.default_rng()
            if self.do_degrade:
                mouth = degrade(mouth, train_res(rng, self.p_degrade), rng, self.codec_p)
            if self.augment:
                mouth = augment(mouth, length, rng)
        item = {"mouth": mouth, "length": length, "text": self.texts[i], "key": (self.spks[si], ci),
                "labeled": self.is_lab[i]}
        if self.kd:
            for k in KD_KEYS:
                item[k] = np.array(a[k][ci])
        return item


def collate(batch):
    out = {
        "mouth": torch.from_numpy(np.stack([b["mouth"] for b in batch])),
        "lengths": torch.tensor([b["length"] for b in batch]),
        "texts": [b["text"] for b in batch],
        "keys": [b["key"] for b in batch],
        "labeled": torch.tensor([b.get("labeled", True) for b in batch]),
    }
    tg = [vocab.encode(b["text"]) for b in batch]
    out["targets"] = torch.tensor([c for t in tg for c in t], dtype=torch.long)
    out["target_lengths"] = torch.tensor([len(t) for t in tg])
    if "eface" in batch[0]:
        for k in KD_KEYS:
            out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
    return out


def to_input(mouth_u8, device):
    """uint8 (B,T,88,88) -> normalised float (B,1,T,88,88) on device."""
    x = mouth_u8.to(device, non_blocking=True).float().div_(255).sub_(MEAN).div_(STD)
    return x.unsqueeze(1)


def worker_init(_):
    import cv2
    cv2.setNumThreads(0)
