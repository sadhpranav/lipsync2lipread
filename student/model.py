"""Student lip reader: 3D-conv + ResNet-18 frontend, Conformer/BiGRU backend, CTC head, KD heads."""
import math
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from data.vocab import NUM_CLASSES


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False), nn.BatchNorm3d(64), nn.ReLU(True),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))
        r = torchvision.models.resnet18(weights=None)
        self.trunk = nn.Sequential(r.layer1, r.layer2, r.layer3, r.layer4)

    def forward(self, x):  # (B,1,T,H,W) -> (B,T,512)
        B, T = x.shape[0], x.shape[2]
        x = self.stem(x).transpose(1, 2).flatten(0, 1)
        return self.trunk(x).mean((2, 3)).view(B, T, -1)


class FeedForward(nn.Module):
    def __init__(self, d, mult=4, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(p),
                                 nn.Linear(d * mult, d), nn.Dropout(p))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d, k=15, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x, pad_mask):
        y = self.ln(x).masked_fill(pad_mask[..., None], 0).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = self.pw2(F.silu(self.bn(self.dw(y))))
        return self.drop(y.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d, heads=4, k=15, p=0.1):
        super().__init__()
        self.ff1, self.ff2 = FeedForward(d, p=p), FeedForward(d, p=p)
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)
        self.conv = ConvModule(d, k, p)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x, pad_mask):
        x = x + 0.5 * self.ff1(x)
        y = self.ln_att(x)
        x = x + self.drop(self.att(y, y, y, key_padding_mask=pad_mask, need_weights=False)[0])
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Conformer(nn.Module):
    def __init__(self, d_in=512, d=256, n_layers=4, heads=4, k=15, p=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.Dropout(p))
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, k, p) for _ in range(n_layers)])
        self.out_dim = d

    def forward(self, x, lengths):
        T = x.shape[1]
        pad_mask = torch.arange(T, device=x.device)[None] >= lengths[:, None]
        x = self.inp(x) + sinusoid(T, self.inp[0].out_features, x.device).to(x.dtype)
        for b in self.blocks:
            x = b(x, pad_mask)
        return x


class BiGRU(nn.Module):
    def __init__(self, d_in=512, d=256, n_layers=2, p=0.2):
        super().__init__()
        self.rnn = nn.GRU(d_in, d, n_layers, batch_first=True, bidirectional=True, dropout=p)
        self.drop = nn.Dropout(p)
        self.out_dim = 2 * d

    def forward(self, x, lengths):
        T = x.shape[1]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=T)
        return self.drop(y)


class LipReader(nn.Module):
    def __init__(self, backend="conformer", d_model=256, n_layers=4):
        super().__init__()
        self.cfg = {"backend": backend, "d_model": d_model, "n_layers": n_layers}
        self.frontend = Frontend()
        self.backend = Conformer(d=d_model, n_layers=n_layers) if backend == "conformer" else BiGRU(d=d_model)
        self.ctc = nn.Linear(self.backend.out_dim, NUM_CLASSES)
        self.kd_window = nn.Sequential(nn.Conv1d(512, 512, 5), nn.GELU())  # window t covers frames t..t+4
        self.h_face, self.h_audio = nn.Linear(512, 512), nn.Linear(512, 512)

    def forward(self, x, lengths):
        feat = self.frontend(x)
        logits = self.ctc(self.backend(feat, lengths))
        w = self.kd_window(feat.transpose(1, 2)).transpose(1, 2)
        return {"logits": logits, "feat": feat,
                "h_face": F.normalize(self.h_face(w).float(), dim=-1),
                "h_audio": F.normalize(self.h_audio(w).float(), dim=-1)}


def init_stem_from_syncnet(model, syncnet_ckpt_path):
    """M1: SyncNet's first conv is (32, 15=5 frames x 3 BGR, 7, 7); averaging colour gives a (32, 5, 7, 7)
    spatio-temporal kernel that fits the student's Conv3d(1, 64, (5, 7, 7)) stem exactly."""
    sd = torch.load(syncnet_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    w = sd["face_encoder.0.conv_block.0.weight"].float().view(32, 5, 3, 7, 7).mean(2).unsqueeze(1)
    stem = model.frontend.stem[0].weight
    w = w.to(stem.device)
    with torch.no_grad():
        stem[:32] = (w * (stem.std() / w.std())).to(stem)
    return model


def save_ckpt(path, model, cfg, **extra):
    torch.save({"model": model.state_dict(), "cfg": cfg, **extra}, path)


def load_student(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = LipReader(cfg.get("backend", "conformer"), cfg.get("d_model", 256), cfg.get("n_layers", 4))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg
