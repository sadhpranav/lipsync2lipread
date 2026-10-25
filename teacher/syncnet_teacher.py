"""Frozen Wav2Lip expert SyncNet: the lip-sync generator's sync knowledge."""
import os
import numpy as np
import torch
import torch.nn.functional as F

from third_party.wav2lip.syncnet import SyncNet_color
from third_party.wav2lip import audio as w2l_audio

CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "work", "ckpt", "lipsync_expert.pth")
WIN = 5          # video frames per window (0.2 s)
MEL_STEP = 16    # mel columns per window
FPS = 25


def mel_start(t):
    return int(80.0 * (t / float(FPS)))  # Wav2Lip's frame -> mel index mapping (3.2 t)


def melspectrogram(wav16k):
    return w2l_audio.melspectrogram(wav16k).astype(np.float32)  # (80, M)


class SyncNetTeacher:
    def __init__(self, ckpt=CKPT, device="cuda"):
        self.net = SyncNet_color()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        self.net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    @staticmethod
    def face_windows(tface):
        """tface (T, 48, 96, 3) uint8 BGR -> (T-4, 15, 48, 96) float in [0,1], frame-major channels."""
        x = torch.from_numpy(np.ascontiguousarray(tface)).float() / 255.0      # T,H,W,3
        x = x.permute(0, 3, 1, 2)                                              # T,3,H,W
        return torch.cat([x[i:x.shape[0] - WIN + 1 + i] for i in range(WIN)], 1)  # W,15,H,W

    @staticmethod
    def mel_windows(mel, n_windows):
        """mel (80, M) -> (n_windows, 1, 80, 16); pads the end by edge if short."""
        need = mel_start(n_windows - 1) + MEL_STEP
        if mel.shape[1] < need:
            mel = np.pad(mel, ((0, 0), (0, need - mel.shape[1])), mode="edge")
        return torch.from_numpy(np.stack([mel[:, mel_start(t):mel_start(t) + MEL_STEP] for t in range(n_windows)]))[:, None]

    @torch.no_grad()
    def embed_face(self, face_win):
        return F.normalize(self.net.face_encoder(face_win.to(self.device)).flatten(1), dim=1)

    @torch.no_grad()
    def embed_audio(self, mel_win):
        return F.normalize(self.net.audio_encoder(mel_win.to(self.device)).flatten(1), dim=1)

    @torch.no_grad()
    def clip(self, tface, mel):
        """Embeddings for every 5-frame window of a clip -> (eface, eaud) each (T-4, 512)."""
        fw = self.face_windows(tface)
        return self.embed_face(fw), self.embed_audio(self.mel_windows(mel, fw.shape[0]))
