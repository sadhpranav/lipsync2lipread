"""Simulated far-camera degradation of 88x88 mouth crops (M4 student input)."""
import zlib
import cv2
import numpy as np

SIZE = 88
TRAIN_RES = (16, 24, 32)


def degrade(frames, res, rng):
    """frames uint8 (T,88,88) -> same shape, as if the mouth ROI were only res x res pixels."""
    if res >= SIZE:
        return frames
    sigma = rng.uniform(0, 1.2)
    q = int(rng.integers(20, 61))
    alpha, beta = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    out = np.empty(frames.shape, np.float32)
    for i, f in enumerate(frames):
        s = cv2.resize(f, (res, res), interpolation=cv2.INTER_AREA)
        if sigma > 0.1:
            s = cv2.GaussianBlur(s, (0, 0), sigma)
        s = cv2.imdecode(cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
        out[i] = cv2.resize(s, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return np.clip(out * alpha + beta, 0, 255).astype(np.uint8)


def test_rng(spk, idx, res):
    """Deterministic per-clip RNG for test-time degradation (crc32: stable across processes, unlike hash())."""
    return np.random.default_rng(zlib.crc32(f"{spk}/{idx}/{res}".encode()))


def train_res(rng, p_degrade=0.8):
    return int(rng.choice(TRAIN_RES)) if rng.random() < p_degrade else SIZE
