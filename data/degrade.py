"""Simulated far-camera degradation of 88x88 mouth crops (M4 student input)."""
import zlib
import cv2
import numpy as np

SIZE = 88
TRAIN_RES = (16, 24, 32)
# H.264 quantiser range for --codec: brackets the mouth-motion attenuation (0.51-0.82) and PSNR (~35 dB) of the
# 250 kbps GRID-Crowd encoder, measured on the validation speaker's crowd clips (s13), not on the test set
CODEC_QP = (28, 35)


def h264(frames, qp, preset="veryfast"):
    """Round-trip a small grey clip (T,h,w) uint8 (h, w even) through H.264 at a fixed quantiser, in memory.
    One I-frame then predicted frames, as in a low-bitrate stream: small mouth motion is smeared or dropped.
    Grey values go straight into the Y plane (U = V = 128), so qp=0 is an exact round trip. ~10 ms per clip."""
    import io
    import av
    T, h, w = frames.shape
    uv = np.full((h // 2, w), 128, np.uint8)
    buf = io.BytesIO()
    with av.open(buf, "w", format="h264") as out:
        st = out.add_stream("libx264", rate=25)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"qp": str(int(qp)), "preset": preset, "threads": "1"}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.concatenate([f, uv]), format="yuv420p")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    buf.seek(0)
    with av.open(buf, "r", format="h264") as inp:
        vs = inp.streams.video[0]
        vs.thread_type = "NONE"
        dec = [fr.to_ndarray()[:h] for fr in inp.decode(vs)]
    return np.stack((dec + [dec[-1]] * T)[:T]) if dec else frames


def degrade(frames, res, rng, codec_p=0.0, qp_range=CODEC_QP):
    """frames uint8 (T,88,88) -> same shape, as if the mouth ROI were only res x res pixels.
    codec_p: probability of replacing the per-frame JPEG by an H.264 round trip of the whole small clip
    (train-time only; with codec_p=0 the random draws and output are exactly those of the JPEG-only version)."""
    if res >= SIZE:
        return frames
    sigma = rng.uniform(0, 1.2)
    q = int(rng.integers(20, 61))
    alpha, beta = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    small = []
    for f in frames:
        s = cv2.resize(f, (res, res), interpolation=cv2.INTER_AREA)
        if sigma > 0.1:
            s = cv2.GaussianBlur(s, (0, 0), sigma)
        small.append(s)
    if codec_p > 0 and rng.random() < codec_p:
        small = h264(np.stack(small), int(rng.integers(qp_range[0], qp_range[1] + 1)))
    else:
        small = [cv2.imdecode(cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
                 for s in small]
    out = np.empty(frames.shape, np.float32)
    for i, s in enumerate(small):
        out[i] = cv2.resize(s, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return np.clip(out * alpha + beta, 0, 255).astype(np.uint8)


def test_rng(spk, idx, res):
    """Deterministic per-clip RNG for test-time degradation (crc32: stable across processes, unlike hash())."""
    return np.random.default_rng(zlib.crc32(f"{spk}/{idx}/{res}".encode()))


def train_res(rng, p_degrade=0.8):
    return int(rng.choice(TRAIN_RES)) if rng.random() < p_degrade else SIZE
