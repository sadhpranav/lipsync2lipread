"""Face landmarks -> mouth ROI and teacher face crop.

The same crop geometry is used for preprocessing (training/eval) and for the
interface, so demo numbers match evaluation numbers.
"""
import os
import cv2
import numpy as np

CKPT_DIR = os.environ.get("L2L_CKPT", os.path.join(
    os.environ.get("L2L_WORK", os.path.join(os.path.dirname(__file__), "..", "..", "work")), "ckpt"))


def ckpt_path(name):
    """Path of a model file in CKPT_DIR, with a pointer to the download script if it is missing."""
    p = os.path.join(CKPT_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{name} not found in {os.path.abspath(CKPT_DIR)}; run scripts/get_models.sh "
                                "(or set L2L_WORK / L2L_CKPT)")
    return p


MOUTH_SIZE = 88          # student input side
MOUTH_SCALE = 1.8        # ROI side = MOUTH_SCALE * mouth-corner distance
FACE_SIZE = 96           # teacher input side (lower half used -> 48 x 96)

# MediaPipe face-mesh indices
L_CORNER, R_CORNER, UP_LIP, LOW_LIP = 61, 291, 0, 17


class Landmarker:
    """Thin wrapper around MediaPipe FaceLandmarker (IMAGE mode, CPU)."""

    def __init__(self, num_faces=1, min_conf=0.3):
        import mediapipe as mp
        v = mp.tasks.vision
        opts = v.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=ckpt_path("face_landmarker.task")),
            running_mode=v.RunningMode.IMAGE, num_faces=num_faces,
            min_face_detection_confidence=min_conf, min_face_presence_confidence=min_conf)
        self._mp = mp
        self.lm = v.FaceLandmarker.create_from_options(opts)

    def __call__(self, bgr):
        """Return list of (478, 2) pixel landmark arrays, one per face."""
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[:, :, ::-1]))
        res = self.lm.detect(img)
        return [np.array([[p.x * w, p.y * h] for p in face], np.float32) for face in res.face_landmarks]

    def close(self):
        self.lm.close()


def geometry_from_landmarks(pts):
    """pts (478,2) -> (mouth_cx, mouth_cy, mouth_w, face_box[x1,y1,x2,y2])."""
    mouth = pts[[L_CORNER, R_CORNER, UP_LIP, LOW_LIP]]
    cx, cy = mouth.mean(0)
    mw = float(np.linalg.norm(pts[L_CORNER] - pts[R_CORNER]))
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    y2 = y2 + 0.05 * (y2 - y1)  # Wav2Lip pads the chin slightly
    return np.array([cx, cy, mw, x1, y1, x2, y2], np.float32)


def interpolate_track(frame_idx, geoms, n_frames, smooth=5):
    """Linearly interpolate sparse per-frame geometry to all frames, then moving-average smooth."""
    frame_idx = np.asarray(frame_idx)
    geoms = np.asarray(geoms, np.float32)
    out = np.stack([np.interp(np.arange(n_frames), frame_idx, geoms[:, j]) for j in range(geoms.shape[1])], 1)
    if smooth > 1 and n_frames >= smooth:
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
        out = np.stack([np.convolve(padded[:, j], k, mode="valid") for j in range(out.shape[1])], 1)
    return out.astype(np.float32)


def mouth_box(geom):
    cx, cy, mw = geom[:3]
    side = MOUTH_SCALE * mw
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], np.float32)


def crop_square(img, box, size):
    """Crop box (may exceed image bounds; edge-padded) and resize to size x size."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    pad = int(max(0, -x1, -y1, x2 - w, y2 - h)) + 2
    if pad > 2:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    side_w, side_h = x2 - x1, y2 - y1
    if side_w > size or side_h > size:
        # warpAffine silently treats INTER_AREA as bilinear (aliasing on downscale): cut the box out at native
        # resolution (sub-pixel exact), then area-average down like a real sensor would
        nw, nh = max(size, int(np.ceil(side_w))), max(size, int(np.ceil(side_h)))
        M = np.array([[nw / side_w, 0, -x1 * nw / side_w], [0, nh / side_h, -y1 * nh / side_h]], np.float32)
        return cv2.resize(cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR), (size, size),
                          interpolation=cv2.INTER_AREA)
    M = np.array([[size / side_w, 0, -x1 * size / side_w], [0, size / side_h, -y1 * size / side_h]], np.float32)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)


def mouth_crop(bgr, geom):
    """88x88 grayscale mouth ROI (student input)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return crop_square(gray, mouth_box(geom), MOUTH_SIZE)


def teacher_face_crop(bgr, geom):
    """Lower half of a 96x96 face crop, BGR uint8 (48, 96, 3) -- Wav2Lip SyncNet input."""
    x1, y1, x2, y2 = geom[3:7]
    face = crop_square(bgr, (x1, y1, x2, y2), FACE_SIZE)
    return face[FACE_SIZE // 2:]


def track_video(frames, landmarker, stride=5):
    """Single-face track for a clip. Returns geometry (T, 7) or None if no face found."""
    idx, geoms = [], []
    T = len(frames)
    sample = sorted(set(list(range(0, T, stride)) + [T - 1]))
    for i in sample:
        faces = landmarker(frames[i])
        if faces:
            # largest face
            g = max((geometry_from_landmarks(f) for f in faces), key=lambda g: (g[5] - g[3]) * (g[6] - g[4]))
            idx.append(i)
            geoms.append(g)
    if len(idx) < max(2, len(sample) // 3):
        return None
    return interpolate_track(idx, geoms, T)
