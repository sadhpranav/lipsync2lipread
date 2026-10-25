"""Crowd / in-the-wild localisation: face detection (tiled) -> IoU tracking ->
per-track mouth geometry -> active-speaker selection by mouth motion.

Uses the same mouth-crop geometry as preprocessing (data/facetrack.py).
"""
import os
import cv2
import numpy as np

from data import facetrack as ft

DET_SIDE = 640           # frames are resized so the long side is this for detection
TILE, TILE_STRIDE = 192, 128
CLIP_LEN = 75
# mouth/face geometry relative to a BlazeFace box (x, y, w, h), calibrated on 40 GRID frames
FALLBACK_REL = np.array([0.499, 0.746, 0.291, 0.096, -0.02, 0.9, 1.076], np.float32)


def read_video(path, fps=25, max_frames=None, start=0):
    """Decode a video and resample it to `fps` by nearest timestamps. Returns list of BGR frames."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    need = None if max_frames is None else int((start + max_frames) * src_fps / fps) + 2
    frames = []
    while need is None or len(frames) < need:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if frames and abs(src_fps - fps) > 0.5:
        idx = np.round(np.arange(int(len(frames) * fps / src_fps)) * src_fps / fps).astype(int)
        frames = [frames[min(i, len(frames) - 1)] for i in idx]
    frames = frames[start:]
    return frames if max_frames is None else frames[:max_frames]


def iou(a, b):
    """a (N,4), b (M,4) -> (N,M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-6)


def nms(dets, thr=0.3):
    if len(dets) == 0:
        return dets
    dets = dets[np.argsort(-dets[:, 4])]
    keep = []
    for d in dets:
        if not keep or iou(d[None, :4], np.array(keep)[:, :4]).max() < thr:
            keep.append(d)
    return np.array(keep, np.float32)


class FaceDetector:
    """BlazeFace full-range on the whole frame plus overlapping 192 px tiles (for tiny faces)."""

    def __init__(self, min_conf=0.4):
        import mediapipe as mp
        v = mp.tasks.vision
        path = os.path.join(ft.CKPT_DIR, "blaze_face_full_range.tflite")
        if not os.path.exists(path):
            path = os.path.join(ft.CKPT_DIR, "blaze_face_short_range.tflite")
        self._mp = mp
        self.det = v.FaceDetector.create_from_options(v.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=path),
            running_mode=v.RunningMode.IMAGE, min_detection_confidence=min_conf))

    def _detect(self, bgr, x0=0, y0=0, scale=1.0):
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[:, :, ::-1]))
        out = []
        for d in self.det.detect(img).detections:
            b = d.bounding_box
            out.append([x0 + b.origin_x * scale, y0 + b.origin_y * scale,
                        x0 + (b.origin_x + b.width) * scale, y0 + (b.origin_y + b.height) * scale, d.categories[0].score])
        return out

    def __call__(self, bgr):
        """Return (K,5) [x1,y1,x2,y2,score] in original frame pixels."""
        h, w = bgr.shape[:2]
        r = DET_SIDE / max(h, w)
        small = cv2.resize(bgr, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        sh, sw = small.shape[:2]
        dets = self._detect(small)
        ys = sorted(set(list(range(0, max(sh - TILE, 0) + 1, TILE_STRIDE)) + [max(sh - TILE, 0)]))
        xs = sorted(set(list(range(0, max(sw - TILE, 0) + 1, TILE_STRIDE)) + [max(sw - TILE, 0)]))
        for y in ys:
            for x in xs:
                tile = small[y:y + TILE, x:x + TILE]
                k = TILE / max(tile.shape[:2])
                up = cv2.resize(tile, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
                dets += self._detect(up, x, y, 1 / k)
        dets = np.array(dets, np.float32).reshape(-1, 5)
        # drop huge tile false positives cut by tile borders, merge duplicates
        dets = nms(dets)
        dets[:, :4] /= r
        return dets


def track_faces(dets_by_frame, n_frames, iou_thr=0.2, max_gap=12, min_frac=0.3):
    """Greedy IoU tracker over sparse detections. Returns list of (T,4) box arrays."""
    tracks = []  # each: dict(f=[frame idx], b=[box])
    for fi in sorted(dets_by_frame):
        dets = dets_by_frame[fi]
        active = [t for t in tracks if fi - t["f"][-1] <= max_gap]
        used = set()
        if len(active) and len(dets):
            m = iou(np.array([t["b"][-1] for t in active]), dets[:, :4])
            for _ in range(min(m.shape)):
                i, j = np.unravel_index(np.argmax(m), m.shape)
                if m[i, j] < iou_thr:
                    break
                active[i]["f"].append(fi); active[i]["b"].append(dets[j, :4]); used.add(j)
                m[i, :] = -1; m[:, j] = -1
        for j in range(len(dets)):
            if j not in used:
                tracks.append({"f": [fi], "b": [dets[j, :4]]})
    n_sampled = max(1, len(dets_by_frame))
    keep = [t for t in tracks if len(t["f"]) >= max(2, min_frac * n_sampled)]
    return [ft.interpolate_track(t["f"], t["b"], n_frames, smooth=3) for t in keep]


def track_geometry(frames, boxes, landmarker, stride=5):
    """Mouth/face geometry (T,7) for one face track: landmarks on upscaled face crops, box-ratio fallback."""
    idx, geoms = [], []
    for i in range(0, len(frames), stride):
        x1, y1, x2, y2 = boxes[i]
        bw, bh = x2 - x1, y2 - y1
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, 2.0 * max(bw, bh)
        crop_box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        faces = landmarker(ft.crop_square(frames[i], crop_box, 256))
        if faces:
            k = side / 256
            pts = min(faces, key=lambda p: np.abs(p.mean(0) - 128).sum()) * k + np.array(crop_box[:2], np.float32)
            g = ft.geometry_from_landmarks(pts)
        else:
            rel = FALLBACK_REL
            g = np.array([x1 + rel[0] * bw, y1 + rel[1] * bh, rel[2] * bw,
                          x1 + rel[3] * bw, y1 + rel[4] * bh, x1 + rel[5] * bw, y1 + rel[6] * bh], np.float32)
        idx.append(i); geoms.append(g)
    return ft.interpolate_track(idx, geoms, len(frames))


def motion_score(frames, geom):
    """Mouth motion minus upper-face motion (cancels camera/head jitter)."""
    def energy(crops):
        c = np.asarray(crops, np.float32)
        c = (c - c.mean()) / (c.std() + 1e-6)
        return float(np.abs(np.diff(c, axis=0)).mean()) if len(c) > 1 else 0.0
    mouth, upper = [], []
    for f, g in zip(frames, geom):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        mouth.append(ft.mouth_crop(gray, g))
        fx1, fy1, fx2, fy2 = g[3:7]
        fw, fh = fx2 - fx1, fy2 - fy1
        upper.append(ft.crop_square(gray, (fx1 + 0.2 * fw, fy1 + 0.2 * fh, fx2 - 0.2 * fw, fy1 + 0.55 * fh), 48))
    return energy(mouth) - energy(upper), np.stack(mouth)


_DET, _LM = None, None


def _models():
    global _DET, _LM
    if _DET is None:
        _DET, _LM = FaceDetector(), ft.Landmarker()
    return _DET, _LM


def box_geometry(boxes):
    """Mouth/face geometry (T,7) from detector boxes alone (calibrated ratios)."""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    bw, bh = boxes[:, 2] - x1, boxes[:, 3] - y1
    r = FALLBACK_REL
    return np.stack([x1 + r[0] * bw, y1 + r[1] * bh, r[2] * bw, x1 + r[3] * bw, y1 + r[4] * bh,
                     x1 + r[5] * bw, y1 + r[6] * bh], 1).astype(np.float32)


def localise(frames, det_stride=6, detector=None, landmarker=None, min_mouth_px=8.0):
    """frames: list of BGR (any size). Returns dict(tracks, chosen, mouth88 (T,88,88), mouth_px, geom).

    tracks: list of dict(boxes (T,4), geom (T,7), score, mouth88, mouth_px). chosen = -1 if no face found.
    Active speaker = track whose mouth moves most relative to its upper face.
    """
    if detector is None or landmarker is None:
        detector, landmarker = _models()
    T = len(frames)
    dets = {i: detector(frames[i]) for i in sorted(set(range(0, T, det_stride)) | {T - 1})}
    tracks = []
    for boxes in track_faces(dets, T, max_gap=3 * det_stride, min_frac=0.5):
        if np.median(ft.MOUTH_SCALE * box_geometry(boxes)[:, 2]) < min_mouth_px:
            continue  # implausibly tiny face: tile false positive
        geom = track_geometry(frames, boxes, landmarker)
        score, mouth = motion_score(frames, geom)
        tracks.append({"boxes": boxes, "geom": geom, "score": score, "mouth88": mouth,
                       "mouth_px": float(np.median(ft.MOUTH_SCALE * geom[:, 2]))})
    if not tracks:
        return {"tracks": [], "chosen": -1, "mouth88": None, "mouth_px": None, "geom": None}
    chosen = int(np.argmax([t["score"] for t in tracks]))
    t = tracks[chosen]
    return {"tracks": tracks, "chosen": chosen, "mouth88": t["mouth88"], "mouth_px": t["mouth_px"], "geom": t["geom"]}


def pad_clip(mouth88, length=CLIP_LEN):
    """Edge-pad / trim a mouth sequence to `length` frames. Returns (clip, real_length)."""
    n = min(len(mouth88), length)
    clip = mouth88[:length]
    if n < length:
        clip = np.concatenate([clip, np.repeat(clip[-1:], length - n, 0)], 0)
    return clip, n
