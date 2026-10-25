# Interface contract (internal, for parallel development)

Repo root: `E:\College\IE_Project\lipsync2lipread` (run everything from here; `python -m pkg.module` or scripts that add repo root to sys.path).
Work dir (outside git): `E:\College\IE_Project\work` -> referred to as `WORK`. Override with env `L2L_WORK`.

## Raw data
- `WORK/raw/videos/<spk>/<id>.mpg` — GRID clips, 360x288 MPEG-1, 25 fps, 75 frames, audio mp2 44.1 kHz stereo.
- `WORK/raw/align/<spk>/<id>.align` — lines `start end word`; frame index = units / 1000. Words `sil`, `sp` are non-speech.
- Speakers: test `s1 s2 s20 s22`; val `s13`; train `s3..s12` (+ more if downloaded; see `data/splits.py`).

## Preprocessed data: `WORK/proc/<spk>/`  (written by `data/preprocess.py` + `teacher/precompute.py`)
| file | dtype / shape | meaning |
|---|---|---|
| `mouth.npy` | uint8 (N, 75, 88, 88) | grayscale mouth ROI, clean (student input before degradation) |
| `geom.npy` | float32 (N, 75, 7) | per-frame `[mouth_cx, mouth_cy, mouth_w, face_x1, face_y1, face_x2, face_y2]` in original 360x288 pixels |
| `nframes.npy` | int16 (N,) | real frame count (<=75; shorter clips edge-padded) |
| `smask.npy` | bool (N, 71) | window t (frames t..t+4) overlaps speech per alignment |
| `eface.npy` | float16 (N, 71, 512) | SyncNet face embedding of clean window t (L2-normalised) |
| `eaud.npy` | float16 (N, 71, 512) | SyncNet audio embedding of window t (mel cols int(3.2t)..+16) |
| `tconf.npy` | float16 (N, 71) | cos(eface, eaud): teacher sync confidence |
| `meta.json` | `{"ids": [...], "texts": [...]}` | texts lowercase words, sil/sp removed, single spaces |

Windows: `W = 71 = 75 - 4`. Student window feature t must cover frames t..t+4.

## Shared modules (already written)
- `data/vocab.py`: `CHARS`, `BLANK=0`, `NUM_CLASSES=28`, `encode(text)`, `decode(ids)`, `GRID_SLOTS`.
- `data/facetrack.py`: `Landmarker()`, `track_video(frames, lm) -> geom (T,7) | None`, `mouth_crop(bgr, geom_t) -> (88,88) uint8`, `teacher_face_crop(bgr, geom_t) -> (48,96,3)`, `mouth_box(geom_t)`, `geometry_from_landmarks(pts)`, `crop_square(img, box, size)`.
- Teacher: `work/ckpt/lipsync_expert.pth` (`state_dict` keys `face_encoder.0.conv_block.0.weight` = (32,15,7,7); the 15 channels are 5 frames x 3 BGR channels, frame-major). Vendored model: `third_party/wav2lip/syncnet.py: SyncNet_color`.

## Student input normalisation
`x = (uint8 / 255 - 0.421) / 0.165`, shape `(B, 1, T, 88, 88)` float.

## Degradation (`data/degrade.py`, Fork A)
`degrade(frames_uint8 (T,88,88), res:int, rng:np.random.Generator) -> (T,88,88) uint8`: INTER_AREA down to res x res, Gaussian blur sigma U[0,1.2] (applied at low res), per-clip JPEG quality U[20,60] per frame, bilinear upsample to 88, brightness/contrast jitter (alpha U[0.8,1.2], beta U[-20,20]). `res=88` -> return clean. Test-time: `rng = np.random.default_rng(hash((spk, idx, res)) & 0xffffffff)` -> deterministic.

## Student (`student/model.py`, Fork A)
- `LipReader(backend="conformer"|"bigru", d_model=256, n_layers=4)`.
- `forward(x (B,1,T,88,88), lengths (B,)) -> dict(logits=(B,T,28) raw, feat=(B,T,512) frontend feats, h_face=(B,T-4,512) L2-normed, h_audio=(B,T-4,512) L2-normed)`.
- `init_stem_from_syncnet(model, syncnet_ckpt_path)` (M1).
- `save_ckpt(path, model, cfg, **extra)` / `load_student(path, device) -> (model, cfg)`; cfg is a plain dict (backend etc.).

## Metrics (`eval/metrics.py`, Fork A)
`ctc_greedy(logits (B,T,C), lengths) -> list[str]`; `ctc_greedy_with_conf(logits_1clip) -> (text, [(word, conf)])`; `wer(refs, hyps)`, `cer(refs, hyps)`, `slot_acc(refs, hyps) -> {"letter":..,"digit":..}`, `bootstrap_ci(per_clip_errors, per_clip_counts)`.

## Training (`distill/train.py`, Fork A)
CLI flags: `--name --kd {none,m2,m3,m2m3,nce,m2m3nce} --m1 --degrade --init <ckpt> --epochs --lr --bs --accum --seed --backend --lambda_v --lambda_a --lambda_nce --train_spk ... --max_minutes`. Output `WORK/runs/<name>/{last.pt,best.pt,log.csv,config.json}`. Resume from `last.pt` automatically.

## Crowd set (`data/crowd.py`, Fork B)
`WORK/crowd/videos/<cid>.mp4` + `WORK/crowd/meta.json`: list of `{cid, spk, idx, text, target_px, frame_size, target_boxes:(75,4) face box, target_mouth:(75,4), distractors:[{spk,idx,boxes}]}`.

## Inference (`app/infer.py`, Fork B)
`read_video(path, fps=25) -> list[bgr]`; `localise(frames) -> {tracks, chosen, mouth88 (T,88,88), mouth_px}`; `Reader(ckpt).read(mouth88) -> {text, words:[(w,conf)], logits}`.
