# lipsync2lipread

**Distilling a lip-sync generator's knowledge into a lip reader for small, distant, crowded faces.**
IE 643 (Deep Learning: Theory and Practice, IIT Bombay, 2026) course project. Team PieTorch: Pranav Sadh (23B3960).

The frozen **expert SyncNet of Wav2Lip** (the component that teaches the Wav2Lip generator what in-sync lips look like) is used as a teacher. Its 512-d audio-visual sync embeddings, computed on *clean, high-resolution* faces and *real audio*, supervise a lip-reading student that only ever sees *muted, degraded* mouth crops (16–32 px). No synthetic video is generated: transfer happens entirely in the teacher's latent space.

```
clean face (96 px) --> SyncNet face enc --> e_face  --\
real audio (mel)   --> SyncNet audio enc --> e_audio --+--> KD losses (training only)
                                                        |
degraded mouth (16-32 px) --> 3D-conv + ResNet-18 --> KD heads (h_face, h_audio)
                                     |
                                     +--> Conformer (4 x 256) --> CTC --> text
```

Transfer mechanisms (flags of `distill/train.py`):

| | mechanism | flag |
|---|---|---|
| M1 | student stem initialised from SyncNet's first conv: (32,15,7,7) = 5 frames x 3 colours -> colour-averaged 3D kernel (32,5,7,7) | `--m1` |
| M2 | visual-feature KD: `1 - cos(h_face, e_face)` on speech windows | `--kd m2` |
| M3 | cross-modal KD: `1 - cos(h_audio, e_audio)`, weighted by teacher sync confidence, speech windows only | `--kd m3` |
| NCE | sync-contrastive KD: the student's window must pick its own audio embedding among the clip's other windows (InfoNCE, tau 0.1) | `--kd m2m3nce` |
| M4 | cross-resolution: student input degraded (16/24/32 px, blur, JPEG, jitter), teacher targets always from clean frames | `--degrade` |

## Setup

Tested on Windows 11, Python 3.11, RTX 3070 Laptop (8 GB), CUDA 13.0.

```bash
pip install -r requirements.txt          # torch wheels: use the CUDA index for your system
```

Large files live outside the repo in a work directory (default `../work`, override with `L2L_WORK`):

```
work/ckpt/lipsync_expert.pth        # Wav2Lip expert SyncNet, https://huggingface.co/Nekochu/Wav2Lip
                                    # SHA-256 9b9936c721696446eeed353032cab242a8cf0eed8c46cde540366f6ae5493be5
work/ckpt/face_landmarker.task      # MediaPipe face landmarker
work/ckpt/blaze_face_full_range.tflite, blaze_face_short_range.tflite
```
Download links are in `scripts/get_models.sh`.

## Data (GRID corpus, Zenodo record 3625687)

```bash
bash data/download_grid.sh ../work/raw/a s1 s2 s3 ...       # speaker zips + alignments.zip
python -m data.preprocess --watch --spk s1 s2 s3 ... s34     # extract, track, crop, mel, teacher targets
```
Notes: the Zenodo `alignments/sN` folders are permuted relative to the `sN.zip` videos (e.g. videos s5 <-> alignments s6). `preprocess.py` matches them by file-name overlap (every speaker matches 1000/1000). Speaker s21 has no video.

Output per speaker in `work/proc/<spk>/`: 88x88 grayscale mouth crops, face/mouth geometry, speech mask from the alignments, and precomputed teacher embeddings (`eface`, `eaud`, `tconf`). See `INTERFACES.md` for exact shapes.

Splits (speaker-independent, LipNet's unseen-speaker test split): **test** s1, s2, s20, s22; **val** s13 (fixed 300-clip subset); **train** all other speakers with video.

GRID-Crowd (muted crowd composites with 16/24/32 px target mouths and 2–5 distractor faces):
```bash
python -m data.crowd --n 300
```

## Training

```bash
pwsh scripts/run_queue.ps1      # full experiment queue, resumable
# single run, e.g. distilled fine-tune from the clean baseline:
python distill/train.py --name R3_ft_m2m3 --init ../work/runs/R1_base_clean/best.pt --degrade --kd m2m3 --max_minutes 25 --bs 16 --accum 1
```
Low-label experiment (teacher targets need audio but no transcripts, so KD can use unlabeled video):
```bash
pwsh scripts/run_lowlabel.ps1   # LL_noKD: CTC on s3-s5 only; LL_m2m3: + M2/M3 KD on all 28 training speakers
python distill/train.py --name LL_m2m3 --labeled_spk s3 s4 s5 --kd m2m3 --degrade --max_minutes 60
```
With `--labeled_spk`, only those speakers' transcripts are used; their clips are oversampled to about half of each epoch, and the other speakers' clips contribute only the KD losses.

AdamW (lr 3e-4 scratch / 1.5e-4 fine-tune, wd 1e-2), linear warm-up then cosine decay to the wall-clock budget, fp16 AMP, grad-clip 5, batch 16. Train-time augmentation (all runs): +-4 px shift, horizontal flip, 2 time masks of <= 10 frames.

## Evaluation

```bash
python eval/evaluate.py --ckpt ../work/runs/R3_ft_m2m3/best.pt          # clean, 32/24/16 px; raw + grammar/TTA
python -m eval.eval_crowd --ckpt ../work/runs/R2_ft_noKD/best.pt ../work/runs/R3_ft_m2m3/best.pt
python -m eval.figures && python -m eval.figures --decode grammar
python -m eval.qualitative --baseline R2_ft_noKD --distilled R3_ft_m2m3
python -m teacher.sanity_sweep --spk s1 s2 s20 s22                       # teacher transfer check
```
Metrics: WER, CER, letter- and digit-slot accuracy (the only unpredictable GRID slots), 95% bootstrap CIs, and a sync score (cosine between the student's `h_audio` and the teacher's audio embedding). Two decoders are always reported: **raw** greedy CTC, and **grammar+TTA** (GRID-grammar-constrained CTC beam search over the 6 word slots, with horizontal-flip test-time averaging). Model selection uses raw greedy WER only.

## Interface

```bash
python -m app.gradio_app --baseline ../work/runs/R2_ft_noKD/best.pt --distilled ../work/runs/R3_ft_m2m3/best.pt [--share]
```
Upload a muted video, record one with the webcam, or pick a demo crowd clip. The app detects and tracks every face, picks the active speaker by mouth motion, and shows the ROI overlay, the mouth crops the model saw (with their pixel size), baseline vs distilled transcripts with per-word confidence, and WER if a reference transcript is given. The vocabulary is the GRID grammar, e.g. "bin blue at f two now".

## Repository map

| path | content |
|---|---|
| `data/` | download, preprocessing, face tracking, degradation, dataset, splits, GRID-Crowd builder, crowd localisation |
| `teacher/` | frozen SyncNet wrapper, target precomputation, transfer sanity sweep |
| `student/model.py` | 3D-conv + ResNet-18 frontend, Conformer / BiGRU backend, CTC head, KD heads, M1 init |
| `distill/` | losses (CTC, cosine KD with speech mask and confidence weighting, InfoNCE) and training loop |
| `eval/` | metrics, evaluation, grammar-constrained decoding, crowd evaluation, figures, qualitative panels |
| `app/` | Gradio interface and shared inference code |
| `third_party/wav2lip/` | vendored SyncNet model and audio code from Rudrabha/Wav2Lip |
| `LOG.md` | dated experiment log; every reported number traces here |

## Acknowledgements

Wav2Lip and its expert SyncNet (Prajwal et al., 2020; github.com/Rudrabha/Wav2Lip), GRID corpus (Cooke et al., 2006), MediaPipe, Conformer (Gulati et al., 2020), LipNet (Assael et al., 2016), Auto-AVSR (Ma et al., 2023).
