# Experiment log

Every number in the report must trace to an entry here. Hardware: RTX 3070 Laptop (8 GB), i7-12650H, 64 GB RAM, Windows 11, torch 2.11+cu130.

## 2026-10-25

**01:20 Data.** GRID from Zenodo record 3625687 (s*.zip 350–490 MB each, alignments.zip 20 MB). Clips: 360x288 MPEG-1, 25 fps, 75 frames, mp2 audio 44.1 kHz stereo (embedded, so audio_25k.zip not needed).

**01:25 Teacher.** `lipsync_expert.pth` from huggingface.co/Nekochu/Wav2Lip, SHA-256 9b9936c7...be5 verified. Wav2Lip SyncNet_color loads; face (B,15,48,96) and audio (B,1,80,16) -> 512-d L2-normalised.

**01:28 Face tracking.** MediaPipe FaceLandmarker (mediapipe 1.1.1), landmarks every 5th frame, linear interpolation + 5-frame moving average. Mouth ROI = square of side 1.8 x mouth-corner distance -> 88x88 gray. Teacher crop = landmark bbox (+5% chin) -> 96x96 -> lower half 48x96 BGR. 0.12 s/clip tracking.

**01:31 Preprocess s1.** 1000/1000 clips, 232 s with 8 workers (+ teacher precompute ~60 s on GPU).

**01:33 Teacher on GRID s1 (all 1000 clips, speech windows only, 0.4 s offset negatives).** In-sync cos 0.552 vs off-sync 0.175; sync accuracy 85.8% (chance 50%). Teacher confidence cos(e_face,e_audio) is *higher* in non-speech windows (0.760) than speech windows (0.551) -> silence is trivially "in sync"; motivates the speech mask in KD.

**01:40 Teacher resolution sweep (s1, 150 clips, 6056 speech windows; face crop shrunk to r px then upsampled).**

| face px | 96 | 64 | 48 | 32 | 24 | 16 |
|---|---|---|---|---|---|---|
| sync acc | 0.867 | 0.871 | 0.877 | 0.882 | 0.876 | 0.848 |
| cos(hi, lo) | 1.000 | 0.990 | 0.978 | 0.948 | 0.856 | 0.719 |

**01:42 Data issue, resolved 01:55.**
- Zenodo s5.zip video names do not match alignments/s5 (0/1000 overlap).
- The full check showed the alignment folders are *permuted* relative to the video zips: videos s5↔align s6, s6↔s5, s10↔s13, s11↔s10, s12↔s11, s13↔s12, s14↔s15, s15↔s14, s18↔s19, s19↔s18, s20↔s21, s22↔s23. Every video folder matches exactly one alignment folder 1000/1000.
- Final count over all speakers: 19 of 33 video speakers needed remapping. The remapped pairs were s5/6, s10→13, s11→10, s12→11, s13→12, s14/15, s18/19, s20→21, s22→23, s23→22, s26/27, s28/29 and s32/33.
- **04:30 Preprocessing complete:** 33 speakers, 32,878 clips (s8: 895/1000, s15: 983/1000, all others 1000; failures are undecodable or untrackable clips).
- Content check: in all 34,000 .align files the words match the sentence encoded in the file name (GRID writes digit 0 as 'z'). So the remap is safe.
- `preprocess.align_dir_for` picks the alignment folder by filename overlap.
- Decision (approach B): use all GRID speakers with video (s21 has none): ~27 train, s13 val, s1/s2/s20/s22 test.

Sync accuracy holds ~85–88% down to 16 px but the embedding drifts -> teacher targets computed on clean frames only (M4). Raw: `work/runs/teacher_sanity.json`. (To be re-run on all test speakers for the final figure.)

**05:14 R1_base_clean (scratch, clean training, Conformer, bs 16, 60 min, 12,763 steps, ~62 clips/s; 28 train speakers).** Best val (s13, 300 clips) raw WER 13.7% clean / 42.7% @24px.
Test s1 s2 s20 s22 (4,000 clips):

| cond | raw WER [95% CI] | CER | letter | digit | grammar+TTA WER | letter | digit |
|---|---|---|---|---|---|---|---|
| clean | 17.22 [16.74,17.75] | 7.94 | 0.541 | 0.880 | 15.38 | 0.538 | 0.916 |
| 32 px | 33.85 [33.17,34.58] | 18.64 | 0.364 | 0.681 | 29.21 | 0.368 | 0.758 |
| 24 px | 44.06 [43.32,44.77] | 25.20 | 0.276 | 0.528 | 37.99 | 0.270 | 0.630 |
| 16 px | 62.45 [61.65,63.21] | 38.32 | 0.149 | 0.310 | 54.58 | 0.143 | 0.426 |

Sync score (cos h_audio vs e_audio) ~0.01 (KD heads untrained in R1, as expected).
GRID-Crowd (300 clips): localisation face recall 1.00, target recall 1.00, speaker selection 87.7%. R1 WER oracle boxes 64.8% (grammar 56.3%), detected 65.4% (grammar 57.6%); by size 16/24/32 px detected raw 75.0/64.5/56.8.

**06:27 R2_ft_noKD vs R3_ft_m2m3 (both fine-tuned from R1 with degradation, 25 min, ~5,420 steps, identical schedule; the only difference is the KD term).** Test s1 s2 s20 s22, 4,000 clips:

| cond | R2 raw WER [95% CI] | R3 raw WER [95% CI] | R2 grammar+TTA | R3 grammar+TTA | R2 sync | R3 sync |
|---|---|---|---|---|---|---|
| clean | 16.44 [15.99,16.92] | 16.78 [16.32,17.30] | 14.53 | 14.65 | 0.008 | 0.710 |
| 32 px | 18.58 [18.12,19.05] | 18.83 [18.33,19.32] | 16.45 | 16.26 | 0.009 | 0.701 |
| 24 px | 20.32 [19.84,20.81] | 20.24 [19.74,20.75] | 17.87 | 17.65 | 0.010 | 0.693 |
| 16 px | 24.74 [24.20,25.32] | 24.62 [24.05,25.22] | 21.38 | 21.32 | 0.011 | 0.669 |

CER clean/32/24/16: R2 7.93/9.01/10.18/13.03, R3 7.87/9.02/9.99/12.99. Letter acc 16 px: R2 0.476, R3 0.478; digit 0.737 / 0.731.
- **Degradation-aware fine-tuning (M4 data side) is the large effect:** 16 px WER 62.5 -> 24.7 (R1 -> R2).
- **M2+M3 KD gives no significant WER change** with 28 labeled speakers (all differences < 0.4 pt, CIs overlap), although the student does learn the teacher space (sync score 0.67-0.71 vs ~0.01; final KD losses m2 0.14, m3 0.25 in 1-cos units).
- Hypothesis: with 28k transcribed clips CTC already supplies the information the teacher offers. KD needs no transcripts, so it should matter when labels are scarce -> added low-label experiment (below).

**06:35 Low-label experiment added (`scripts/run_lowlabel.ps1`, starts after the main queue).** New flag `train.py --labeled_spk`: only s3, s4, s5 (3,000 clips) keep transcripts; their clips are oversampled to ~50% of each epoch (24,000 of 48,878 samples) and the other 25 speakers contribute only the KD losses (per-clip CTC weight 0; with all weights 1 the loss equals the old mean exactly, checked). LL_noKD = CTC on s3-s5 only; LL_m2m3 = same + M2+M3 KD on all 28 speakers. Both from scratch, degraded, 60 min. Note the KD run gets ~half as many labeled samples per minute (conservative for KD).

**08:20 KD ablations (fine-tuned from R1 with degradation, 25 min, same schedule as R2/R3).** Raw greedy WER % [95% CI], test s1 s2 s20 s22:

| run | KD | clean | 32 px | 24 px | 16 px | 16 px grammar+TTA | 16 px letter / digit | sync | steps | clips/s |
|---|---|---|---|---|---|---|---|---|---|---|
| R2 | none | 16.44 | 18.58 | 20.32 | 24.74 [24.20,25.32] | 21.38 | 0.476 / 0.737 | 0.011 | 5416 | 58.9 |
| R3 | M2+M3 | 16.78 | 18.83 | 20.24 | 24.62 [24.05,25.22] | 21.32 | 0.478 / 0.731 | 0.669 | 5422 | 58.9 |
| R5 | M3 | 16.99 | 18.74 | 20.72 | 24.95 [24.38,25.50] | 21.36 | 0.463 / 0.743 | 0.669 | 5340 | 58.5 |
| R6 | M2 | 19.72 | 21.75 | 23.50 | 28.55 [27.98,29.16] | 24.61 | 0.360 / 0.740 | -0.002 | 5494 | 58.5 |
| R8 | M2+M3+NCE | 16.85 | 19.13 | 20.91 | 26.47 [25.86,27.04] | 22.29 | 0.471 / 0.709 | 0.665 | 5084 | 50.5 |

- M3 (audio-sync KD) alone and M2+M3: no significant change vs R2.
- **M2 (face-embedding KD) alone hurts significantly** (+3.8 pt at 16 px, CIs disjoint; letter accuracy 0.476 -> 0.360). e_face encodes the clean colour lower face (jaw, cheeks, identity), which a 16 px grey mouth crop cannot predict; regressing it competes with CTC for the frontend. Adding M3 (R3) removes the damage.
- R8 (InfoNCE) is worse at 16 px (+1.7 pt) but its per-clip NCE loop cut throughput to 50.5 clips/s, so it got 6% fewer steps in the same 25 min; the NCE loss stayed high (1.93 nats vs log(~50) ~ 3.9 chance), i.e. the student only partly learns the sync ranking.

**09:40-10:45 Interface QA, bugs found and fixed, crowd domain gap.**
- App (`app/gradio_app.py`), tested in a headless browser at 1440x1000 and 1280x720, light and dark themes, and on edge cases. Fixed:
  1. **crash on machines without a usable GPU.** `torch.cuda.is_available()` was True with `CUDA_VISIBLE_DEVICES=""`, so the CUDA checkpoint load failed. Now also checks `device_count()`, plus a `--device` flag.
  2. The demo dropdown showed a clip as selected but loaded none. It now preloads the first demo and its reference text, and orders demos easy to hard.
  3. Per-word confidence text was unreadable (grey on blue); now theme-aware.
  4. The overlay label was unreadable. It is now a filled tag ("speaker - mouth ROI N px") above a bold box on the chosen face.
  5. **crash on clips too short for a GRID sentence** (8 frames). The grammar forced alignment returned no path. It now falls back to greedy + lexicon snap (`eval/decode.py`); outputs for normal-length clips are unchanged.
  6. Warning shown for clips < 2 s. Verified: no-face video, 6 s 30 fps clip (2 chunks), portrait 1080x1920, webcam-style 1280x720 VP8 .webm.
- **`facetrack.crop_square` aliasing bug:** `cv2.warpAffine(..., INTER_AREA)` is silently bilinear (verified: identical output to INTER_LINEAR). Downscaled crops were therefore aliased. Fixed by cutting at native resolution, then `cv2.resize(INTER_AREA)` (matches resize exactly).
  - Affected the GRID-Crowd composites (~4x downscale of the face), so the crowd test set was rebuilt with the same seed: identical scenes, sentences, distractors and boxes, only the rendering changed.
  - Also affected the teacher face crops (~1.5x downscale; teacher targets were computed before the fix and not recomputed) and app crops of high-resolution video. Training mouth crops are upscaled and unaffected.
  - Old crowd results are kept as `runs/R{1,2,3}/eval_crowd_aliased_v1.json`.
- **Crowd results with the old (aliased) set:** R2 detected/oracle WER 57.4 / 55.4; R3 57.5 / 56.4. R2 oracle by size, 16/24/32 px: 66.3 / 54.7 / 45.2, against 24.7 / 20.3 / 18.6 on synthetic degradation.
- **Diagnosis of the crowd gap (R2, oracle boxes, after the fix):**
  - 250 kbps: 55.2% WER (16/24/32 px: 64.2 / 57.5 / 44.0).
  - Same scenes at 8 Mbps (`crowd_hq`): **18.2%** (19.2 / 17.7 / 17.7).
  - => the gap is the **H.264 codec at low bitrate**, not resolution, compositing or localisation. On the s13 dev crowd set, the 250 kbps mouth ROI vs 8 Mbps has PSNR ~35 dB, but frame-to-frame mouth motion is attenuated to 0.63 / 0.69 / 0.70 (range 0.51-0.82) at 16 / 24 / 32 px: P-frame smearing of small motions, which per-frame JPEG never produces.
- **Codec-aware degradation** (`degrade.h264`, PyAV, ~10 ms/clip; `train.py --codec p`):
  - With probability p, H.264 round-trips the small clip at QP ~ U{28..35}; otherwise JPEG as before.
  - QP range calibrated on the s13 dev crowd set (synthetic QP 28 -> motion 0.81-0.83, QP 32 -> 0.64-0.72, QP 36 -> 0.43-0.58).
  - Test-time degradation is bit-identical to before (checked on 44 clip/resolution pairs).
- **Speaker selection** on test crowd: s1 0.84, s2 0.93, s20 0.77, s22 0.96; 16/24/32 px: 0.81 / 0.92 / 0.90. Dev crowd sets (s13 target): 0.98 (s13 distractors), 0.97 (training-speaker distractors). Not tuned on test.
- **Load confound:** R2s1 got 4,761 steps in its 25 min vs 5,416 for R2, because my CPU work (crowd builds, localisation, app tests) slowed its data loading. **Remaining runs use step budgets** (`--max_minutes 0 --max_steps N`).
- New queue `scripts/run_followup.ps1` (replaces run_lowlabel.ps1; starts after the main queue):
  - R9_ft_codec (R2 + `--codec 0.5`, 5,416 steps);
  - R10_ft_codec_m2m3 (R9 + KD);
  - LL_noKD and LL_m2m3 (12,763 steps each);
  - then crowd evaluation of all runs on the rebuilt set.

**10:51 R4 (M1, scratch, clean, 60 min) and seed reruns.** Raw WER % [95% CI], test s1 s2 s20 s22:

| run | steps | clean | 32 px | 24 px | 16 px | 16 px grammar+TTA |
|---|---|---|---|---|---|---|
| R1 | 12,763 | 17.22 [16.74,17.75] | 33.85 | 44.06 | 62.45 | 54.58 |
| R4 (M1) | **13,913** | 16.23 [15.78,16.72] | 35.17 | 46.44 | 65.91 | 56.21 |
| R2 | 5,416 | 16.44 | 18.58 | 20.32 | 24.74 [24.20,25.32] | 21.38 |
| R2s1 | 4,761 | 16.94 | 18.80 | 20.56 | 25.42 [24.86,26.02] | 22.03 |
| R3 | 5,422 | 16.78 | 18.83 | 20.24 | 24.62 [24.05,25.22] | 21.32 |
| R3s1 | 5,192 | 17.71 | 19.44 | 20.96 | 25.40 [24.85,25.98] | 21.73 |

- **R4's clean gain is confounded:** it got 9% more steps than R1 in the same 60 min. Validation WER at equal steps interleaves; R1/R4 at 1k steps: 62.9 / 57.9; 5k: 22.6 / 19.8; 10k: 13.9 / 13.2; 12k: 13.7 / 13.8. That is a possible early-training speed-up only. Step-matched rerun R4b (12,763 steps) is queued in `scripts/run_m1_matched.ps1` after the follow-up queue.
- **Seeds:** KD vs no-KD stays tied in both seeds (16 px: 24.7 vs 24.6; 25.4 vs 25.4). The seed-1 runs got fewer steps (load from my concurrent CPU work), which is a plausible reason they are about 0.7 pt worse.
