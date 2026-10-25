#!/usr/bin/env bash
# Download the frozen teacher and MediaPipe face models into $L2L_WORK/ckpt (default ../work/ckpt).
set -e
OUT="${L2L_WORK:-$(dirname "$0")/../../work}/ckpt"
mkdir -p "$OUT"
curl -L -o "$OUT/lipsync_expert.pth" https://huggingface.co/Nekochu/Wav2Lip/resolve/main/lipsync_expert.pth
echo "9b9936c721696446eeed353032cab242a8cf0eed8c46cde540366f6ae5493be5  $OUT/lipsync_expert.pth" | sha256sum -c
MP=https://storage.googleapis.com/mediapipe-models
curl -L -o "$OUT/face_landmarker.task" $MP/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o "$OUT/blaze_face_full_range.tflite" $MP/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite
curl -L -o "$OUT/blaze_face_short_range.tflite" $MP/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
