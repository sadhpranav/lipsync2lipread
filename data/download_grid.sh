#!/usr/bin/env bash
# Download GRID corpus speakers + alignments from Zenodo record 3625687.
# Usage: bash download_grid.sh <out_dir> s1 s2 ...
set -u
OUT="$1"; shift
mkdir -p "$OUT"
BASE="https://zenodo.org/records/3625687/files"
for f in alignments "$@"; do
  if [ -f "$OUT/$f.zip.done" ]; then echo "skip $f"; continue; fi
  echo "[$(date +%T)] downloading $f"
  curl -L --retry 5 --retry-delay 5 -C - -o "$OUT/$f.zip" "$BASE/$f.zip?download=1" && touch "$OUT/$f.zip.done"
  echo "[$(date +%T)] done $f ($(du -h "$OUT/$f.zip" | cut -f1))"
done
