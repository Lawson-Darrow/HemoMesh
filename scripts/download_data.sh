#!/usr/bin/env bash
# Download the Suk et al. coronary-mesh dataset (CC-BY 4.0) into the layout
# hemomesh expects: vessel-datasets/stead/{single,bifurcating}/raw/database.hdf5
#
# The source (SURFdrive) serves a folder as an on-the-fly zip: no resume, no
# per-file access. It also rate-limits repeat pullers. So this script is
# SPEED-GATED: it samples the endpoint and only commits to the full pull when
# throughput is high enough to finish in a reasonable window; otherwise it waits
# and retries. Safe to run in the background or in a Colab/Linux runtime.
#
# Usage: bash scripts/download_data.sh [DEST_DIR]
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
DAV="https://surfdrive.surf.nl/public.php/dav/files/prElf2HkN0x3JOY?accept=zip"
MIN_MBPS="${MIN_MBPS:-0.8}"     # commit to full download at/above this rate
SAMPLE_SECS=15                  # speed-probe duration
WAIT_SECS="${WAIT_SECS:-1800}"  # sleep between retries when throttled (30 min)
MAX_ATTEMPTS="${MAX_ATTEMPTS:-16}"
TMP="$DEST/.dl"

single="$DEST/vessel-datasets/stead/single/raw/database.hdf5"
bifur="$DEST/vessel-datasets/stead/bifurcating/raw/database.hdf5"

if [ -f "$single" ] && [ -f "$bifur" ]; then
  echo "[done] dataset already present under $DEST/vessel-datasets/stead/"
  exit 0
fi

mkdir -p "$TMP"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[$(date +%T)] attempt $attempt/$MAX_ATTEMPTS — probing endpoint speed (${SAMPLE_SECS}s)..."
  rm -f "$TMP/spd.part"
  curl -s --max-time "$SAMPLE_SECS" "$DAV" -o "$TMP/spd.part" 2>/dev/null || true
  bytes=$(stat -f%z "$TMP/spd.part" 2>/dev/null || stat -c%s "$TMP/spd.part" 2>/dev/null || echo 0)
  mbps=$(echo "scale=3; $bytes/1048576/$SAMPLE_SECS" | bc 2>/dev/null || echo 0)
  rm -f "$TMP/spd.part"
  echo "[$(date +%T)]   ~${mbps} MB/s"
  if [ "$(echo "$mbps >= $MIN_MBPS" | bc 2>/dev/null || echo 0)" = "1" ]; then
    echo "[$(date +%T)] throughput OK — downloading full 2.5 GB zip..."
    if curl -L --fail --max-time 7200 "$DAV" -o "$TMP/suk.zip" \
        && unzip -tqq "$TMP/suk.zip"; then
      echo "[$(date +%T)] extracting into $DEST ..."
      unzip -oq "$TMP/suk.zip" -d "$DEST"
      rm -rf "$TMP"
      echo "[$(date +%T)] verifying md5 sums..."
      for sub in single bifurcating; do
        d="$DEST/vessel-datasets/stead/$sub/raw"
        [ -f "$d/md5_sum" ] || continue
        got=$( (cd "$d" && { md5 -q database.hdf5 2>/dev/null || md5sum database.hdf5 2>/dev/null | awk '{print $1}'; }) )
        exp=$(awk '{print $1}' "$d/md5_sum")
        [ "$got" = "$exp" ] && echo "  [$sub] md5 OK ($got)" || echo "  [$sub] MD5 MISMATCH got=$got exp=$exp"
      done
      echo "DONE — dataset at $DEST/vessel-datasets/stead/"
      exit 0
    else
      echo "[$(date +%T)] download/verify failed mid-pull (likely throttle drop); will retry."
      rm -f "$TMP/suk.zip"
    fi
  fi
  echo "[$(date +%T)] throttled — sleeping ${WAIT_SECS}s before retry."
  sleep "$WAIT_SECS"
done
echo "GAVE_UP after $MAX_ATTEMPTS attempts — SURFdrive still throttled. Try again later or use a Drive copy/mirror."
exit 1
