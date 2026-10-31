#!/bin/bash
# Place exact corresponding source and build recipe alongside the signed app.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${1:-"$ROOT/.build-macos/lgpl-pyav"}
DEST=${2:-"$ROOT/.build-macos/LGPL-corresponding-source.tar.gz"}
STAGE="$WORK/source-offer"
rm -rf -- "$STAGE"
mkdir -p "$STAGE"
cp "$WORK/sources/ffmpeg-7.1.1.tar.xz" "$WORK/sources/av-14.2.0.tar.gz" \
  "$WORK/configure-flags.txt" "$ROOT/scripts/build_lgpl_pyav.sh" "$STAGE/"
cp "$WORK/ffmpeg-7.1.1/COPYING.LGPLv2.1" "$STAGE/LGPL-2.1.txt"
cp "$WORK/av-14.2.0/LICENSE.txt" "$STAGE/PyAV-BSD-LICENSE.txt"
: > "$STAGE/changes.diff"
printf '%s\n' 'FFmpeg 7.1.1 and PyAV 14.2.0 are unmodified upstream source releases.' \
  'Run build_lgpl_pyav.sh on macOS arm64 to rebuild the shared libraries and PyAV wheel.' \
  'configure-flags.txt records the flags used for the distributed FFmpeg libraries.' \
  > "$STAGE/README.txt"
tar -czf "$DEST" -C "$STAGE" .
shasum -a 256 "$DEST"
