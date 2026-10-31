#!/bin/bash
# Place exact corresponding source and build recipe alongside the signed app.
set -euo pipefail
export COPYFILE_DISABLE=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${1:-"$ROOT/.build-macos/lgpl-pyav"}
DEST=${2:-"$ROOT/.build-macos/LGPL-corresponding-source.tar.gz"}
STAGE="$WORK/source-offer"
rm -rf -- "$STAGE"
mkdir -p "$STAGE/scripts" "$STAGE/packaging" "$STAGE/.build-macos/lgpl-pyav/sources"
cp "$WORK/sources/ffmpeg-7.1.1.tar.xz" "$WORK/sources/av-14.2.0.tar.gz" \
  "$STAGE/.build-macos/lgpl-pyav/sources/"
cp "$WORK/configure-flags.txt" "$STAGE/"
cp "$ROOT/scripts/build_lgpl_pyav.sh" "$STAGE/scripts/"
cp "$ROOT/packaging/requirements-lgpl-build.txt" "$STAGE/packaging/"
cp "$WORK/ffmpeg-7.1.1/COPYING.LGPLv2.1" "$STAGE/LGPL-2.1.txt"
cp "$WORK/av-14.2.0/LICENSE.txt" "$STAGE/PyAV-BSD-LICENSE.txt"
: > "$STAGE/changes.diff"
printf '%s\n' 'FFmpeg 7.1.1 and PyAV 14.2.0 are unmodified upstream source releases.' \
  'Extract this archive and run scripts/build_lgpl_pyav.sh .build-macos/lgpl-pyav /path/to/python3.12.11 on macOS arm64 with uv installed.' \
  'The pinned source tarballs are included under .build-macos/lgpl-pyav/sources, so the script does not download FFmpeg or PyAV source again.' \
  'configure-flags.txt records the flags used for the distributed FFmpeg libraries.' \
  > "$STAGE/README.txt"
# Stable bytes matter: the notice embeds this archive's SHA-256, including on retries.
find "$STAGE" -type f -exec touch -t 202001010000.00 {} +
FILELIST="$WORK/source-offer-files.txt"
(cd "$STAGE" && find . -type f | LC_ALL=C sort) > "$FILELIST"
tar -c --format=ustar --no-mac-metadata --no-xattrs \
  --uid 0 --gid 0 --uname '' --gname '' \
  -f - -C "$STAGE" -T "$FILELIST" | gzip -n > "$DEST.tmp"
mv "$DEST.tmp" "$DEST"
shasum -a 256 "$DEST"
