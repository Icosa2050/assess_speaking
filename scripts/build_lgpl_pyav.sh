#!/bin/bash
# Build the pinned PyAV release against a deliberately small LGPL FFmpeg.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${1:-"$ROOT/.build-macos/lgpl-pyav"}
PYTHON=${2:-"$ROOT/.build-macos/venv/bin/python"}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
PYTHON=$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")
[[ "$WORK" == "$ROOT/.build-macos/"* ]] || { echo 'Build output must be under .build-macos' >&2; exit 1; }
FFMPEG_VERSION=7.1.1
PYAV_VERSION=14.2.0
FFMPEG_SHA256=733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1
PYAV_SHA256=132b5d52ca262b97b0356e8f48cbbe54d0ac232107a722ab8cc8c0c19eafa17b

test "$(uname -s)-$(uname -m)" = Darwin-arm64 || { echo 'Requires macOS arm64' >&2; exit 1; }
export MACOSX_DEPLOYMENT_TARGET=14.0
mkdir -p "$WORK/sources" "$WORK/wheels"
fetch() {
  local url=$1 dest=$2 sha=$3
  if ! test -f "$dest"; then
    curl --fail --location --retry 3 --proto '=https' --tlsv1.2 "$url" --output "$dest"
  fi
  echo "$sha  $dest" | shasum -a 256 --check -
}
fetch "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz" "$WORK/sources/ffmpeg-$FFMPEG_VERSION.tar.xz" "$FFMPEG_SHA256"
fetch "https://files.pythonhosted.org/packages/source/a/av/av-$PYAV_VERSION.tar.gz" "$WORK/sources/av-$PYAV_VERSION.tar.gz" "$PYAV_SHA256"
rm -rf -- "$WORK/ffmpeg-$FFMPEG_VERSION" "$WORK/av-$PYAV_VERSION" "$WORK/ffmpeg-install" "$WORK/wheels"
mkdir -p "$WORK/wheels"
tar -xf "$WORK/sources/ffmpeg-$FFMPEG_VERSION.tar.xz" -C "$WORK"
tar -xf "$WORK/sources/av-$PYAV_VERSION.tar.gz" -C "$WORK"

PREFIX="$WORK/ffmpeg-install"
FLAGS=(
  "--prefix=$PREFIX" '--install-name-dir=@rpath' '--arch=arm64'
  '--enable-shared' '--disable-static' '--disable-programs' '--disable-doc'
  '--disable-network' '--disable-autodetect' '--disable-everything'
  '--enable-avcodec' '--enable-avformat' '--enable-avdevice' '--enable-avfilter'
  '--enable-swresample' '--enable-swscale'
  '--enable-protocol=file,pipe'
  '--enable-demuxer=aac,aiff,flac,matroska,mov,mp3,ogg,wav'
  '--enable-decoder=aac,flac,mp3,mp3float,opus,pcm_alaw,pcm_f32be,pcm_f32le,pcm_f64be,pcm_f64le,pcm_mulaw,pcm_s16be,pcm_s16le,pcm_s24be,pcm_s24le,pcm_s32be,pcm_s32le,pcm_u8'
  '--enable-parser=aac,flac,mpegaudio,opus'
  '--enable-encoder=flac' '--enable-muxer=flac'
  '--enable-filter=abuffer,abuffersink,aformat,aresample'
)
printf '%s\n' "${FLAGS[@]}" > "$WORK/configure-flags.txt"
cd "$WORK/ffmpeg-$FFMPEG_VERSION"
./configure "${FLAGS[@]}"
make -j "$(sysctl -n hw.ncpu)"
make install
test "$(cat "$PREFIX/include/libavutil/ffversion.h" | grep -c 'FFMPEG_VERSION')" -gt 0
if grep -E -- '--enable-(gpl|nonfree|version3|libx26[45])' ffbuild/config.mak; then
  echo 'Forbidden FFmpeg configure flag' >&2; exit 1
fi

uv venv --clear --python "$PYTHON" "$WORK/build-venv"
uv pip install --python "$WORK/build-venv/bin/python" 'Cython==3.0.12' 'setuptools==75.8.0' 'wheel==0.45.1'
cd "$WORK/av-$PYAV_VERSION"
MACOSX_DEPLOYMENT_TARGET=14.0 ARCHFLAGS='-arch arm64' \
  "$WORK/build-venv/bin/python" setup.py "--ffmpeg-dir=$PREFIX" bdist_wheel --dist-dir "$WORK/wheels"

# The wheel and corresponding FFmpeg dylibs remain separate and replaceable.
# build_macos.py installs this wheel after the hashed production sync.
echo "Built $WORK/wheels/av-$PYAV_VERSION-*.whl"
