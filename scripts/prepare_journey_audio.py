"""Build >=31-second synthetic upload fixtures without altering shipped samples.

Repeated authored TTS is suitable for codec/storage UI tests, never CEFR accuracy.
"""
from pathlib import Path
import wave


def prepare(destination: Path, source_root: Path | None = None):
    root = source_root or Path(__file__).resolve().parents[1] / 'samples/cefr'
    for source in root.rglob('*.wav'):
        with wave.open(str(source),'rb') as original:
            params=original.getparams(); frames=original.readframes(original.getnframes())
        frame_size=params.nchannels*params.sampwidth
        wanted=params.framerate*31
        if not frames: raise ValueError('Empty synthetic sample')
        repeated=(frames*((wanted*frame_size+len(frames)-1)//len(frames)))[:wanted*frame_size]
        target=destination/source.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        with wave.open(str(target),'wb') as output:
            output.setparams(params);output.writeframes(repeated)
