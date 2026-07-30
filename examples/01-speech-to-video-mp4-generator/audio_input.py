"""Load speech audio from any common file (WAV, MP3, M4A, ...) as mono 16-bit PCM.

A mono 16-bit WAV is read straight from the stdlib. Everything else — an MP3, an
M4A, a stereo or 24-bit WAV — is decoded by the ffmpeg binary that ships with
imageio-ffmpeg (the same dependency `mp4_writer.py` already uses, so there is
nothing extra to install).
"""

from __future__ import annotations

import pathlib
import subprocess
import wave

import imageio_ffmpeg

TARGET_RATE = 16000  # what the Face model runs at; ffmpeg resamples to it


class AudioDecodeError(Exception):
    """Raised when ffmpeg cannot turn the input file into PCM audio."""


def load_audio(path: pathlib.Path) -> tuple[bytes, int]:
    """Read `path` as (pcm_bytes, sample_rate), mono 16-bit, decoding if needed."""
    already_pcm = _read_mono_wav(path)
    if already_pcm is not None:
        return already_pcm
    return _decode(path), TARGET_RATE


def _read_mono_wav(path: pathlib.Path) -> tuple[bytes, int] | None:
    """Return the PCM of an already-mono 16-bit WAV, or None to let ffmpeg do it."""
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                return None  # stereo / 24-bit: ffmpeg down-mixes and requantizes
            return wav.readframes(wav.getnframes()), wav.getframerate()
    except (wave.Error, EOFError):
        return None  # not a WAV at all (MP3, M4A, ...)


def _decode(path: pathlib.Path) -> bytes:
    """Decode any ffmpeg-readable file to raw mono 16-bit PCM at `TARGET_RATE`."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",  # ignore any video track (e.g. an MP4 dropped in)
        "-ac",
        "1",
        "-ar",
        str(TARGET_RATE),
        "-f",
        "s16le",  # raw samples on stdout, no container
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioDecodeError(proc.stderr.decode(errors="replace").strip())
    if not proc.stdout:
        raise AudioDecodeError("the file decoded to zero samples (no audio track?)")
    return proc.stdout
