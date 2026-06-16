"""Stitch avatar RGB video frames and the source WAV into one MP4 (with audio).

Uses imageio-ffmpeg, which ships its own ffmpeg binary (no system install needed):
raw RGB frames are streamed straight into the encoder and the original WAV is muxed
in as the audio track, so the result is a single MP4 with both picture and sound.
"""

from __future__ import annotations

import pathlib

import imageio_ffmpeg


class Mp4Writer:
    """Encode RGB24 frames to H.264 and mux a WAV file in as the audio track."""

    def __init__(
        self,
        path: pathlib.Path,
        width: int,
        height: int,
        fps: int,
        audio_wav: pathlib.Path,
    ) -> None:
        """Open the encoder for `width`x`height` @ `fps`, muxing in `audio_wav`."""
        self.frames = 0
        self._writer = imageio_ffmpeg.write_frames(
            str(path),
            (width, height),
            fps=fps,
            pix_fmt_in="rgb24",  # exactly what STVVideoFrame.rgb already is
            audio_path=str(audio_wav),
            audio_codec="aac",
            output_params=["-shortest"],  # end at whichever track finishes first
        )
        self._writer.send(None)  # prime the generator (starts ffmpeg)

    def write(self, rgb: bytes) -> None:
        """Append one RGB24 frame (width * height * 3 bytes)."""
        self._writer.send(rgb)
        self.frames += 1

    def close(self) -> None:
        """Flush and finalize the MP4."""
        self._writer.close()
