"""Unit tests for ojin.stv.output (sink + async-iterator adapter)."""

import asyncio

from ojin.stv.frames import STVAudioFrame, STVVideoFrame
from ojin.stv.output import QueueOutput


def _audio(pts: int) -> STVAudioFrame:
    return STVAudioFrame(pcm=b"\x00\x00", sample_rate=16000, num_channels=1, pts=pts)


def _video(pts: int) -> STVVideoFrame:
    return STVVideoFrame(
        rgb=None, source_bytes=bytes([pts]), width=1, height=1, frame_type=1, pts=pts
    )


def test_queue_output_yields_writes_in_order() -> None:
    """Both audio and video writes surface through the single stream, in order."""

    async def run() -> None:
        q = QueueOutput(max_video=4)
        await q.write_audio(_audio(0))
        await q.write_video(_video(1))
        it = q.stream()
        got = [await it.__anext__(), await it.__anext__()]
        assert isinstance(got[0], STVAudioFrame)
        assert isinstance(got[1], STVVideoFrame)

    asyncio.run(run())


def test_queue_output_drops_oldest_video_on_overflow() -> None:
    """Over max_video, the oldest video is dropped; audio is never dropped."""

    async def run() -> None:
        q = QueueOutput(max_video=2)
        for i in range(5):
            await q.write_video(_video(i))
        vids = []
        it = q.stream()
        try:
            while True:
                vids.append(await asyncio.wait_for(it.__anext__(), 0.05))
        except asyncio.TimeoutError:
            pass
        assert [v.pts for v in vids] == [3, 4]  # oldest three dropped

    asyncio.run(run())


def test_queue_output_keeps_all_audio() -> None:
    """Audio is retained even past the video cap."""

    async def run() -> None:
        q = QueueOutput(max_video=1)
        for i in range(5):
            await q.write_audio(_audio(i))
        out = []
        it = q.stream()
        try:
            while True:
                out.append(await asyncio.wait_for(it.__anext__(), 0.05))
        except asyncio.TimeoutError:
            pass
        assert [a.pts for a in out] == [0, 1, 2, 3, 4]

    asyncio.run(run())
