"""Realtime Speech-To-Video: your mic drives an Ojin avatar, live in the browser.

Run:
    pip install -r requirements.txt
    python server.py            # then open http://localhost:8000

The browser captures your mic (16 kHz mono) and webcam. This backend runs
voice-activity detection (VAD) on the mic stream: when you finish an utterance it
sends that audio to the Ojin avatar, which speaks it back lip-synced; when you
start talking again it interrupts the avatar (barge-in). Avatar audio + video are
streamed back to the browser.
"""

import asyncio
import pathlib
import sys
from collections import deque
from contextlib import suppress

import uvicorn
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ojin import MissingCredentialsError, load_env, resolve_credentials
from ojin.stv import OjinSTVClient, STVAudioFrame, STVEvent, STVVideoFrame

RATE = 16000  # everything (mic, VAD, avatar audio) runs at 16 kHz mono
HERE = pathlib.Path(__file__).parent

load_env(base_dir=HERE)  # optional .env beside this file
try:
    CREDS = resolve_credentials(load_env_file=False)
except MissingCredentialsError as exc:
    sys.exit(str(exc))


class VadSegmenter:
    """Split a 16 kHz mono int16 stream into utterances using WebRTC VAD.

    Feed it raw PCM with ``process()``; it yields ``("start", None)`` when speech
    begins and ``("end", pcm_bytes)`` when it ends.
    """

    def __init__(
        self,
        rate: int = RATE,
        aggressiveness: int = 3,
        frame_ms: int = 20,
        start_frames: int = 3,
        end_frames: int = 50,
    ) -> None:
        """Configure the segmenter (end_frames * frame_ms = trailing silence)."""
        super().__init__()
        self._vad = webrtcvad.Vad(aggressiveness)
        self._rate = rate
        self._frame_bytes = int(rate * frame_ms / 1000) * 2  # 20 ms @ 16 kHz = 640 B
        self._start_frames = start_frames
        self._end_frames = end_frames
        self._buf = b""
        self._preroll: deque[bytes] = deque(maxlen=start_frames)
        self._triggered = False
        self._voiced = 0
        self._silence = 0
        self._utterance = bytearray()

    def process(self, pcm: bytes) -> list[tuple[str, bytes | None]]:
        """Consume a PCM chunk and return any speech start/end events."""
        self._buf += pcm
        events: list[tuple[str, bytes | None]] = []
        while len(self._buf) >= self._frame_bytes:
            frame, self._buf = (
                self._buf[: self._frame_bytes],
                self._buf[self._frame_bytes :],
            )
            speech = self._vad.is_speech(frame, self._rate)
            if not self._triggered:
                self._preroll.append(frame)
                self._voiced = self._voiced + 1 if speech else 0
                if self._voiced >= self._start_frames:
                    self._triggered = True
                    self._utterance = bytearray(b"".join(self._preroll))  # keep onset
                    self._voiced = self._silence = 0
                    events.append(("start", None))
            else:
                self._utterance += frame
                self._silence = 0 if speech else self._silence + 1
                if self._silence >= self._end_frames:
                    self._triggered = False
                    events.append(("end", bytes(self._utterance)))
                    self._utterance = bytearray()
                    self._silence = 0
        return events


app = FastAPI()


@app.get("/")
async def index() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.websocket("/ws")
async def avatar(ws: WebSocket) -> None:
    """Bridge one browser session to one Ojin avatar session."""
    await ws.accept()
    client = OjinSTVClient(
        api_key=CREDS.api_key, config_id=CREDS.config_id, image_size=(512, 512)
    )

    # The pump task and the error relay both write this socket, so funnel every
    # browser-bound message through one lock-guarded sender.
    send_lock = asyncio.Lock()

    async def send(payload: bytes | str) -> None:
        """Send one message to the browser, serialized and failure-tolerant."""
        async with send_lock:
            with suppress(Exception):
                if isinstance(payload, str):
                    await ws.send_text(payload)
                else:
                    await ws.send_bytes(payload)

    async def on_error(message: str = "", **_: object) -> None:
        """Relay a fatal Ojin error to the browser."""
        await send(f"Ojin error: {message}")

    client.add_listener(STVEvent.ERROR, on_error)
    await client.start()

    async def forward() -> None:
        """Stream avatar audio (0x00) + JPEG video (0x01) to the browser."""
        async for frame in client.output_stream():
            if isinstance(frame, STVVideoFrame):
                # Held/repeat ticks carry empty source_bytes; skip them and the
                # browser simply keeps showing the last frame it received.
                if frame.source_bytes:
                    await send(b"\x01" + frame.source_bytes)  # raw JPEG
            elif isinstance(frame, STVAudioFrame):
                await send(b"\x00" + frame.pcm)  # int16 PCM @ 16 kHz

    pump = asyncio.create_task(forward())
    vad = VadSegmenter()
    try:
        while True:
            chunk = await ws.receive_bytes()  # mic PCM: 16 kHz mono int16
            if not client.is_connected:
                continue  # session not ready yet — ignore early mic audio
            for event, utterance in vad.process(chunk):
                if event == "start":
                    await client.interrupt()  # barge-in (no-op if idle)
                elif utterance:
                    await client.say(utterance, RATE, 1)  # avatar speaks your audio
    except WebSocketDisconnect:
        pass
    finally:
        pump.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await pump
        await client.close()


if __name__ == "__main__":
    print("  Ready. Open http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
